"""Trade selection (architecture §5) with the fixed-holding-period exit rule
(exit_rule_spec.md): rankings -> a budget-constrained list of entry AND exit
events.

A position entered on period `d` is closed on period `d + hold_days`,
unconditionally — no stop-loss, no take-profit, no signal-reversal exit
(those are deferred to the ablation table, spec §7). What this module
enforces:

- long-only (no short selling, no margin)
- an entry must clear round-trip cost by ``min_edge_margin`` before it's
  worth acting on (``config/cost_model.py``) — cost is charged once, at
  entry, since the exit is already committed and mechanical
- no entering a ticker already held, and no re-entering a ticker on the
  same period it just exited
- equal-weight position sizing, capped at ``position_cap_pct``
- budget accounting per spec §3: every entry obligates a future exit, so
  the 200-trade cap is ~100 round trips, not 200 opportunities
- an entry cutoff (spec §5) so nothing is entered too late to exit inside
  the competition window, plus a force-close safety net (logged, since it
  firing means the cutoff didn't do its job)

Sizing here is a *recommendation*, not an executed allocation — WInS has no
API, so a human sizes and places every real order (CLAUDE.md).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from config import cost_model
from src.config import TradingConfig
from src.selection.portfolio_state import PortfolioState, empty_portfolio

_EVENT_COLUMNS = [
    "ticker",
    "timestamp",
    "action",
    "period",
    "entry_period",
    "scheduled_exit_period",
    "position_size_pct",
    "combined_score",
    "rank",
    "forced",
]


@dataclass
class OpenPosition:
    entry_period: int
    #: None means unknown (e.g. seeded from a real position with no
    #: scheduled_exit_date on file) — it will never auto-fire an exit, only
    #: block re-entry, until force-closed at the end of the run.
    scheduled_exit_period: Optional[int]
    position_size_pct: float


class TradeSelector:
    def __init__(self, config: TradingConfig, portfolio: PortfolioState | None = None):
        self.config = config
        self.portfolio = portfolio if portfolio is not None else empty_portfolio()
        self.hold_days = config.hold_days
        self.window_trading_days = config.window_trading_days
        self._executed = 0  # E: entries + exits executed so far
        self._open: dict[str, OpenPosition] = {}
        self._just_exited: set[str] = set()
        self.force_close_log: list[dict] = []
        self._log: list[dict] = []

    @property
    def trades_taken(self) -> int:
        """E — entries and exits both count against the budget (spec §3)."""
        return self._executed

    @property
    def remaining_budget(self) -> int:
        """R = TRADE_BUDGET - E."""
        return self.config.trade_budget - self._executed

    @property
    def open_positions_count(self) -> int:
        """P — currently open positions."""
        return len(self._open)

    # -- eligibility -----------------------------------------------------

    def _eligible(self, period_ranked: pd.DataFrame) -> pd.DataFrame:
        """Rows it's legal to enter: long-only, cost-clearing, not already
        open, not exited this same period (spec §6 same-day re-entry ban)."""
        eligible = period_ranked
        if not self.config.allow_short:
            eligible = eligible[eligible["combined_score"] > 0]

        # Cost charged once (entry only) but at the round-trip rate — the
        # exit is already mechanically committed, not a second decision.
        cost_floor = self.config.min_edge_margin * cost_model.round_trip_cost_fraction()
        eligible = eligible[eligible["combined_score"].abs() > cost_floor]

        if eligible.empty:
            return eligible
        blocked = self._open.keys() | self._just_exited
        excluded = eligible.index.get_level_values("ticker").isin(blocked)
        return eligible[~excluded]

    def _size_positions(self, selected: pd.DataFrame) -> pd.Series:
        """Equal-weight, capped at ``position_cap_pct`` (architecture §5).
        Without a confirmed cap, falls back to equal-weight within this
        period's batch — a placeholder, not a real capital allocation."""
        if selected.empty:
            return pd.Series(dtype=float, index=selected.index)
        cap = self.config.position_cap_pct
        size = cap if cap is not None else 1.0 / len(selected)
        return pd.Series(size, index=selected.index)

    # -- per-period mechanics ---------------------------------------------

    def _process_exits(self, period_number: int, timestamp) -> list[dict]:
        due = [ticker for ticker, pos in self._open.items() if pos.scheduled_exit_period == period_number]
        rows = []
        for ticker in due:
            position = self._open.pop(ticker)
            self._executed += 1
            rows.append(self._event(ticker, timestamp, "exit", period_number, position))
        self._just_exited = set(due)
        return rows

    def _entries_allowed(self) -> int:
        """spec §3-4: budget reserves one exit-unit per open position, and
        each new entry obligates its own future exit (2 units); capital
        additionally caps concurrent positions via position_cap_pct."""
        remaining, open_count = self.remaining_budget, self.open_positions_count
        free = max(0, remaining - open_count)
        by_budget = free // 2

        cap = self.config.position_cap_pct
        by_capital = math.floor(1 / cap) - open_count if cap else self.config.top_n_per_period
        return max(0, min(by_budget, by_capital, self.config.top_n_per_period))

    def _process_entries(self, period_number: int, timestamp, period_ranked: pd.DataFrame) -> list[dict]:
        # Entry cutoff (spec §5): a position entered here must be able to
        # exit by window close, d + hold_days <= window_trading_days.
        if period_number > self.window_trading_days - self.hold_days:
            return []

        n = min(self._entries_allowed(), len(period_ranked))
        if n <= 0:
            return []
        eligible = self._eligible(period_ranked)
        n = min(n, len(eligible))
        if n <= 0:
            return []

        selected = eligible.sort_values("rank").iloc[:n]
        sizes = self._size_positions(selected)

        rows = []
        for (ticker, ts), row in selected.iterrows():
            exit_period = period_number + self.hold_days
            position = OpenPosition(
                entry_period=period_number,
                scheduled_exit_period=exit_period,
                position_size_pct=float(sizes.loc[(ticker, ts)]),
            )
            self._open[ticker] = position
            self._executed += 1
            rows.append(
                self._event(
                    ticker, ts, "entry", period_number, position,
                    combined_score=row["combined_score"], rank=row["rank"],
                )
            )
        return rows

    def _force_close_remaining(self, period_number: int, timestamp, reason: str) -> list[dict]:
        rows = []
        for ticker in list(self._open.keys()):
            position = self._open.pop(ticker)
            self._executed += 1
            self.force_close_log.append(
                {
                    "ticker": ticker,
                    "scheduled_exit_period": position.scheduled_exit_period,
                    "forced_at_period": period_number,
                    "reason": reason,
                }
            )
            rows.append(self._event(ticker, timestamp, "exit", period_number, position, forced=True))
        return rows

    def _event(
        self,
        ticker,
        timestamp,
        action: str,
        period_number: int,
        position: OpenPosition,
        combined_score: float = float("nan"),
        rank: float = float("nan"),
        forced: bool = False,
    ) -> dict:
        return {
            "ticker": ticker,
            "timestamp": timestamp,
            "action": action,
            "period": period_number,
            "entry_period": position.entry_period,
            "scheduled_exit_period": position.scheduled_exit_period,
            "position_size_pct": position.position_size_pct,
            "combined_score": combined_score,
            "rank": rank,
            "forced": forced,
        }

    def _assert_invariant(self) -> None:
        e, p = self._executed, self.open_positions_count
        if e + p > self.config.trade_budget:
            raise RuntimeError(
                f"invariant violated: E({e}) + P({p}) > TRADE_BUDGET({self.config.trade_budget})"
            )

    def _seed_from_portfolio(self, timestamps: list) -> None:
        """Real starting holdings block re-entry for the whole run; a
        resolvable scheduled_exit_date also lets them exit naturally within
        this run (spec §6)."""
        for ticker, real_position in self.portfolio.positions.items():
            exit_period = None
            if real_position.scheduled_exit_date:
                try:
                    exit_ts = pd.Timestamp(real_position.scheduled_exit_date)
                    idx = int(np.searchsorted(timestamps, exit_ts, side="left"))
                    if idx < len(timestamps):
                        exit_period = idx + 1  # 1-indexed periods
                except (ValueError, TypeError):
                    exit_period = None
            self._open[ticker] = OpenPosition(
                entry_period=0, scheduled_exit_period=exit_period, position_size_pct=0.0
            )

    # -- entry point --------------------------------------------------------

    def run(self, ranked: pd.DataFrame) -> pd.DataFrame:
        """Walk every period in ``ranked`` (grouped by timestamp,
        chronologically, 1-indexed as trading days) processing due exits
        then new entries, until the run ends or periods are exhausted.
        Force-closes anything still open when the run ends.
        """
        timestamps = sorted(pd.unique(ranked.index.get_level_values("timestamp")))
        self._seed_from_portfolio(timestamps)

        all_rows: list[dict] = []
        last_period, last_timestamp = 0, None
        for period_number, timestamp in enumerate(timestamps, start=1):
            last_period, last_timestamp = period_number, timestamp
            period_rows = ranked.xs(timestamp, level="timestamp", drop_level=False)

            all_rows.extend(self._process_exits(period_number, timestamp))
            all_rows.extend(self._process_entries(period_number, timestamp, period_rows))
            self._assert_invariant()

        if self._open:
            # Expected only if the window's real day 50 was reached and the
            # entry cutoff somehow let a non-exitable position through
            # (spec §5: "if it fires, log it"), or if `ranked` simply ran
            # out of periods before every open position's scheduled exit.
            reason = (
                "cutoff was not enforced"
                if last_period >= self.window_trading_days
                else f"data ended at period {last_period}, before window close ({self.window_trading_days})"
            )
            all_rows.extend(self._force_close_remaining(last_period, last_timestamp, reason))

        self._log = all_rows
        return self._rows_to_frame(all_rows)

    @staticmethod
    def _rows_to_frame(rows: list[dict]) -> pd.DataFrame:
        if not rows:
            empty = pd.DataFrame(columns=_EVENT_COLUMNS[2:])
            empty.index = pd.MultiIndex.from_tuples([], names=["ticker", "timestamp"])
            return empty
        frame = pd.DataFrame(rows, columns=_EVENT_COLUMNS)
        return frame.set_index(["ticker", "timestamp"])
