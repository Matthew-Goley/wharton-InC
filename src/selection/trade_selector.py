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

#: Generous, deliberately loose absolute-fraction gap allowed between
#: budget-consumed and window-elapsed before _assert_pacing_invariant
#: fires -- a coarse safety net against gross front-loading (the bug this
#: was added for hit 80 percentage points: 100% consumed at ~20% elapsed),
#: not a precision pacing target. Derived pacing should normally keep this
#: far tighter in realistic configurations.
_PACING_TOLERANCE = 0.25

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


@dataclass
class EligibilityStats:
    """Per-filter-stage rejection counts, accumulated across every period
    _eligible() actually runs on (a period with zero pacing-derived entry
    slots never calls it at all -- see _process_entries -- so these
    reflect candidates actually evaluated, not literally every period in
    the window).

    A funnel, not independent totals: each candidate is counted at the
    FIRST stage that rejects it, then removed from consideration, so a
    row rejected for direction is never also double-counted against cost.
    This is what makes "how many did the cost filter actually reject"
    (as opposed to "how many entries got made") answerable at all --
    TradeSelector's own output (the trades frame) only ever shows what
    was ADMITTED, never what was screened out or why.
    """

    considered: int = 0
    rejected_direction: int = 0  # long-only: combined_score <= 0
    rejected_cost: int = 0  # |combined_score| <= cost floor
    rejected_earnings: int = 0  # known earnings date inside the hold window
    rejected_blocked: int = 0  # already open, just exited, or frequency-capped


class TradeSelector:
    def __init__(
        self,
        config: TradingConfig,
        portfolio: PortfolioState | None = None,
        trades_executed_so_far: int = 0,
        days_to_next_earnings: Optional[pd.Series] = None,
    ):
        self.config = config
        self.portfolio = portfolio if portfolio is not None else empty_portfolio()
        self.hold_days = config.hold_days
        self.window_trading_days = config.window_trading_days
        #: (ticker, timestamp)-indexed, e.g. table.frame["days_to_next_
        #: earnings"] (RealDataProvider._earnings_features_for_ticker) --
        #: kept as a SEPARATE aligned series rather than baked into
        #: `ranked`, the same pattern OverallModel.rank(risk=...) already
        #: uses for volatility_20. Not a ranking feature (tested and
        #: nulled for that, baseline_system_architecture.md open question
        #: 24) -- an EVENT-RISK eligibility control, settled two rounds
        #: before this was built: excludes an entry only when an
        #: announcement is KNOWN to fall inside the hold window, see
        #: _eligible.
        self.days_to_next_earnings = days_to_next_earnings
        #: Per-filter-stage rejection counts, see EligibilityStats.
        self.eligibility_stats = EligibilityStats()
        #: period_number -> the derived pacing allowance actually used
        #: that period (entries_allowed's return value, snapshotted at
        #: the moment _process_entries computed it -- see there for why
        #: this can't just be recomputed after the fact).
        self.period_pacing: dict[int, int] = {}
        #: E: entries + exits executed so far. 0 by default (every
        #: existing backtest call site unchanged) -- a live run seeds
        #: this from PortfolioState.trades_executed_so_far
        #: (src.selection.portfolio_state's trade_log), since a single
        #: day's ranked frame passed to run() carries no memory of trades
        #: already made earlier this season (open question 27).
        self._executed = trades_executed_so_far
        self._open: dict[str, OpenPosition] = {}
        self._just_exited: set[str] = set()
        self.force_close_log: list[dict] = []
        self._log: list[dict] = []

        # Per-ticker entry frequency cap (see TradingConfig.max_ticker_share_pct).
        # `trade_budget // 2` is the total-entries denominator: a stable
        # upper bound known before the run starts (the true final entry
        # count depends on eligibility as the run unfolds, so it can't be
        # the denominator the cap is computed from).
        self._entry_counts: dict[str, int] = {}
        self._max_entries_per_ticker: Optional[int] = (
            max(1, math.floor(config.max_ticker_share_pct * (config.trade_budget // 2)))
            if config.max_ticker_share_pct is not None
            else None
        )

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
        open, not exited this same period (spec §6 same-day re-entry ban),
        no known earnings announcement inside the hold window.

        Updates self.eligibility_stats as it goes -- a funnel, so each
        candidate is counted at the first stage that rejects it (see
        EligibilityStats)."""
        eligible = period_ranked
        self.eligibility_stats.considered += len(period_ranked)

        if not self.config.allow_short:
            before = len(eligible)
            eligible = eligible[eligible["combined_score"] > 0]
            self.eligibility_stats.rejected_direction += before - len(eligible)

        # Cost charged once (entry only) but at the round-trip rate — the
        # exit is already mechanically committed, not a second decision.
        cost_floor = self.config.min_edge_margin * cost_model.round_trip_cost_fraction()
        before = len(eligible)
        eligible = eligible[eligible["combined_score"].abs() > cost_floor]
        self.eligibility_stats.rejected_cost += before - len(eligible)

        if self.days_to_next_earnings is not None and not eligible.empty:
            # Event-risk control, not a return predictor -- settled two
            # rounds before this was built (days_to_next_earnings was
            # tested and nulled as a RANKING feature, baseline_system_
            # architecture.md open question 24; its "more natural role...
            # eligibility control" was the explicit conclusion there).
            # Excludes an entry whose next known announcement falls on or
            # before the day it would exit (<=, not < -- a position
            # entered day d and exited day d+hold_days is still open
            # through that exit day's own session, so an announcement
            # exactly on the exit day is still inside the exposure
            # window, not outside it).
            #
            # NaN (no resolvable next-earnings date -- no earnings data
            # for this ticker at all, e.g. the ~8% delisted/acquired
            # population, or past the known calendar's range) is
            # deliberately NOT excluded. Unknown isn't risky, it's
            # unknown -- treating every unmapped ticker as ineligible
            # would silently block ~8% of the universe from ever being
            # entered, for a reason unrelated to actual event risk. A
            # different missingness convention than vol_adjusted's own
            # fillna(max) (that's a RANKING penalty, where "assume the
            # worst" is safe; this is a hard ELIGIBILITY gate, where the
            # same assumption would be a severe, disproportionate cost).
            before = len(eligible)
            dte = self.days_to_next_earnings.reindex(eligible.index)
            earnings_in_window = dte.notna() & (dte <= self.hold_days)
            eligible = eligible[~earnings_in_window]
            self.eligibility_stats.rejected_earnings += before - len(eligible)

        if eligible.empty:
            return eligible
        blocked = self._open.keys() | self._just_exited
        if self._max_entries_per_ticker is not None:
            blocked = blocked | {
                t for t, n in self._entry_counts.items() if n >= self._max_entries_per_ticker
            }
        before = len(eligible)
        excluded = eligible.index.get_level_values("ticker").isin(blocked)
        eligible = eligible[~excluded]
        self.eligibility_stats.rejected_blocked += before - len(eligible)
        return eligible

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
        # <=, not == : exact equality is the wrong tool for "has a
        # monotonically-increasing counter reached or passed a
        # threshold" in general -- a real, latent bug in this function on
        # its own (open question 27), independent of whether live mode
        # exists. It happened to never matter before because the only
        # caller (a full-window backtest) always walks every period in
        # contiguous order, so == and <= could never actually disagree.
        # A live run breaks that assumption: it can be skipped on the
        # exact day a position was due out (the team doesn't run the
        # script that day), and under == that exit would then silently
        # never fire on any LATER run either, since period_number has
        # already moved past it. <= fires it as soon as a run catches up.
        due = [
            ticker for ticker, pos in self._open.items()
            if pos.scheduled_exit_period is not None and pos.scheduled_exit_period <= period_number
        ]
        rows = []
        for ticker in due:
            position = self._open.pop(ticker)
            self._executed += 1
            rows.append(self._event(ticker, timestamp, "exit", period_number, position))
        self._just_exited = set(due)
        return rows

    def entries_allowed(self, period_number: int) -> int:
        """spec §3-4: budget reserves one exit-unit per open position, and
        each new entry obligates its own future exit (2 units); capital
        additionally caps concurrent positions via position_cap_pct.

        Per-period pacing is DERIVED — remaining entry capacity divided by
        remaining entry-eligible periods — not a fixed ceiling. A fixed
        ``top_n_per_period`` was found (real full-sector-coverage backtest)
        to blow through an entire multi-year run's budget in the first
        ~100 of 497 days the moment eligible candidates became abundant
        (as few as one sector expert blended in was enough), leaving ~19
        months of the window completely untested — and along the way,
        meant the selector was taking essentially every eligible candidate
        rather than choosing among many, so the thing actually under test
        (selection quality, not just eligibility) was never exercised.
        This was flagged as a known divergence from the architecture doc's
        own spec ("paces itself...") from very early in this project;
        deriving pacing here is that deferred fix landing, not new scope.

        Self-correcting: recomputed fresh every period from the CURRENT
        remaining budget, so a period with no eligible candidates doesn't
        lose its share — the unspent capacity stays in remaining_budget,
        pushing the derived rate for whatever periods are left back up.

        ``TradingConfig.top_n_per_period``, when explicitly set, still acts
        as an additional hard ceiling on top of the derived rate — off
        (None) by default, since the derived rate is now what governs
        pacing in the realistic (non-degenerate) case.
        """
        remaining, open_count = self.remaining_budget, self.open_positions_count
        free = max(0, remaining - open_count)
        by_budget = free // 2

        cap = self.config.position_cap_pct
        by_capital = math.floor(1 / cap) - open_count if cap else by_budget

        entry_periods_left = max(1, (self.window_trading_days - self.hold_days) - period_number + 1)
        derived = round(by_budget / entry_periods_left)

        n = min(by_budget, by_capital, derived)
        if self.config.top_n_per_period is not None:
            n = min(n, self.config.top_n_per_period)
        return max(0, n)

    def _process_entries(self, period_number: int, timestamp, period_ranked: pd.DataFrame) -> list[dict]:
        # Entry cutoff (spec §5): a position entered here must be able to
        # exit by window close, d + hold_days <= window_trading_days.
        if period_number > self.window_trading_days - self.hold_days:
            return []

        # Recorded at the moment it's computed, not re-derivable
        # afterward: calling entries_allowed(period_number) again once
        # this period's entries have already been processed would read
        # post-period state (budget already spent), not the allowance
        # that actually governed this period's decision. A caller (e.g.
        # score_live.py, reporting "what pacing said this period") needs
        # this snapshot, not a recomputation.
        allowed = self.entries_allowed(period_number)
        self.period_pacing[period_number] = allowed
        n = min(allowed, len(period_ranked))
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
            self._entry_counts[ticker] = self._entry_counts.get(ticker, 0) + 1
            rows.append(
                self._event(
                    ticker, ts, "entry", period_number, position,
                    combined_score=row["combined_score"], rank=row["rank"],
                )
            )
        return rows

    def _force_close_remaining(
        self, period_number: int, timestamp, reason: str, tickers: Optional[list[str]] = None
    ) -> list[dict]:
        """Force-close ``tickers`` (default: everything currently open --
        every existing call site's unchanged behavior). A live-mode run
        end (open question 27) passes a narrower subset -- only positions
        with no resolvable scheduled_exit_period -- rather than closing
        every genuinely-still-pending position just because today's
        single-day call ended."""
        rows = []
        for ticker in list(tickers) if tickers is not None else list(self._open.keys()):
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

    def _assert_pacing_invariant(self, period_number: int) -> None:
        """General form of the bug that motivated derived pacing (see
        entries_allowed's docstring): any resource-constrained simulation
        should track its resource consumption against elapsed time, not
        just check it never exceeds the final ceiling. This is deliberately
        ONE-SIDED — it only fires when budget is being consumed FASTER than
        the window is elapsing (front-loading, the actual bug found: 100%
        of a 497-day budget spent by day ~100). Under-consumption
        (consumed_frac < elapsed_frac) never fires here — that's a
        legitimate, already-documented open question (architecture doc
        open question 7: was the selector too conservative, or correctly
        selective), not a pacing bug, and firing on it would also make
        this trip on any sparse/short synthetic test that never intends to
        spend its whole budget.
        """
        if not self.config.trade_budget or not self.window_trading_days:
            return
        consumed_frac = self._executed / self.config.trade_budget
        elapsed_frac = period_number / self.window_trading_days
        if consumed_frac - elapsed_frac > _PACING_TOLERANCE:
            raise RuntimeError(
                f"pacing invariant violated at period {period_number}/{self.window_trading_days}: "
                f"budget {consumed_frac:.1%} consumed vs window {elapsed_frac:.1%} elapsed "
                f"(tolerance {_PACING_TOLERANCE:.0%}) -- entries are front-loading faster than "
                f"the window is elapsing, the same failure mode derived pacing exists to prevent."
            )

    def _seed_from_portfolio(
        self,
        timestamps: list,
        starting_period_number: int = 1,
        today: Optional[pd.Timestamp] = None,
    ) -> None:
        """Real starting holdings block re-entry for the whole run; a
        resolvable scheduled_exit_date also lets them exit naturally within
        this run (spec §6).

        Two ways to resolve a real position's ``scheduled_exit_date`` to a
        period number, chosen by whether ``today`` is supplied:

        - ``today is None`` (default -- every existing backtest call site,
          UNCHANGED behavior): frame-relative -- find where the exit date
          falls among ``timestamps``, the dates actually present in this
          call's ``ranked`` frame. Correct for a full-window backtest,
          where ``timestamps`` spans the whole simulated window.
        - ``today`` supplied (live-mode, open question 27): calendar-
          relative -- ``starting_period_number + busday_count(today,
          exit_date)``, independent of what's in ``timestamps`` at all.
          This is the only correct option for a live call, whose
          ``ranked`` frame is typically a single day -- the frame-relative
          approach would read every real exit date as "period 1" (the
          only date in the frame), regardless of how far away it actually
          is. An exit date already in the past relative to ``today``
          yields a period number BELOW ``starting_period_number`` on
          purpose, not clamped -- ``_process_exits``'s ``<=`` check (not
          ``==``) fires it immediately on this run rather than silently
          losing it, exactly the overdue-exit case that fix exists for.
        """
        for ticker, real_position in self.portfolio.positions.items():
            exit_period = None
            if real_position.scheduled_exit_date:
                try:
                    exit_ts = pd.Timestamp(real_position.scheduled_exit_date)
                    if today is not None:
                        offset = int(np.busday_count(today.date(), exit_ts.date()))
                        exit_period = starting_period_number + offset
                    else:
                        idx = int(np.searchsorted(timestamps, exit_ts, side="left"))
                        if idx < len(timestamps):
                            exit_period = idx + 1  # 1-indexed periods
                except (ValueError, TypeError):
                    exit_period = None
            self._open[ticker] = OpenPosition(
                entry_period=0, scheduled_exit_period=exit_period, position_size_pct=0.0
            )

    # -- entry point --------------------------------------------------------

    def run(
        self,
        ranked: pd.DataFrame,
        starting_period_number: int = 1,
        today: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        """Walk every period in ``ranked`` (grouped by timestamp,
        chronologically) processing due exits then new entries, until the
        run ends or periods are exhausted. Force-closes anything still
        open when the run ends.

        Periods are numbered starting from ``starting_period_number``
        (default 1, matching every existing backtest call site's
        unchanged behavior -- a full-window backtest's first timestamp
        genuinely is period 1). A live call passes the real, calendar-
        derived period number for "today" (``config.hold_period.
        period_number_for``) together with a single-day ``ranked`` frame,
        and ``today`` itself (see ``_seed_from_portfolio``) -- see open
        question 27. ``entries_allowed``'s derived pacing needs no
        changes at all to support this: it already recomputes fresh from
        ``self._executed``/``self.remaining_budget`` and ``period_number``
        every call, so correctly-seeded inputs are the whole fix.
        """
        timestamps = sorted(pd.unique(ranked.index.get_level_values("timestamp")))
        self._seed_from_portfolio(timestamps, starting_period_number=starting_period_number, today=today)

        all_rows: list[dict] = []
        last_period, last_timestamp = 0, None
        for period_number, timestamp in enumerate(timestamps, start=starting_period_number):
            last_period, last_timestamp = period_number, timestamp
            period_rows = ranked.xs(timestamp, level="timestamp", drop_level=False)

            all_rows.extend(self._process_exits(period_number, timestamp))
            all_rows.extend(self._process_entries(period_number, timestamp, period_rows))
            self._assert_invariant()
            self._assert_pacing_invariant(period_number)

        if today is None:
            # Backtest mode, EXACTLY the original behavior: `ranked` here
            # is understood to be ALL the data there ever will be for this
            # run, so anything still open when it runs out gets force-
            # closed regardless of why -- an unresolvable
            # scheduled_exit_period (None: e.g. a real position with no
            # scheduled_exit_date on file, force-closed as a safety net
            # since there's no later call coming that could ever resolve
            # it), a resolvable one the data simply didn't reach yet, or
            # the entry cutoff failing to keep something exitable (spec
            # §5: "if it fires, log it").
            if self._open:
                reason = (
                    "cutoff was not enforced"
                    if last_period >= self.window_trading_days
                    else f"data ended at period {last_period}, before window close ({self.window_trading_days})"
                )
                all_rows.extend(self._force_close_remaining(last_period, last_timestamp, reason))
        else:
            # Live mode (open question 27): unlike a backtest, more calls
            # ARE coming -- `ranked` here is one day's snapshot, not "all
            # the data there ever will be". Force-closing everything still
            # open at the end of today's single-period loop would
            # incorrectly close every genuinely-still-pending position on
            # every live invocation. Only two cases legitimately warrant
            # it: (a) the true window close was actually reached this
            # call, or (b) a position has NO resolvable
            # scheduled_exit_period at all (seeded from a real position
            # with no scheduled_exit_date on file) -- waiting is pointless
            # when there's no known date to wait for, so this is flagged
            # immediately rather than held open indefinitely. Everything
            # else with a known, merely-not-yet-reached exit period is
            # left in self._open for a later call to resolve naturally.
            at_window_close = last_period >= self.window_trading_days
            if at_window_close and self._open:
                all_rows.extend(
                    self._force_close_remaining(last_period, last_timestamp, "cutoff was not enforced")
                )
            else:
                unresolvable = [t for t, pos in self._open.items() if pos.scheduled_exit_period is None]
                if unresolvable:
                    all_rows.extend(
                        self._force_close_remaining(
                            last_period, last_timestamp,
                            "no resolvable scheduled_exit_date on this real position",
                            tickers=unresolvable,
                        )
                    )

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
