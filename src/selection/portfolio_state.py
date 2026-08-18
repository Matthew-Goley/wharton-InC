"""Manually-tracked portfolio state (architecture §5; CLAUDE.md "Portfolio state").

WInS has no API, so the selection layer can't query current holdings itself
— a human updates `data/processed/portfolio_state.json` by hand after every
real trade, and this module is the one place that reads (and writes) it.
`TradeSelector` takes a `PortfolioState` as an explicit input rather than
assuming it starts from flat/empty, since by the time this runs for real
there will usually be open positions it doesn't otherwise know about.

``trade_log`` (baseline_system_architecture.md open question 27): an
append-only record of every REAL executed trade, entries and exits alike.
Added specifically so a live run can know ``trades_executed_so_far`` (E) —
``positions`` alone only tells you what's currently open, not how many
trades (including ones already closed) have happened this season, which
is exactly what ``TradeSelector``'s derived pacing needs across separate
live invocations. ``len(state.trade_log)`` is always E — never stored
separately, so it can't drift from the log itself (same "don't keep two
structures that can independently drift" principle already applied once
in this project, ``earnings_tickers`` -> ``earnings_events``).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from config import hold_period

#: data/processed/portfolio_state.json at the repo root.
DEFAULT_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "portfolio_state.json"

_VALID_ACTIONS = ("entry", "exit")


@dataclass(frozen=True)
class Position:
    shares: float
    avg_price: float
    #: When this position was opened, and when the fixed-hold exit rule
    #: (exit_rule_spec.md §1, §6) says it's due to close — both persisted so
    #: a scheduled exit survives across manual execution sessions. "" if
    #: unknown (e.g. a position logged before this field existed).
    entry_date: str = ""
    scheduled_exit_date: str = ""


@dataclass(frozen=True)
class TradeLogEntry:
    """One REAL executed WInS order — both entries and exits count (both
    count against the 200-trade budget, per exit_rule_spec.md §3)."""

    ticker: str
    action: str  # "entry" or "exit"
    date: str
    shares: float
    price: float


@dataclass(frozen=True)
class PortfolioState:
    as_of: str
    cash: float
    positions: dict[str, Position] = field(default_factory=dict)
    trade_log: list[TradeLogEntry] = field(default_factory=list)

    def shares_held(self, ticker: str) -> float:
        position = self.positions.get(ticker)
        return 0.0 if position is None else position.shares

    def position_value(self, ticker: str, current_price: float) -> float:
        return self.shares_held(ticker) * current_price

    def total_value(self, prices: dict[str, float]) -> float:
        """Mark-to-market total value: cash + every held position at ``prices``
        (falling back to cost basis for any ticker ``prices`` doesn't cover)."""
        held = sum(
            position.shares * prices.get(ticker, position.avg_price)
            for ticker, position in self.positions.items()
        )
        return self.cash + held

    @property
    def trades_executed_so_far(self) -> int:
        """E — always len(trade_log), never a separately-tracked counter
        (see the module docstring). What TradeSelector(trades_executed_
        so_far=...) needs to seed its derived pacing correctly on a live
        run that isn't starting from period 1."""
        return len(self.trade_log)


def empty_portfolio() -> PortfolioState:
    """All-cash, no positions, no trade history — the starting state before
    any real trade is logged."""
    return PortfolioState(as_of="", cash=0.0, positions={}, trade_log=[])


def load_portfolio_state(path: Path = DEFAULT_PATH) -> PortfolioState:
    """Read portfolio_state.json, or an empty portfolio if it doesn't exist yet
    (hand-maintained; won't exist until the first real trade is logged)."""
    if not path.exists():
        return empty_portfolio()

    raw = json.loads(path.read_text())
    positions = {
        ticker: Position(
            shares=p["shares"],
            avg_price=p["avg_price"],
            entry_date=p.get("entry_date", ""),
            scheduled_exit_date=p.get("scheduled_exit_date", ""),
        )
        for ticker, p in raw.get("positions", {}).items()
    }
    trade_log = [
        TradeLogEntry(ticker=t["ticker"], action=t["action"], date=t["date"], shares=t["shares"], price=t["price"])
        for t in raw.get("trade_log", [])
    ]
    return PortfolioState(
        as_of=raw.get("as_of", ""), cash=raw.get("cash", 0.0), positions=positions, trade_log=trade_log
    )


def save_portfolio_state(state: PortfolioState, path: Path = DEFAULT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "as_of": state.as_of,
        "cash": state.cash,
        "positions": {
            ticker: {
                "shares": p.shares,
                "avg_price": p.avg_price,
                "entry_date": p.entry_date,
                "scheduled_exit_date": p.scheduled_exit_date,
            }
            for ticker, p in state.positions.items()
        },
        "trade_log": [
            {"ticker": t.ticker, "action": t.action, "date": t.date, "shares": t.shares, "price": t.price}
            for t in state.trade_log
        ],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def log_trade(
    state: PortfolioState,
    ticker: str,
    action: str,
    date: str,
    shares: float,
    price: float,
    hold_days: int = hold_period.HOLD_DAYS,
) -> PortfolioState:
    """The one sanctioned way to record a REAL executed trade (open_
    system_architecture.md open question 27) — appends to trade_log AND
    updates positions/cash together, in the same call, so the two can't
    independently drift the way two hand-maintained structures could.
    Direct JSON editing stays possible (it's still a plain file); this is
    the documented path going forward, not the only physically possible
    one.

    ``action="entry"``: ticker must not already be held (an entry into an
    already-open position is almost certainly a logging mistake, not a
    real second entry — this project's own selection rules forbid
    re-entering a ticker that's already open). scheduled_exit_date is
    computed automatically as `date` + `hold_days` trading days
    (numpy.busday_offset, same weekends-only approximation as
    config.hold_period.period_number_for — not a real NYSE calendar).
    Cash decreases by shares*price; raises if that would go negative
    (long-only, no margin — CLAUDE.md's hard competition rule, and a
    plausible data-entry-typo catch).

    ``action="exit"``: ticker must currently be held. Removes the
    position; cash increases by shares*price.

    Returns a NEW PortfolioState (this dataclass is frozen) — does not
    write to disk; call save_portfolio_state(...) on the result.
    """
    if action not in _VALID_ACTIONS:
        raise ValueError(f"action must be one of {_VALID_ACTIONS}, got {action!r}")

    entry = TradeLogEntry(ticker=ticker, action=action, date=date, shares=shares, price=price)
    positions = dict(state.positions)

    if action == "entry":
        if ticker in positions:
            raise ValueError(
                f"{ticker} is already held ({positions[ticker]}) -- an entry log call for an "
                f"already-open position is almost certainly a mistake, not a real second entry "
                f"(this project's own selection rules forbid re-entering an already-held ticker)."
            )
        cash = state.cash - shares * price
        if cash < 0:
            raise ValueError(
                f"logging this entry ({shares} x {ticker} @ {price}) would take cash negative "
                f"({state.cash} -> {cash}) -- long-only, no margin (CLAUDE.md competition rules)."
            )
        scheduled_exit_date = str(np.busday_offset(date, hold_days, roll="forward"))
        positions[ticker] = Position(
            shares=shares, avg_price=price, entry_date=date, scheduled_exit_date=scheduled_exit_date
        )
    else:  # "exit"
        if ticker not in positions:
            raise ValueError(f"{ticker} is not currently held -- nothing to exit.")
        cash = state.cash + shares * price
        del positions[ticker]

    return PortfolioState(
        as_of=date, cash=cash, positions=positions, trade_log=[*state.trade_log, entry]
    )
