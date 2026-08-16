"""Manually-tracked portfolio state (architecture §5; CLAUDE.md "Portfolio state").

WInS has no API, so the selection layer can't query current holdings itself
— a human updates `data/processed/portfolio_state.json` by hand after every
real trade, and this module is the one place that reads (and writes) it.
`TradeSelector` takes a `PortfolioState` as an explicit input rather than
assuming it starts from flat/empty, since by the time this runs for real
there will usually be open positions it doesn't otherwise know about.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

#: data/processed/portfolio_state.json at the repo root.
DEFAULT_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "portfolio_state.json"


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
class PortfolioState:
    as_of: str
    cash: float
    positions: dict[str, Position] = field(default_factory=dict)

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


def empty_portfolio() -> PortfolioState:
    """All-cash, no positions — the starting state before any real trade is logged."""
    return PortfolioState(as_of="", cash=0.0, positions={})


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
    return PortfolioState(as_of=raw.get("as_of", ""), cash=raw.get("cash", 0.0), positions=positions)


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
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
