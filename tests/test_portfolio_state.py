"""portfolio_state.py's trade_log extension and log_trade (baseline_
system_architecture.md open question 27) -- the schema a live run reads
trades_executed_so_far from, and the one sanctioned way to record a real
trade.
"""
from __future__ import annotations

import json

import pytest

from src.selection.portfolio_state import (
    PortfolioState,
    empty_portfolio,
    load_portfolio_state,
    log_trade,
    save_portfolio_state,
)


def test_empty_portfolio_has_zero_trades_executed():
    assert empty_portfolio().trades_executed_so_far == 0


def test_trades_executed_so_far_is_always_len_trade_log():
    state = PortfolioState(as_of="", cash=10_000.0, positions={}, trade_log=[])
    state = log_trade(state, "AAPL", "entry", "2026-08-17", shares=5, price=190.0)
    assert state.trades_executed_so_far == 1
    state = log_trade(state, "AAPL", "exit", "2026-08-24", shares=5, price=195.0)
    assert state.trades_executed_so_far == 2


# --- log_trade: entry ---------------------------------------------------------

def test_log_trade_entry_opens_a_position_and_debits_cash():
    state = PortfolioState(as_of="", cash=10_000.0, positions={}, trade_log=[])
    out = log_trade(state, "AAPL", "entry", "2026-08-17", shares=5, price=190.0)
    assert out.cash == pytest.approx(10_000.0 - 5 * 190.0)
    assert out.positions["AAPL"].shares == 5
    assert out.positions["AAPL"].avg_price == 190.0
    assert out.positions["AAPL"].entry_date == "2026-08-17"


def test_log_trade_entry_computes_scheduled_exit_date_hold_days_forward(monkeypatch):
    from config import hold_period
    monkeypatch.setattr(hold_period, "HOLD_DAYS", 5)
    state = PortfolioState(as_of="", cash=10_000.0, positions={}, trade_log=[])
    out = log_trade(state, "AAPL", "entry", "2026-08-17", shares=5, price=190.0, hold_days=5)
    # Monday 2026-08-17 + 5 trading days -> Monday 2026-08-24 (skips the weekend).
    assert out.positions["AAPL"].scheduled_exit_date == "2026-08-24"


def test_log_trade_entry_rejects_an_already_held_ticker():
    """Deliberate failing-case check: a duplicate entry log call for a
    ticker already open is almost certainly a mistake, not a real second
    entry -- this project's own selection rules forbid it."""
    state = PortfolioState(as_of="", cash=10_000.0, positions={}, trade_log=[])
    state = log_trade(state, "AAPL", "entry", "2026-08-17", shares=5, price=190.0)
    with pytest.raises(ValueError, match="already held"):
        log_trade(state, "AAPL", "entry", "2026-08-18", shares=5, price=191.0)


def test_log_trade_entry_rejects_a_purchase_that_would_go_cash_negative():
    """Long-only, no margin (CLAUDE.md competition rules) -- also catches
    a plausible share-count typo before it corrupts the recorded state."""
    state = PortfolioState(as_of="", cash=100.0, positions={}, trade_log=[])
    with pytest.raises(ValueError, match="negative"):
        log_trade(state, "AAPL", "entry", "2026-08-17", shares=5, price=190.0)


# --- log_trade: exit -----------------------------------------------------------

def test_log_trade_exit_closes_the_position_and_credits_cash():
    state = PortfolioState(as_of="", cash=10_000.0, positions={}, trade_log=[])
    state = log_trade(state, "AAPL", "entry", "2026-08-17", shares=5, price=190.0)
    out = log_trade(state, "AAPL", "exit", "2026-08-24", shares=5, price=195.0)
    assert "AAPL" not in out.positions
    assert out.cash == pytest.approx(state.cash + 5 * 195.0)


def test_log_trade_exit_rejects_a_ticker_not_currently_held():
    state = PortfolioState(as_of="", cash=10_000.0, positions={}, trade_log=[])
    with pytest.raises(ValueError, match="not currently held"):
        log_trade(state, "AAPL", "exit", "2026-08-24", shares=5, price=195.0)


def test_log_trade_rejects_an_unrecognized_action():
    state = PortfolioState(as_of="", cash=10_000.0, positions={}, trade_log=[])
    with pytest.raises(ValueError, match="action must be one of"):
        log_trade(state, "AAPL", "hold", "2026-08-17", shares=5, price=190.0)


def test_log_trade_does_not_mutate_the_input_state():
    """PortfolioState is frozen -- log_trade must return a new instance,
    not mutate positions/trade_log in place (both are plain mutable dict/
    list under the frozen dataclass, so this isn't automatic)."""
    state = PortfolioState(as_of="", cash=10_000.0, positions={}, trade_log=[])
    log_trade(state, "AAPL", "entry", "2026-08-17", shares=5, price=190.0)
    assert state.positions == {}
    assert state.trade_log == []


# --- round-trip through JSON ----------------------------------------------------

def test_trade_log_round_trips_through_save_and_load(tmp_path):
    path = tmp_path / "portfolio_state.json"
    state = PortfolioState(as_of="", cash=10_000.0, positions={}, trade_log=[])
    state = log_trade(state, "AAPL", "entry", "2026-08-17", shares=5, price=190.0)
    state = log_trade(state, "MSFT", "entry", "2026-08-17", shares=2, price=300.0)
    save_portfolio_state(state, path)

    reloaded = load_portfolio_state(path)
    assert reloaded.trades_executed_so_far == 2
    assert reloaded.positions.keys() == {"AAPL", "MSFT"}
    assert reloaded.cash == pytest.approx(state.cash)


def test_loading_an_old_schema_file_without_trade_log_defaults_to_empty(tmp_path):
    """Real portfolio_state.json predates this field -- must not KeyError
    on a file that simply doesn't have "trade_log" yet."""
    path = tmp_path / "portfolio_state.json"
    path.write_text(json.dumps({"as_of": "", "cash": 0.0, "positions": {}}))
    state = load_portfolio_state(path)
    assert state.trade_log == []
    assert state.trades_executed_so_far == 0
