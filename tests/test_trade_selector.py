"""Covers exit_rule_spec.md §8's test cases against the fixed-hold TradeSelector."""
import pandas as pd
import pytest

from config import cost_model
from src.config import TradingConfig
from src.selection.portfolio_state import PortfolioState, Position
from src.selection.trade_selector import TradeSelector

PERIODS = [pd.Timestamp("2020-01-01") + pd.Timedelta(days=i) for i in range(60)]

FLOOR = cost_model.MIN_EDGE_MARGIN * cost_model.round_trip_cost_fraction()
EDGE = FLOOR * 3  # comfortably clears the cost floor


def _ranked(rows: dict) -> pd.DataFrame:
    """rows: {(ticker, timestamp): (combined_score, rank)}"""
    index = pd.MultiIndex.from_tuples(rows.keys(), names=["ticker", "timestamp"])
    return pd.DataFrame(
        {"combined_score": [v[0] for v in rows.values()], "rank": [v[1] for v in rows.values()]},
        index=index,
    )


def _config(**overrides) -> TradingConfig:
    defaults = dict(trade_budget=200, top_n_per_period=10, hold_days=2, window_trading_days=50)
    defaults.update(overrides)
    return TradingConfig(**defaults)


def _entries(trades: pd.DataFrame) -> pd.DataFrame:
    return trades[trades["action"] == "entry"]


def _exits(trades: pd.DataFrame) -> pd.DataFrame:
    return trades[trades["action"] == "exit"]


# 1. Every entry has exactly one matching exit, N trading days later.
def test_entry_gets_exactly_one_exit_n_periods_later():
    selector = TradeSelector(_config(hold_days=3))
    rows = {("AAA", PERIODS[0]): (EDGE, 1)}
    for i in range(1, 6):
        rows[("FILLER", PERIODS[i])] = (FLOOR / 2, 1)  # below cost floor: never eligible, just fills the period axis
    trades = selector.run(_ranked(rows))
    entries, exits = _entries(trades), _exits(trades)
    assert len(entries) == 1 and len(exits) == 1
    assert entries.iloc[0]["period"] == 1
    assert exits.iloc[0]["period"] == 1 + 3
    assert not exits.iloc[0]["forced"]


# 2. Total executed trades never exceeds TRADE_BUDGET.
def test_total_executed_never_exceeds_budget():
    budget = 6
    selector = TradeSelector(_config(trade_budget=budget, hold_days=1, top_n_per_period=10))
    rows = {(f"T{i}", PERIODS[i]): (EDGE, 1) for i in range(20)}
    trades = selector.run(_ranked(rows))
    assert selector.trades_taken <= budget
    assert len(trades) <= budget


# 3. E + P <= TRADE_BUDGET at every period (enforced internally; completing
#    without RuntimeError from _assert_invariant *is* the check).
def test_invariant_holds_under_pressure():
    budget = 7
    selector = TradeSelector(_config(trade_budget=budget, hold_days=2, top_n_per_period=10))
    rows = {(f"T{i}", PERIODS[i % 15]): (EDGE, 1) for i in range(30)}
    selector.run(_ranked(rows))  # raises RuntimeError internally if the invariant is ever violated
    assert selector.trades_taken + selector.open_positions_count <= budget


# 4. No entries proposed after day 50 - N (using a smaller window here).
def test_no_entries_after_cutoff():
    hold_days, window = 5, 20
    selector = TradeSelector(_config(hold_days=hold_days, window_trading_days=window, top_n_per_period=10))
    rows = {(f"T{i}", PERIODS[i]): (EDGE, 1) for i in range(window)}
    trades = selector.run(_ranked(rows))
    entries = _entries(trades)
    assert entries["period"].max() <= window - hold_days


# 5. No positions open after window close.
def test_no_positions_open_after_window_close():
    hold_days, window = 3, 10
    selector = TradeSelector(_config(hold_days=hold_days, window_trading_days=window, top_n_per_period=10))
    rows = {(f"T{i}", PERIODS[i]): (EDGE, 1) for i in range(window)}
    selector.run(_ranked(rows))
    assert selector.open_positions_count == 0
    assert not selector.force_close_log  # cutoff working means the safety net shouldn't have to fire


# 6. No ticker entered while already held (starting portfolio).
def test_already_held_ticker_from_starting_portfolio_is_skipped():
    portfolio = PortfolioState(as_of="2020-01-01", cash=0.0, positions={"AAA": Position(shares=10, avg_price=100.0)})
    selector = TradeSelector(_config(top_n_per_period=5), portfolio=portfolio)
    rows = {("AAA", PERIODS[0]): (EDGE, 1), ("BBB", PERIODS[0]): (EDGE, 2)}
    trades = selector.run(_ranked(rows))
    assert list(_entries(trades).index.get_level_values("ticker")) == ["BBB"]


def test_run_does_not_repick_a_ticker_bought_earlier_in_the_same_run():
    selector = TradeSelector(_config(hold_days=10, top_n_per_period=5))
    rows = {
        ("AAA", PERIODS[0]): (EDGE, 1),
        ("AAA", PERIODS[1]): (EDGE * 2, 1),  # same ticker, later period, higher-ranked
        ("BBB", PERIODS[1]): (EDGE, 2),
    }
    trades = selector.run(_ranked(rows))
    entries = _entries(trades)
    assert list(entries.index.get_level_values("ticker")) == ["AAA", "BBB"]
    assert entries.iloc[0]["period"] == 1


# 7. No ticker re-entered on its own exit day.
def test_no_same_day_reentry_on_exit_day():
    hold_days = 2
    selector = TradeSelector(_config(hold_days=hold_days, top_n_per_period=10))
    rows = {
        ("AAA", PERIODS[0]): (EDGE, 1),  # period 1: entry
        ("AAA", PERIODS[2]): (EDGE * 2, 1),  # period 3 = its own exit day, tempting high score
        ("FILLER", PERIODS[1]): (FLOOR / 2, 1),
    }
    trades = selector.run(_ranked(rows))
    entries = _entries(trades)
    assert len(entries) == 1
    assert entries.iloc[0]["period"] == 1


# 8. N = 1 produces no same-day round trip.
def test_hold_days_one_produces_no_same_day_round_trip():
    selector = TradeSelector(_config(hold_days=1, top_n_per_period=10))
    rows = {("AAA", PERIODS[0]): (EDGE, 1), ("FILLER", PERIODS[1]): (FLOOR / 2, 1)}
    trades = selector.run(_ranked(rows))
    entry_period = _entries(trades)["period"].iloc[0]
    exit_period = _exits(trades)["period"].iloc[0]
    assert exit_period == entry_period + 1


# 9. Label horizon vs. hold_days mismatch is covered in tests/test_pipeline.py
#    (that assertion lives in train_baseline, not TradeSelector).


# 10. Capital ceiling throttles entries when it's tighter than the budget.
def test_entries_throttled_by_capital_cap_not_budget():
    selector = TradeSelector(_config(hold_days=5, top_n_per_period=10, trade_budget=200, position_cap_pct=0.5))
    rows = {(f"T{i}", PERIODS[0]): (EDGE, i + 1) for i in range(5)}
    trades = selector.run(_ranked(rows))
    assert len(_entries(trades)) == 2  # floor(1/0.5) = 2, far below budget's own limit


def test_top_n_per_period_caps_entries_within_one_period():
    selector = TradeSelector(_config(top_n_per_period=1, hold_days=5))
    rows = {("AAA", PERIODS[0]): (EDGE, 1), ("BBB", PERIODS[0]): (EDGE, 2)}
    trades = selector.run(_ranked(rows))
    assert list(_entries(trades).index.get_level_values("ticker")) == ["AAA"]


def test_long_only_excludes_negative_scores_by_default():
    selector = TradeSelector(_config(top_n_per_period=5))
    rows = {("AAA", PERIODS[0]): (EDGE, 1), ("BBB", PERIODS[0]): (-EDGE, 2)}
    trades = selector.run(_ranked(rows))
    assert list(_entries(trades).index.get_level_values("ticker")) == ["AAA"]


def test_allow_short_permits_negative_scores():
    selector = TradeSelector(_config(top_n_per_period=5, allow_short=True))
    rows = {("AAA", PERIODS[0]): (EDGE, 1), ("BBB", PERIODS[0]): (-EDGE, 2)}
    trades = selector.run(_ranked(rows))
    assert set(_entries(trades).index.get_level_values("ticker")) == {"AAA", "BBB"}


def test_edge_below_cost_floor_is_excluded():
    selector = TradeSelector(_config(top_n_per_period=5))
    rows = {("AAA", PERIODS[0]): (FLOOR * 2, 1), ("BBB", PERIODS[0]): (FLOOR / 2, 2)}
    trades = selector.run(_ranked(rows))
    assert list(_entries(trades).index.get_level_values("ticker")) == ["AAA"]


def test_position_size_is_equal_weight_within_period_by_default():
    selector = TradeSelector(_config(top_n_per_period=5, hold_days=5))
    rows = {("AAA", PERIODS[0]): (EDGE, 1), ("BBB", PERIODS[0]): (EDGE, 2)}
    trades = selector.run(_ranked(rows))
    assert (_entries(trades)["position_size_pct"] == 0.5).all()


def test_position_size_respects_cap():
    selector = TradeSelector(_config(top_n_per_period=5, hold_days=5, position_cap_pct=0.1))
    rows = {("AAA", PERIODS[0]): (EDGE, 1), ("BBB", PERIODS[0]): (EDGE, 2)}
    trades = selector.run(_ranked(rows))
    assert (_entries(trades)["position_size_pct"] == 0.1).all()


def test_real_position_without_scheduled_exit_is_force_closed_and_logged():
    portfolio = PortfolioState(as_of="2020-01-01", cash=0.0, positions={"AAA": Position(shares=10, avg_price=100.0)})
    selector = TradeSelector(_config(top_n_per_period=5), portfolio=portfolio)
    rows = {("BBB", PERIODS[0]): (EDGE, 1)}
    selector.run(_ranked(rows))
    assert any(entry["ticker"] == "AAA" for entry in selector.force_close_log)


def test_derived_pacing_spreads_entries_across_the_whole_window_not_just_the_start():
    """Regression for the actual bug: a fixed top_n_per_period let a real
    full-sector-coverage backtest blow through its entire multi-year budget
    in the first ~100 of 497 days the moment eligible candidates became
    abundant, leaving ~19 months completely untested. Here: 60 periods,
    2 eligible candidates every single period (abundant), budget sized for
    roughly one entry every 3 periods on average -- entries must appear
    past the halfway point, not be exhausted early."""
    hold_days, window = 2, 60
    budget = 40  # ~20 round trips over 60 periods
    selector = TradeSelector(_config(trade_budget=budget, hold_days=hold_days, window_trading_days=window))
    rows = {}
    for i in range(window):
        rows[(f"A{i}", PERIODS[i])] = (EDGE, 1)
        rows[(f"B{i}", PERIODS[i])] = (EDGE / 2, 2)
    trades = selector.run(_ranked(rows))
    entries = _entries(trades)
    periods_used = entries["period"]
    assert len(entries) > 0
    assert periods_used.max() > window / 2  # entries still happening in the second half
    assert periods_used.min() > 1 or len(entries[entries["period"] == 1]) < len(entries)  # not all crammed into period 1


def test_derived_pacing_is_self_correcting_after_an_idle_gap():
    """A stretch with no eligible candidates shouldn't permanently lose
    that budget share -- it should still mostly get spent once eligible
    candidates reappear, not sit unused for the rest of the run."""
    hold_days, window = 1, 30
    budget = 20  # ~10 round trips over 30 periods
    selector = TradeSelector(_config(trade_budget=budget, hold_days=hold_days, window_trading_days=window))
    rows = {}
    # First 15 periods: nothing clears the cost floor (idle gap).
    for i in range(15):
        rows[(f"DEAD{i}", PERIODS[i])] = (FLOOR / 2, 1)
    # Remaining periods: abundant eligible candidates.
    for i in range(15, window):
        rows[(f"A{i}", PERIODS[i])] = (EDGE, 1)
        rows[(f"B{i}", PERIODS[i])] = (EDGE / 2, 2)
    trades = selector.run(_ranked(rows))
    entries = _entries(trades)
    # Budget denominator is trade_budget//2 = 10 round trips; with only 15
    # periods left once eligibility opens up, most of that should still
    # get spent, not stall at just 1-2 entries because the derived rate
    # was computed once at the (idle) start and never revisited.
    assert len(entries) >= 6


def test_pacing_invariant_fires_on_gross_front_loading():
    """Directly exercises _assert_pacing_invariant with contrived internal
    state -- the actual bug found (100% of a 497-day budget consumed by
    ~20% of the window elapsed) is exactly what this should catch."""
    selector = TradeSelector(_config(trade_budget=100, window_trading_days=500))
    selector._executed = 100  # 100% of budget consumed
    with pytest.raises(RuntimeError, match="pacing invariant violated"):
        selector._assert_pacing_invariant(period_number=100)  # only 20% of the window elapsed


def test_pacing_invariant_does_not_fire_on_under_consumption():
    """One-sided by design: spending SLOWER than the window elapses is a
    legitimate, already-documented open question (was the selector too
    conservative?), not a pacing bug -- must never trip this invariant."""
    selector = TradeSelector(_config(trade_budget=1000, window_trading_days=100))
    selector._executed = 5  # far under-spent
    selector._assert_pacing_invariant(period_number=90)  # window almost over -- must not raise


def test_max_ticker_share_pct_stops_repeated_entries_once_capped():
    # hold_days=1 + the same-day re-entry ban means AAA's fastest possible
    # cadence is every other period (enter, exit, blocked same day, re-enter)
    # -- 3 entries in 6 periods uncapped (see the sibling test below).
    # window_trading_days=6 matches the 6 periods of data actually used --
    # derived pacing needs a realistic budget/window ratio to not round its
    # per-period allotment down to 0 (a huge default window like 50 against
    # a tiny trade_budget=20 would otherwise starve every period here,
    # unrelated to what this test is actually checking). trade_budget=20 ->
    # total-entries denominator = 20//2 = 10. max_ticker_share_pct=0.2 ->
    # max(1, floor(0.2*10)) = 2, below that natural 3.
    selector = TradeSelector(_config(
        trade_budget=20, hold_days=1, window_trading_days=6, top_n_per_period=5, max_ticker_share_pct=0.2,
    ))
    rows = {}
    for i in range(6):
        rows[("AAA", PERIODS[i])] = (EDGE, 1)  # always ranks 1 -- would otherwise recur every cycle
        rows[("BBB", PERIODS[i])] = (EDGE / 2, 2)
    trades = selector.run(_ranked(rows))
    entries = _entries(trades)
    aaa_entries = entries[entries.index.get_level_values("ticker") == "AAA"]
    assert len(aaa_entries) == 2  # capped below its natural 3-entry cadence
    assert len(entries[entries.index.get_level_values("ticker") == "BBB"]) > 0  # BBB fills in once AAA is capped


def test_max_ticker_share_pct_none_is_uncapped_by_default():
    # window_trading_days=6: see the sibling test above for why this needs
    # to match the data's actual span rather than inheriting _config()'s
    # much larger default.
    selector = TradeSelector(_config(trade_budget=20, hold_days=1, window_trading_days=6, top_n_per_period=5))
    rows = {("AAA", PERIODS[i]): (EDGE, 1) for i in range(6)}
    trades = selector.run(_ranked(rows))
    # No default limit: AAA re-enters on its natural cadence (every other
    # period, given hold_days=1 and the same-day re-entry ban), not held
    # below it by any cap.
    assert len(_entries(trades)) == 3


def test_max_ticker_share_pct_floors_to_at_least_one_entry():
    # A share so small it would floor to 0 must still allow 1 entry per ticker,
    # not lock every ticker out after its very first pick.
    selector = TradeSelector(_config(
        trade_budget=200, hold_days=1, top_n_per_period=5, max_ticker_share_pct=0.001,
    ))
    rows = {("AAA", PERIODS[i]): (EDGE, 1) for i in range(3)}
    trades = selector.run(_ranked(rows))
    assert len(_entries(trades)) == 1  # capped at 1, not 0


def test_real_position_with_scheduled_exit_date_exits_naturally():
    portfolio = PortfolioState(
        as_of="2019-12-01",
        cash=0.0,
        positions={
            "AAA": Position(
                shares=10, avg_price=100.0, entry_date="2019-12-01",
                scheduled_exit_date=str(PERIODS[2].date()),
            )
        },
    )
    selector = TradeSelector(_config(top_n_per_period=5, hold_days=5), portfolio=portfolio)
    rows = {("BBB", PERIODS[i]): (FLOOR / 2, 1) for i in range(5)}  # filler, ineligible
    trades = selector.run(_ranked(rows))
    aaa_exits = _exits(trades)[_exits(trades).index.get_level_values("ticker") == "AAA"]
    assert len(aaa_exits) == 1
    assert not aaa_exits.iloc[0]["forced"]
