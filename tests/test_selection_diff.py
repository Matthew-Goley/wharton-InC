import numpy as np
import pandas as pd
import pytest

from src.evaluation.selection_diff import decompose_selection_change

#: 10 consecutive trading days, hold_days=2 -> 5 non-overlapping blocks.
DAYS = pd.date_range("2024-01-01", periods=10, freq="D")
HOLD_DAYS = 2


def _entries(pairs: list[tuple[str, int]]) -> pd.DataFrame:
    """pairs: [(ticker, day_index), ...] -> a minimal entries-like frame."""
    index = pd.MultiIndex.from_tuples([(t, DAYS[d]) for t, d in pairs], names=["ticker", "timestamp"])
    return pd.DataFrame({"action": "entry"}, index=index)


def _returns(values: dict[tuple[str, int], float]) -> pd.Series:
    index = pd.MultiIndex.from_tuples([(t, DAYS[d]) for t, d in values], names=["ticker", "timestamp"])
    return pd.Series(list(values.values()), index=index)


def test_common_only_a_only_b_partition_correctly():
    entries_a = _entries([("AAA", 0), ("BBB", 0), ("CCC", 2)])
    entries_b = _entries([("AAA", 0), ("DDD", 4)])
    raw = _returns({
        ("AAA", 0): 0.01, ("BBB", 0): -0.02, ("CCC", 2): 0.03, ("DDD", 4): 0.05,
    })
    result = decompose_selection_change(entries_a, entries_b, raw, DAYS.to_numpy(), HOLD_DAYS)

    assert result["common"]["n"] == 1
    assert result["common"]["mean_return"] == pytest.approx(0.01)
    assert result["only_a"]["n"] == 2  # BBB, CCC
    assert result["only_a"]["mean_return"] == pytest.approx((-0.02 + 0.03) / 2)
    assert result["only_b"]["n"] == 1  # DDD
    assert result["only_b"]["mean_return"] == pytest.approx(0.05)


def test_distinct_blocks_counts_calendar_spread_not_trade_count():
    # 4 trades, all inside block 0 (days 0-1) -> 1 block despite n=4.
    concentrated = _entries([("AAA", 0), ("BBB", 0), ("CCC", 1), ("DDD", 1)])
    # 4 trades, one per block (days 0,2,4,6 land in blocks 0,1,2,3) -> 4 blocks.
    spread = _entries([("EEE", 0), ("FFF", 2), ("GGG", 4), ("HHH", 6)])
    empty_b = _entries([])
    raw = _returns({
        ("AAA", 0): 0.0, ("BBB", 0): 0.0, ("CCC", 1): 0.0, ("DDD", 1): 0.0,
        ("EEE", 0): 0.0, ("FFF", 2): 0.0, ("GGG", 4): 0.0, ("HHH", 6): 0.0,
    })

    result = decompose_selection_change(concentrated, spread, raw, DAYS.to_numpy(), HOLD_DAYS)
    assert result["only_a"]["n"] == 4
    assert result["only_a"]["distinct_blocks"] == 1
    assert result["only_a"]["trades_per_active_block"] == pytest.approx(4.0)
    assert result["only_b"]["n"] == 4
    assert result["only_b"]["distinct_blocks"] == 4
    assert result["only_b"]["trades_per_active_block"] == pytest.approx(1.0)


def test_significance_is_none_when_either_side_empty():
    entries_a = _entries([("AAA", 0)])
    entries_b = _entries([("AAA", 0)])  # fully common -- no only_a, no only_b
    raw = _returns({("AAA", 0): 0.01})
    result = decompose_selection_change(entries_a, entries_b, raw, DAYS.to_numpy(), HOLD_DAYS)
    assert result["only_a"]["n"] == 0
    assert result["only_b"]["n"] == 0
    assert result["significance_only_a_vs_only_b"] is None


def test_n_blocks_total_matches_calendar_length():
    entries_a = _entries([])
    entries_b = _entries([])
    raw = pd.Series(dtype=float, index=pd.MultiIndex.from_tuples([], names=["ticker", "timestamp"]))
    result = decompose_selection_change(entries_a, entries_b, raw, DAYS.to_numpy(), HOLD_DAYS)
    assert result["n_blocks_total"] == 5  # 10 days / hold_days=2
