import numpy as np
import pandas as pd
import pytest

from src.evaluation.ic_comparison import paired_ic_comparison
from src.evaluation.metrics import MIN_NAMES_PER_DATE


def _panel(n_dates=60, n_names=30, seed=0):
    rng = np.random.default_rng(seed)
    dates = np.repeat(pd.date_range("2020-01-01", periods=n_dates), n_names)
    y = rng.normal(size=n_dates * n_names)
    return dates, y, rng


def test_identical_predictions_have_zero_diff_and_zero_variance():
    dates, y, _ = _panel()
    result = paired_ic_comparison(dates, y, y, y)
    assert result["mean_diff"] == pytest.approx(0.0)
    assert result["mean_ic_a"] == pytest.approx(result["mean_ic_b"])
    assert result["ci_low"] <= 0.0 <= result["ci_high"]


def test_a_strictly_better_predictor_shows_a_positive_diff():
    """pred_b tracks y perfectly; pred_a is pure noise -- b should score a
    higher IC, and the paired difference should reflect that."""
    dates, y, rng = _panel(n_dates=80, n_names=40, seed=3)
    pred_a = rng.normal(size=y.size)  # unrelated to y
    pred_b = y  # perfect
    result = paired_ic_comparison(dates, y, pred_a, pred_b)
    assert result["mean_ic_b"] == pytest.approx(1.0)
    assert result["mean_ic_a"] < 0.3  # noise, should be near zero
    assert result["mean_diff"] > 0.5
    assert result["t"] > 0


def test_n_dates_is_the_intersection_not_either_sides_own_count():
    """If one side drops a thin date the other doesn't, the comparison
    should only use dates BOTH sides have a qualifying IC for."""
    dates, y, rng = _panel(n_dates=10, n_names=MIN_NAMES_PER_DATE + 5, seed=5)
    pred_a = y.copy()
    pred_b = y.copy()
    # Make pred_b constant (no variation -> no IC) on the very first date's rows.
    first_date_mask = dates == dates[0]
    pred_b[first_date_mask] = 0.0
    result = paired_ic_comparison(dates, y, pred_a, pred_b)
    assert result["n_dates"] == 9  # 10 dates minus the one pred_b drops


def test_empty_when_no_dates_qualify():
    thin = np.repeat(pd.date_range("2020-01-01", periods=2), MIN_NAMES_PER_DATE - 1)
    y = np.arange(thin.size, dtype=float)
    result = paired_ic_comparison(thin, y, y, y)
    assert result["n_dates"] == 0
    assert result["mean_diff"] is None


def test_ci_width_reflects_alpha():
    dates, y, rng = _panel(n_dates=100, n_names=30, seed=9)
    pred_a = 0.3 * y + rng.normal(size=y.size)
    pred_b = 0.5 * y + rng.normal(size=y.size)
    wide = paired_ic_comparison(dates, y, pred_a, pred_b, alpha=0.20)
    narrow = paired_ic_comparison(dates, y, pred_a, pred_b, alpha=0.01)
    wide_width = wide["ci_high"] - wide["ci_low"]
    narrow_width = narrow["ci_high"] - narrow["ci_low"]
    assert narrow_width > wide_width  # a stricter (lower alpha) CI must be wider
