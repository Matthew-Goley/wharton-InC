import numpy as np
import pandas as pd
import pytest

from src.evaluation.ic_comparison import keep_or_discard, paired_ic_comparison
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


# --- keep_or_discard --------------------------------------------------------
# Regression coverage for the actual bug: an inline "KEEP if |diff| > threshold"
# check doesn't look at sign, so a real, decisive DEGRADATION prints KEEP.
# Each case is a synthetic, hand-built result dict -- no need to run a real
# paired comparison to exercise the decision logic in isolation.

def _result(mean_diff, se):
    return {"n_dates": 100, "mean_ic_a": 0.02, "mean_ic_b": 0.02 + mean_diff,
            "mean_diff": mean_diff, "se": se, "t": mean_diff / se}


def test_keep_on_a_real_improvement():
    assert keep_or_discard(_result(mean_diff=0.02, se=0.005)) == "KEEP"  # 4 SE improvement


def test_discard_on_a_real_degradation_not_keep():
    """THE regression test: a real, decisive move in the wrong direction
    (this is the actual sector-momentum result's shape: diff=-0.0114,
    se=0.0042, |t|~2.7) must DISCARD, not KEEP just because |diff| cleared
    the bar -- an earlier inline version of this check did exactly that."""
    result = _result(mean_diff=-0.0114, se=0.0042)
    assert result["t"] < -2.5  # comfortably past the 2 SE bar, in magnitude
    decision = keep_or_discard(result)
    assert decision.startswith("DISCARD")
    assert "wrong direction" in decision


def test_discard_on_no_detectable_effect():
    assert keep_or_discard(_result(mean_diff=0.001, se=0.005)) == "DISCARD (no detectable effect)"  # 0.2 SE


def test_discard_on_no_comparable_dates():
    empty = {"n_dates": 0, "mean_diff": None, "se": None}
    assert keep_or_discard(empty) == "DISCARD (no comparable dates)"


def test_threshold_is_configurable():
    result = _result(mean_diff=0.011, se=0.005)  # 2.2 SE -- keeps at 2.0, not at 3.0
    assert keep_or_discard(result, min_se_multiple=2.0) == "KEEP"
    assert keep_or_discard(result, min_se_multiple=3.0).startswith("DISCARD")
