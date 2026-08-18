import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from src.evaluation.metrics import (
    MIN_NAMES_PER_DATE,
    cross_sectional_ic,
    information_coefficient,
    per_date_cross_sectional_ic,
)


def _panel(n_dates=20, n_names=30, seed=0):
    rng = np.random.default_rng(seed)
    dates = np.repeat(pd.date_range("2020-01-01", periods=n_dates), n_names)
    y = rng.normal(size=n_dates * n_names)
    return dates, y, rng


def test_perfect_per_date_ranking_scores_one():
    dates, y, _ = _panel()
    assert cross_sectional_ic(dates, y, y) == pytest.approx(1.0)


def test_reversed_per_date_ranking_scores_minus_one():
    dates, y, _ = _panel()
    assert cross_sectional_ic(dates, y, -y) == pytest.approx(-1.0)


def test_matches_scipy_spearman_averaged_over_dates():
    """The grouped-sums shortcut must equal the obvious slow implementation."""
    dates, y, rng = _panel(n_dates=15, n_names=40, seed=7)
    pred = 0.4 * y + rng.normal(size=y.size)  # partial signal

    frame = pd.DataFrame({"d": dates, "y": y, "p": pred})
    expected = frame.groupby("d").apply(
        lambda g: spearmanr(g["y"], g["p"]).statistic, include_groups=False
    ).mean()

    assert cross_sectional_ic(dates, y, pred) == pytest.approx(expected, abs=1e-12)


def test_handles_ties_like_spearman():
    dates = np.repeat(pd.date_range("2020-01-01", periods=3), 12)
    y = np.tile([1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0, 6.0], 3)
    pred = np.tile(np.arange(12, dtype=float), 3)
    frame = pd.DataFrame({"d": dates, "y": y, "p": pred})
    expected = frame.groupby("d").apply(
        lambda g: spearmanr(g["y"], g["p"]).statistic, include_groups=False
    ).mean()
    assert cross_sectional_ic(dates, y, pred) == pytest.approx(expected, abs=1e-12)


def test_pooled_and_cross_sectional_disagree_under_a_market_factor():
    """Why the fitness function changed.

    Give every date a common shock that the predictor also sees. Pooled IC
    rewards tracking that shock; per-date IC correctly ignores it, because
    within a date the shock is a constant offset for every name.
    """
    rng = np.random.default_rng(3)
    n_dates, n_names = 40, 25
    dates = np.repeat(pd.date_range("2020-01-01", periods=n_dates), n_names)
    market = np.repeat(rng.normal(0, 5.0, n_dates), n_names)  # big common move
    idio = rng.normal(0, 1.0, n_dates * n_names)
    y = market + idio
    pred = market  # knows the market perfectly, nothing about single names

    pooled = information_coefficient(y, pred)
    per_date = cross_sectional_ic(dates, y, pred)
    assert pooled > 0.5, "pooled IC should look impressive here"
    assert abs(per_date) < 1e-9, "per-date IC should see no stock-picking skill"


def test_dates_with_too_few_names_are_skipped():
    thin = np.repeat(pd.date_range("2020-01-01", periods=2), MIN_NAMES_PER_DATE - 1)
    y = np.arange(thin.size, dtype=float)
    assert cross_sectional_ic(thin, y, y) == 0.0


def test_constant_predictions_score_zero():
    dates, y, _ = _panel()
    assert cross_sectional_ic(dates, y, np.zeros_like(y)) == 0.0


def test_non_finite_rows_are_dropped_not_fatal():
    dates, y, rng = _panel()
    pred = y.copy()
    pred[:5] = np.nan
    pred[5:8] = np.inf
    assert cross_sectional_ic(dates, y, pred) == pytest.approx(1.0)


def test_empty_input_scores_zero():
    empty = np.array([], dtype="datetime64[ns]")
    assert cross_sectional_ic(empty, np.array([]), np.array([])) == 0.0


def test_per_date_series_averages_to_the_aggregate():
    """cross_sectional_ic and per_date_cross_sectional_ic must share the
    same underlying computation (refactored to a common helper) -- the
    aggregate is just the series' mean, not a separately-derived number."""
    dates, y, rng = _panel(n_dates=15, n_names=40, seed=11)
    pred = 0.4 * y + rng.normal(size=y.size)
    series = per_date_cross_sectional_ic(dates, y, pred)
    assert series.index.nunique() == 15  # one value per date
    assert float(series.mean()) == pytest.approx(cross_sectional_ic(dates, y, pred))


def test_per_date_series_empty_when_nothing_qualifies():
    thin = np.repeat(pd.date_range("2020-01-01", periods=2), MIN_NAMES_PER_DATE - 1)
    y = np.arange(thin.size, dtype=float)
    series = per_date_cross_sectional_ic(thin, y, y)
    assert series.empty
