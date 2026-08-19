"""newey_west_se / block_bootstrap_se (open question 45): the naive
std(series)/sqrt(n) SE silently assumes independence that overlapping
hold_days-forward-return labels don't have. Both corrections get a
deliberate failing-case test (a synthetic series with STRONG, KNOWN
autocorrelation where the naive SE is badly wrong) before being trusted,
per this project's standing rule for any new diagnostic tool -- plus
newey_west_se is cross-validated directly against statsmodels' own HAC
implementation, not just checked against the hand-derived formula that
produced it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.block_se import block_bootstrap_se, newey_west_se


# --- newey_west_se ---------------------------------------------------------

def test_newey_west_at_lag_zero_matches_the_naive_se():
    """Deliberate anchor check: with no autocorrelation correction at all
    (lag=0), Newey-West must reduce exactly to the population-style
    std/sqrt(n) (ddof=0, matching gamma_0's own 1/n normalization) --
    confirms the zero-lag base case is right before trusting any lag>0
    result built on top of it."""
    rng = np.random.default_rng(0)
    values = rng.normal(size=200)
    nw = newey_west_se(values, lag=0)
    naive = values.std(ddof=0) / np.sqrt(len(values))
    assert nw == pytest.approx(naive, rel=1e-9)


def test_newey_west_matches_statsmodels_hac_on_a_constant_regression():
    """Cross-validated against a real, independently-implemented HAC
    estimator (statsmodels), not just self-consistent with its own
    formula. OLS of `values` on a constant, with cov_type='HAC', gives
    the HAC standard error of the sample mean directly -- mathematically
    the same quantity newey_west_se computes."""
    import statsmodels.api as sm

    rng = np.random.default_rng(1)
    # Real, non-trivial serial correlation (AR(1)-ish), not white noise --
    # a stronger cross-check than an already-independent series would be.
    n = 300
    values = np.zeros(n)
    values[0] = rng.normal()
    for t in range(1, n):
        values[t] = 0.5 * values[t - 1] + rng.normal()

    for lag in (0, 4, 10):
        ours = newey_west_se(values, lag=lag)
        X = np.ones((n, 1))
        model = sm.OLS(values, X).fit(cov_type="HAC", cov_kwds={"maxlags": lag})
        theirs = float(np.sqrt(model.cov_params().iloc[0, 0]) if hasattr(model.cov_params(), "iloc")
                        else np.sqrt(model.cov_params()[0, 0]))
        assert ours == pytest.approx(theirs, rel=0.02), f"lag={lag}: ours={ours} theirs={theirs}"


def test_newey_west_deliberate_failing_case_catches_understated_naive_se():
    """The actual point of this whole module: construct a series with
    STRONG, KNOWN autocorrelation (each of 50 independent draws repeated
    5x consecutively -- maximal within-block correlation, zero between-
    block) where the true effective n is 50, not 250. The naive SE
    (sqrt(250)) must be badly too small.

    NW does NOT fully recover the true 50-independent-draws SE here, and
    it shouldn't be expected to: this construction's autocovariance is
    itself a linear taper (gamma_k = sigma^2*(5-k)/5 in expectation, for
    a hard block of 5), and the Bartlett kernel applies a SECOND linear
    taper on top of it (weight_k = 1-k/(lag+1)), compounding into an
    effective quadratic down-weighting. Worked by hand and confirmed
    empirically (mean ratio 0.81, std 0.026 across 30 seeds) that NW at
    lag=4 recovers ~82.5% of the true block-level SE for this specific
    hard-block-boundary construction -- a real, derivable, near-
    deterministic property of the estimator, not noise. This is exactly
    why the module reports block_bootstrap_se too: it has no taper at
    all and is the better-matched estimator whenever the true
    independence unit is a hard, known-width block (as it is here, and
    as this project asserts it is for hold_days-derived series)."""
    rng = np.random.default_rng(2)
    block_draws = rng.normal(loc=0.02, scale=0.05, size=50)
    values = np.repeat(block_draws, 5)  # 250 values, only 50 independent

    naive_se = values.std(ddof=1) / np.sqrt(len(values))
    true_se_50_independent = block_draws.std(ddof=1) / np.sqrt(50)
    nw_se = newey_west_se(values, lag=4)

    assert naive_se < true_se_50_independent * 0.5  # naive badly understates -- the bug this exists to catch
    assert nw_se > naive_se * 1.5  # NW still corrects substantially even though it under-recovers
    assert nw_se == pytest.approx(true_se_50_independent * 0.825, rel=0.2)  # matches the derived Bartlett-taper expectation, not the untapered truth


# --- block_bootstrap_se ------------------------------------------------------

def _dates(n, start="2024-01-01"):
    return pd.bdate_range(start, periods=n).to_numpy()


def test_block_bootstrap_se_on_independent_data_matches_the_naive_se_roughly():
    """Positive control: with genuinely independent per-date values (no
    real block structure), the block-bootstrap SE should land close to
    the naive SE -- confirms the method isn't inflating SEs
    unconditionally, only when real within-block correlation exists."""
    rng = np.random.default_rng(3)
    values = rng.normal(size=250)
    dates = _dates(250)
    result = block_bootstrap_se(values, dates, block_days=5, n_bootstrap=5000, seed=0)
    naive_se = values.std(ddof=1) / np.sqrt(250)
    assert result["n_blocks"] == 50
    assert result["se"] == pytest.approx(naive_se, rel=0.25)


def test_block_bootstrap_deliberate_failing_case_catches_understated_naive_se():
    """Same synthetic construction as the Newey-West failing case (50
    independent draws, each repeated 5x consecutively) -- the block
    bootstrap, resampling whole 5-day blocks, must recover something
    close to the TRUE 50-independent-draws SE, not the naive
    sqrt(250)-based one, which is badly too small for this series."""
    rng = np.random.default_rng(4)
    block_draws = rng.normal(loc=0.02, scale=0.05, size=50)
    values = np.repeat(block_draws, 5)
    dates = _dates(250)

    naive_se = values.std(ddof=1) / np.sqrt(len(values))
    true_se_50_independent = block_draws.std(ddof=1) / np.sqrt(50)
    result = block_bootstrap_se(values, dates, block_days=5, n_bootstrap=10_000, seed=5)

    assert naive_se < true_se_50_independent * 0.5
    assert result["se"] == pytest.approx(true_se_50_independent, rel=0.15)


def test_block_bootstrap_drops_a_partial_final_block():
    """253 dates at block_days=5 -- the trailing 3 must be dropped, not
    silently included in a short 4th block or padded."""
    rng = np.random.default_rng(6)
    values = rng.normal(size=253)
    dates = _dates(253)
    result = block_bootstrap_se(values, dates, block_days=5, n_bootstrap=1000, seed=0)
    assert result["n_blocks"] == 50  # 253 // 5, not 51


def test_block_bootstrap_sorts_by_date_not_input_order():
    """Deliberate failing-case check: shuffling the input order must not
    change the result -- blocks are defined by chronological order, not
    array position."""
    rng = np.random.default_rng(7)
    values = rng.normal(size=100)
    dates = _dates(100)
    shuffle = rng.permutation(100)

    ordered = block_bootstrap_se(values, dates, block_days=5, n_bootstrap=2000, seed=1)
    shuffled = block_bootstrap_se(values[shuffle], dates[shuffle], block_days=5, n_bootstrap=2000, seed=1)
    assert ordered["se"] == pytest.approx(shuffled["se"], rel=1e-9)
    assert ordered["mean"] == pytest.approx(shuffled["mean"], rel=1e-9)
