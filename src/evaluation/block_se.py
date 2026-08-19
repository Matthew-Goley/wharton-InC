"""Autocorrelation-aware standard errors for per-date IC series (open
question 45): the naive ``std(series, ddof=1) / sqrt(n)`` SE used
throughout ``paired_ic_comparison``/``cross_sectional_ic`` assumes the n
per-date observations are independent. They are not, when the underlying
label is a ``hold_days``-forward return with overlapping windows — a
label at date t and date t+1 share ``hold_days - 1`` days of the SAME
future price path, inducing real serial correlation in the per-date IC
series that ``sqrt(n)`` silently treats as if it weren't there.

This project's own established independence unit elsewhere
(``src.evaluation.selection_diff.decompose_selection_change``) is the
non-overlapping ``hold_days``-wide block, not the individual date; these
functions operationalize that same philosophy for a per-date SERIES
rather than a trade count.

Two independent corrections, reported and cross-checked against each
other rather than either one trusted alone:

- ``newey_west_se``: the standard HAC long-run-variance estimator (Newey
  & West 1987), Bartlett-kernel-weighted autocovariances up to ``lag``.
  Cross-validated directly against ``statsmodels``' own
  ``OLS(..., cov_type="HAC")`` on a constant-only regression
  (mathematically equivalent to the sample mean's HAC SE) — see
  ``tests/test_block_se.py`` — rather than trusted from the formula alone.
- ``block_bootstrap_se``: resamples the non-overlapping ``hold_days``-wide
  blocks with replacement, many times, and takes the standard deviation
  of the resulting bootstrap-mean distribution as the SE estimate — a
  direct, minimal-assumption operationalization of "the blocks, not the
  dates, are what's actually independent here."

KNOWN, DERIVED PROPERTY (see ``tests/test_block_se.py``'s deliberate
failing-case test): for a series whose true independence unit is a hard,
known-width block (exactly this project's situation), ``newey_west_se``
at ``lag = hold_days - 1`` systematically UNDER-recovers the true
block-level SE by a knowable amount (~17.5% short, empirically 0.81 mean
ratio / 0.026 std across seeds, for a 5-wide block) — the Bartlett
kernel's own linear taper compounds with the block structure's own
linear-decaying autocovariance into an effective quadratic
down-weighting. This is expected behavior, not a bug (cross-validated
against ``statsmodels`` separately, on a smoothly-decaying AR(1) series
where no such compounding occurs). Practical consequence: NW is a
useful cross-check but ``block_bootstrap_se`` is the better-matched,
untapered estimator whenever the true correlation structure is a hard
block boundary rather than a smooth decay — prefer it as the primary
reported figure when the two disagree.
"""
from __future__ import annotations

import numpy as np


def newey_west_se(values: np.ndarray, lag: int) -> float:
    """HAC standard error of the sample MEAN of ``values`` (Newey & West
    1987, Bartlett/triangular kernel). ``lag`` should be at least
    ``hold_days - 1`` for a series derived from a ``hold_days``-forward
    return — the maximum possible overlap between two labels' price
    windows. Reduces to the naive ``std(values, ddof=0) / sqrt(n)`` at
    ``lag=0`` (no correction), which is the deliberate failing-case check
    in the test suite that this function's own zero-lag behavior is
    correct before trusting it at a real lag.
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n < 2:
        return float("nan")
    demeaned = values - values.mean()
    gamma_0 = float(np.sum(demeaned**2)) / n
    long_run_var = gamma_0
    for k in range(1, min(lag, n - 1) + 1):
        gamma_k = float(np.sum(demeaned[k:] * demeaned[:-k])) / n
        weight = 1.0 - k / (lag + 1)  # Bartlett kernel -- tapers to 0 at k=lag+1
        long_run_var += 2 * weight * gamma_k
    return float(np.sqrt(max(long_run_var, 0.0) / n))


def block_bootstrap_se(
    values: np.ndarray,
    dates: np.ndarray,
    block_days: int,
    n_bootstrap: int = 10_000,
    seed: int = 42,
) -> dict:
    """Block-bootstrap SE of the sample mean of ``values``, resampling
    non-overlapping ``block_days``-wide chunks of CONSECUTIVE dates (this
    project's own established independence unit) with replacement.

    ``dates`` need not be pre-sorted; sorted internally. Any remainder
    dates that don't fill a whole final block are dropped — consistent
    with how ``n_blocks = n_dates // block_days`` is computed everywhere
    else in this project (e.g. the data_capacity_curves eval-block
    report). Returns a dict, not a bare float, so ``n_blocks`` (the
    actual effective-n this SE reflects) travels with the estimate
    rather than needing to be recomputed separately and trusted to match.
    """
    values = np.asarray(values, dtype=float)
    order = np.argsort(dates)
    values = values[order]
    n = len(values)
    n_blocks = n // block_days
    if n_blocks < 2:
        return {"se": float("nan"), "n_blocks": n_blocks, "mean": float(values.mean()) if n else float("nan")}
    usable = n_blocks * block_days
    blocks = values[:usable].reshape(n_blocks, block_days)

    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sampled = rng.integers(0, n_blocks, size=n_blocks)
        boot_means[i] = blocks[sampled].mean()

    return {
        "se": float(boot_means.std(ddof=1)),
        "n_blocks": int(n_blocks),
        "mean": float(blocks.mean()),
    }
