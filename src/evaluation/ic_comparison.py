"""Paired per-date IC comparison — the tool the step-4/5 measurement-target
pivot promotes to standard practice for comparing architectures, features,
and hyperparameters (see baseline_system_architecture.md).

Backtest return, filtered through TradeSelector's budget/eligibility/
holding-period mechanics, turned out to be a coarse instrument for this:
step 5's 6-arm ablation found arms whose return differed by 0.2–0.6
percentage points, against a resolving power of roughly ±0.7–1.0pp given
~100 independent 5-day blocks in a 497-day window — the near-certain
result was a null, and it was. Per-date IC, computed directly across
~500+ names a day over the same window, resolves roughly an order of
magnitude finer: paired per-date IC comparisons already run in this
project (broad vs. cross-sectionally-normalized features) had SE≈0.004–
0.005, vs. backtest return's SE≈0.4–0.5pp on comparable data. Every
remaining architecture/feature/GA question in this project is answerable
against IC and was not reliably answerable against return — this is why
the GA gate (baseline_system_architecture.md) is now stated in terms of
IC movement, not backtest-return significance.

Paired, not two independent samples: both prediction sets are scored
against the *same* dates/tickers/labels, so differencing removes the
shared day-to-day market-regime noise that dominates each individual IC
estimate — the same reason a paired t-test beats an unpaired one whenever
the pairing is real. This has been reimplemented ad hoc four times across
this project's diagnostics already (the normalization gate check, the
per-sector significance table, the materials sign-stability check, this
docstring's own citations); promoted here so it isn't reimplemented a
fifth time.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

from src.evaluation.metrics import MIN_NAMES_PER_DATE, per_date_cross_sectional_ic

#: Default bar for "IC moved" under this project's feature-work protocol
#: (baseline_system_architecture.md open question 12): keep only on an
#: improvement past this many SE, not just any significant movement.
DEFAULT_KEEP_SE_MULTIPLE = 2.0


def paired_ic_comparison(
    dates: np.ndarray,
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    min_names: int = MIN_NAMES_PER_DATE,
    alpha: float = 0.05,
) -> dict:
    """Compare two prediction sets on the identical (date, ticker, y_true)
    rows via paired per-date IC.

    ``pred_a``/``pred_b`` must be aligned to the same ``dates``/``y_true``
    arrays (same order, same rows) — e.g. two different models' predictions
    on the same held-out set, or the same model's predictions before/after
    a feature change.

    Returns a dict: each side's own mean IC (``mean_ic_a``/``mean_ic_b``),
    the paired difference ``b - a`` (mean/SE/t/a two-sided confidence
    interval at ``alpha``), and ``n_dates`` — the number of dates both
    sides had a qualifying IC for (the actual sample size the comparison
    is based on, since either side can independently drop a thin date).

    A confidence interval that excludes 0 is this project's criterion for
    "IC moved" (see the restated GA gate) — report the interval, not just
    the point estimate or a bare significance flag.
    """
    ic_a = per_date_cross_sectional_ic(dates, y_true, pred_a, min_names)
    ic_b = per_date_cross_sectional_ic(dates, y_true, pred_b, min_names)
    common = ic_a.index.intersection(ic_b.index)
    n = len(common)

    if n == 0:
        return {
            "n_dates": 0, "mean_ic_a": None, "mean_ic_b": None,
            "mean_diff": None, "se": None, "t": None, "ci_low": None, "ci_high": None,
        }

    diff = ic_b.loc[common] - ic_a.loc[common]
    mean_diff = float(diff.mean())
    se = float(diff.std() / np.sqrt(n)) if n > 1 else float("nan")
    t = mean_diff / se if se and se > 0 else float("nan")
    z = float(norm.ppf(1 - alpha / 2))

    return {
        "n_dates": n,
        "mean_ic_a": float(ic_a.loc[common].mean()),
        "mean_ic_b": float(ic_b.loc[common].mean()),
        "mean_diff": mean_diff,
        "se": se,
        "t": t,
        "ci_low": mean_diff - z * se,
        "ci_high": mean_diff + z * se,
    }


def keep_or_discard(result: dict, min_se_multiple: float = DEFAULT_KEEP_SE_MULTIPLE) -> str:
    """Interpret a :func:`paired_ic_comparison` result under this project's
    feature-work protocol: KEEP only on a real improvement past
    ``min_se_multiple`` SE, DISCARD on a real move in the wrong direction
    or on no detectable effect.

    Regression-tested against the exact bug this fixes: an earlier,
    inline version of this decision (in the first feature-work diagnostic
    script) returned "KEEP" for any ``|diff| > threshold``, without
    checking sign — so a real, decisive DEGRADATION (the sector-momentum
    candidate, paired t=-2.69) printed "KEEP" instead of "DISCARD". This
    is the fourth diagnostic-tooling bug this project has caught (after
    the vol_normalized_rank_check NaN crash, the split-sample year-
    boundary leak, and the Step 5 Bonferroni threshold needing to be
    recomputed rather than assumed) — worth a deliberate failing-case
    test before trusting any new diagnostic's output, not just this one.
    """
    if result["n_dates"] == 0 or result["mean_diff"] is None:
        return "DISCARD (no comparable dates)"
    diff, se = result["mean_diff"], result["se"]
    threshold = min_se_multiple * se
    if diff > threshold:
        return "KEEP"
    elif diff < -threshold:
        return "DISCARD (moved the wrong direction)"
    else:
        return "DISCARD (no detectable effect)"
