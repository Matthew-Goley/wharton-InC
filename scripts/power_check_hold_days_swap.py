"""Power check for B1 (the hold_days sweep, HOLD_DAYS_SWEEP = (1, 3, 5, 10)
in config/hold_period.py) -- run BEFORE any B1 work begins, per instruction.
Not B1 itself: no 10-day-hold model is trained here. This estimates
whether a cross-horizon paired IC comparison (5-day baseline vs. a 10-day
arm) would even be resolvable, so B1 isn't spent finding out the hard way.

Why this needs its own estimate, not the existing feature-work SE figures:
every paired IC comparison run so far in this project (the tight loop,
data_capacity_curves.py) compares two prediction sets against the SAME
label series (same hold_days=5 forward return) -- the day-to-day "how
predictable was the cross-section today" component is common-mode between
the two arms and cancels almost entirely in the paired difference, which
is exactly why observed paired SEs (~0.004-0.006, see feature_candidate_
common.py's corrected figures) have been so much smaller than either arm's
own single-arm noise floor (~0.0077-0.0125, item 45). A 5-day-vs-10-day
comparison differs in a way that breaks that cancellation: the two arms
would be scored against DIFFERENT label series (5-day vs. 10-day forward
return), which are correlated (a 10-day return's first half IS the 5-day
return) but not identical -- so some, not all, of the common-mode noise
that normally cancels won't.

METHOD (a cheap, real-data proxy, not the real B1 experiment): hold the
model fixed -- reuse broad's already-trained, already-committed 5-day
baseline (BEST_GENOME, TRAIN window, hold_days=5 label) -- and score its
SAME predictions against both the 5-day and the 10-day forward-return
label series on the TEST window. This isolates exactly the piece B1's
extra SE would come from (the label difference) without paying for a
second model. It is a LOWER BOUND on B1's real cross-horizon SE, not an
exact prediction: a genuinely-refit 10-day model would differ from the
5-day model in more than just its scoring target, adding decorrelation
this proxy can't see. Stated explicitly, not left implicit.

Usage:
    python -m scripts.power_check_hold_days_swap
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import BROAD_EXPERT, BaselineConfig, ExpertSpec  # noqa: E402
from src.data_layer import RealDataProvider  # noqa: E402
from src.evaluation.block_se import block_bootstrap_se  # noqa: E402
from src.evaluation.metrics import cross_sectional_ic, per_date_cross_sectional_ic  # noqa: E402
from src.experts.base import ExpertModel  # noqa: E402
from scripts.backtest_broad_real import BEST_GENOME, TRAIN_START, TRAIN_END, TEST_START, TEST_END  # noqa: E402

HOLD_DAYS_SWEEP = (1, 3, 5, 10)  # config/hold_period.py -- the real B1 family, 4 arms
BASELINE_HOLD_DAYS = 5
ARM_HOLD_DAYS = 10
FAMILY_SIZE = len(HOLD_DAYS_SWEEP)  # 4 -- matches the real sweep, not an assumed count


def main() -> None:
    config = BaselineConfig()
    seed = config.ga.random_seed
    provider = RealDataProvider()

    print(f"Reusing broad's already-trained {BASELINE_HOLD_DAYS}-day baseline (BEST_GENOME, seed={seed}) "
          f"-- no new model trained, this is a power ESTIMATE, not B1 itself.")

    print(f"\nLoading TRAIN window ({TRAIN_START}..{TRAIN_END}), hold_days={BASELINE_HOLD_DAYS}...")
    train_table = provider.get_feature_table(TRAIN_START, TRAIN_END, hold_days=BASELINE_HOLD_DAYS)
    train_frame = train_table.frame
    feature_cols = train_table.feature_columns
    model = ExpertModel(BROAD_EXPERT, BEST_GENOME, random_seed=seed).fit(
        train_frame[feature_cols], train_frame[train_table.target_column]
    )

    print(f"Loading TEST window ({TEST_START}..{TEST_END}) under BOTH label horizons...")
    test_5 = provider.get_feature_table(TEST_START, TEST_END, hold_days=BASELINE_HOLD_DAYS).frame
    test_10 = provider.get_feature_table(TEST_START, TEST_END, hold_days=ARM_HOLD_DAYS).frame
    print(f"  hold_days={BASELINE_HOLD_DAYS}: {len(test_5):,} rows, "
          f"{test_5.index.get_level_values('timestamp').nunique()} dates")
    print(f"  hold_days={ARM_HOLD_DAYS}: {len(test_10):,} rows, "
          f"{test_10.index.get_level_values('timestamp').nunique()} dates  "
          f"(fewer -- the last {ARM_HOLD_DAYS} trading days of the window can't have a realized "
          f"{ARM_HOLD_DAYS}-day forward return yet, dropped, not a bug)")

    # Common (ticker, timestamp) rows only -- a fair paired comparison
    # needs the identical row set under both labels, not each label's own
    # (differently-sized) universe.
    common_idx = test_5.index.intersection(test_10.index)
    print(f"  common rows (both labels realized): {len(common_idx):,}")

    t5 = test_5.loc[common_idx]
    t10 = test_10.loc[common_idx]
    dates = t5.index.get_level_values("timestamp").to_numpy()
    preds = model.predict(t5[feature_cols])["predicted_return"].to_numpy()

    y5 = t5[train_table.target_column].to_numpy()
    y10 = t10[train_table.target_column].to_numpy()

    ic_vs_5 = per_date_cross_sectional_ic(dates, y5, preds)
    ic_vs_10 = per_date_cross_sectional_ic(dates, y10, preds)
    common_dates = ic_vs_5.index.intersection(ic_vs_10.index)
    diff = (ic_vs_10.loc[common_dates] - ic_vs_5.loc[common_dates])

    n = len(diff)
    mean_diff = float(diff.mean())
    naive_se = float(diff.std(ddof=1) / np.sqrt(n))
    naive_t = mean_diff / naive_se if naive_se else float("nan")
    boot = block_bootstrap_se(diff.to_numpy(), common_dates.to_numpy(), block_days=ARM_HOLD_DAYS)
    corrected_se = boot["se"]
    corrected_t = mean_diff / corrected_se if corrected_se and corrected_se > 0 else float("nan")

    print(f"\n=== Same-model, cross-label proxy (n_dates={n}) ===")
    print(f"  mean IC vs {BASELINE_HOLD_DAYS}-day label:  {float(ic_vs_5.loc[common_dates].mean()):+.4f}")
    print(f"  mean IC vs {ARM_HOLD_DAYS}-day label: {float(ic_vs_10.loc[common_dates].mean()):+.4f}")
    print(f"  paired diff: {mean_diff:+.4f}")
    print(f"  SE naive:     {naive_se:.4f}  t naive:     {naive_t:+.2f}")
    print(f"  SE corrected: {corrected_se:.4f}  t corrected: {corrected_t:+.2f}  "
          f"(block bootstrap, n_blocks={boot['n_blocks']}, block_days={ARM_HOLD_DAYS})")
    print(f"  NOTE: this is a LOWER BOUND on B1's real cross-horizon SE (same model, only the label "
          f"differs) -- a genuinely refit {ARM_HOLD_DAYS}-day model would add further decorrelation "
          f"this proxy cannot see.")

    mde_2se = 2.0 * corrected_se
    bonf_z = float(norm.ppf(1 - 0.05 / (2 * FAMILY_SIZE)))
    mde_bonf = bonf_z * corrected_se

    print(f"\n=== Minimum detectable IC difference (B1, {FAMILY_SIZE}-arm family: {HOLD_DAYS_SWEEP}) ===")
    print(f"  at 2x corrected SE (no correction):        {mde_2se:.4f}")
    print(f"  at Bonferroni threshold ({bonf_z:.3f}x SE, {FAMILY_SIZE} arms): {mde_bonf:.4f}")

    print(f"\n=== Context: plausible horizon-driven IC differences, from this project's own measured effects ===")
    print(f"  item 19 (20-day -> 5-day feature-window cohort swap, a comparably large horizon change): "
          f"IC moved {-0.0072:+.4f} (borderline, t=-1.97 naive)")
    print(f"  item 45 capacity sweep, largest real point-estimate swing (max_depth 2 vs 8): "
          f"IC moved {0.0038 - 0.0251:+.4f}")
    print(f"  item 45 data-scaling table, largest point-estimate gap (1yr vs 3yr): {0.0086:+.4f}")
    if mde_2se > 0.02 or mde_bonf > 0.02:
        print(f"\n  MDE is at or beyond the largest already-observed real effects in this project's own "
              f"history above -- a 5-vs-10-day IC experiment at this SE risks being unable to resolve "
              f"even a genuinely large horizon effect from noise.")
    else:
        print(f"\n  MDE is within the range of already-observed real effects -- a 5-vs-10-day IC "
              f"experiment has a plausible chance of resolving a real effect if one exists.")


if __name__ == "__main__":
    main()
