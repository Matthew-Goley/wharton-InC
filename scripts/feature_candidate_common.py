"""Shared harness for the feature-work tight loop (baseline_system_
architecture.md open question 12): add/swap a feature, run
paired_ic_comparison against broad's established baseline, decide via
keep_or_discard. Promoted out of the first candidate script
(feature_candidate_sector_momentum.py) rather than copy-pasted into the
second -- the loop is meant to run repeatedly, so the harness should too.

Corrected SE added (item 47): paired_ic_comparison's own SE is naive
(std(diff)/sqrt(n_dates)), which assumes independent dates -- false for
this project's hold_days-derived per-date IC series (item 45's finding).
Every candidate test run through this harness, past or future, now also
reports block_bootstrap_se over the non-overlapping hold_days-wide
blocks alongside the naive figure -- paired_ic_comparison itself is left
untouched (still used for the point estimate and the existing
keep_or_discard decision path), the correction is layered on top locally,
same pattern as scripts/data_capacity_curves.py.
"""
from __future__ import annotations

import numpy as np

from src.config import BROAD_EXPERT, BaselineConfig, ExpertSpec
from src.data_layer import RealDataProvider
from src.evaluation import keep_or_discard, paired_ic_comparison
from src.evaluation.block_se import block_bootstrap_se
from src.evaluation.metrics import per_date_cross_sectional_ic
from src.experts.base import ExpertModel
from scripts.backtest_broad_real import BEST_GENOME, TRAIN_START, TRAIN_END, TEST_START, TEST_END


def test_candidate_columns(candidate_columns: list[str], label: str) -> dict:
    """Trains broad on ``candidate_columns`` (same genome/seed as the
    established baseline model, which is retrained fresh here on the
    baseline 15 for an exact apples-to-apples paired comparison) and
    compares out-of-sample IC against it via paired_ic_comparison.

    ``candidate_columns`` can differ from the baseline by ADDING a column
    (a brand-new feature, like sector_momentum) or SWAPPING one out for a
    re-windowed variant (like px_to_ma_20 -> px_to_ma_10) -- either way,
    "baseline" always means the current, committed FEATURE_COLUMNS.
    """
    config = BaselineConfig()
    provider = RealDataProvider()
    broad_spec = ExpertSpec(BROAD_EXPERT, None)

    print(f"[{label}] Loading TRAIN window ({TRAIN_START} .. {TRAIN_END})...")
    train_table = provider.get_feature_table(TRAIN_START, TRAIN_END, hold_days=config.trading.hold_days)
    train_frame = train_table.frame
    baseline_cols = train_table.feature_columns
    y_train = train_frame[train_table.target_column]

    print(f"  baseline: {len(baseline_cols)} features  candidate: {len(candidate_columns)} features")
    print("  Fitting baseline and candidate on TRAIN, same genome/seed...")
    model_baseline = ExpertModel(BROAD_EXPERT, BEST_GENOME, random_seed=config.ga.random_seed).fit(
        train_frame[baseline_cols], y_train
    )
    model_candidate = ExpertModel(BROAD_EXPERT, BEST_GENOME, random_seed=config.ga.random_seed).fit(
        train_frame[candidate_columns], y_train
    )

    print(f"  Loading TEST window ({TEST_START} .. {TEST_END}) -- held out...")
    test_table = provider.get_feature_table(TEST_START, TEST_END, hold_days=config.trading.hold_days)
    test_frame = test_table.frame
    y_test = test_frame[test_table.target_column]

    pred_baseline = model_baseline.predict(test_frame[baseline_cols])["predicted_return"]
    pred_candidate = model_candidate.predict(test_frame[candidate_columns])["predicted_return"]

    dates = test_frame.index.get_level_values("timestamp").to_numpy()
    result = paired_ic_comparison(dates, y_test.to_numpy(), pred_baseline.to_numpy(), pred_candidate.to_numpy())
    decision = keep_or_discard(result)

    # Corrected SE: recompute ic_a/ic_b locally (paired_ic_comparison
    # itself is untouched) so the raw per-date diff series is available
    # for block_bootstrap_se -- same pattern as data_capacity_curves.py.
    hold_days = config.trading.hold_days
    ic_a = per_date_cross_sectional_ic(dates, y_test.to_numpy(), pred_baseline.to_numpy())
    ic_b = per_date_cross_sectional_ic(dates, y_test.to_numpy(), pred_candidate.to_numpy())
    common = ic_a.index.intersection(ic_b.index)
    diff_series = (ic_b.loc[common] - ic_a.loc[common])
    boot = block_bootstrap_se(diff_series.to_numpy(), common.to_numpy(), block_days=hold_days)
    corrected_se = boot["se"]
    corrected_t = result["mean_diff"] / corrected_se if corrected_se and corrected_se > 0 else float("nan")
    # Consistency check against paired_ic_comparison's own naive figure --
    # same mean_diff computed two independent ways must agree.
    assert abs(float(diff_series.mean()) - result["mean_diff"]) < 1e-9, \
        f"{label}: locally recomputed mean_diff disagrees with paired_ic_comparison's"

    print("\n" + "=" * 72)
    print(f"[{label}] n_dates={result['n_dates']}")
    print(f"  baseline IC:  {result['mean_ic_a']:+.4f}")
    print(f"  candidate IC: {result['mean_ic_b']:+.4f}")
    print(f"  paired diff (candidate - baseline): {result['mean_diff']:+.4f}")
    print(f"  SE naive:     {result['se']:.4f}  t naive:     {result['t']:+.2f}")
    print(f"  SE corrected: {corrected_se:.4f}  t corrected: {corrected_t:+.2f}  "
          f"(block bootstrap, n_blocks={boot['n_blocks']}, block_days={hold_days})")
    print(f"  95% CI (naive SE): [{result['ci_low']:+.4f}, {result['ci_high']:+.4f}]")
    print(f"  DECISION (naive-SE rule, unchanged): {decision}")
    corrected_threshold = 2.0 * corrected_se
    if result["mean_diff"] > corrected_threshold:
        corrected_decision = "KEEP"
    elif result["mean_diff"] < -corrected_threshold:
        corrected_decision = "DISCARD (moved the wrong direction)"
    else:
        corrected_decision = "DISCARD (no detectable effect)"
    print(f"  DECISION (corrected-SE, 2x bar): {corrected_decision}")

    return {
        "label": label, **result, "decision": decision,
        "se_corrected": corrected_se, "t_corrected": corrected_t, "n_blocks": boot["n_blocks"],
    }
