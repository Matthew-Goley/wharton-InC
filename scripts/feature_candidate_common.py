"""Shared harness for the feature-work tight loop (baseline_system_
architecture.md open question 12): add/swap a feature, run
paired_ic_comparison against broad's established baseline, decide via
keep_or_discard. Promoted out of the first candidate script
(feature_candidate_sector_momentum.py) rather than copy-pasted into the
second -- the loop is meant to run repeatedly, so the harness should too.
"""
from __future__ import annotations

from src.config import BROAD_EXPERT, BaselineConfig, ExpertSpec
from src.data_layer import RealDataProvider
from src.evaluation import keep_or_discard, paired_ic_comparison
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

    print("\n" + "=" * 72)
    print(f"[{label}] n_dates={result['n_dates']}")
    print(f"  baseline IC:  {result['mean_ic_a']:+.4f}")
    print(f"  candidate IC: {result['mean_ic_b']:+.4f}")
    print(f"  paired diff (candidate - baseline): {result['mean_diff']:+.4f}  SE={result['se']:.4f}  "
          f"t={result['t']:+.2f}")
    print(f"  95% CI: [{result['ci_low']:+.4f}, {result['ci_high']:+.4f}]")
    print(f"  DECISION: {decision}")

    return {"label": label, **result, "decision": decision}
