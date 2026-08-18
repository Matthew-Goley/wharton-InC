"""Feature-work tight loop, iteration 1: sector_momentum_60_skip_10
(Moskowitz & Grinblatt 1999, industry momentum -- see
RealDataProvider._add_sector_momentum's docstring).

Protocol (baseline_system_architecture.md open question 12's roadmap):
add a feature, run paired_ic_comparison, keep only if IC moves by more
than 2 SE. Both models use the SAME genome, TRAIN window, and random seed
-- the only difference is one extra column in X -- so this isolates the
feature's own effect as cleanly as a paired comparison can.

Usage:
    python -m scripts.feature_candidate_sector_momentum
"""
from __future__ import annotations

from src.config import BROAD_EXPERT, BaselineConfig, ExpertSpec
from src.data_layer import RealDataProvider
from src.evaluation import paired_ic_comparison
from src.experts.base import ExpertModel
from scripts.backtest_broad_real import BEST_GENOME, TRAIN_START, TRAIN_END, TEST_START, TEST_END

CANDIDATE_FEATURE = "sector_momentum_60_skip_10"


def main() -> None:
    config = BaselineConfig()
    provider = RealDataProvider()
    broad_spec = ExpertSpec(BROAD_EXPERT, None)

    print(f"Loading TRAIN window ({TRAIN_START} .. {TRAIN_END})...")
    train_table = provider.get_feature_table(TRAIN_START, TRAIN_END, hold_days=config.trading.hold_days)
    train_frame = train_table.frame

    baseline_cols = train_table.feature_columns
    candidate_cols = baseline_cols + [CANDIDATE_FEATURE]
    print(f"  baseline: {len(baseline_cols)} features  candidate: {len(candidate_cols)} features "
          f"(+{CANDIDATE_FEATURE})")

    y_train = train_frame[train_table.target_column]

    print("\nFitting baseline (existing 15 features) and candidate (+1) on TRAIN, same genome/seed...")
    model_baseline = ExpertModel(BROAD_EXPERT, BEST_GENOME, random_seed=config.ga.random_seed).fit(
        train_frame[baseline_cols], y_train
    )
    model_candidate = ExpertModel(BROAD_EXPERT, BEST_GENOME, random_seed=config.ga.random_seed).fit(
        train_frame[candidate_cols], y_train
    )

    print(f"\nLoading TEST window ({TEST_START} .. {TEST_END}) -- held out...")
    test_table = provider.get_feature_table(TEST_START, TEST_END, hold_days=config.trading.hold_days)
    test_frame = test_table.frame
    y_test = test_frame[test_table.target_column]

    pred_baseline = model_baseline.predict(test_frame[baseline_cols])["predicted_return"]
    pred_candidate = model_candidate.predict(test_frame[candidate_cols])["predicted_return"]

    dates = test_frame.index.get_level_values("timestamp").to_numpy()
    result = paired_ic_comparison(dates, y_test.to_numpy(), pred_baseline.to_numpy(), pred_candidate.to_numpy())

    print("\n" + "=" * 72)
    print(f"n_dates={result['n_dates']}")
    print(f"baseline (15 features) IC:  {result['mean_ic_a']:+.4f}")
    print(f"candidate (+sector_mom) IC: {result['mean_ic_b']:+.4f}")
    print(f"paired diff (candidate - baseline): {result['mean_diff']:+.4f}  SE={result['se']:.4f}  "
          f"t={result['t']:+.2f}")
    print(f"95% CI: [{result['ci_low']:+.4f}, {result['ci_high']:+.4f}]")
    two_se = 2 * result["se"]
    diff = result["mean_diff"]
    # KEEP only on a real IMPROVEMENT clearing 2 SE -- a real move in the
    # wrong direction is exactly as decisive a DISCARD as no move at all.
    if diff > two_se:
        print(f"\nKEEP: candidate IC improved by {diff:+.4f} (> 2 SE = {two_se:.4f}).")
    elif diff < -two_se:
        print(f"\nDISCARD: candidate IC got WORSE by {diff:+.4f} (< -2 SE = {-two_se:.4f}) -- a real, "
              f"decisive move in the wrong direction, not just a non-result.")
    else:
        print(f"\nDISCARD: |diff| ({abs(diff):.4f}) <= 2 SE ({two_se:.4f}) -- no detectable effect.")


if __name__ == "__main__":
    main()
