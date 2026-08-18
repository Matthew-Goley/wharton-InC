"""vol_adjusted's own IC audit (baseline_system_architecture.md open
question 18): the standing ranking default (item 13) was adopted on
concentration grounds (PARA's rank-1 share), before the IC pivot (item 12)
existed as a tool. Checks it doesn't quietly cost signal the way
normalize_features did (item "Cross-sectional feature normalization" in
the ablation table -- real IC cost for a concentration fix that barely
worked).

Usage:
    python -m scripts.audit_vol_adjusted_ic
"""
from __future__ import annotations

from src.config import BROAD_EXPERT, BaselineConfig, ExpertSpec
from src.data_layer import RealDataProvider
from src.evaluation import keep_or_discard, paired_ic_comparison
from src.experts.base import ExpertModel
from scripts.backtest_broad_real import BEST_GENOME, TRAIN_START, TRAIN_END, TEST_START, TEST_END


def main() -> None:
    config = BaselineConfig()
    provider = RealDataProvider()
    broad_spec = ExpertSpec(BROAD_EXPERT, None)

    train_table = provider.get_feature_table(TRAIN_START, TRAIN_END, hold_days=config.trading.hold_days)
    train_frame = train_table.frame
    cols = train_table.feature_columns
    y_train = train_frame[train_table.target_column]
    model = ExpertModel(BROAD_EXPERT, BEST_GENOME, random_seed=config.ga.random_seed).fit(train_frame[cols], y_train)

    test_table = provider.get_feature_table(TEST_START, TEST_END, hold_days=config.trading.hold_days)
    test_frame = test_table.frame
    y_test = test_frame[test_table.target_column]
    preds = model.predict(test_frame[cols])["predicted_return"]

    vol = test_frame["volatility_20"]
    vol = vol.fillna(vol.max()).clip(lower=0.005)
    vol_adjusted_score = preds / vol

    dates = test_frame.index.get_level_values("timestamp").to_numpy()
    result = paired_ic_comparison(dates, y_test.to_numpy(), preds.to_numpy(), vol_adjusted_score.to_numpy())

    print(f"n_dates={result['n_dates']}")
    print(f"raw-ranking IC:          {result['mean_ic_a']:+.4f}")
    print(f"vol_adjusted-ranking IC: {result['mean_ic_b']:+.4f}")
    print(f"paired diff (vol_adj - raw): {result['mean_diff']:+.4f}  SE={result['se']:.4f}  t={result['t']:+.2f}")
    print(f"95% CI: [{result['ci_low']:+.4f}, {result['ci_high']:+.4f}]")
    print(f"decision (does vol_adjusted cost IC): {keep_or_discard(result)}")


if __name__ == "__main__":
    main()
