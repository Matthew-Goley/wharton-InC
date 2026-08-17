"""Diagnostic: does dividing the score by recent realized volatility before
ranking stop rank-1 from being a winner's-curse magnet for a handful of
chaotic names?

Winsorizing the prediction doesn't change the sort (capping an outlier
still leaves it the largest value, so it still ranks 1) -- normalizing by
risk does, because it demotes a name whose big prediction is just a big
volatility reading. Uses the existing `volatility_20` feature (already
computed, already backward-looking, no new leakage surface) as the risk
estimate: risk_adjusted = combined_score / volatility_20.

Originally built as a standalone diagnostic that didn't touch OverallModel/
TradeSelector at all, specifically so a null result wouldn't cost an
architecture change. Result was real but partial (see below), which was
enough to promote `ranking_score="vol_adjusted"` into OverallModel.rank()
as a real, opt-in option -- this script now exercises that real API rather
than a hand-rolled copy of it, as a regression check that the formalized
version still reproduces the diagnostic's finding.

RESULT: real, not a fix. Rank-1 (n=62->72, ticker diversity 14->35, PARA's
share 60%->32%) improves from -4.11% to -2.81% mean, but decomposing the
actual composition change (not just the naive dilution counterfactor) shows
the 37 trades common to both rankings are literally unchanged at -4.27% --
100% of the visible improvement comes from the 25 demoted trades (-3.88%
avg) being replaced by 35 different ones (-1.28% avg), which is a real
quality improvement in the swap but still firmly negative. Rank 1 stays the
worst bucket in the risk-adjusted curve too. Residual is likely
argmax/selection bias (the max of many noisy estimates is upward-biased by
selection alone, independent of volatility) -- addressed by an uncertainty
estimate, not a better vol estimate; see OverallModel.rank()'s docstring
for the planned "uncertainty_adjusted" mode once sector experts exist.

Usage:
    python -m scripts.vol_normalized_rank_check
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from src.combiner.overall_model import OverallModel
from src.config import BROAD_EXPERT, BaselineConfig, ExpertSpec
from src.data_layer import RealDataProvider
from src.data_layer.real_provider import RAW_TARGET_COLUMN
from src.experts.base import ExpertModel, XGBConfig
from src.experts.registry import build_expert_data
from src.selection.trade_selector import TradeSelector

TRAIN_START, TRAIN_END = "2018-01-01", "2024-01-01"
TEST_START, TEST_END = "2024-01-01", "2026-01-01"

BEST_GENOME = XGBConfig(
    max_depth=8, learning_rate=0.030856726198249796, n_estimators=98,
    subsample=0.5624618396528216, colsample_bytree=0.9792124018732338,
    min_child_weight=6.383279099154468, reg_alpha=2.6108842799586585,
    reg_lambda=7.752080142539311, gamma=0.11401935514848749,
)

#: Floor on the volatility denominator -- a near-zero recent vol reading
#: shouldn't produce a near-infinite risk-adjusted score.
MIN_VOL = 0.005


def run_curve(ranked: pd.DataFrame, test_frame: pd.DataFrame, config, label: str) -> pd.DataFrame:
    n_periods = ranked.index.get_level_values("timestamp").nunique()
    budget = int(round((200 / 50) * n_periods))
    trading_config = replace(config.trading, window_trading_days=n_periods, trade_budget=budget)
    selector = TradeSelector(trading_config)
    trades = selector.run(ranked)
    entries = trades[trades["action"] == "entry"]
    entry_raw = test_frame.loc[entries.index, RAW_TARGET_COLUMN]

    rank1 = entries[entries["rank"] == 1]
    tickers1 = rank1.index.get_level_values("ticker")
    print(f"\n=== {label} ===")
    print(f"entries: {len(entries)}  rank-1 trades: {len(rank1)}  "
          f"rank-1 unique tickers: {tickers1.nunique()}")
    if len(rank1):
        top = tickers1.value_counts().head(5).to_dict()
        print(f"rank-1 top tickers: {top}")
        r1 = entry_raw[rank1.index]
        print(f"rank-1 mean return: {r1.mean():+.4%}  median: {np.median(r1):+.4%}  win_rate: {(r1 > 0).mean():.1%}")

    print("continuous rank curve (1-10):")
    for r in range(1, 11):
        mask = (entries["rank"] == r).to_numpy()
        n = int(mask.sum())
        if n:
            rr = entry_raw.to_numpy()[mask]
            print(f"  rank {r:>2}: n={n:>3}  mean={rr.mean():+.4%}  median={np.median(rr):+.4%}")
    return entries


def main() -> None:
    config = BaselineConfig()
    provider = RealDataProvider()
    broad_spec = ExpertSpec(BROAD_EXPERT, None)

    train_table = provider.get_feature_table(TRAIN_START, TRAIN_END, hold_days=config.trading.hold_days)
    train_data = build_expert_data(train_table, broad_spec)
    model = ExpertModel(BROAD_EXPERT, BEST_GENOME, random_seed=config.ga.random_seed).fit(
        train_data.X, train_data.y
    )

    test_table = provider.get_feature_table(TEST_START, TEST_END, hold_days=config.trading.hold_days)
    test_data = build_expert_data(test_table, broad_spec)
    preds = model.predict(test_data.X)
    combiner = OverallModel((broad_spec,))
    combined = combiner.combine({BROAD_EXPERT: preds})

    # Baseline: today's actual ranking (by raw combined_score).
    ranked_baseline = combiner.rank(combined)
    run_curve(ranked_baseline, test_data.frame, config, "BASELINE (rank by raw predicted return)")

    # Risk-adjusted, via OverallModel's own ranking_score option (formalized
    # after this diagnostic confirmed the direction was real) -- combined_score
    # itself stays untouched for long-only + cost-clearing, which need real %
    # units, not a risk-adjusted ratio.
    # A handful of rows have no volatility_20 yet (a ticker's own history
    # starts inside the test window, so its 20d rolling window hasn't
    # filled) -- fill with a deliberately large value so those rows get
    # demoted rather than crash the rank on NaN, since "unknown risk" is
    # not "cheap risk."
    vol = test_data.frame["volatility_20"]
    vol = vol.fillna(vol.max())
    ranked_vol = combiner.rank(combined, ranking_score="vol_adjusted", risk=vol, risk_floor=MIN_VOL)
    run_curve(ranked_vol, test_data.frame, config, "RISK-ADJUSTED (rank by predicted_return / volatility_20)")


if __name__ == "__main__":
    main()
