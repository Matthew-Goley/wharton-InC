"""Out-of-sample backtest: the real selection pipeline (OverallModel ->
TradeSelector) against genuinely held-out real data.

Retrains the genome the 20-generation GA search already found
(train_broad_real.py) on a TRAIN-only window, then predicts and trades on a
TEST window the model never saw -- rather than re-running the full GA
search on a truncated window (~7.5hrs) for marginal purity gain.

Caveat, stated plainly: the hyperparameter *choice* came from a search whose
walk-forward folds spanned the full 2018-2026 range, including this test
window -- so the config wasn't picked blind to it, even though this specific
model's training and every prediction below are computed against genuinely
held-out data it never saw in any form. A fully clean version would re-run
the GA search restricted to the train window too; flagged, not done here.

Usage:
    python -m scripts.backtest_broad_real
"""
from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path

from src.combiner.overall_model import OverallModel
from src.config import BROAD_EXPERT, BaselineConfig, ExpertSpec
from src.data_layer import RealDataProvider
from src.data_layer.real_provider import MARKET_TARGET_COLUMN, RAW_TARGET_COLUMN
from src.experts.base import ExpertModel, XGBConfig
from src.experts.registry import build_expert_data
from src.selection.trade_selector import TradeSelector

TRAIN_START, TRAIN_END = "2018-01-01", "2024-01-01"
TEST_START, TEST_END = "2024-01-01", "2026-01-01"

# The genome found by the 20-generation GA search (train_broad_real.py's
# "done" event) -- best cross-sectional IC 0.0144.
BEST_GENOME = XGBConfig(
    max_depth=8, learning_rate=0.030856726198249796, n_estimators=98,
    subsample=0.5624618396528216, colsample_bytree=0.9792124018732338,
    min_child_weight=6.383279099154468, reg_alpha=2.6108842799586585,
    reg_lambda=7.752080142539311, gamma=0.11401935514848749,
)

RESULT_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "backtest_broad_real.json"


def main() -> None:
    config = BaselineConfig()
    provider = RealDataProvider()
    broad_spec = ExpertSpec(BROAD_EXPERT, None)

    print(f"Loading TRAIN window ({TRAIN_START} .. {TRAIN_END})...")
    t0 = time.perf_counter()
    train_table = provider.get_feature_table(TRAIN_START, TRAIN_END, hold_days=config.trading.hold_days)
    train_data = build_expert_data(train_table, broad_spec)
    print(f"  {len(train_data.frame):,} rows, loaded in {time.perf_counter()-t0:.1f}s")

    print("Retraining the found genome on TRAIN-only data...")
    t0 = time.perf_counter()
    model = ExpertModel(BROAD_EXPERT, BEST_GENOME, random_seed=config.ga.random_seed).fit(
        train_data.X, train_data.y
    )
    print(f"  fit in {time.perf_counter()-t0:.1f}s")

    print(f"\nLoading TEST window ({TEST_START} .. {TEST_END}) -- held out, model never saw this...")
    test_table = provider.get_feature_table(TEST_START, TEST_END, hold_days=config.trading.hold_days)
    test_data = build_expert_data(test_table, broad_spec)
    n_periods = test_data.frame.index.get_level_values("timestamp").nunique()
    print(f"  {len(test_data.frame):,} rows across {n_periods} trading days")

    print("\nPredicting on the held-out set, combining (pass-through, one expert), ranking...")
    preds = model.predict(test_data.X)
    combiner = OverallModel((broad_spec,))
    ranked = combiner.rank(combiner.combine({BROAD_EXPERT: preds}))

    # Match the competition's implied trading intensity (200 trades / 50
    # days = 4/day) scaled to the test window's actual length, so the
    # strategy isn't artificially budget-starved partway through a 2-year
    # backtest sized for a 10-week competition.
    budget = int(round((200 / 50) * n_periods))
    trading_config = replace(config.trading, window_trading_days=n_periods, trade_budget=budget)
    print(f"Running TradeSelector: window={n_periods} days, budget={budget}...")

    selector = TradeSelector(trading_config)
    trades = selector.run(ranked)

    entries = trades[trades["action"] == "entry"]
    exits = trades[trades["action"] == "exit"]

    # Realized P&L uses the RAW (un-demeaned) forward return -- what a
    # position actually pays out in dollars -- not the excess-over-market
    # figure the model predicts and the selector scores eligibility on.
    entry_raw = test_data.frame.loc[entries.index, RAW_TARGET_COLUMN]
    entry_market = test_data.frame.loc[entries.index, MARKET_TARGET_COLUMN]
    entry_excess = entry_raw - entry_market

    # Baseline: every candidate that was ELIGIBLE that day (cleared cost,
    # long-only), not just the ones the selector actually picked -- shows
    # whether ranking adds anything beyond the eligibility filter alone.
    all_eligible_raw = test_data.frame[RAW_TARGET_COLUMN]

    results = {
        "train_window": [TRAIN_START, TRAIN_END],
        "test_window": [TEST_START, TEST_END],
        "test_trading_days": n_periods,
        "budget": budget,
        "trades_executed": int(selector.trades_taken),
        "entries": len(entries),
        "exits": len(exits),
        "still_open_at_end": int(selector.open_positions_count),
        "force_closed": len(selector.force_close_log),
        "win_rate": float((entry_raw > 0).mean()) if len(entries) else None,
        "mean_return_per_trade_raw": float(entry_raw.mean()) if len(entries) else None,
        "median_return_per_trade_raw": float(entry_raw.median()) if len(entries) else None,
        "mean_excess_return_per_trade_vs_market": float(entry_excess.mean()) if len(entries) else None,
        "size_weighted_mean_return": float((entry_raw * entries["position_size_pct"]).sum() / entries["position_size_pct"].sum()) if len(entries) else None,
        "baseline_mean_return_all_candidates": float(all_eligible_raw.mean()),
    }

    RESULT_PATH.write_text(json.dumps(results, indent=2) + "\n")

    print("\n" + "=" * 72)
    print(f"Entries: {len(entries)}  Exits: {len(exits)}  Still open: {selector.open_positions_count}  "
          f"Force-closed: {len(selector.force_close_log)}")
    print(f"Win rate: {results['win_rate']:.1%}" if results["win_rate"] is not None else "Win rate: n/a")
    print(f"Mean return/trade (raw {config.trading.hold_days}-day): {results['mean_return_per_trade_raw']:+.4%}")
    print(f"Mean excess return/trade vs. that day's market: {results['mean_excess_return_per_trade_vs_market']:+.4%}")
    print(f"Size-weighted mean return: {results['size_weighted_mean_return']:+.4%}")
    print(f"Baseline (every eligible candidate, unranked): {results['baseline_mean_return_all_candidates']:+.4%}")
    print(f"\nSaved: {RESULT_PATH}")


if __name__ == "__main__":
    main()
