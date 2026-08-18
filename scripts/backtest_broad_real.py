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

FINDING from the first real run (2024-01-01..2026-01-01 holdout): the model's
ranking ability held up out of sample (cross-sectional IC +0.021, even better
than the +0.014 it scored during training) -- but the selector's actual
trades still lost money on average (-0.13%/trade). The gap is entirely
rank-1: the single most-confident pick each day scored -4.1% mean / -3.1%
median / 29% win rate, systematically bad, not a few outliers -- while ranks
2-10 were all flat-to-positive (best: rank 3-5 at +0.86% mean, 53% win
rate). The model's *ranking* is real; blindly taking its single most extreme
prediction each day is not a good way to use it.

SPLIT-SAMPLE VALIDATION (2024 vs. 2025, each independent of the other and of
the pooled analysis above that first surfaced the pattern): rank-1-is-bad and
rank-3-5-is-good both hold their direction in both years separately --
genuine evidence, not a pooled-window fluke. Ranks 5-10 and 10+ do NOT: they
looked fine pooled (blending a good 2024 and a bad 2025) but flip sign
between the two years on their own, which means the pooled number for those
buckets was hiding a regime flip, not describing a dependable pattern. Only
trust rank 1 (avoid) and rank 3-5 (favor) from this; everything else needs
more data before it means anything. See return_by_rank_bucket and
split_sample_validation in the saved JSON before trusting any future run's
headline numbers.

Usage:
    python -m scripts.backtest_broad_real
"""
from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from src.combiner.overall_model import OverallModel
from src.config import BROAD_EXPERT, BaselineConfig, ExpertSpec
from src.data_layer import RealDataProvider
from src.data_layer.real_provider import MARKET_TARGET_COLUMN, RAW_TARGET_COLUMN
from src.evaluation.metrics import cross_sectional_ic
from src.experts.base import ExpertModel, XGBConfig
from src.experts.registry import build_expert_data
from src.selection.trade_selector import TradeSelector

#: Rank-1 (the model's single most confident pick each day) has measured
#: systematically worse -- not just outlier-driven -- than ranks 2-10 on the
#: 2024-2026 holdout: median -3.08% vs. flat-to-positive, win rate 29% vs.
#: 50%+. Broken out explicitly rather than folded into one aggregate number,
#: since that aggregate hid this entirely the first time.
RANK_BUCKETS = [(0, 1), (1, 3), (3, 5), (5, 10), (10, 10_000)]

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


def _rank_bucket_stats(entries: pd.DataFrame, entry_raw: pd.Series) -> list[dict]:
    """Mean/median/win-rate return per RANK_BUCKETS slice, over whatever
    subset of `entries` (already index-aligned with `entry_raw`) is passed
    in -- the full window, or one sub-period of it."""
    stats = []
    for lo, hi in RANK_BUCKETS:
        mask = ((entries["rank"] > lo) & (entries["rank"] <= hi)).to_numpy()
        bucket_returns = entry_raw.to_numpy()[mask]
        stats.append({
            "rank_range": f"({lo}, {hi}]",
            "n": int(mask.sum()),
            "mean_return": float(bucket_returns.mean()) if len(bucket_returns) else None,
            "median_return": float(np.median(bucket_returns)) if len(bucket_returns) else None,
            "win_rate": float((bucket_returns > 0).mean()) if len(bucket_returns) else None,
        })
    return stats


def _split_sample_validation(
    entries: pd.DataFrame, entry_raw: pd.Series, all_trading_days: np.ndarray, hold_days: int
) -> dict:
    """Splits entries by calendar year and computes rank-bucket stats
    independently in each -- checks whether a bucket's direction (not just
    its pooled-window magnitude) actually replicates, since a pooled number
    can blend a good year and a bad year into something that looks like a
    stable, mediocre pattern when neither year alone supports that.

    Purged/embargoed at the boundary: an entry within the last `hold_days`
    TRADING days of a year has its hold_days-forward label resolving into
    the following year, so it can't be cleanly attributed to either side --
    it's dropped from the split rather than left to blur whichever side
    it's nominally dated into (the year split is otherwise not a clean
    train/test-style boundary, since the label horizon crosses it).
    """
    dates = entries.index.get_level_values("timestamp")
    all_years = pd.DatetimeIndex(all_trading_days).year
    years = sorted(pd.unique(all_years))

    embargoed_days = set()
    for year in years[:-1]:  # the last year has no "next year" to bleed into
        year_days = np.sort(all_trading_days[all_years == year])
        embargoed_days.update(year_days[-hold_days:] if len(year_days) >= hold_days else year_days)

    embargo_mask = dates.isin(embargoed_days)
    kept, kept_raw = entries[~embargo_mask], entry_raw[~embargo_mask]
    kept_dates = kept.index.get_level_values("timestamp")

    by_year = {}
    for year in years:
        mask = np.asarray(kept_dates.year == year)
        by_year[str(year)] = _rank_bucket_stats(kept[mask], kept_raw[mask])

    consistent = {}
    for i, (lo, hi) in enumerate(RANK_BUCKETS):
        means = [by_year[str(y)][i]["mean_return"] for y in years]
        means = [m for m in means if m is not None]
        consistent[f"({lo}, {hi}]"] = bool(means) and (all(m > 0 for m in means) or all(m < 0 for m in means))

    return {
        "by_year": by_year,
        "direction_consistent_across_years": consistent,
        "embargoed_entries": int(embargo_mask.sum()),
    }


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

    # Isolates model ranking quality from selection-pipeline behavior --
    # measured against ALL test rows, before any eligibility/budget/holding
    # constraints touch anything, so a bad backtest number can't be
    # misdiagnosed as "the model doesn't generalize" when it's actually the
    # selection logic (or vice versa).
    oos_ic = cross_sectional_ic(
        test_data.frame.index.get_level_values("timestamp").to_numpy(),
        test_data.y.to_numpy(),
        preds["predicted_return"].to_numpy(),
    )
    print(f"  out-of-sample cross-sectional IC (model ranking quality alone): {oos_ic:+.4f}")

    combiner = OverallModel((broad_spec,))
    combined = combiner.combine({BROAD_EXPERT: preds})
    # Read the standing ranking default from config rather than a bare
    # call (baseline_system_architecture.md open questions 30/33 -- this
    # was one of the four bare combiner.rank(combined) sites gap #5 named
    # as still unfixed; fixed here the same way train_baseline.py was).
    ranking_score = config.trading.ranking_score
    risk = test_table.frame["volatility_20"].reindex(combined.index) if ranking_score == "vol_adjusted" else None
    ranked = combiner.rank(combined, ranking_score=ranking_score, risk=risk)

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

    rank_bucket_stats = _rank_bucket_stats(entries, entry_raw)
    all_trading_days = test_data.frame.index.get_level_values("timestamp").unique().to_numpy()
    split_validation = _split_sample_validation(entries, entry_raw, all_trading_days, config.trading.hold_days)

    results = {
        "train_window": [TRAIN_START, TRAIN_END],
        "test_window": [TEST_START, TEST_END],
        "test_trading_days": n_periods,
        "out_of_sample_cross_sectional_ic": oos_ic,
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
        "return_by_rank_bucket": rank_bucket_stats,
        "split_sample_validation": split_validation,
    }

    RESULT_PATH.write_text(json.dumps(results, indent=2) + "\n")

    print("\n" + "=" * 72)
    print(f"Out-of-sample cross-sectional IC: {oos_ic:+.4f}")
    print(f"Entries: {len(entries)}  Exits: {len(exits)}  Still open: {selector.open_positions_count}  "
          f"Force-closed: {len(selector.force_close_log)}")
    print(f"Win rate: {results['win_rate']:.1%}" if results["win_rate"] is not None else "Win rate: n/a")
    print(f"Mean return/trade (raw {config.trading.hold_days}-day): {results['mean_return_per_trade_raw']:+.4%}")
    print(f"Mean excess return/trade vs. that day's market: {results['mean_excess_return_per_trade_vs_market']:+.4%}")
    print(f"Size-weighted mean return: {results['size_weighted_mean_return']:+.4%}")
    print(f"Baseline (every eligible candidate, unranked): {results['baseline_mean_return_all_candidates']:+.4%}")
    print("\nReturn by rank bucket (is the top pick actually the best pick?):")
    for row in rank_bucket_stats:
        if row["n"]:
            print(f"  rank {row['rank_range']:>10}: n={row['n']:>4}  mean={row['mean_return']:+.4%}  "
                  f"median={row['median_return']:+.4%}  win_rate={row['win_rate']:.1%}")

    print(f"\nSplit-sample validation (embargoed {split_validation['embargoed_entries']} entries within "
          f"{config.trading.hold_days} trading days of the year boundary -- their label resolves into the next "
          f"year, so they can't be cleanly attributed to either side):")
    for year, year_stats in split_validation["by_year"].items():
        print(f"  {year}:")
        for row in year_stats:
            if row["n"]:
                print(f"    rank {row['rank_range']:>10}: n={row['n']:>4}  mean={row['mean_return']:+.4%}  "
                      f"win_rate={row['win_rate']:.1%}")
    print("  consistent across years:", {
        k: ("YES" if v else "no") for k, v in split_validation["direction_consistent_across_years"].items()
    })
    print(f"\nSaved: {RESULT_PATH}")


if __name__ == "__main__":
    main()
