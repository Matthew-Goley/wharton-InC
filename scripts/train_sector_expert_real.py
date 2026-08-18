"""DEPRECATED (baseline_system_architecture.md open question 35) -- this
checkpoint's own job (validate the training path end to end before
committing compute to the remaining sectors) was completed once step 3
(``scripts/train_all_sector_experts_real.py``) ran successfully; step 3
now trains this same sector (Information Technology) as part of the full
roster, deriving it live from ``config.EXPERTS`` rather than hand-picking
it, so nothing here is still the unique source of anything. Not deleted
(historical record -- this is the actual IT checkpoint referenced from
``src/config.py``'s ``EXPERTS`` docstring and elsewhere), but do not run
it expecting a current result: it still has gap #5's bare
``combiner.rank(combined)`` call (open questions 26/30), deliberately left
unfixed here rather than threaded with ``risk`` like the still-active
scripts, since fixing a script whose job is already done just to keep it
runnable isn't a good use of that effort. If this checkpoint's validation
role is ever needed again (e.g. debugging a single sector in isolation
without paying for all 10), fix the bare call first.

Step 2 checkpoint: train ONE sector expert (Information Technology, the
largest sector at 90/90 tickers with both sector and price data) and run it
through the real OverallModel + TradeSelector + diagnostic pipeline for the
first time with genuine ``has_sector=True`` rows.

The point isn't a conclusion about IT stocks -- it's validating the training
path end to end (sector slicing, GA search on a much smaller universe, the
combiner's join/agreement-bonus logic exercised on real non-degenerate data
for the first time, diagnostics running without crashing) for the cost of
one model, before committing compute to the remaining ~10 sectors (step 3).

GA search is deliberately SMALL (population=12, generations=6 -- ~14x fewer
genome evaluations than the full population=50/generations=20 search), per
the explicit instruction not to spend real search compute until step 5
(uncertainty-adjusted ranking) validates its central bet. The broad expert
is NOT re-searched -- it reuses the genome the full search already found
(BEST_GENOME, imported from backtest_broad_real), retrained on TRAIN-only
data exactly like that script does, since re-validating the broad model
isn't this checkpoint's job.

Usage:
    python -m scripts.train_sector_expert_real
"""
from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path

import joblib

from scripts.backtest_broad_real import (
    BEST_GENOME,
    TEST_END,
    TEST_START,
    TRAIN_END,
    TRAIN_START,
    _rank_bucket_stats,
    _split_sample_validation,
)
from src.combiner.overall_model import OverallModel
from src.config import BROAD_EXPERT, BaselineConfig, EXPERTS, ExpertSpec
from src.data_layer import RealDataProvider
from src.data_layer.real_provider import MARKET_TARGET_COLUMN, RAW_TARGET_COLUMN
from src.evaluation.metrics import cross_sectional_ic
from src.experts.base import ExpertModel
from src.experts.registry import build_expert_data
from src.ga.search import GeneticSearch, default_max_workers
from src.selection.trade_selector import TradeSelector

DATA_PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"
LOG_PATH = DATA_PROCESSED / "sector_expert_it_search_log.jsonl"
MODEL_PATH = DATA_PROCESSED / "sector_expert_it_model.joblib"
RESULT_PATH = DATA_PROCESSED / "backtest_sector_it_real.json"

#: The one sector expert this checkpoint trains -- pulled from the real
#: roster (config.EXPERTS) rather than hand-typed, so its name/sector string
#: can never drift from what step 3 will use for the same sector.
IT_SPEC = next(spec for spec in EXPERTS if spec.sector == "Information Technology")

#: ~14x fewer genome evaluations than the full search (12*6=72 vs 50*20=1000)
#: on top of a ~7x smaller universe slice (90 tickers vs 653) -- a cheap
#: checkpoint, not a real search. Re-run with the full GAConfig in step 3
#: only after step 5 validates the design's central bet.
SMALL_GA = dict(population_size=12, n_generations=6, elite_count=2, tournament_size=3)


def log(record: dict) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a") as f:
        f.write(json.dumps(record) + "\n")
    print(record)


def main() -> None:
    config = BaselineConfig()
    ga_config = replace(config.ga, **SMALL_GA)
    workers = default_max_workers()
    provider = RealDataProvider()
    broad_spec = ExpertSpec(BROAD_EXPERT, None)

    print(f"Loading TRAIN window ({TRAIN_START} .. {TRAIN_END})...")
    t0 = time.perf_counter()
    train_table = provider.get_feature_table(TRAIN_START, TRAIN_END, hold_days=config.trading.hold_days)
    broad_train = build_expert_data(train_table, broad_spec)
    it_train = build_expert_data(train_table, IT_SPEC)
    print(f"  broad: {len(broad_train.frame):,} rows  |  {IT_SPEC.name} ({IT_SPEC.sector}): "
          f"{len(it_train.frame):,} rows, {it_train.frame.index.get_level_values('ticker').nunique()} tickers "
          f"-- loaded in {time.perf_counter()-t0:.1f}s")
    if it_train.frame.empty:
        raise ValueError(
            f"Sector filter '{IT_SPEC.sector}' matched zero TRAIN rows -- sector_map.py / "
            f"RealDataProvider's sector wiring is broken, stop here rather than silently training on nothing."
        )

    print("\nRetraining the broad expert's already-found genome on TRAIN-only data...")
    t0 = time.perf_counter()
    broad_model = ExpertModel(BROAD_EXPERT, BEST_GENOME, random_seed=config.ga.random_seed).fit(
        broad_train.X, broad_train.y
    )
    print(f"  fit in {time.perf_counter()-t0:.1f}s")

    print(f"\nSmall GA search for {IT_SPEC.name} ({SMALL_GA['population_size']} x "
          f"{SMALL_GA['n_generations']} = {SMALL_GA['population_size']*SMALL_GA['n_generations']} "
          f"evaluations, workers={workers})...")
    t0 = time.perf_counter()
    log({"event": "start", "expert": IT_SPEC.name, "sector": IT_SPEC.sector, "rows": len(it_train.frame),
         "tickers": int(it_train.frame.index.get_level_values("ticker").nunique()),
         "workers": workers, **SMALL_GA})

    def on_generation(record) -> None:
        log({
            "event": "generation", "expert": IT_SPEC.name, "generation": record.generation,
            "best_fitness": record.best_fitness, "mean_fitness": record.mean_fitness,
            "best_genome": record.best_genome.as_xgb_params(),
            "elapsed_s": round(time.perf_counter() - t0, 1),
        })

    search = GeneticSearch(ga_config, config.hyperparam_space, config.walk_forward, max_workers=workers)
    result = search.run(it_train.frame, train_table.feature_columns, train_table.target_column,
                         on_generation=on_generation)
    log({"event": "done", "expert": IT_SPEC.name, "best_fitness": result.best_fitness,
         "best_genome": result.best_genome.as_xgb_params(), "elapsed_s": round(time.perf_counter() - t0, 1)})
    print(f"  done in {time.perf_counter()-t0:.1f}s -- best fitness (cross-sectional IC): {result.best_fitness:+.4f}")

    print(f"\nTraining the final {IT_SPEC.name} expert on the best config found...")
    it_model = ExpertModel(IT_SPEC.name, result.best_genome, random_seed=config.ga.random_seed).fit(
        it_train.X, it_train.y
    )
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"broad": broad_model, IT_SPEC.name: it_model}, MODEL_PATH)

    print(f"\nLoading TEST window ({TEST_START} .. {TEST_END}) -- held out, neither model saw this...")
    test_table = provider.get_feature_table(TEST_START, TEST_END, hold_days=config.trading.hold_days)
    broad_test = build_expert_data(test_table, broad_spec)
    it_test = build_expert_data(test_table, IT_SPEC)
    n_periods = broad_test.frame.index.get_level_values("timestamp").nunique()
    print(f"  broad: {len(broad_test.frame):,} rows across {n_periods} days  |  "
          f"{IT_SPEC.name}: {len(it_test.frame):,} rows, "
          f"{it_test.frame.index.get_level_values('ticker').nunique()} tickers")

    broad_preds = broad_model.predict(broad_test.X)
    it_preds = it_model.predict(it_test.X)

    oos_ic_broad = cross_sectional_ic(
        broad_test.frame.index.get_level_values("timestamp").to_numpy(),
        broad_test.y.to_numpy(), broad_preds["predicted_return"].to_numpy(),
    )
    oos_ic_it = cross_sectional_ic(
        it_test.frame.index.get_level_values("timestamp").to_numpy(),
        it_test.y.to_numpy(), it_preds["predicted_return"].to_numpy(),
    )
    print(f"  out-of-sample cross-sectional IC -- broad: {oos_ic_broad:+.4f}  {IT_SPEC.name}: {oos_ic_it:+.4f}")

    print("\nCombining via the real OverallModel -- has_sector=True for real rows, for the first time...")
    combiner = OverallModel((broad_spec, IT_SPEC))
    combined = combiner.combine({BROAD_EXPERT: broad_preds, IT_SPEC.name: it_preds})
    n_with_sector = int((combined["n_experts"] == 2).sum())
    print(f"  {n_with_sector:,} / {len(combined):,} rows have a sector-expert contribution "
          f"({n_with_sector / len(combined):.1%}) -- should equal {IT_SPEC.name}'s TEST row count if the "
          f"join is correct.")
    if n_with_sector != len(it_test.frame):
        print(f"  ** MISMATCH: expected {len(it_test.frame):,} (== {IT_SPEC.name} TEST rows). "
              f"Check the sector join / index alignment before trusting anything below. **")

    ranked = combiner.rank(combined)

    budget = int(round((200 / 50) * n_periods))
    trading_config = replace(config.trading, window_trading_days=n_periods, trade_budget=budget)
    print(f"Running TradeSelector: window={n_periods} days, budget={budget}...")
    selector = TradeSelector(trading_config)
    trades = selector.run(ranked)

    entries = trades[trades["action"] == "entry"]
    exits = trades[trades["action"] == "exit"]
    # entries only ever come from `combined`'s (ticker, timestamp) index,
    # which is broad_test's full index (broad always contributes) -- so
    # entry P&L always comes from broad_test's frame, sector membership or
    # not.
    entry_raw = broad_test.frame.loc[entries.index, RAW_TARGET_COLUMN]
    entry_market = broad_test.frame.loc[entries.index, MARKET_TARGET_COLUMN]
    entry_excess = entry_raw - entry_market
    entry_has_sector = combined.loc[entries.index, "n_experts"] == 2

    rank_bucket_stats = _rank_bucket_stats(entries, entry_raw)
    all_trading_days = broad_test.frame.index.get_level_values("timestamp").unique().to_numpy()
    split_validation = _split_sample_validation(entries, entry_raw, all_trading_days, config.trading.hold_days)

    results = {
        "checkpoint": "step2_single_sector_expert",
        "sector_expert": {"name": IT_SPEC.name, "sector": IT_SPEC.sector, "ga": SMALL_GA,
                           "best_genome": result.best_genome.as_xgb_params(), "train_fitness": result.best_fitness},
        "train_window": [TRAIN_START, TRAIN_END],
        "test_window": [TEST_START, TEST_END],
        "test_trading_days": n_periods,
        "out_of_sample_cross_sectional_ic_broad": oos_ic_broad,
        f"out_of_sample_cross_sectional_ic_{IT_SPEC.name}": oos_ic_it,
        "rows_with_sector_expert": n_with_sector,
        "rows_total": len(combined),
        "budget": budget,
        "trades_executed": int(selector.trades_taken),
        "entries": len(entries),
        "exits": len(exits),
        "entries_with_sector_expert": int(entry_has_sector.sum()),
        "win_rate": float((entry_raw > 0).mean()) if len(entries) else None,
        "mean_return_per_trade_raw": float(entry_raw.mean()) if len(entries) else None,
        "median_return_per_trade_raw": float(entry_raw.median()) if len(entries) else None,
        "mean_excess_return_per_trade_vs_market": float(entry_excess.mean()) if len(entries) else None,
        "return_by_rank_bucket": rank_bucket_stats,
        "split_sample_validation": split_validation,
    }
    RESULT_PATH.write_text(json.dumps(results, indent=2) + "\n")

    print("\n" + "=" * 72)
    print(f"[{IT_SPEC.name}] out-of-sample IC: broad {oos_ic_broad:+.4f}  sector {oos_ic_it:+.4f}")
    print(f"Entries: {len(entries)}  ({int(entry_has_sector.sum())} with sector-expert input)  Exits: {len(exits)}")
    print(f"Win rate: {results['win_rate']:.1%}" if results["win_rate"] is not None else "Win rate: n/a")
    print(f"Mean return/trade (raw {config.trading.hold_days}-day): {results['mean_return_per_trade_raw']:+.4%}")
    print("\nReturn by rank bucket:")
    for row in rank_bucket_stats:
        if row["n"]:
            print(f"  rank {row['rank_range']:>10}: n={row['n']:>4}  mean={row['mean_return']:+.4%}  "
                  f"median={row['median_return']:+.4%}  win_rate={row['win_rate']:.1%}")
    print(f"\nSaved: {RESULT_PATH}")
    print(f"Model:  {MODEL_PATH}")
    print(f"Log:    {LOG_PATH}")


if __name__ == "__main__":
    main()
