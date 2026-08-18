"""Step 3: train the full sector-expert roster (all 11 sectors in
config.EXPERTS), then run the real combine -> rank -> select -> diagnose
pipeline with genuine multi-sector has_sector=True coverage for the first
time.

Straight to full coverage, not one checkpoint per expert added -- partial
coverage is a structural confound (a bonus multiplier only reachable by
covered tickers over-selects them regardless of merit, and the over-
selection ratio shifts with every sector added), so intermediate
checkpoints wouldn't be comparable to each other or to the final result.
Only the full-coverage run is interpretable; see baseline_system_architecture.md
open question 8 and the "Agreement bonus on vs. off" ablation row.

Same deliberately SMALL GA per expert as the step-2 checkpoint (12 x 6 = 72
evaluations) -- eleven more small searches, not eleven full ones. The broad
expert reuses its already-found genome, exactly as before.

Usage:
    python -m scripts.train_all_sector_experts_real
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
from src.config import BROAD_EXPERT, BaselineConfig, EXPERTS
from src.data_layer import RealDataProvider
from src.data_layer.real_provider import MARKET_TARGET_COLUMN, RAW_TARGET_COLUMN
from src.evaluation.metrics import cross_sectional_ic
from src.evaluation.selection_diff import decompose_selection_change
from src.experts.base import ExpertModel
from src.experts.registry import build_expert_data
from src.ga.search import GeneticSearch, default_max_workers
from src.selection.trade_selector import TradeSelector

DATA_PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"
LOG_PATH = DATA_PROCESSED / "all_sector_experts_search_log.jsonl"
MODEL_PATH = DATA_PROCESSED / "all_sector_experts_models.joblib"
RESULT_PATH = DATA_PROCESSED / "backtest_full_sector_real.json"
BROAD_BASELINE_PATH = DATA_PROCESSED / "backtest_broad_real.json"

SECTOR_SPECS = tuple(spec for spec in EXPERTS if spec.sector is not None)

#: Same budget as the step-2 checkpoint, per sector expert -- see that
#: script's docstring for the reasoning (~14x fewer evaluations than the
#: full population=50/generations=20 search).
SMALL_GA = dict(population_size=12, n_generations=6, elite_count=2, tournament_size=3)


def log(record: dict) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a") as f:
        f.write(json.dumps(record) + "\n")
    print(record)


def train_one_sector(spec, train_table, config, ga_config, workers: int, t_start: float):
    edata = build_expert_data(train_table, spec)
    n_tickers = edata.frame.index.get_level_values("ticker").nunique() if len(edata.frame) else 0
    print(f"\n[{spec.name}] sector={spec.sector!r}  rows={len(edata.frame):,}  tickers={n_tickers}")
    if edata.frame.empty:
        log({"event": "skipped_empty", "expert": spec.name, "sector": spec.sector})
        print(f"  ** SKIPPED: zero TRAIN rows for sector {spec.sector!r} -- check sector_map.py coverage. **")
        return None, None

    log({"event": "start", "expert": spec.name, "sector": spec.sector, "rows": len(edata.frame),
         "tickers": n_tickers, "workers": workers, **SMALL_GA})

    def on_generation(record) -> None:
        log({"event": "generation", "expert": spec.name, "generation": record.generation,
             "best_fitness": record.best_fitness, "mean_fitness": record.mean_fitness,
             "elapsed_s": round(time.perf_counter() - t_start, 1)})

    t0 = time.perf_counter()
    search = GeneticSearch(ga_config, config.hyperparam_space, config.walk_forward, max_workers=workers)
    result = search.run(edata.frame, train_table.feature_columns, train_table.target_column,
                         on_generation=on_generation)
    fit_s = time.perf_counter() - t0
    log({"event": "done", "expert": spec.name, "best_fitness": result.best_fitness,
         "best_genome": result.best_genome.as_xgb_params(), "elapsed_s": round(fit_s, 1)})
    print(f"  GA done in {fit_s:.1f}s -- best fitness (cross-sectional IC): {result.best_fitness:+.4f}")

    model = ExpertModel(spec.name, result.best_genome, random_seed=config.ga.random_seed).fit(edata.X, edata.y)
    return model, result


def main() -> None:
    config = BaselineConfig()
    ga_config = replace(config.ga, **SMALL_GA)
    workers = default_max_workers()
    provider = RealDataProvider()
    broad_spec = next(s for s in EXPERTS if s.sector is None)

    print(f"Loading TRAIN window ({TRAIN_START} .. {TRAIN_END})...")
    t0 = time.perf_counter()
    train_table = provider.get_feature_table(TRAIN_START, TRAIN_END, hold_days=config.trading.hold_days)
    print(f"  {len(train_table.frame):,} rows loaded in {time.perf_counter()-t0:.1f}s")

    broad_train = build_expert_data(train_table, broad_spec)
    print("\nRetraining the broad expert's already-found genome on TRAIN-only data...")
    t0 = time.perf_counter()
    broad_model = ExpertModel(BROAD_EXPERT, BEST_GENOME, random_seed=config.ga.random_seed).fit(
        broad_train.X, broad_train.y
    )
    print(f"  fit in {time.perf_counter()-t0:.1f}s")

    models = {BROAD_EXPERT: broad_model}
    sector_fitness: dict[str, float] = {}
    t_start = time.perf_counter()
    for spec in SECTOR_SPECS:
        model, result = train_one_sector(spec, train_table, config, ga_config, workers, t_start)
        if model is not None:
            models[spec.name] = model
            sector_fitness[spec.name] = result.best_fitness

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(models, MODEL_PATH)
    print(f"\nAll experts trained in {time.perf_counter()-t_start:.1f}s total. Saved: {MODEL_PATH}")

    trained_sector_specs = [s for s in SECTOR_SPECS if s.name in models]
    combiner = OverallModel(tuple([broad_spec] + trained_sector_specs))

    print(f"\nLoading TEST window ({TEST_START} .. {TEST_END}) -- held out, nothing saw this...")
    test_table = provider.get_feature_table(TEST_START, TEST_END, hold_days=config.trading.hold_days)
    broad_test = build_expert_data(test_table, broad_spec)
    n_periods = broad_test.frame.index.get_level_values("timestamp").nunique()

    expert_predictions = {BROAD_EXPERT: broad_model.predict(broad_test.X)}
    oos_ic = {}
    oos_ic["broad"] = cross_sectional_ic(
        broad_test.frame.index.get_level_values("timestamp").to_numpy(),
        broad_test.y.to_numpy(), expert_predictions[BROAD_EXPERT]["predicted_return"].to_numpy(),
    )
    for spec in trained_sector_specs:
        edata = build_expert_data(test_table, spec)
        if edata.frame.empty:
            continue
        preds = models[spec.name].predict(edata.X)
        expert_predictions[spec.name] = preds
        oos_ic[spec.name] = cross_sectional_ic(
            edata.frame.index.get_level_values("timestamp").to_numpy(), edata.y.to_numpy(),
            preds["predicted_return"].to_numpy(),
        )
    print("\nOut-of-sample cross-sectional IC per expert:")
    for name, ic in oos_ic.items():
        print(f"  {name:>24}: {ic:+.4f}")

    print("\nCombining via the real OverallModel -- full sector coverage, for the first time...")
    combined = combiner.combine(expert_predictions)
    n_with_sector = int((combined["n_experts"] == 2).sum())
    print(f"  {n_with_sector:,} / {len(combined):,} rows ({n_with_sector/len(combined):.1%}) have a sector-expert "
          f"contribution.")
    # Read the standing ranking default from config rather than a bare
    # call (baseline_system_architecture.md open questions 30/33 -- one of
    # the four bare combiner.rank(combined) sites gap #5 named as still
    # unfixed; fixed here the same way train_baseline.py was).
    ranking_score = config.trading.ranking_score
    risk = test_table.frame["volatility_20"].reindex(combined.index) if ranking_score == "vol_adjusted" else None
    ranked = combiner.rank(combined, ranking_score=ranking_score, risk=risk)

    budget = int(round((200 / 50) * n_periods))
    trading_config = replace(config.trading, window_trading_days=n_periods, trade_budget=budget)
    print(f"Running TradeSelector: window={n_periods} days, budget={budget}...")
    selector = TradeSelector(trading_config)
    trades = selector.run(ranked)
    entries = trades[trades["action"] == "entry"]
    exits = trades[trades["action"] == "exit"]

    entry_raw = broad_test.frame.loc[entries.index, RAW_TARGET_COLUMN]
    entry_market = broad_test.frame.loc[entries.index, MARKET_TARGET_COLUMN]
    entry_excess = entry_raw - entry_market
    entry_has_sector = combined.loc[entries.index, "n_experts"] == 2

    rank_bucket_stats = _rank_bucket_stats(entries, entry_raw)
    all_trading_days = broad_test.frame.index.get_level_values("timestamp").unique().to_numpy()
    split_validation = _split_sample_validation(entries, entry_raw, all_trading_days, config.trading.hold_days)

    # Compare against the original broad-only baseline via the standard
    # decomposition, not a headline mean -- same discipline the agreement-
    # bonus fix got, applied to the actual step-3 comparison this time.
    diff_vs_broad = None
    if BROAD_BASELINE_PATH.exists():
        # The broad-only baseline script doesn't persist its per-trade entry
        # index, only aggregate stats -- so recompute broad-only entries here
        # from the same already-fit broad_model, on the same TEST window, for
        # an apples-to-apples comparison against this run's entries.
        broad_only_combiner = OverallModel((broad_spec,))
        broad_only_combined = broad_only_combiner.combine({BROAD_EXPERT: expert_predictions[BROAD_EXPERT]})
        broad_only_risk = (
            test_table.frame["volatility_20"].reindex(broad_only_combined.index)
            if ranking_score == "vol_adjusted" else None
        )
        broad_only_ranked = broad_only_combiner.rank(
            broad_only_combined, ranking_score=ranking_score, risk=broad_only_risk
        )
        broad_only_trades = TradeSelector(trading_config).run(broad_only_ranked)
        broad_only_entries = broad_only_trades[broad_only_trades["action"] == "entry"]
        diff_vs_broad = decompose_selection_change(
            broad_only_entries, entries, broad_test.frame[RAW_TARGET_COLUMN], all_trading_days,
            config.trading.hold_days,
        )
        diff_vs_broad["label_a"] = "broad_only"
        diff_vs_broad["label_b"] = "full_sector_coverage"

    results = {
        "checkpoint": "step3_full_sector_coverage",
        "n_sector_experts_trained": len(trained_sector_specs),
        "sector_experts": [s.name for s in trained_sector_specs],
        "train_window": [TRAIN_START, TRAIN_END],
        "test_window": [TEST_START, TEST_END],
        "test_trading_days": n_periods,
        "out_of_sample_cross_sectional_ic": oos_ic,
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
        "decomposition_vs_broad_only": diff_vs_broad,
    }
    RESULT_PATH.write_text(json.dumps(results, indent=2) + "\n")

    print("\n" + "=" * 72)
    print(f"Entries: {len(entries)}  ({int(entry_has_sector.sum())} with sector-expert input)  Exits: {len(exits)}")
    print(f"Win rate: {results['win_rate']:.1%}" if results["win_rate"] is not None else "Win rate: n/a")
    print(f"Mean return/trade (raw {config.trading.hold_days}-day): {results['mean_return_per_trade_raw']:+.4%}")
    print("\nReturn by rank bucket:")
    for row in rank_bucket_stats:
        if row["n"]:
            print(f"  rank {row['rank_range']:>10}: n={row['n']:>4}  mean={row['mean_return']:+.4%}  "
                  f"median={row['median_return']:+.4%}  win_rate={row['win_rate']:.1%}")
    if diff_vs_broad:
        print("\nDecomposition vs. broad-only baseline (same TEST window, same broad model):")
        print(f"  common: n={diff_vs_broad['common']['n']}  mean={diff_vs_broad['common']['mean_return']}")
        print(f"  only broad-only: n={diff_vs_broad['only_a']['n']}  "
              f"blocks={diff_vs_broad['only_a']['distinct_blocks']}  mean={diff_vs_broad['only_a']['mean_return']}")
        print(f"  only full-coverage: n={diff_vs_broad['only_b']['n']}  "
              f"blocks={diff_vs_broad['only_b']['distinct_blocks']}  mean={diff_vs_broad['only_b']['mean_return']}")
        if diff_vs_broad["significance_only_a_vs_only_b"]:
            print(f"  block-effective t: {diff_vs_broad['significance_only_a_vs_only_b']['block_effective_t']:+.2f}")
    print(f"\nSaved: {RESULT_PATH}")
    print(f"Models: {MODEL_PATH}")
    print(f"Log:    {LOG_PATH}")


if __name__ == "__main__":
    main()
