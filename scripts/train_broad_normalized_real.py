"""Step 1 of the cross-sectional-normalization plan: retrain the broad
expert on cross-sectionally-normalized features (RealDataProvider's new
``normalize_features=True``), then measure PARA's raw daily argmax share
on the same TEST window as every other checkpoint here (245/497 = 49.3%
with raw features -- see baseline_system_architecture.md's ablation table).

This is a "confirm or kill" gate, not a full checkpoint: if PARA's argmax
share doesn't drop meaningfully, the normalization diagnosis is wrong and
steps 2 onward (full diagnostic suite, retraining every sector expert on
normalized features) aren't worth running. Small GA (12x6=72 evaluations,
same budget as every other checkpoint since this project's rule "keep the
GA small" hasn't cleared its own gate yet either) -- cheap enough that a
kill here costs little.

Usage:
    python -m scripts.train_broad_normalized_real
"""
from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path

import joblib

from src.config import BROAD_EXPERT, BaselineConfig, ExpertSpec
from src.data_layer import RealDataProvider
from src.evaluation.metrics import cross_sectional_ic
from src.experts.base import ExpertModel
from src.experts.registry import build_expert_data
from src.ga.search import GeneticSearch, default_max_workers
from scripts.backtest_broad_real import TEST_START, TEST_END, TRAIN_START, TRAIN_END

DATA_PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"
LOG_PATH = DATA_PROCESSED / "broad_normalized_search_log.jsonl"
MODEL_PATH = DATA_PROCESSED / "broad_normalized_model.joblib"
RESULT_PATH = DATA_PROCESSED / "broad_normalized_argmax_check.json"

#: Same small budget as every sector-expert checkpoint -- see module docstring.
SMALL_GA = dict(population_size=12, n_generations=6, elite_count=2, tournament_size=3)

#: The number this gate is checked against -- PARA's share of the raw daily
#: argmax winner (before any trading mechanics) on RAW, unnormalized
#: features, measured in scripts/rerun_full_sector_with_ticker_cap.py.
BASELINE_PARA_ARGMAX_DAYS = 245
BASELINE_TOTAL_DAYS = 497


def log(record: dict) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a") as f:
        f.write(json.dumps(record) + "\n")
    print(record)


def main() -> None:
    config = BaselineConfig()
    ga_config = replace(config.ga, **SMALL_GA)
    workers = default_max_workers()
    provider = RealDataProvider(normalize_features=True)
    broad_spec = ExpertSpec(BROAD_EXPERT, None)

    print("Loading TRAIN window with normalize_features=True...")
    t0 = time.perf_counter()
    train_table = provider.get_feature_table(TRAIN_START, TRAIN_END, hold_days=config.trading.hold_days)
    train_data = build_expert_data(train_table, broad_spec)
    print(f"  {len(train_data.frame):,} rows loaded in {time.perf_counter()-t0:.1f}s")

    print(f"\nSmall GA search ({SMALL_GA['population_size']}x{SMALL_GA['n_generations']}="
          f"{SMALL_GA['population_size']*SMALL_GA['n_generations']} evaluations, workers={workers})...")
    t0 = time.perf_counter()
    log({"event": "start", "rows": len(train_data.frame), "workers": workers, **SMALL_GA})

    def on_generation(record) -> None:
        log({"event": "generation", "generation": record.generation, "best_fitness": record.best_fitness,
             "mean_fitness": record.mean_fitness, "best_genome": record.best_genome.as_xgb_params(),
             "elapsed_s": round(time.perf_counter() - t0, 1)})

    search = GeneticSearch(ga_config, config.hyperparam_space, config.walk_forward, max_workers=workers)
    result = search.run(train_data.frame, train_table.feature_columns, train_table.target_column,
                         on_generation=on_generation)
    log({"event": "done", "best_fitness": result.best_fitness, "best_genome": result.best_genome.as_xgb_params(),
         "elapsed_s": round(time.perf_counter() - t0, 1)})
    print(f"  done in {time.perf_counter()-t0:.1f}s -- best fitness (cross-sectional IC): {result.best_fitness:+.4f}")

    print("\nTraining the final broad expert on the best config found...")
    model = ExpertModel(BROAD_EXPERT, result.best_genome, random_seed=config.ga.random_seed).fit(
        train_data.X, train_data.y
    )
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print(f"\nLoading TEST window ({TEST_START} .. {TEST_END}) -- held out, model never saw this...")
    test_table = provider.get_feature_table(TEST_START, TEST_END, hold_days=config.trading.hold_days)
    test_data = build_expert_data(test_table, broad_spec)
    n_periods = test_data.frame.index.get_level_values("timestamp").nunique()

    preds = model.predict(test_data.X)
    oos_ic = cross_sectional_ic(
        test_data.frame.index.get_level_values("timestamp").to_numpy(),
        test_data.y.to_numpy(), preds["predicted_return"].to_numpy(),
    )
    print(f"  out-of-sample cross-sectional IC (normalized features): {oos_ic:+.4f}  "
          f"(raw-feature broad model scored +0.0209 on the same window)")

    # THE GATE CHECK: does PARA still win the raw daily argmax on anywhere
    # near 245/497 (49.3%) of days, or did normalization actually change
    # who wins?
    frame = test_data.frame.copy()
    frame["pred"] = preds["predicted_return"].values
    daily_winner = frame.loc[frame.groupby(level="timestamp")["pred"].idxmax()]
    winner_tickers = daily_winner.index.get_level_values("ticker")
    para_days = int((winner_tickers == "PARA").sum())
    para_share = para_days / n_periods
    baseline_share = BASELINE_PARA_ARGMAX_DAYS / BASELINE_TOTAL_DAYS

    print("\n" + "=" * 72)
    print(f"GATE CHECK -- PARA's share of the raw daily argmax winner:")
    print(f"  RAW features (baseline):        {BASELINE_PARA_ARGMAX_DAYS}/{BASELINE_TOTAL_DAYS} = {baseline_share:.1%}")
    print(f"  NORMALIZED features (this run): {para_days}/{n_periods} = {para_share:.1%}")
    print(f"  distinct tickers ever #1 (raw):        46")
    print(f"  distinct tickers ever #1 (normalized): {winner_tickers.nunique()}")
    verdict = "CONFIRMED -- normalization meaningfully reduces PARA's dominance" if para_share < baseline_share * 0.5 else (
        "PARTIAL -- some reduction but not dramatic" if para_share < baseline_share else
        "KILLED -- normalization did not reduce PARA's dominance"
    )
    print(f"  verdict: {verdict}")

    results = {
        "out_of_sample_ic_normalized": oos_ic,
        "out_of_sample_ic_raw_baseline": 0.0209,
        "para_argmax_days_normalized": para_days,
        "para_argmax_days_raw_baseline": BASELINE_PARA_ARGMAX_DAYS,
        "total_days": n_periods,
        "para_argmax_share_normalized": para_share,
        "para_argmax_share_raw_baseline": baseline_share,
        "distinct_argmax_winners_normalized": int(winner_tickers.nunique()),
        "distinct_argmax_winners_raw_baseline": 46,
        "top_argmax_winners_normalized": winner_tickers.value_counts().head(10).to_dict(),
        "verdict": verdict,
        "best_genome": result.best_genome.as_xgb_params(),
    }
    RESULT_PATH.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nSaved: {RESULT_PATH}")
    print(f"Model: {MODEL_PATH}")


if __name__ == "__main__":
    main()
