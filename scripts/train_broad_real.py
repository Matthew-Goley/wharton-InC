"""Real training run: broad expert only, on actual market data.

Sector data doesn't exist yet (config/sector_map.py isn't built), so this
restricts BaselineConfig to just the broad expert -- sector experts would
train on zero rows otherwise (RealDataProvider tags every row
sector="Unknown"). Full production GA settings (population=50,
generations=20), parallelized across a safe worker pool
(src.ga.search.default_max_workers -- leaves 2 cores free).

Usage:
    python -m scripts.train_broad_real
"""
from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path

import joblib

from src.config import BROAD_EXPERT, BaselineConfig, ExpertSpec
from src.data_layer import RealDataProvider
from src.experts.base import ExpertModel
from src.experts.registry import build_expert_data
from src.ga.search import GeneticSearch, default_max_workers

DATA_PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"
LOG_PATH = DATA_PROCESSED / "broad_expert_search_log.jsonl"
MODEL_PATH = DATA_PROCESSED / "broad_expert_model.joblib"


def log(record: dict) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a") as f:
        f.write(json.dumps(record) + "\n")
    print(record)


def main() -> None:
    config = replace(BaselineConfig(), experts=(ExpertSpec(name=BROAD_EXPERT, sector=None),))
    workers = default_max_workers()

    print("Loading real data...")
    t0 = time.perf_counter()
    provider = RealDataProvider()
    table = provider.get_feature_table("2018-01-01", "2026-01-01", hold_days=config.trading.hold_days)
    print(f"  {len(table.frame):,} rows, {table.frame.index.get_level_values('ticker').nunique()} tickers, "
          f"{len(provider.skipped_tickers)} skipped, loaded in {time.perf_counter()-t0:.1f}s")

    edata = build_expert_data(table, config.experts[0])
    print(f"broad expert: {len(edata.frame):,} rows")
    print(f"GA: population={config.ga.population_size} generations={config.ga.n_generations} workers={workers}")
    log({"event": "start", "rows": len(edata.frame), "workers": workers,
         "population": config.ga.population_size, "generations": config.ga.n_generations})

    def on_generation(record) -> None:
        # Log the champion's actual genome every generation, not just its
        # fitness -- stopping the run early (a real plateau, a crash, just
        # wanting results sooner) should never mean losing the only usable
        # artifact of however many generations already ran.
        log({
            "event": "generation", "generation": record.generation,
            "best_fitness": record.best_fitness, "mean_fitness": record.mean_fitness,
            "best_genome": record.best_genome.as_xgb_params(),
            "elapsed_s": round(time.perf_counter() - t0, 1),
        })

    search = GeneticSearch(config.ga, config.hyperparam_space, config.walk_forward, max_workers=workers)
    result = search.run(edata.frame, table.feature_columns, table.target_column, on_generation=on_generation)

    log({"event": "done", "best_fitness": result.best_fitness, "best_genome": result.best_genome.as_xgb_params(),
         "elapsed_s": round(time.perf_counter() - t0, 1)})

    print("\nTraining the final broad expert on the best config found...")
    model = ExpertModel(BROAD_EXPERT, result.best_genome, random_seed=config.ga.random_seed).fit(edata.X, edata.y)

    # A ~7-8 hour search producing a model that only exists in this
    # process's memory is a real trap -- the moment the process exits, the
    # only artifact left is the genome (recoverable, but a full walk-forward
    # search away from being trained again). Persist the actual fitted
    # model, not just what it took to find it.
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print("Done. Best genome:", result.best_genome)
    print("Log:", LOG_PATH)
    print("Model:", MODEL_PATH)


if __name__ == "__main__":
    main()
