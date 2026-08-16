"""Genetic hyperparameter search (architecture §3) — one run per expert.

Fitness is mean out-of-fold Information Coefficient (rank correlation
between predicted and realized return) across walk-forward validation
folds: "train on past, test on unseen future," per the architecture doc.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd

from src.config import GAConfig, HyperparamSpace, WalkForwardConfig
from src.evaluation.metrics import information_coefficient
from src.evaluation.walk_forward import generate_walk_forward_splits
from src.experts.base import ExpertModel, XGBConfig
from src.ga.population import Individual, Population

#: Fitness assigned to a genome that fails to train (e.g. a degenerate
#: combination of hyperparameters) so one bad candidate can't crash a run.
_FAILED_FITNESS = -1.0


@dataclass(frozen=True)
class GenerationRecord:
    generation: int
    best_fitness: float
    mean_fitness: float
    best_genome: XGBConfig


@dataclass(frozen=True)
class SearchResult:
    best_genome: XGBConfig
    best_fitness: float
    history: list[GenerationRecord]


def _evaluate_fitness(
    genome: XGBConfig,
    frame: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    wf_config: WalkForwardConfig,
    random_seed: int,
) -> float:
    fold_scores = []
    for split in generate_walk_forward_splits(frame, wf_config):
        try:
            model = ExpertModel("candidate", genome, random_seed=random_seed).fit(
                split.train[feature_columns], split.train[target_column]
            )
            preds = model.predict(split.val[feature_columns])
            fold_scores.append(
                information_coefficient(
                    split.val[target_column].to_numpy(),
                    preds["predicted_return"].to_numpy(),
                )
            )
        except Exception:
            fold_scores.append(_FAILED_FITNESS)

    if not fold_scores:
        raise ValueError(
            "Not enough history for even one walk-forward fold — lower "
            "WalkForwardConfig.min_train_periods/val_periods or provide more data."
        )
    return float(np.mean(fold_scores))


class GeneticSearch:
    """Evolves XGBoost configs for a single expert's data slice."""

    def __init__(self, ga_config: GAConfig, space: HyperparamSpace, wf_config: WalkForwardConfig):
        self.ga_config = ga_config
        self.space = space
        self.wf_config = wf_config

    def run(
        self,
        frame: pd.DataFrame,
        feature_columns: list[str],
        target_column: str,
        on_generation: Optional[Callable[[GenerationRecord], None]] = None,
    ) -> SearchResult:
        rng = np.random.default_rng(self.ga_config.random_seed)
        population = Population.random(self.space, self.ga_config, rng)
        history: list[GenerationRecord] = []

        for gen in range(self.ga_config.n_generations):
            self._score_unevaluated(population, frame, feature_columns, target_column)

            best = population.best()
            record = GenerationRecord(
                generation=gen,
                best_fitness=best.fitness,
                mean_fitness=float(np.mean([ind.fitness for ind in population.individuals])),
                best_genome=best.genome,
            )
            history.append(record)
            if on_generation is not None:
                on_generation(record)

            is_last = gen == self.ga_config.n_generations - 1
            if not is_last:
                population = population.next_generation(self.space, self.ga_config, rng)

        best = population.best()
        return SearchResult(best_genome=best.genome, best_fitness=best.fitness, history=history)

    def _score_unevaluated(
        self,
        population: Population,
        frame: pd.DataFrame,
        feature_columns: list[str],
        target_column: str,
    ) -> None:
        for individual in population.individuals:
            if individual.fitness is None:  # elites carry their fitness over
                individual.fitness = _evaluate_fitness(
                    individual.genome, frame, feature_columns, target_column,
                    self.wf_config, self.ga_config.random_seed,
                )
