"""Population management: selection, elitism, and generational replacement."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.config import GAConfig, HyperparamSpace
from src.experts.base import XGBConfig
from src.ga.genome import crossover, mutate, random_genome


@dataclass
class Individual:
    genome: XGBConfig
    fitness: float | None = None


class Population:
    def __init__(self, individuals: list[Individual]):
        self.individuals = individuals

    @classmethod
    def random(cls, space: HyperparamSpace, config: GAConfig, rng: np.random.Generator) -> "Population":
        return cls([Individual(random_genome(space, rng)) for _ in range(config.population_size)])

    def best(self) -> Individual:
        return max(self.individuals, key=lambda ind: _fitness_or_worst(ind))

    def _tournament_select(self, rng: np.random.Generator, size: int) -> Individual:
        idx = rng.integers(0, len(self.individuals), size=size)
        contenders = [self.individuals[i] for i in idx]
        return max(contenders, key=lambda ind: _fitness_or_worst(ind))

    def next_generation(
        self, space: HyperparamSpace, config: GAConfig, rng: np.random.Generator
    ) -> "Population":
        """Elitism (unchanged, fitness preserved) + tournament-selected
        crossover/mutation for the rest of the population."""
        ranked = sorted(self.individuals, key=lambda ind: _fitness_or_worst(ind), reverse=True)
        next_individuals = [Individual(ind.genome, ind.fitness) for ind in ranked[: config.elite_count]]

        while len(next_individuals) < config.population_size:
            parent_a = self._tournament_select(rng, config.tournament_size)
            parent_b = self._tournament_select(rng, config.tournament_size)
            if rng.random() < config.crossover_rate:
                child = crossover(parent_a.genome, parent_b.genome, rng)
            else:
                child = parent_a.genome
            child = mutate(child, space, rng, config.mutation_rate)
            next_individuals.append(Individual(child))

        return Population(next_individuals)


def _fitness_or_worst(individual: Individual) -> float:
    return individual.fitness if individual.fitness is not None else float("-inf")
