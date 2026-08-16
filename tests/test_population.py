import numpy as np

from src.config import GAConfig, HyperparamSpace
from src.ga.population import Individual, Population

SPACE = HyperparamSpace()


def test_best_picks_max_fitness():
    rng = np.random.default_rng(0)
    config = GAConfig(population_size=10, random_seed=0)
    pop = Population.random(SPACE, config, rng)
    for i, ind in enumerate(pop.individuals):
        ind.fitness = float(i)
    assert pop.best().fitness == 9.0


def test_next_generation_preserves_population_size():
    rng = np.random.default_rng(0)
    config = GAConfig(population_size=12, elite_count=3, tournament_size=3, random_seed=0)
    pop = Population.random(SPACE, config, rng)
    for i, ind in enumerate(pop.individuals):
        ind.fitness = float(i)
    next_pop = pop.next_generation(SPACE, config, rng)
    assert len(next_pop.individuals) == config.population_size


def test_elites_carry_over_unchanged_with_fitness():
    rng = np.random.default_rng(0)
    config = GAConfig(population_size=10, elite_count=2, tournament_size=3, random_seed=0)
    pop = Population.random(SPACE, config, rng)
    for i, ind in enumerate(pop.individuals):
        ind.fitness = float(i)

    ranked = sorted(pop.individuals, key=lambda ind: ind.fitness, reverse=True)
    top_two = {(ranked[0].genome, ranked[0].fitness), (ranked[1].genome, ranked[1].fitness)}

    next_pop = pop.next_generation(SPACE, config, rng)
    carried = {(ind.genome, ind.fitness) for ind in next_pop.individuals if ind.fitness is not None}
    assert top_two <= carried


def test_non_elite_offspring_have_no_fitness_yet():
    rng = np.random.default_rng(0)
    config = GAConfig(population_size=10, elite_count=2, tournament_size=3, random_seed=0)
    pop = Population.random(SPACE, config, rng)
    for i, ind in enumerate(pop.individuals):
        ind.fitness = float(i)
    next_pop = pop.next_generation(SPACE, config, rng)
    unevaluated = [ind for ind in next_pop.individuals if ind.fitness is None]
    assert len(unevaluated) == config.population_size - config.elite_count
