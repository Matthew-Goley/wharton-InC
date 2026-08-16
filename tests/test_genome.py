import numpy as np
import pytest

from src.config import HyperparamSpace
from src.experts.base import XGBConfig
from src.ga.genome import crossover, mutate, random_genome

SPACE = HyperparamSpace()


def _assert_in_bounds(genome: XGBConfig, space: HyperparamSpace = SPACE):
    for name in space.__dataclass_fields__:
        lo, hi = getattr(space, name)
        value = getattr(genome, name)
        assert lo <= value <= hi, f"{name}={value} outside [{lo}, {hi}]"


def test_random_genome_within_bounds():
    rng = np.random.default_rng(0)
    for _ in range(200):
        _assert_in_bounds(random_genome(SPACE, rng))


def test_random_genome_integer_fields_are_ints():
    rng = np.random.default_rng(0)
    genome = random_genome(SPACE, rng)
    assert isinstance(genome.max_depth, int)
    assert isinstance(genome.n_estimators, int)


def test_mutate_rate_zero_is_identity():
    rng = np.random.default_rng(0)
    genome = random_genome(SPACE, rng)
    assert mutate(genome, SPACE, rng, rate=0.0) == genome


def test_mutate_rate_one_changes_every_field_and_stays_in_bounds():
    rng = np.random.default_rng(1)
    genome = random_genome(SPACE, rng)
    mutated = mutate(genome, SPACE, rng, rate=1.0)
    _assert_in_bounds(mutated)
    assert mutated != genome


def test_crossover_each_gene_comes_from_a_parent():
    rng = np.random.default_rng(2)
    a = random_genome(SPACE, rng)
    b = random_genome(SPACE, rng)
    child = crossover(a, b, rng)
    for name in SPACE.__dataclass_fields__:
        assert getattr(child, name) in (getattr(a, name), getattr(b, name))
    _assert_in_bounds(child)
