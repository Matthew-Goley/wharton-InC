"""Genome representation and genetic operators.

A genome *is* an :class:`~src.experts.base.XGBConfig` — there's no separate
encoding step. Operators sample/perturb/mix directly within the ranges
defined by :class:`~src.config.HyperparamSpace`, so a genome is always a
valid, trainable XGBoost config.
"""
from __future__ import annotations

from dataclasses import asdict

import numpy as np

from src.config import HyperparamSpace
from src.experts.base import XGBConfig

#: Fields sampled as integers; every other field in HyperparamSpace is float.
_INT_FIELDS = frozenset({"max_depth", "n_estimators"})


def _sample_field(rng: np.random.Generator, name: str, bounds: tuple[float, float]):
    lo, hi = bounds
    if name in _INT_FIELDS:
        return int(rng.integers(int(lo), int(hi) + 1))
    return float(rng.uniform(lo, hi))


def random_genome(space: HyperparamSpace, rng: np.random.Generator) -> XGBConfig:
    """Sample a uniformly-random genome from the search space."""
    kwargs = {
        name: _sample_field(rng, name, getattr(space, name))
        for name in space.__dataclass_fields__
    }
    return XGBConfig(**kwargs)


def mutate(genome: XGBConfig, space: HyperparamSpace, rng: np.random.Generator, rate: float) -> XGBConfig:
    """Reset-mutation: each gene independently has probability ``rate`` of
    being resampled uniformly within its bounds. Always in-range by
    construction, so a mutated genome never needs re-clamping."""
    kwargs = asdict(genome)
    for name in space.__dataclass_fields__:
        if rng.random() < rate:
            kwargs[name] = _sample_field(rng, name, getattr(space, name))
    return XGBConfig(**kwargs)


def crossover(parent_a: XGBConfig, parent_b: XGBConfig, rng: np.random.Generator) -> XGBConfig:
    """Uniform crossover: each gene independently comes from one parent or the other."""
    a, b = asdict(parent_a), asdict(parent_b)
    kwargs = {name: (a[name] if rng.random() < 0.5 else b[name]) for name in a}
    return XGBConfig(**kwargs)
