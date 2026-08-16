"""Central configuration for the baseline training system.

Everything that isn't a data concern lives here: which experts exist, the GA
search settings, the hyperparameter search space, walk-forward validation
settings, and the trading budget. Keeping these in one module means the GA,
experts, combiner, and pipeline all agree on the same knobs without importing
each other.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from config import cost_model, hold_period


# ---------------------------------------------------------------------------
# Experts (architecture §2)
# ---------------------------------------------------------------------------

#: Name of the always-on, full-universe expert. Its config key in
#: ``EXPERTS`` maps to ``sector=None``, meaning "no sector filter".
BROAD_EXPERT = "broad"


@dataclass(frozen=True)
class ExpertSpec:
    """One row of the architecture doc's expert table."""

    name: str
    sector: Optional[str]  # None => trains on the full universe


#: The expert roster. Add a row here (and nowhere else) to add a new sector
#: expert — the GA, registry, and combiner all derive from this list.
EXPERTS: tuple[ExpertSpec, ...] = (
    ExpertSpec(name=BROAD_EXPERT, sector=None),
    ExpertSpec(name="tech", sector="Technology"),
    ExpertSpec(name="financials", sector="Financials"),
    ExpertSpec(name="real_estate", sector="Real Estate"),
)


# ---------------------------------------------------------------------------
# Genetic algorithm (architecture §3)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GAConfig:
    population_size: int = 50
    n_generations: int = 20
    elite_count: int = 4  # top performers copied unchanged into next gen
    tournament_size: int = 3
    crossover_rate: float = 0.7
    mutation_rate: float = 0.3  # probability *per gene* of mutating
    random_seed: int = 42


@dataclass(frozen=True)
class HyperparamSpace:
    """Inclusive [low, high] ranges the GA samples XGBoost configs from."""

    max_depth: tuple[int, int] = (2, 10)
    learning_rate: tuple[float, float] = (0.005, 0.3)
    n_estimators: tuple[int, int] = (50, 600)
    subsample: tuple[float, float] = (0.5, 1.0)
    colsample_bytree: tuple[float, float] = (0.5, 1.0)
    min_child_weight: tuple[float, float] = (1.0, 10.0)
    reg_alpha: tuple[float, float] = (0.0, 5.0)
    reg_lambda: tuple[float, float] = (0.1, 10.0)
    gamma: tuple[float, float] = (0.0, 5.0)


# ---------------------------------------------------------------------------
# Walk-forward validation (used as the GA fitness function)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WalkForwardConfig:
    n_splits: int = 4
    min_train_periods: int = 500  # rows, not calendar time
    val_periods: int = 100
    expanding: bool = True  # expanding window vs. fixed-size rolling window


# ---------------------------------------------------------------------------
# Trade selection (architecture §5)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TradingConfig:
    trade_budget: int = 200
    budget_period_weeks: int = 10
    top_n_per_period: int = 10  # cap on trades taken per rebalance period

    # Competition rules (see baseline_system_architecture.md "Competition
    # Constraints" and CLAUDE.md). Numeric defaults come from
    # config/cost_model.py — the single place those get hardcoded once
    # confirmed — so this dataclass never drifts out of sync with it.
    allow_short: bool = False  # long-only: no short selling, no margin
    min_edge_margin: float = cost_model.MIN_EDGE_MARGIN  # predicted edge must clear cost by this multiple
    position_cap_pct: Optional[float] = cost_model.POSITION_CAP_PCT  # per-stock % of portfolio; TBD
    max_trades_per_stock_per_day: int = 2  # real rule, expected non-binding (see doc)

    # Fixed-holding-period exit rule (exit_rule_spec.md). hold_days (N) also
    # defines the expert training label (N-day forward return) — the two
    # must always agree, so this comes from config/hold_period.py, the same
    # place the label builder reads it from. A sweep over N (spec §2) means
    # constructing several BaselineConfigs, one per N, via dataclasses.replace
    # — see src.pipeline.sweep.
    hold_days: int = hold_period.HOLD_DAYS
    window_trading_days: int = hold_period.WINDOW_TRADING_DAYS


@dataclass(frozen=True)
class BaselineConfig:
    ga: GAConfig = field(default_factory=GAConfig)
    hyperparam_space: HyperparamSpace = field(default_factory=HyperparamSpace)
    walk_forward: WalkForwardConfig = field(default_factory=WalkForwardConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    experts: tuple[ExpertSpec, ...] = EXPERTS
