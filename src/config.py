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
#:
#: Sector strings must match ``config.sector_map.SECTOR_MAP``'s GICS-
#: normalized values exactly (a mismatch silently trains on zero rows rather
#: than erroring, since ``ExpertData.slice_for`` is a plain filter). All
#: eleven of the map's sectors are wired up here (validated end to end on
#: Information Technology alone first — see scripts/train_sector_expert_real.py
#: and its checkpoint result, data/processed/backtest_sector_it_real.json —
#: including a real bug found and fixed there: OverallModel's agreement
#: bonus was additive in raw-return units and dominated real predictions;
#: now multiplicative). The ~70 universe tickers with no resolved sector
#: (config/sector_map.py) stay broad-only, same as before this roster
#: existed — nothing downstream needs to special-case them.
EXPERTS: tuple[ExpertSpec, ...] = (
    ExpertSpec(name=BROAD_EXPERT, sector=None),
    ExpertSpec(name="tech", sector="Information Technology"),
    ExpertSpec(name="industrials", sector="Industrials"),
    ExpertSpec(name="financials", sector="Financials"),
    ExpertSpec(name="consumer_discretionary", sector="Consumer Discretionary"),
    ExpertSpec(name="health_care", sector="Health Care"),
    ExpertSpec(name="consumer_staples", sector="Consumer Staples"),
    ExpertSpec(name="real_estate", sector="Real Estate"),
    ExpertSpec(name="utilities", sector="Utilities"),
    ExpertSpec(name="energy", sector="Energy"),
    ExpertSpec(name="communication_services", sector="Communication Services"),
    ExpertSpec(name="materials", sector="Materials"),
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
    """Fold geometry for the GA's fitness estimate.

    Sized against the measurement noise, not by eye. Cross-sectional IC has a
    standard deviation of about 0.19 across dates on the real universe, while
    the signal being hunted is around 0.013. A fold's standard error is
    therefore ~0.19/sqrt(val_periods), and the earlier 100-period windows put
    it at 0.019 — larger than the effect itself, which meant the search was
    ranking candidates mostly on luck.

    The values below give ~0.0107 per fold and ~0.0054 once four folds are
    averaged: a signal-to-noise ratio of roughly 2.4x, up from 1.4x. They are
    also chosen so the last fold's validation window lands on the final day of
    the current ~2,006-day history (746 = 2006 - 4*315) instead of stopping 17
    months short and never testing recent data.
    """

    n_splits: int = 4
    min_train_periods: int = 746  # periods, not calendar time (~3 years)
    val_periods: int = 315  # ~15 months; short windows cannot resolve IC~0.013
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
