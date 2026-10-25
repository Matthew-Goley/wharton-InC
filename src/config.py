"""Central configuration for the baseline training system.

Everything that isn't a data concern lives here: which experts exist, the GA
search settings, the hyperparameter search space, walk-forward validation
settings, and the trading budget. Keeping these in one module means the GA,
experts, combiner, and pipeline all agree on the same knobs without importing
each other.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from config import cost_model, hold_period
from src.data_layer.interface import FeatureTable

if TYPE_CHECKING:
    # Type-checking only, deliberately -- src.config is meant to be a
    # foundational module other layers (experts, combiner, pipeline)
    # import FROM, not the reverse (see this module's own docstring). A
    # real runtime import here would invert that. assert_config_
    # consistency's `artifact` parameter only ever duck-types against
    # ExpertBundle's three fields (models/hold_days/feature_columns), so
    # no runtime dependency is actually needed.
    from src.experts.base import ExpertBundle


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
#: than erroring, since ``ExpertData.slice_for`` is a plain filter). 10 of
#: the map's 11 sectors are wired up here (validated end to end on
#: Information Technology alone first — see scripts/train_sector_expert_real.py
#: and its checkpoint result, data/processed/backtest_sector_it_real.json —
#: including a real bug found and fixed there: OverallModel's agreement
#: bonus was additive in raw-return units and dominated real predictions;
#: now multiplicative). The ~70 universe tickers with no resolved sector
#: (config/sector_map.py) stay broad-only, same as before this roster
#: existed — nothing downstream needs to special-case them.
#:
#: Materials is EXCLUDED, decided independently of the step-5 ranking_score/
#: agreement_bonus ablation rather than as a third axis in it: its own
#: out-of-sample IC is sign-stable negative across both split-sample halves
#: (2024 t=-3.75, 2025 t=-1.37 — real, though the full-window t=-3.52 is
#: mostly a 2024 result), which is decisive by the pre-registered rule
#: (exclude if the sign holds in both halves). Cutting it as an ablation
#: axis instead of testing "materials in/out" alongside everything else
#: takes step 5 from 12 arms to 6 (3 ranking_score x 2 agreement_bonus),
#: materially improving power on the question that axis was actually
#: built to test — see baseline_system_architecture.md open question 9's
#: sign-stability entry and the step-5 power-analysis note.
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
    #: Optional ADDITIONAL hard ceiling on entries per period, on top of
    #: TradeSelector's own derived pacing (remaining entry capacity /
    #: remaining entry-eligible periods). None by default -- a fixed
    #: ceiling here used to BE the pacing mechanism, and was found to blow
    #: through an entire multi-year backtest's budget in the first ~100 of
    #: 497 days once eligible candidates became abundant (see
    #: TradeSelector._entries_allowed's docstring). Set this only to
    #: deliberately cap entries below whatever the derived rate would
    #: otherwise allow.
    top_n_per_period: Optional[int] = None

    # Competition rules (see baseline_system_architecture.md "Competition
    # Constraints" and CLAUDE.md). Numeric defaults come from
    # config/cost_model.py — the single place those get hardcoded once
    # confirmed — so this dataclass never drifts out of sync with it.
    allow_short: bool = False  # long-only: no short selling, no margin
    min_edge_margin: float = cost_model.MIN_EDGE_MARGIN  # predicted edge must clear cost by this multiple
    #: Fixed per-entry size as a fraction of portfolio value (also sets max
    #: concurrent positions = floor(1/size)). Despite the name, this is the
    #: SIZE the selector trades at, defaulting to cost_model.POSITION_SIZE_PCT;
    #: the competition's real per-stock ceiling is cost_model.POSITION_CAP_PCT,
    #: and score_live.py's preflight refuses a live run sized above it (unit
    #: tests deliberately use other sizes to exercise selector mechanics).
    position_cap_pct: Optional[float] = cost_model.POSITION_SIZE_PCT
    max_trades_per_stock_per_day: int = 2  # real rule, expected non-binding (see doc)

    #: Max share of the window's total ENTRIES (not entries+exits) any one
    #: ticker may take, as a fraction of ``trade_budget // 2`` (a stable,
    #: known-in-advance denominator — the true final entry count isn't known
    #: until the run finishes, since it depends on eligibility and open
    #: positions at window close). None disables the cap entirely (default,
    #: backward-compatible). Not a competition rule — a portfolio-
    #: construction control added after a real finding: the single most
    #: extreme-scoring ticker's share of the rank-1 bucket was 60-81% across
    #: every scoring architecture tried (broad-only, vol-adjusted, IT-only,
    #: full sector coverage) — the concentration was invariant to the
    #: architecture change, meaning it's a property of unconstrained argmax
    #: selection, not any one ticker or fixable by a better ensemble (see
    #: baseline_system_architecture.md open questions 9). Ticker-agnostic by
    #: design — no hardcoded exclusion list, since the offending ticker is
    #: expected to differ by test period (no real book puts most of its top
    #: conviction in one name regardless of who that name is).
    max_ticker_share_pct: Optional[float] = None

    # Fixed-holding-period exit rule (exit_rule_spec.md). hold_days (N) also
    # defines the expert training label (N-day forward return) — the two
    # must always agree, so this comes from config/hold_period.py, the same
    # place the label builder reads it from. A sweep over N (spec §2) means
    # constructing several BaselineConfigs, one per N, via dataclasses.replace
    # — see src.pipeline.sweep.
    hold_days: int = hold_period.HOLD_DAYS
    window_trading_days: int = hold_period.WINDOW_TRADING_DAYS

    #: The standing ranking default (baseline_system_architecture.md open
    #: question 13: "vol_adjusted", not "raw" — decided on concentration
    #: grounds, PARA's rank-1 share 42.6% -> 3.0%). Declared HERE, as a real
    #: config value, specifically so a pipeline reads this rather than
    #: hardcoding its own literal or relying on OverallModel.rank()'s own
    #: separate code-level default — that gap (item 26's gap #5) is exactly
    #: how train_baseline.py silently ranked by "raw" while this decision
    #: was already standing; open question 30's audit found four MORE
    #: independent bare `combiner.rank(combined)` call sites with the same
    #: bug (scripts/train_sector_expert_real.py, train_all_sector_experts_
    #: real.py x2, backtest_broad_real.py, rerun_full_sector_with_ticker_
    #: cap.py) — none yet fixed to read this field, tracked as known,
    #: outstanding landmines rather than silently left for later discovery.
    ranking_score: str = "vol_adjusted"

    #: The standing agreement-bonus default (baseline_system_architecture.md
    #: open question 13: 0, "no evidence bonus=on helps, at return or via
    #: the PARA count"). Declared HERE for the identical reason
    #: ranking_score is: OverallModel.__init__'s own code default was
    #: 0.1, not 0 -- a second instance of gap #5's exact shape, found
    #: while verifying a live dry run (open question 42), not caught by
    #: item 33's fix, since that only touched rank()'s default, not
    #: OverallModel's constructor. Both score_live.py and scripts/dry_run_
    #: multi_day.py called `OverallModel(config.experts)` bare, silently
    #: running agreement_bonus=0.1 -- affecting combined_score directly
    #: (it's multiplied into it), and therefore every downstream cost-
    #: clearing, ranking, and sizing decision in every real/dry run
    #: reported before this fix.
    agreement_bonus: float = 0.0


@dataclass(frozen=True)
class BaselineConfig:
    ga: GAConfig = field(default_factory=GAConfig)
    hyperparam_space: HyperparamSpace = field(default_factory=HyperparamSpace)
    walk_forward: WalkForwardConfig = field(default_factory=WalkForwardConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    experts: tuple[ExpertSpec, ...] = EXPERTS


class ConfigConsistencyError(ValueError):
    """A pipeline's EFFECTIVE inputs (what a run actually used) diverged
    from its DECLARED config (what src.config / config/*.py say should
    govern it). Recurring bug class in this project (baseline_system_
    architecture.md open question 30 names four prior instances before
    this axis: the "Technology" vs "Information Technology" sector-name
    mismatch, the stale materials key in a saved joblib vs. the current
    EXPERTS roster, the Bonferroni threshold reused across a different arm
    count, and gap #5 — train_baseline.py's bare combiner.rank(combined)
    silently ranking by "raw" while vol_adjusted was the declared
    standing default). A single named exception type so every caller
    handles this the same way, rather than each check inventing its own
    ValueError message shape."""


def assert_config_consistency(
    config: BaselineConfig,
    *,
    table: Optional[FeatureTable] = None,
    used_ranking_score: Optional[str] = None,
    trained_expert_names: Optional[set[str]] = None,
    artifact: Optional["ExpertBundle"] = None,
) -> None:
    """Check EFFECTIVE inputs against DECLARED config, across every axis
    this project has already been burned by at least once. Each check only
    runs if its corresponding input is supplied, so a caller can invoke
    this with whatever context it has on hand at a given point in a
    pipeline — no single call site necessarily has all four at once.

    - ``table``: label horizon (table.target_horizon_days) must equal
      config.trading.hold_days -- generalizes the check that used to live
      only inline in train_baseline.py (the ONE real pipeline that already
      had it; every ad hoc backtest script does not, since none of them
      call train_baseline() at all -- see open question 30). Also checks
      config.trading.hold_days is one of config/hold_period.py's declared
      HOLD_DAYS_SWEEP candidates -- NOT strict equality to the single
      HOLD_DAYS default, since src.pipeline.sweep deliberately constructs
      configs with other values from that set, and every one of them is
      legitimate; membership still catches a genuine mistake (e.g.
      hold_days confused with window_trading_days) without flagging every
      intentional sweep value. window_trading_days, unlike hold_days, has
      no legitimate reason to ever vary (it's the competition's actual
      window length, not a tunable), so it's checked by strict equality
      against WINDOW_TRADING_DAYS. Also the PACING sanity check
      _entries_allowed's own ``max(1, ...)`` clamps currently paper over
      silently: hold_days must leave at least one entry-eligible period
      inside the window (hold_days <= window_trading_days), not just
      degrade to "pace as if there's one period left" without ever saying
      why.
    - ``used_ranking_score``: must equal config.trading.ranking_score --
      the direct guard against gap #5's exact failure mode (a caller using
      a different value, bare-default or otherwise, than the declared
      standing default).
    - ``trained_expert_names``: must exactly match {spec.name for spec in
      config.experts} -- no extras (the stale-materials-key shape), no
      omissions (a sector that silently failed to train). Symmetric
      difference reported both ways, not just "sets don't match", so the
      failure mode is legible without a second investigation.
    - ``artifact``: an ``src.experts.base.ExpertBundle`` loaded from disk
      (``load_experts``) -- validates a SAVED artifact wholesale in one
      call (open question 36), the thing a bare pickle of weights could
      never support. Checks its ``hold_days`` against
      ``config.trading.hold_days`` (a model trained on one label horizon,
      loaded under a config declaring a different one, is a real and
      serious correctness bug -- the same class of check as the ``table``
      one above, just for a persisted artifact instead of a freshly-built
      FeatureTable) and, if ``table`` is also supplied, its
      ``feature_columns`` against ``table.feature_columns`` (as sets --
      order isn't itself a drift signal, content is). Also seeds
      ``trained_expert_names`` from ``artifact.models`` if that wasn't
      separately supplied, so the roster check above runs for free rather
      than needing to be duplicated.
    """
    if artifact is not None:
        if trained_expert_names is None:
            trained_expert_names = set(artifact.models)
        if artifact.hold_days != config.trading.hold_days:
            raise ConfigConsistencyError(
                f"artifact label horizon mismatch: this artifact was trained with hold_days="
                f"{artifact.hold_days} (saved_at={artifact.saved_at!r}) but config.trading."
                f"hold_days={config.trading.hold_days} -- predicting with it now would use a "
                f"model fit on the wrong-horizon label for the selector's current exit rule."
            )
        if table is not None and set(artifact.feature_columns) != set(table.feature_columns):
            extra = set(artifact.feature_columns) - set(table.feature_columns)
            missing = set(table.feature_columns) - set(artifact.feature_columns)
            raise ConfigConsistencyError(
                f"artifact feature_columns mismatch: extra (in the artifact, not the current "
                f"feature table): {sorted(extra) or 'none'}. missing (in the current feature "
                f"table, not the artifact): {sorted(missing) or 'none'}."
            )
    if table is not None:
        if table.target_horizon_days != config.trading.hold_days:
            raise ConfigConsistencyError(
                f"label horizon mismatch: FeatureTable.target_horizon_days="
                f"{table.target_horizon_days} but config.trading.hold_days="
                f"{config.trading.hold_days}. Experts would be trained on the "
                f"wrong-horizon target for the selector's exit rule."
            )
    if config.trading.hold_days not in hold_period.HOLD_DAYS_SWEEP:
        # NOT a strict equality check against hold_period.HOLD_DAYS -- the
        # sweep (src.pipeline.sweep, exit_rule_spec.md §2) deliberately
        # constructs configs with hold_days values other than the single
        # default, and every one of them is legitimate. Membership in the
        # pre-declared sweep candidates still catches a genuine mistake
        # (e.g. hold_days confused with window_trading_days) without
        # flagging every intentional sweep value as "diverged".
        raise ConfigConsistencyError(
            f"config.trading.hold_days={config.trading.hold_days} is not one of "
            f"config/hold_period.py's declared HOLD_DAYS_SWEEP candidates "
            f"{hold_period.HOLD_DAYS_SWEEP} -- if this is a deliberate new value, add it "
            f"there first (the single source of truth for which values are sanctioned); "
            f"if it isn't deliberate, this is exactly the drift this check exists to catch."
        )
    if config.trading.window_trading_days != hold_period.WINDOW_TRADING_DAYS:
        raise ConfigConsistencyError(
            f"config.trading.window_trading_days={config.trading.window_trading_days} has "
            f"diverged from config/hold_period.py's WINDOW_TRADING_DAYS="
            f"{hold_period.WINDOW_TRADING_DAYS}."
        )
    if config.trading.hold_days > config.trading.window_trading_days:
        raise ConfigConsistencyError(
            f"pacing is impossible: hold_days={config.trading.hold_days} > "
            f"window_trading_days={config.trading.window_trading_days} -- every position "
            f"entered would need to exit past the end of the window. "
            f"TradeSelector._entries_allowed's max(1, ...) clamp would silently degrade "
            f"pacing to 'one period left' rather than surface this."
        )

    if used_ranking_score is not None and used_ranking_score != config.trading.ranking_score:
        raise ConfigConsistencyError(
            f"ranking_score mismatch: this run used {used_ranking_score!r} but "
            f"config.trading.ranking_score declares {config.trading.ranking_score!r} as the "
            f"standing default (baseline_system_architecture.md open question 13)."
        )

    if trained_expert_names is not None:
        declared = {spec.name for spec in config.experts}
        extra = trained_expert_names - declared
        missing = declared - trained_expert_names
        if extra or missing:
            raise ConfigConsistencyError(
                f"roster mismatch: trained experts {sorted(trained_expert_names)} do not match "
                f"declared config.experts {sorted(declared)}. "
                f"extra (trained but not declared): {sorted(extra) or 'none'}. "
                f"missing (declared but not trained): {sorted(missing) or 'none'}."
            )
