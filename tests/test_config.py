"""assert_config_consistency: the config-vs-code drift guard (baseline_
system_architecture.md open question 30). Fifth instance of this bug
class in this project (sector-name mismatch, the stale-materials-key
joblib, a Bonferroni threshold reused across a different arm count, and
gap #5's bare combiner.rank(combined) all preceded it) -- each axis below
gets a deliberate failing-case test, per the standing rule for any new
diagnostic tool in this project.
"""
from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from config import hold_period
from src.config import BROAD_EXPERT, BaselineConfig, ConfigConsistencyError, ExpertSpec, assert_config_consistency
from src.data_layer.interface import FeatureTable


def _table(target_horizon_days: int) -> FeatureTable:
    index = pd.MultiIndex.from_tuples([("AAA", pd.Timestamp("2020-01-01"))], names=["ticker", "timestamp"])
    frame = pd.DataFrame({"f1": [0.0], "fwd_return": [0.0]}, index=index)
    return FeatureTable(
        frame=frame, feature_columns=["f1"], target_column="fwd_return",
        target_horizon_days=target_horizon_days,
    )


# --- a no-op call (nothing supplied) must never spuriously fail -------------

def test_no_inputs_supplied_is_a_silent_no_op():
    assert_config_consistency(BaselineConfig())  # must not raise


# --- label horizon (table) ---------------------------------------------------

def test_passes_when_table_horizon_matches_hold_days():
    config = BaselineConfig()
    assert_config_consistency(config, table=_table(config.trading.hold_days))


def test_raises_on_label_horizon_mismatch():
    config = BaselineConfig()
    with pytest.raises(ConfigConsistencyError, match="label horizon mismatch"):
        assert_config_consistency(config, table=_table(config.trading.hold_days + 1))


# --- hold_days / window_trading_days vs config/hold_period.py's own values --

def test_raises_when_hold_days_is_not_a_declared_sweep_candidate():
    """NOT a strict-equality check against HOLD_DAYS -- src.pipeline.sweep
    deliberately uses other HOLD_DAYS_SWEEP values, all legitimate. This
    checks membership catches a genuine mistake (a value nobody declared,
    e.g. hold_days confused with window_trading_days) without flagging
    intentional sweep values."""
    assert hold_period.HOLD_DAYS_SWEEP == (1, 3, 5, 10)
    config = replace(BaselineConfig(), trading=replace(BaselineConfig().trading, hold_days=50))
    with pytest.raises(ConfigConsistencyError, match="HOLD_DAYS_SWEEP"):
        assert_config_consistency(config)


def test_every_declared_sweep_candidate_is_accepted():
    """The positive counterpart -- src.pipeline.sweep's whole premise is
    that every one of these is a legitimate, non-drifted value."""
    for hold_days in hold_period.HOLD_DAYS_SWEEP:
        config = replace(BaselineConfig(), trading=replace(BaselineConfig().trading, hold_days=hold_days))
        assert_config_consistency(config)  # must not raise for any of them


def test_raises_when_window_trading_days_has_diverged_from_hold_period_py():
    """Unlike hold_days, window_trading_days has no legitimate reason to
    vary (it's the competition's actual window length, not a tunable) --
    checked by strict equality, not membership."""
    config = replace(
        BaselineConfig(),
        trading=replace(BaselineConfig().trading, window_trading_days=hold_period.WINDOW_TRADING_DAYS + 1),
    )
    with pytest.raises(ConfigConsistencyError, match="window_trading_days"):
        assert_config_consistency(config)


# --- pacing: hold_days must leave at least one entry-eligible period --------
# Exercised via monkeypatching hold_period's own constants, not just
# TradingConfig -- with today's real values (HOLD_DAYS_SWEEP maxing out at
# 10, WINDOW_TRADING_DAYS=50) hold_days > window_trading_days can never
# happen simultaneously with the other two checks passing, by
# construction. The scenario this guards against is a FUTURE edit to
# hold_period.py itself (e.g. a new sweep candidate added without
# checking it still fits the window), not a value reachable today.

def test_raises_when_hold_days_exceeds_window_trading_days(monkeypatch):
    """Deliberate failing-case check: without this, TradeSelector._entries_
    allowed's max(1, ...) clamp would silently degrade pacing to 'one
    period left' instead of surfacing a nonsensical hold_period.py edit."""
    monkeypatch.setattr(hold_period, "HOLD_DAYS_SWEEP", (1, 3, 5, 10, 60))
    trading = replace(BaselineConfig().trading, hold_days=60, window_trading_days=hold_period.WINDOW_TRADING_DAYS)
    config = replace(BaselineConfig(), trading=trading)
    with pytest.raises(ConfigConsistencyError, match="pacing is impossible"):
        assert_config_consistency(config)


def test_hold_days_equal_to_window_trading_days_is_allowed(monkeypatch):
    """Boundary case: exactly one entry-eligible period (day 1) is legal,
    not off-by-one excluded."""
    monkeypatch.setattr(hold_period, "HOLD_DAYS_SWEEP", (1, 3, 5, 10, 50))
    trading = replace(BaselineConfig().trading, hold_days=50, window_trading_days=hold_period.WINDOW_TRADING_DAYS)
    config = replace(BaselineConfig(), trading=trading)
    assert_config_consistency(config)  # must not raise


# --- ranking score ------------------------------------------------------------

def test_passes_when_used_ranking_score_matches_declared_default():
    config = BaselineConfig()
    assert_config_consistency(config, used_ranking_score=config.trading.ranking_score)


def test_raises_on_ranking_score_drift():
    """The direct regression test for gap #5: a caller silently using
    'raw' while config declares 'vol_adjusted' standing must be caught,
    not pass silently the way train_baseline.py's bare combiner.rank(
    combined) call did before this guard existed."""
    config = BaselineConfig()
    assert config.trading.ranking_score == "vol_adjusted"  # the declared standing default (open question 13)
    with pytest.raises(ConfigConsistencyError, match="ranking_score mismatch"):
        assert_config_consistency(config, used_ranking_score="raw")


# --- roster --------------------------------------------------------------------

def _config_with_experts(*names: str) -> BaselineConfig:
    experts = tuple(ExpertSpec(name, None if name == BROAD_EXPERT else name) for name in names)
    return replace(BaselineConfig(), experts=experts)


def test_passes_when_trained_roster_exactly_matches_declared():
    config = _config_with_experts(BROAD_EXPERT, "tech")
    assert_config_consistency(config, trained_expert_names={BROAD_EXPERT, "tech"})


def test_raises_on_an_extra_trained_expert_not_in_declared_roster():
    """The direct regression test for the stale-materials-key finding: an
    artifact trained with a member the current config doesn't declare."""
    config = _config_with_experts(BROAD_EXPERT, "tech")
    with pytest.raises(ConfigConsistencyError, match="extra"):
        assert_config_consistency(config, trained_expert_names={BROAD_EXPERT, "tech", "materials"})


def test_raises_on_a_missing_declared_expert_not_actually_trained():
    config = _config_with_experts(BROAD_EXPERT, "tech", "financials")
    with pytest.raises(ConfigConsistencyError, match="missing"):
        assert_config_consistency(config, trained_expert_names={BROAD_EXPERT, "tech"})
