"""Outer loop over hold_days (exit_rule_spec.md §2).

`hold_days` defines the training label, so each value requires a full
retrain of every expert and a full GA re-run — it's not a GA gene, it's an
outer loop around the whole `train_baseline` pipeline. Cost multiplies by
the number of values swept, so keep the sweep coarse
(`config.hold_period.HOLD_DAYS_SWEEP` is the suggested starting set).
"""
from __future__ import annotations

from dataclasses import replace
from typing import Optional

from src.config import BaselineConfig
from src.data_layer.interface import DataProvider
from src.pipeline.train_baseline import BaselineResult, ExpertDoneCallback, GenerationCallback, train_baseline
from src.selection.portfolio_state import PortfolioState


def sweep_hold_days(
    data_provider: DataProvider,
    config: BaselineConfig,
    hold_days_values: tuple[int, ...],
    start: str,
    end: str,
    portfolio: Optional[PortfolioState] = None,
    on_generation: Optional[GenerationCallback] = None,
    on_expert_done: Optional[ExpertDoneCallback] = None,
) -> dict[int, BaselineResult]:
    """Run the full pipeline once per value of N, each with its own
    from-scratch GA search and expert retrain (§2's "not a member of the GA
    population"). Returns one :class:`BaselineResult` per N, so variants can
    be compared against each other directly."""
    results: dict[int, BaselineResult] = {}
    for hold_days in hold_days_values:
        config_n = replace(config, trading=replace(config.trading, hold_days=hold_days))
        results[hold_days] = train_baseline(
            data_provider,
            config_n,
            start,
            end,
            portfolio=portfolio,
            on_generation=on_generation,
            on_expert_done=on_expert_done,
        )
    return results
