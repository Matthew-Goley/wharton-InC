"""Orchestrates the full baseline training system, end to end.

    DataProvider                                  (architecture §1 — stubbed)
         |
    FeatureTable
         |
    per-expert data slices  ──►  GeneticSearch  ──►  best XGBConfig   (§3)
         |                                                |
         └──────────────►  ExpertModel.fit(best config)  ─┘           (§2)
                                    |
                         expert predictions (return, confidence)
                                    |
                              OverallModel.combine + rank              (§4)
                                    |
                              TradeSelector.run                        (§5)
                                    |
                             final trade list

This module only wires the pieces together — every step's real logic lives
in its own module (``src.experts``, ``src.ga``, ``src.combiner``,
``src.selection``). Swapping the Data Layer stub for a real implementation
means changing the ``data_provider`` argument here; nothing else moves.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import pandas as pd

from src.combiner.overall_model import OverallModel
from src.config import BaselineConfig
from src.data_layer.interface import DataProvider
from src.experts.base import ExpertModel
from src.experts.registry import ExpertData, build_all_expert_data
from src.ga.search import GenerationRecord, GeneticSearch, SearchResult
from src.selection.portfolio_state import PortfolioState, load_portfolio_state
from src.selection.trade_selector import TradeSelector


@dataclass
class BaselineResult:
    search_results: dict[str, SearchResult]
    trained_experts: dict[str, ExpertModel]
    expert_predictions: dict[str, pd.DataFrame]
    ranked: pd.DataFrame
    trades: pd.DataFrame
    selector: TradeSelector


#: Called after each expert finishes GA search, with (expert_name, SearchResult).
ExpertDoneCallback = Callable[[str, SearchResult], None]
#: Called after every GA generation, with (expert_name, GenerationRecord).
GenerationCallback = Callable[[str, GenerationRecord], None]


def train_baseline(
    data_provider: DataProvider,
    config: BaselineConfig,
    start: str,
    end: str,
    portfolio: Optional[PortfolioState] = None,
    trading_window_periods: Optional[int] = None,
    on_generation: Optional[GenerationCallback] = None,
    on_expert_done: Optional[ExpertDoneCallback] = None,
) -> BaselineResult:
    """Run the full baseline pipeline and return every intermediate artifact.

    ``data_provider`` is where architecture §1 plugs in — pass
    :class:`src.data_layer.MockDataProvider` until the real one exists.

    ``portfolio`` is current holdings (architecture §5) — defaults to
    whatever's on disk at ``data/processed/portfolio_state.json`` (an empty
    portfolio if that file doesn't exist yet).

    Experts train on the full ``[start, end)`` history, but real trade
    selection only ever runs over the live competition window — set
    ``trading_window_periods`` (e.g. ``config.trading.window_trading_days``)
    to restrict what's handed to ``TradeSelector`` to the most recent N
    periods, so the exit rule's entry cutoff and window close are exercised
    against something resembling the real ~50-trading-day window rather
    than the whole training history at once.
    """
    if portfolio is None:
        portfolio = load_portfolio_state()
    table = data_provider.get_feature_table(start, end, hold_days=config.trading.hold_days)
    if table.target_horizon_days != config.trading.hold_days:
        # exit_rule_spec.md §2/§9: the label horizon and the selector's
        # exit timing must always agree — a provider that ignores the
        # requested hold_days (e.g. a stale cached feature table) is a
        # silent correctness bug, not something to proceed past.
        raise ValueError(
            f"label horizon mismatch: FeatureTable.target_horizon_days="
            f"{table.target_horizon_days} but config.trading.hold_days="
            f"{config.trading.hold_days}. Experts would be trained on the "
            f"wrong-horizon target for the selector's exit rule."
        )
    expert_data = build_all_expert_data(table, config.experts)

    search_results: dict[str, SearchResult] = {}
    trained_experts: dict[str, ExpertModel] = {}
    expert_predictions: dict[str, pd.DataFrame] = {}

    for name, edata in expert_data.items():
        result = _search_expert(name, edata, table, config, on_generation)
        search_results[name] = result
        if on_expert_done is not None:
            on_expert_done(name, result)

        model = ExpertModel(name, result.best_genome, random_seed=config.ga.random_seed).fit(
            edata.X, edata.y
        )
        trained_experts[name] = model
        expert_predictions[name] = model.predict(edata.X)

    combiner = OverallModel(config.experts)
    combined = combiner.combine(expert_predictions)
    ranked = combiner.rank(combined)

    trading_window = ranked
    if trading_window_periods is not None:
        all_timestamps = sorted(pd.unique(ranked.index.get_level_values("timestamp")))
        live_timestamps = set(all_timestamps[-trading_window_periods:])
        trading_window = ranked[ranked.index.get_level_values("timestamp").isin(live_timestamps)]

    selector = TradeSelector(config.trading, portfolio=portfolio)
    trades = selector.run(trading_window)

    return BaselineResult(
        search_results=search_results,
        trained_experts=trained_experts,
        expert_predictions=expert_predictions,
        ranked=ranked,
        trades=trades,
        selector=selector,
    )


def _search_expert(
    name: str,
    edata: ExpertData,
    table,
    config: BaselineConfig,
    on_generation: Optional[GenerationCallback],
) -> SearchResult:
    search = GeneticSearch(config.ga, config.hyperparam_space, config.walk_forward)
    callback = (lambda record: on_generation(name, record)) if on_generation else None
    return search.run(edata.frame, table.feature_columns, table.target_column, on_generation=callback)
