"""End-to-end smoke test for the baseline training system.

Runs the full pipeline (GA search -> experts -> combiner -> trade selection)
against synthetic data from ``MockDataProvider`` with small, fast settings,
so it finishes in well under a minute. This exists to prove the wiring
works, not to produce a real result — for a production-scale run, use the
defaults in ``src.config.BaselineConfig`` (much larger population/generations)
against a real ``DataProvider``.

Usage:
    python -m scripts.run_baseline_demo
"""
from __future__ import annotations

from dataclasses import replace

from src.config import BaselineConfig, GAConfig, TradingConfig, WalkForwardConfig
from src.data_layer import MockDataProvider
from src.pipeline.train_baseline import train_baseline


def build_demo_config() -> BaselineConfig:
    base = BaselineConfig()
    return replace(
        base,
        ga=replace(base.ga, population_size=8, n_generations=4, elite_count=2, tournament_size=3),
        walk_forward=replace(base.walk_forward, n_splits=2, min_train_periods=150, val_periods=40),
        trading=replace(base.trading, trade_budget=200, top_n_per_period=10),
    )


def main() -> None:
    config = build_demo_config()
    data_provider = MockDataProvider(n_tickers_per_sector=10, n_periods=400, seed=0)

    def on_generation(expert_name: str, record) -> None:
        print(
            f"  [{expert_name}] gen {record.generation:>2}: "
            f"best IC={record.best_fitness:+.4f}  mean IC={record.mean_fitness:+.4f}"
        )

    def on_expert_done(expert_name: str, result) -> None:
        print(f"[{expert_name}] search done — best IC={result.best_fitness:+.4f}\n"
              f"[{expert_name}] best config: {result.best_genome}\n")

    print("Running baseline pipeline on synthetic data (Data Layer stubbed)...\n")
    result = train_baseline(
        data_provider,
        config,
        start="2018-01-01",
        end="2026-01-01",
        # Experts train on the full history above; trade selection only
        # runs over the most recent `window_trading_days` periods — a stand
        # in for the real ~50-trading-day competition window.
        trading_window_periods=config.trading.window_trading_days,
        on_generation=on_generation,
        on_expert_done=on_expert_done,
    )

    trades = result.trades
    entries = trades[trades["action"] == "entry"]
    exits = trades[trades["action"] == "exit"]

    print("=" * 72)
    print(f"Trades executed (entries + exits): {result.selector.trades_taken} / {config.trading.trade_budget} budget")
    print(f"Entries: {len(entries)}   Exits: {len(exits)}   Still open at end: {result.selector.open_positions_count}")
    if result.selector.force_close_log:
        print(f"Force-closed (safety net fired): {len(result.selector.force_close_log)}")
    print("\nFirst 10 entries:")
    print(
        entries.sort_values("period")
        .head(10)[["period", "scheduled_exit_period", "combined_score", "position_size_pct"]]
        .to_string()
    )
    print("\nFirst 10 exits:")
    print(exits.sort_values("period").head(10)[["period", "entry_period", "forced"]].to_string())


if __name__ == "__main__":
    main()
