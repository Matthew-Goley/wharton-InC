"""Multi-day dry run: simulates several consecutive score_live.py-style
invocations against real data and the real trained roster, carrying
portfolio state forward exactly the way real day-to-day usage would (a
FRESH TradeSelector each day, seeded from the previous day's resulting
positions) -- the thing a single-day run structurally cannot exercise:
overlapping fixed-hold positions accumulating across repeated entries,
the position-sizing/capital-constraint interaction, and budget/pacing
drawdown over several real days.

Reuses real historical dates already present in the live-fetched data
(the most recent N trading days) rather than waiting for N real calendar
days to pass -- a faithful simulation of "if this had been run once a day
for the last N trading days," not a synthetic fixture.

Usage:
    python -m scripts.dry_run_multi_day --tickers AAPL MSFT ... --days 5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import hold_period  # noqa: E402
from config.hold_period import period_number_for  # noqa: E402
from src.combiner.overall_model import OverallModel  # noqa: E402
from src.config import BaselineConfig, assert_config_consistency  # noqa: E402
from src.data_layer import RealDataProvider  # noqa: E402
from src.data_pipeline import fetch_live  # noqa: E402
from src.experts.base import load_experts  # noqa: E402
from src.experts.registry import build_expert_data  # noqa: E402
from src.selection.portfolio_state import PortfolioState, Position, empty_portfolio  # noqa: E402
from src.selection.trade_selector import TradeSelector  # noqa: E402
from scripts.train_all_sector_experts_real import MODEL_PATH  # noqa: E402


def _rebuild_positions(portfolio: PortfolioState, trades: pd.DataFrame, today: pd.Timestamp, hold_days: int) -> PortfolioState:
    """Update in-memory portfolio state from one day's TradeSelector
    output: new entries become open positions (scheduled_exit_date
    computed the same way log_trade does), exits (natural or forced)
    remove them."""
    import numpy as np

    positions = dict(portfolio.positions)
    for (ticker, _ts), row in trades.iterrows():
        if row["action"] == "entry":
            scheduled_exit = str(np.busday_offset(today.date(), hold_days, roll="forward"))
            positions[ticker] = Position(
                shares=0.0, avg_price=0.0,  # dry run: sizing is tracked via position_size_pct, not dollars
                entry_date=str(today.date()), scheduled_exit_date=scheduled_exit,
            )
        else:  # exit (natural or forced)
            positions.pop(ticker, None)
    return PortfolioState(as_of=str(today.date()), cash=portfolio.cash, positions=positions, trade_log=portfolio.trade_log)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tickers", nargs="+", required=True)
    parser.add_argument("--days", type=int, default=5)
    parser.add_argument(
        "--window-start", default=None,
        help=(
            "Override config/hold_period.py's WINDOW_START_DATE for this run. Explicit on "
            "purpose, not just a convenience -- an arbitrary/uncalibrated anchor silently "
            "changes what window PERIOD the simulated dates land on, which changes the derived "
            "pacing rate (fewer entry-periods remaining reads as 'behind schedule' and inflates "
            "the allowance). Pass the first of --days dates here to simulate a true fresh-window "
            "start (period 1); leave unset to use the real, currently-confirmed season start."
        ),
    )
    args = parser.parse_args(argv)

    if args.window_start is not None:
        hold_period.WINDOW_START_DATE = args.window_start
    elif not hold_period.WINDOW_START_DATE_CONFIRMED:
        # The module default is now a non-None PLACEHOLDER (baseline_
        # system_architecture.md item 47/48), not "unset" -- relying on
        # it silently here would break this docstring's own promise
        # ("leave unset to use the real, currently-confirmed season
        # start") exactly the way an uncalibrated anchor already caused a
        # real mis-anchoring bug once (item 42). Pass --window-start
        # explicitly for a simulation run instead.
        print(
            f"config/hold_period.py's WINDOW_START_DATE ({hold_period.WINDOW_START_DATE}) is still a "
            "PLACEHOLDER (WINDOW_START_DATE_CONFIRMED=False) and --window-start wasn't passed -- pass "
            "--window-start explicitly for a simulation run, or confirm the real date first.",
            file=sys.stderr,
        )
        return 1
    if hold_period.WINDOW_START_DATE is None:
        print("config/hold_period.py's WINDOW_START_DATE is not set (and --window-start wasn't "
              "passed either).", file=sys.stderr)
        return 1

    config = BaselineConfig()
    print(f"Refreshing live data for {len(args.tickers)} ticker(s)...")
    fetch_live.main(["--tickers", *args.tickers])

    provider = RealDataProvider(
        universe=args.tickers, data_dir=fetch_live.LIVE_DIR,
        apply_point_in_time_filter=False, require_labeled_rows=False,
    )
    today_calendar = pd.Timestamp.now().normalize()
    start = (today_calendar - pd.Timedelta(days=450)).strftime("%Y-%m-%d")
    table = provider.get_feature_table(start, today_calendar.strftime("%Y-%m-%d"), hold_days=config.trading.hold_days)

    bundle = load_experts(MODEL_PATH)
    assert_config_consistency(config, artifact=bundle, table=table)

    expert_predictions = {}
    for spec in config.experts:
        if spec.name not in bundle.models:
            continue
        edata = build_expert_data(table, spec)
        if edata.X.empty:
            continue
        expert_predictions[spec.name] = bundle.models[spec.name].predict(edata.X)

    combiner = OverallModel(config.experts, agreement_bonus=config.trading.agreement_bonus)
    combined = combiner.combine(expert_predictions)
    risk = table.frame["volatility_20"].reindex(combined.index)
    ranked = combiner.rank(combined, ranking_score=config.trading.ranking_score, risk=risk)

    all_dates = sorted(ranked.index.get_level_values("timestamp").unique())
    sim_dates = all_dates[-args.days:]
    if len(sim_dates) < args.days:
        print(f"only {len(sim_dates)} trading date(s) available, requested {args.days}", file=sys.stderr)

    portfolio = empty_portfolio()
    trades_executed = 0
    print(f"\nSimulating {len(sim_dates)} consecutive day(s): {[d.date() for d in sim_dates]}\n")
    print(f"{'='*100}")

    for i, today in enumerate(sim_dates, start=1):
        period_number = period_number_for(today)
        today_ranked = ranked.xs(today, level="timestamp", drop_level=False)

        selector = TradeSelector(
            config.trading, portfolio=portfolio, trades_executed_so_far=trades_executed,
            days_to_next_earnings=table.frame["days_to_next_earnings"],
        )
        trades = selector.run(today_ranked, starting_period_number=period_number, today=today)
        entries = trades[trades["action"] == "entry"]
        exits = trades[trades["action"] == "exit"]

        portfolio = _rebuild_positions(portfolio, trades, today, config.trading.hold_days)
        trades_executed = selector.trades_taken
        cap = config.trading.position_cap_pct
        cumulative_exposure = len(portfolio.positions) * cap

        print(f"Day {i}  {today.date()}  (window period {period_number})")
        print(f"  entries: {len(entries)}  exits: {len(exits)}  "
              f"({', '.join(entries.index.get_level_values('ticker')) or 'none'})")
        if len(entries):
            print(f"  entry size: {cap:.1%} each (fixed at position_cap_pct)")
        stats = selector.eligibility_stats
        print(f"  eligibility: {stats.considered} considered, "
              f"{stats.considered - stats.rejected_direction - stats.rejected_cost - stats.rejected_earnings - stats.rejected_blocked} admitted "
              f"(direction={stats.rejected_direction}, cost={stats.rejected_cost}, "
              f"earnings={stats.rejected_earnings}, blocked={stats.rejected_blocked})")
        print(f"  pacing allowance this period: {selector.period_pacing.get(period_number)}")
        print(f"  concurrent open positions: {len(portfolio.positions)}  "
              f"cumulative exposure: {cumulative_exposure:.1%}  "
              f"remaining budget: {config.trading.trade_budget - trades_executed}")
        print(f"{'='*100}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
