"""score_live.py — the live-scoring entry point (CLAUDE.md; baseline_
system_architecture.md open question 26 scoped this, 27/28/33/36/37 built
what it needs). Run this to get today's trade recommendations.

End to end:

    1. Read config/stock_list.py (Wharton's approved list).
    2. Refresh price data via src.data_pipeline.fetch_live.
    3. Build features in LIVE mode (RealDataProvider with
       apply_point_in_time_filter=False, require_labeled_rows=False --
       open questions 26/28/34) -- so "today", which structurally has no
       realized forward return yet, is still a scoreable row.
    4. Load the trained roster (src.experts.base.load_experts) and
       validate it against the live config in one call
       (assert_config_consistency(artifact=...) -- open question 36) --
       refuses to score with a stale or mismatched artifact rather than
       silently doing so.
    5. Predict per expert, combine, rank under the standing default
       (config.trading.ranking_score, "vol_adjusted" -- open questions
       13/33).
    6. Run TradeSelector seeded from data/processed/portfolio_state.json
       (trades_executed_so_far, today=, the calendar-derived
       starting_period_number -- open question 27) with the
       days_to_next_earnings eligibility filter applied (open question
       37).
    7. Emit an executable list: every recommended action plus the
       reasoning CLAUDE.md requires (which sector model fired, predicted
       edge, why it was sized the way it was) -- joined back from the
       ranked frame, which TradeSelector's own output drops (open
       question 26 gap #8).

Every known precondition is checked up front (_preflight) so a real
failure reads as one clear, named message, not a stack trace three layers
deep -- most pointedly, an empty config/stock_list.py.

Usage:
    python -m scripts.score_live
    python -m scripts.score_live --tickers AAPL MSFT   # override STOCK_LIST, for a smoke test
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import cost_model, hold_period  # noqa: E402
from config.hold_period import period_number_for  # noqa: E402
from config.stock_list import STOCK_LIST  # noqa: E402
from scripts.train_all_sector_experts_real import MODEL_PATH  # noqa: E402
from src.combiner.overall_model import OverallModel  # noqa: E402
from src.config import BaselineConfig, assert_config_consistency  # noqa: E402
from src.data_layer import RealDataProvider  # noqa: E402
from src.data_pipeline import fetch_live  # noqa: E402
from src.experts.base import load_experts  # noqa: E402
from src.experts.registry import build_expert_data  # noqa: E402
from src.selection.portfolio_state import load_portfolio_state  # noqa: E402
from src.selection.trade_selector import TradeSelector  # noqa: E402

OUTPUT_DIR = REPO_ROOT / "data" / "processed" / "live_recommendations"

#: How far back a feature-table request reaches. Generous, not exact --
#: fetch_live.py's own INITIAL_LOOKBACK_CALENDAR_DAYS (400) already bounds
#: what's actually on file per ticker; get_feature_table's own start/end
#: window filter just needs to not clip it. Matches fetch_live's own
#: reasoning (252 trading days is this project's longest rolling window).
FEATURE_LOOKBACK_CALENDAR_DAYS = 450


def _preflight(tickers: list[str], using_real_stock_list: bool = True) -> Optional[str]:
    """Every known precondition, checked up front. Returns a message
    naming exactly what's missing, or None if ready to run -- the
    explicit alternative to letting any one of these surface as a stack
    trace partway through the pipeline.

    ``using_real_stock_list`` distinguishes a real run (no --tickers
    override, scoring config/stock_list.py's actual approved list) from
    a --tickers override (a smoke test or the deliberate placeholder-
    universe daily monitoring run, baseline_system_architecture.md item
    46) -- an unconfirmed WINDOW_START_DATE hard-blocks the former (real
    tickers + a fake calendar anchor is exactly the disaster item 42
    found) but not the latter, which is allowed through with a loud,
    unmissable banner instead (see main()) so it can keep exercising the
    pipeline while genuinely blocked-on-input items get resolved.
    """
    problems = []
    if not tickers:
        problems.append(
            "config/stock_list.py is empty -- populate it with Wharton's approved list "
            "(baseline_system_architecture.md open question 26), or pass --tickers to override "
            "for a one-off run."
        )
    if hold_period.WINDOW_START_DATE is None:
        problems.append(
            "config/hold_period.py's WINDOW_START_DATE is not set -- confirm the competition's "
            "real start date and set it there before a live period number can be computed."
        )
    elif using_real_stock_list and not hold_period.WINDOW_START_DATE_CONFIRMED:
        problems.append(
            f"config/hold_period.py's WINDOW_START_DATE ({hold_period.WINDOW_START_DATE}) is still a "
            "PLACEHOLDER (WINDOW_START_DATE_CONFIRMED=False) -- STOCK_LIST is populated, so this would "
            "be a real run against a fake calendar anchor. Confirm the real competition start date, set "
            "it, and flip WINDOW_START_DATE_CONFIRMED to True in the same change before running for real."
        )
    size = BaselineConfig().trading.position_cap_pct
    if size is not None and size > cost_model.POSITION_CAP_PCT:
        problems.append(
            f"per-entry size {size:.0%} (TradingConfig.position_cap_pct, default "
            f"config/cost_model.py's POSITION_SIZE_PCT) exceeds the competition's per-stock "
            f"cap POSITION_CAP_PCT={cost_model.POSITION_CAP_PCT:.0%} -- every entry would break the rule."
        )
    if not MODEL_PATH.exists():
        problems.append(
            f"{MODEL_PATH} does not exist -- run `python -m scripts.train_all_sector_experts_real` "
            f"first to produce a trained roster."
        )
    if not problems:
        return None
    return "score_live.py cannot run:\n" + "\n".join(f"  - {p}" for p in problems)


def _build_reasoning_report(trades: pd.DataFrame, ranked: pd.DataFrame, config: BaselineConfig) -> pd.DataFrame:
    """Join TradeSelector's trade events back against the ranked frame's
    reasoning columns (sector_expert, broad_return, sector_return,
    ranking_score) -- present in `ranked` but dropped by the time
    TradeSelector emits its own output, which is exactly gap #8 from the
    original score_live.py scope (baseline_system_architecture.md open
    question 26): CLAUDE.md requires every recommendation show which
    sector model fired, predicted edge, and why it was sized the way it
    was, and nothing before this assembled those three into one place.
    """
    reasoning_cols = ["sector_expert", "broad_return", "sector_return", "ranking_score"]
    if trades.empty:
        return trades.assign(**{c: pd.Series(dtype="float64") for c in reasoning_cols}).assign(
            reasoning=pd.Series(dtype="object")
        )

    joined = trades.join(ranked[reasoning_cols], how="left")
    joined["reasoning"] = [
        _reason_for_row(row, config) for _, row in joined.iterrows()
    ]
    return joined


def _reason_for_row(row: pd.Series, config: BaselineConfig) -> str:
    if row["action"] == "exit":
        if row["forced"]:
            return (
                f"forced close at period {row['period']} (entered period {row['entry_period']}, "
                f"scheduled exit never resolved -- see TradeSelector.force_close_log for why)"
            )
        return f"scheduled exit -- entered period {row['entry_period']}, hold_days elapsed"

    # entry
    pieces = [f"broad model predicts {row['broad_return']:+.2%}"]
    if pd.notna(row["sector_expert"]):
        agree = (row["broad_return"] > 0) == (row["sector_return"] > 0)
        pieces.append(
            f"{row['sector_expert']} sector model predicts {row['sector_return']:+.2%} "
            f"({'agree' if agree else 'disagree'})"
        )
    else:
        pieces.append("no sector coverage for this ticker")
    ranking_note = f"ranked by {config.trading.ranking_score} score={row['ranking_score']:+.4f}"
    sizing_note = (
        f"sized at position_cap_pct={config.trading.position_cap_pct:.1%}"
        if config.trading.position_cap_pct is not None
        else f"sized equal-weight at {row['position_size_pct']:.1%} among this period's entries"
    )
    return f"{'; '.join(pieces)}. {ranking_note}. {sizing_note}."


def _print_eligibility_summary(selector: TradeSelector, period_number: int) -> None:
    """Why the allocation layer admitted what it admitted, and rejected
    what it rejected -- CLAUDE.md's reasoning requirement extends to this,
    not just to what got recommended. TradeSelector's own output never
    shows candidates that were screened out, so this is the only place
    that number exists at all."""
    stats = selector.eligibility_stats
    admitted = stats.considered - stats.rejected_direction - stats.rejected_cost - stats.rejected_earnings - stats.rejected_blocked
    print(f"\nEligibility this period: {stats.considered} candidate(s) considered, {admitted} admitted.")
    print(f"  rejected -- direction (short, long-only): {stats.rejected_direction}")
    print(f"  rejected -- cost floor not cleared:        {stats.rejected_cost}")
    print(f"  rejected -- earnings inside hold window:   {stats.rejected_earnings}")
    print(f"  rejected -- already held / blocked:        {stats.rejected_blocked}")
    # Read back the snapshot TradeSelector recorded at the moment this
    # period's pacing decision was actually made -- NOT re-derived from
    # selector.entries_allowed(period_number) now, which after run() has
    # completed would read post-period state (budget already spent by
    # whatever just got entered), not the allowance that governed it.
    pacing = selector.period_pacing.get(period_number)
    print(f"  pacing allowance this period: {pacing} entries "
          f"(remaining_budget={selector.remaining_budget}, open_positions={selector.open_positions_count})")


def _emit(report: pd.DataFrame, today: pd.Timestamp) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{today.date()}.json"
    payload = {
        "window_start_date": hold_period.WINDOW_START_DATE,
        "window_start_date_confirmed": hold_period.WINDOW_START_DATE_CONFIRMED,
        "recommendations": json.loads(report.reset_index().to_json(orient="records", date_format="iso")),
    }
    path.write_text(json.dumps(payload, indent=2))

    print(f"\n{'=' * 72}\nLIVE RECOMMENDATIONS for {today.date()} -- {len(report)} action(s)\n{'=' * 72}")
    if not hold_period.WINDOW_START_DATE_CONFIRMED:
        print(f"  NOTE: window_start_date_confirmed=False in this file -- {hold_period.WINDOW_START_DATE} "
              f"is a placeholder anchor, not the confirmed competition start date.")
    if report.empty:
        print("  (none -- nothing eligible this period, or nothing due to exit)")
    for (ticker, _ts), row in report.iterrows():
        size_note = "closing position" if row["action"] == "exit" else f"{row['position_size_pct']:.1%} of portfolio"
        print(f"\n  {row['action'].upper():<5} {ticker}  ({size_note})")
        print(f"    {row['reasoning']}")
    print(f"\nSaved: {path}")
    return path


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tickers", nargs="+", default=None, help="override config/stock_list.py, for a smoke test")
    args = parser.parse_args(argv)

    using_real_stock_list = args.tickers is None
    tickers = args.tickers or STOCK_LIST
    error = _preflight(tickers, using_real_stock_list=using_real_stock_list)
    if error:
        print(error, file=sys.stderr)
        return 1

    if not hold_period.WINDOW_START_DATE_CONFIRMED:
        print("!" * 72)
        print(f"! WINDOW_START_DATE = {hold_period.WINDOW_START_DATE} IS A PLACEHOLDER, NOT CONFIRMED.")
        print("! Every period_number / pacing figure below is computed against a FAKE")
        print("! calendar anchor. Do not act on this output as if it were real.")
        print("!" * 72)

    config = BaselineConfig()

    print(f"Refreshing live data for {len(tickers)} ticker(s)...")
    fetch_status = fetch_live.main(["--tickers", *tickers])
    if fetch_status != 0:
        print("  (some tickers failed to fetch -- continuing with whatever data is available)")

    print("Building live features (apply_point_in_time_filter=False, require_labeled_rows=False)...")
    provider = RealDataProvider(
        universe=tickers,
        data_dir=fetch_live.LIVE_DIR,
        apply_point_in_time_filter=False,
        require_labeled_rows=False,
    )
    today_calendar = pd.Timestamp.now().normalize()
    start = (today_calendar - pd.Timedelta(days=FEATURE_LOOKBACK_CALENDAR_DAYS)).strftime("%Y-%m-%d")
    end = today_calendar.strftime("%Y-%m-%d")
    table = provider.get_feature_table(start, end, hold_days=config.trading.hold_days)
    if table.frame.empty:
        print("no usable feature rows for any approved ticker -- nothing to score", file=sys.stderr)
        return 1
    if provider.skipped_tickers:
        print(f"  skipped (no live price file found): {sorted(provider.skipped_tickers)}")

    print(f"Loading trained roster from {MODEL_PATH}...")
    bundle = load_experts(MODEL_PATH)
    # Refuses to score with a stale or mismatched artifact rather than
    # silently doing so -- roster, label horizon, and feature columns all
    # checked in this one call (open question 36).
    assert_config_consistency(config, artifact=bundle, table=table)

    print("Predicting per expert...")
    expert_predictions = {}
    for spec in config.experts:
        if spec.name not in bundle.models:
            continue
        edata = build_expert_data(table, spec)
        if edata.X.empty:
            continue
        expert_predictions[spec.name] = bundle.models[spec.name].predict(edata.X)

    # Read from config explicitly rather than trusting OverallModel's own
    # bare-call default -- the same principle already applied to
    # ranking_score (open question 33); this was gap #5's exact shape a
    # second time (open question 42).
    combiner = OverallModel(config.experts, agreement_bonus=config.trading.agreement_bonus)
    combined = combiner.combine(expert_predictions)
    ranking_score = config.trading.ranking_score
    risk = table.frame["volatility_20"].reindex(combined.index) if ranking_score == "vol_adjusted" else None
    ranked = combiner.rank(combined, ranking_score=ranking_score, risk=risk)

    # The most recent date actually present, not calendar "today" -- a
    # weekend/holiday run (or a partial fetch) must not silently score
    # against an empty slice just because datetime.now() doesn't line up
    # with the last real trading day.
    today = ranked.index.get_level_values("timestamp").max()
    period_number = period_number_for(today)
    today_ranked = ranked.xs(today, level="timestamp", drop_level=False)
    placeholder_note = " [PLACEHOLDER anchor, not confirmed]" if not hold_period.WINDOW_START_DATE_CONFIRMED else ""
    print(f"Scoring against {today.date()} (window period {period_number}{placeholder_note})...")

    portfolio = load_portfolio_state()
    selector = TradeSelector(
        config.trading,
        portfolio=portfolio,
        trades_executed_so_far=portfolio.trades_executed_so_far,
        days_to_next_earnings=table.frame["days_to_next_earnings"],
    )
    trades = selector.run(today_ranked, starting_period_number=period_number, today=today)
    _print_eligibility_summary(selector, period_number)

    report = _build_reasoning_report(trades, ranked, config)
    _emit(report, today)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
