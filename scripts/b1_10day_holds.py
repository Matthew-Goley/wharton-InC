"""B1, re-scoped (not the 4-arm HOLD_DAYS_SWEEP sweep -- per instruction,
after scripts/power_check_hold_days_swap.py showed the 4-arm version's
power was thin enough to be worth re-scoping first).

This is ONE single arm: a real retrain at hold_days=10, broad only, same
genome/seed/features as the 5-day baseline. Two independent questions,
kept separate on purpose:

(a) Is 10-day-hold prediction quality acceptable? A GUARDRAIL, not a
    promotion test -- pre-registered, one-sided: 10-day is acceptable
    unless its own IC (vs. its own label) is more than 2 corrected SE
    BELOW the 5-day baseline's own IC (vs. its own label, on the same
    492 common dates -- the 10-day label's tail truncation is applied to
    BOTH arms for a fair paired comparison). A positive diff is NOT
    reported as an improvement -- this script does not make that claim
    either direction, only the guardrail's one-sided pass/fail.

(b) What does the FIXED trade-budget/exit-rule mechanics actually do at
    hold_days=10? This is the primary output, not a side note: run
    TradeSelector itself (not hand arithmetic) under an "abundant
    eligible candidates" synthetic scenario at hold_days=5 (sanity check
    against this project's own already-observed real numbers) and
    hold_days=10, and report concurrent positions, entries, round trips
    achieved vs. the 100 nominal, and the position_cap_pct that WOULD be
    needed to spend the budget fully at hold_days=10 (not a recommendation
    to change it -- config/cost_model.py's POSITION_CAP_PCT is untouched
    by this script).

No config file is modified by this script. TradingConfig variants are
built via dataclasses.replace, in-memory only.

Usage:
    python -m scripts.b1_10day_holds
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import BROAD_EXPERT, BaselineConfig  # noqa: E402
from src.data_layer import RealDataProvider  # noqa: E402
from src.evaluation.block_se import block_bootstrap_se  # noqa: E402
from src.evaluation.metrics import cross_sectional_ic, per_date_cross_sectional_ic  # noqa: E402
from src.experts.base import ExpertModel  # noqa: E402
from src.selection.trade_selector import TradeSelector  # noqa: E402
from scripts.backtest_broad_real import BEST_GENOME, TRAIN_START, TRAIN_END, TEST_START, TEST_END  # noqa: E402
from config import cost_model as cost_model_module  # noqa: E402

BASELINE_HOLD_DAYS = 5
ARM_HOLD_DAYS = 10


# === (a) real retrain + one-sided guardrail ==================================

def run_ic_guardrail(config: BaselineConfig) -> dict:
    provider = RealDataProvider()
    seed = config.ga.random_seed

    print(f"=== (a) Real retrain at hold_days={ARM_HOLD_DAYS}, single arm, guardrail vs "
          f"hold_days={BASELINE_HOLD_DAYS} baseline ===")

    results = {}
    for hd, key in ((BASELINE_HOLD_DAYS, "5"), (ARM_HOLD_DAYS, "10")):
        print(f"  Training hold_days={hd}...")
        train = provider.get_feature_table(TRAIN_START, TRAIN_END, hold_days=hd)
        cols = train.feature_columns
        model = ExpertModel(BROAD_EXPERT, BEST_GENOME, random_seed=seed).fit(
            train.frame[cols], train.frame[train.target_column], hold_days=hd
        )
        test = provider.get_feature_table(TEST_START, TEST_END, hold_days=hd)
        results[key] = {"model": model, "test": test, "cols": cols}

    # Common (ticker, timestamp) rows only -- the 10-day label's tail
    # truncation (last 10 trading days can't have a realized 10-day
    # forward return) applied to BOTH arms for a fair paired comparison.
    common_idx = results["5"]["test"].frame.index.intersection(results["10"]["test"].frame.index)
    print(f"  common rows (both labels realized): {len(common_idx):,} "
          f"({results['5']['test'].frame.index.get_level_values('timestamp').nunique()} dates at 5-day, "
          f"{results['10']['test'].frame.index.get_level_values('timestamp').nunique()} at 10-day, "
          f"restricting both to the intersection)")

    dates = None
    ic_series = {}
    for key in ("5", "10"):
        r = results[key]
        frame = r["test"].frame.loc[common_idx]
        preds = r["model"].predict(frame[r["cols"]])["predicted_return"]
        y = frame[r["test"].target_column]
        dates = frame.index.get_level_values("timestamp").to_numpy()
        ic_series[key] = per_date_cross_sectional_ic(dates, y.to_numpy(), preds.to_numpy())
        print(f"  hold_days={key}: mean IC (on common dates) = {float(ic_series[key].mean()):+.4f}")

    common_dates = ic_series["5"].index.intersection(ic_series["10"].index)
    diff = (ic_series["10"].loc[common_dates] - ic_series["5"].loc[common_dates])  # candidate - baseline
    n = len(diff)
    mean_diff = float(diff.mean())
    naive_se = float(diff.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    boot = block_bootstrap_se(diff.to_numpy(), common_dates.to_numpy(), block_days=ARM_HOLD_DAYS)
    corrected_se = boot["se"]
    t_corrected = mean_diff / corrected_se if corrected_se else float("nan")

    print(f"\n  diff (10-day IC − 5-day IC): {mean_diff:+.4f}")
    print(f"  SE naive: {naive_se:.4f}  SE corrected: {corrected_se:.4f}  "
          f"(block bootstrap, n_blocks={boot['n_blocks']}, block_days={ARM_HOLD_DAYS})")
    print(f"  t corrected: {t_corrected:+.2f}")

    guardrail_threshold = -2.0 * corrected_se
    passes = mean_diff > guardrail_threshold
    print(f"\n  PRE-REGISTERED GUARDRAIL (one-sided, not a promotion test): 10-day acceptable unless "
          f"diff < -2*corrected_SE = {guardrail_threshold:+.4f}")
    print(f"  diff = {mean_diff:+.4f}  ->  {'PASSES (10-day IC not meaningfully worse)' if passes else 'FAILS (10-day IC is meaningfully worse)'}")
    if mean_diff > 0:
        print(f"  NOTE: diff is positive, but per instruction this is NOT reported as an improvement claim "
              f"-- the guardrail is one-sided by design.")

    return {
        "ic_5": float(ic_series["5"].mean()), "ic_10": float(ic_series["10"].mean()),
        "diff": mean_diff, "se_naive": naive_se, "se_corrected": corrected_se,
        "t_corrected": t_corrected, "n_blocks": boot["n_blocks"], "guardrail_pass": passes,
    }


# === (b) trade-budget/exit-rule mechanics, simulated via TradeSelector =======

def _synthetic_abundant_ranked(n_periods: int, n_tickers: int = 30, base_score: float = 0.02) -> pd.DataFrame:
    """A synthetic ranked frame with n_tickers candidates every period,
    all long-eligible and well clear of the cost floor (~0.004) -- an
    'abundant eligible candidates' scenario so budget/concurrency
    mechanics are what's binding, not candidate scarcity. Same ticker
    pool reused every period (realistic: a real universe recurs), scores
    spread slightly so `rank` is well-defined without ties."""
    tickers = [f"SYN{i:02d}" for i in range(n_tickers)]
    dates = pd.bdate_range("2025-01-02", periods=n_periods)
    spread = np.linspace(0.002, -0.001, n_tickers)  # all positive, descending
    order = np.argsort(-spread)
    ranks = np.empty(n_tickers, dtype=int)
    ranks[order] = np.arange(1, n_tickers + 1)

    rows = []
    for date in dates:
        for i, t in enumerate(tickers):
            rows.append({"ticker": t, "timestamp": date, "combined_score": base_score + spread[i], "rank": int(ranks[i])})
    frame = pd.DataFrame(rows).set_index(["ticker", "timestamp"])
    return frame


#: Stand-in for "uncapped" -- position_cap_pct=None crashes _size_positions
#: by design (the deliberate fix for the position-sizing bug requires a
#: real cap to size anything at all). 0.1% is small enough that
#: by_capital=floor(1/0.001)=1000 is never the binding constraint for a
#: 100-round-trip budget, so it measures what concurrency the SCHEDULE
#: (budget + pacing alone) naturally wants, not what a real,
#: unconstrained portfolio would look like -- only used to derive the
#: implied cap below, never reported as a real position size.
_GENEROUS_CAP = 0.001


def simulate_budget_mechanics(trading_config, hold_days: int, position_cap_pct: float | None = None) -> dict:
    tc = replace(trading_config, hold_days=hold_days)
    if position_cap_pct is not None:
        tc = replace(tc, position_cap_pct=position_cap_pct)
    ranked = _synthetic_abundant_ranked(n_periods=tc.window_trading_days)
    selector = TradeSelector(tc)
    trades = selector.run(ranked, starting_period_number=1)  # backtest mode: today=None

    entries = trades[trades["action"] == "entry"]
    exits = trades[trades["action"] == "exit"]
    forced = exits[exits["forced"]] if "forced" in exits.columns else exits.iloc[0:0]

    # Concurrent open-position count per period, reconstructed from the
    # entry/exit event log itself (not re-derived by a second formula).
    max_concurrent = 0
    open_count = 0
    for period in sorted(trades["period"].unique()):
        period_events = trades[trades["period"] == period]
        open_count -= len(period_events[period_events["action"] == "exit"])
        open_count += len(period_events[period_events["action"] == "entry"])
        max_concurrent = max(max_concurrent, open_count)

    n_entries = len(entries)
    n_exits = len(exits)
    round_trips = min(n_entries, n_exits)  # a forced-close at window end still counts as an exit
    nominal_round_trips = tc.trade_budget // 2
    budget_used = selector.trades_taken
    budget_utilization = budget_used / tc.trade_budget
    turns = tc.window_trading_days / hold_days  # how many full hold-and-exit cycles a single slot could complete

    cost_per_round_trip = cost_model_module.round_trip_cost_fraction()
    total_cost_drag = round_trips * cost_per_round_trip

    return {
        "hold_days": hold_days, "n_entries": n_entries, "n_exits": n_exits,
        "n_forced_closes": len(forced), "round_trips": round_trips,
        "nominal_round_trips": nominal_round_trips, "budget_used": budget_used,
        "budget_utilization": budget_utilization, "max_concurrent": max_concurrent,
        "position_cap_pct_used": tc.position_cap_pct,
        "turns": turns, "cost_per_round_trip": cost_per_round_trip, "total_cost_drag": total_cost_drag,
    }


def report_budget_mechanics(config: BaselineConfig) -> None:
    print(f"\n=== (b) Trade-budget / exit-rule mechanics, simulated via TradeSelector "
          f"(synthetic abundant-candidates scenario, not hand-derived) ===")
    print(f"  trade_budget={config.trading.trade_budget}  window_trading_days={config.trading.window_trading_days}  "
          f"position_cap_pct (current, real)={config.trading.position_cap_pct}")

    results = {}
    for hd in (BASELINE_HOLD_DAYS, ARM_HOLD_DAYS):
        r = simulate_budget_mechanics(config.trading, hd)  # current, real position_cap_pct
        r_uncapped = simulate_budget_mechanics(config.trading, hd, position_cap_pct=_GENEROUS_CAP)  # schedule's natural demand
        results[hd] = r
        implied_cap = round(1.0 / r_uncapped["max_concurrent"], 4) if r_uncapped["max_concurrent"] > 0 else None
        print(f"\n  hold_days={hd}:")
        print(f"    entries={r['n_entries']}  exits={r['n_exits']} (forced={r['n_forced_closes']})  "
              f"round_trips={r['round_trips']} of {r['nominal_round_trips']} nominal "
              f"({r['round_trips']/r['nominal_round_trips']:.1%})")
        print(f"    budget used: {r['budget_used']}/{config.trading.trade_budget} ({r['budget_utilization']:.1%})")
        print(f"    max concurrent open positions (current cap={r['position_cap_pct_used']}): {r['max_concurrent']}")
        print(f"    max concurrent DEMANDED by budget+pacing alone, cap not binding (measured, not derived): "
              f"{r_uncapped['max_concurrent']}  -> implied position_cap_pct to not bind here: ~{implied_cap} "
              f"(vs. current {r['position_cap_pct_used']})")
        print(f"    turns (window_trading_days / hold_days): {r['turns']:.1f}")
        print(f"    cost drag: {r['round_trips']} round trips x {r['cost_per_round_trip']:.4%} = "
              f"{r['total_cost_drag']:.4%} of portfolio (cumulative, not annualized)")
        results[hd]["implied_cap"] = implied_cap
        results[hd]["uncapped_max_concurrent"] = r_uncapped["max_concurrent"]

    r5, r10 = results[BASELINE_HOLD_DAYS], results[ARM_HOLD_DAYS]
    print(f"\n  === Comparison, 10-day vs 5-day, current config unchanged ===")
    print(f"  round trips achieved: {r10['round_trips']} vs {r5['round_trips']} "
          f"({r10['round_trips']-r5['round_trips']:+d}, {r10['budget_utilization']-r5['budget_utilization']:+.1%} "
          f"budget-utilization points)")
    print(f"  cost drag: {r10['total_cost_drag']:.4%} vs {r5['total_cost_drag']:.4%} "
          f"({r10['total_cost_drag']-r5['total_cost_drag']:+.4%})")
    if r10["budget_utilization"] < 0.9:
        print(f"  -> At hold_days=10, the CURRENT position_cap_pct={config.trading.position_cap_pct} leaves "
              f"{(1-r10['budget_utilization']):.0%} of the trade budget unspent -- concurrency becomes the "
              f"binding constraint before the budget does, not a hand-derived estimate.")


def main() -> None:
    config = BaselineConfig()
    guardrail = run_ic_guardrail(config)
    report_budget_mechanics(config)

    print(f"\n=== (d) _MIN_TICKER_ROWS and other data-layer constants at hold_days={ARM_HOLD_DAYS} ===")
    from src.data_layer.real_provider import _MIN_TICKER_ROWS, JUMP_VOL_WINDOW
    print(f"  _MIN_TICKER_ROWS = {_MIN_TICKER_ROWS} (=60 rolling-window days + {JUMP_VOL_WINDOW} jump-vol days) -- "
          f"a pure FEATURE-lookback requirement, computed with no reference to hold_days at all. Confirmed by "
          f"reading its definition: identical at hold_days=10, hold_days=5, or any other value in HOLD_DAYS_SWEEP. "
          f"The only hold_days-dependent effect on data availability is the label-side tail truncation already "
          f"measured in (a) above (fewer usable TEST-window rows/dates, not a constant behaving differently).")


if __name__ == "__main__":
    main()
