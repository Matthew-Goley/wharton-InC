"""Re-runs the full-sector-coverage backtest (step 3) with
``max_ticker_share_pct`` enabled, reusing the already-trained models from
``scripts/train_all_sector_experts_real.py`` -- the cap only changes
TradeSelector's selection logic, not the experts, so no retraining is
needed.

Motivation: PARA's share of the rank-1 ENTRY bucket was 60-81% across
every scoring architecture tried so far -- the concentration was invariant
to architecture changes, meaning "rank 1 is structurally bad" and "PARA
specifically was bad" had been indistinguishable, since rank 1 effectively
*was* PARA. The underlying mechanism turns out to be more extreme than
that framing suggested: PARA is the single highest-*scoring* ticker in the
whole cross-section on 245 of 497 test days (49.3%), before any trading
mechanics (eligibility, budget, already-open exclusion) even apply.

Two things learned calibrating the cap that weren't obvious going in:

1. A cap on total ENTRY count doesn't bind at any "reasonable-sounding"
   share (5%, 2%) because PARA's total entry count (16 of 994, 1.6%) isn't
   unusual -- TSLA (14) and ALB (13) are close behind it. PARA's pathology
   is concentrated in *which rank* those entries win (15 of its 16 are
   rank-1), not in overall trade volume. The cap has to sit below ~1% to
   bind at all -- swept empirically (see TICKER_CAP) rather than assumed.

2. The static `rank` column is computed once per day and never
   recomputed after a ticker is excluded from eligibility, so capping
   PARA out doesn't "promote" a substitute into the rank==1 *label* --
   the substitute gets entered but keeps its own original rank number,
   which shrinks the rank(0,1] bucket rather than diversifying its
   composition. The metric that actually answers "what does the best pick
   each day look like, PARA-independent" is the entry with the lowest
   rank number among whatever WAS actually taken that day ("period
   champion") -- not literally rank==1.

Result at the calibrated cap (1%, ~10 entries/ticker): period-champion mean
return flips from -0.80% (uncapped, n=101, PARA=15 champion-days) to
+0.25% (capped, n=103, PARA=8 champion-days), unique tickers 49->61.
Decomposed via decompose_selection_change rather than trusted as a headline
delta (same standing rule as everywhere else here): real direction, t=1.41
on the differing subset -- not conventionally significant, same as
essentially every other finding in this project measured this way. Real,
suggestive, not established.

Cross-sectional feature normalization was tried as an alternative,
model-level route to a PARA-independent read and rejected (see
baseline_system_architecture.md's ablation table): it's structurally
incapable of fixing persistent argmax dominance (percentile rank is
order-preserving, so a name that's the day's most extreme value stays the
day's most extreme value after transforming), and it cost a real chunk of
the model's out-of-sample IC finding that out. This script -- the cap,
which never touches the model at all -- is the actual PARA-independent
read: it now runs the SAME standard diagnostic suite (rank buckets,
split-sample validation) as scripts/train_all_sector_experts_real.py, just
under a capped TradeSelector, alongside the period-champion view above.

Usage:
    python -m scripts.rerun_full_sector_with_ticker_cap
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import joblib

from scripts.backtest_broad_real import TEST_START, TEST_END, _rank_bucket_stats, _split_sample_validation
from src.combiner.overall_model import OverallModel
from src.config import BaselineConfig, EXPERTS
from src.data_layer import RealDataProvider
from src.data_layer.real_provider import RAW_TARGET_COLUMN
from src.evaluation.selection_diff import decompose_selection_change
from src.experts.registry import build_expert_data
from src.selection.trade_selector import TradeSelector

DATA_PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"
MODEL_PATH = DATA_PROCESSED / "all_sector_experts_models.joblib"
RESULT_PATH = DATA_PROCESSED / "backtest_full_sector_capped_real.json"

#: Swept empirically (0.05, 0.02 -> didn't bind; 0.01 -> binds and moves
#: the period-champion mean; see module docstring). Not a claimed-optimal
#: value or a production default -- TradingConfig.max_ticker_share_pct
#: stays None (off) by default, same "confirm before hardcoding" posture
#: as position_cap_pct.
TICKER_CAP = 0.01


def champions(entries):
    """The entry with the lowest (best) static rank among whatever WAS
    actually taken each day -- the PARA-independent analog of "rank 1",
    since the static rank label doesn't get recomputed after a capped
    ticker is excluded from eligibility (see module docstring, point 2)."""
    if entries.empty:
        return entries
    return entries.loc[entries.groupby(level="timestamp")["rank"].idxmin()]


def champion_stats(champs, raw) -> dict:
    if champs.empty:
        return {"n": 0}
    r = raw.loc[champs.index]
    tickers = champs.index.get_level_values("ticker")
    return {
        "n": int(len(champs)),
        "unique_tickers": int(tickers.nunique()),
        "mean_return": float(r.mean()),
        "median_return": float(r.median()),
        "win_rate": float((r > 0).mean()),
        "top_tickers": tickers.value_counts().head(10).to_dict(),
    }


def main() -> None:
    config = BaselineConfig()
    provider = RealDataProvider()
    broad_spec = next(s for s in EXPERTS if s.sector is None)
    sector_specs = [s for s in EXPERTS if s.sector is not None]
    models = joblib.load(MODEL_PATH)

    test_table = provider.get_feature_table(TEST_START, TEST_END, hold_days=config.trading.hold_days)
    broad_test = build_expert_data(test_table, broad_spec)
    trained_specs = [s for s in sector_specs if s.name in models]

    expert_predictions = {"broad": models["broad"].predict(broad_test.X)}
    for spec in trained_specs:
        edata = build_expert_data(test_table, spec)
        if not edata.frame.empty:
            expert_predictions[spec.name] = models[spec.name].predict(edata.X)

    combiner = OverallModel(tuple([broad_spec] + trained_specs))
    ranked = combiner.rank(combiner.combine(expert_predictions))
    n_periods = broad_test.frame.index.get_level_values("timestamp").nunique()
    budget = int(round((200 / 50) * n_periods))
    raw = broad_test.frame[RAW_TARGET_COLUMN]
    all_trading_days = broad_test.frame.index.get_level_values("timestamp").unique().to_numpy()

    # The raw scoring dominance, before any trading mechanics apply --
    # context for why the entry-level cap needs to be this tight.
    daily_winner = ranked[ranked["rank"] == 1]
    n_days = ranked.index.get_level_values("timestamp").nunique()
    daily_winner_tickers = daily_winner.index.get_level_values("ticker")
    print(f"Raw daily #1-ranked ticker, before any trading mechanics: PARA on "
          f"{(daily_winner_tickers == 'PARA').sum()}/{n_days} days "
          f"({(daily_winner_tickers == 'PARA').mean():.1%}); "
          f"{daily_winner_tickers.nunique()} distinct tickers ever hold the #1 slot.")

    uncapped_cfg = replace(config.trading, window_trading_days=n_periods, trade_budget=budget)
    capped_cfg = replace(uncapped_cfg, max_ticker_share_pct=TICKER_CAP)

    entries_uncapped = TradeSelector(uncapped_cfg).run(ranked)
    entries_uncapped = entries_uncapped[entries_uncapped["action"] == "entry"]
    entries_capped = TradeSelector(capped_cfg).run(ranked)
    entries_capped = entries_capped[entries_capped["action"] == "entry"]

    champs_uncapped = champions(entries_uncapped)
    champs_capped = champions(entries_capped)
    stats_uncapped = champion_stats(champs_uncapped, raw)
    stats_capped = champion_stats(champs_capped, raw)

    print(f"\nTICKER_CAP={TICKER_CAP:.0%} (~{TICKER_CAP * (budget // 2):.0f} entries/ticker of "
          f"{budget // 2} total-entries denominator)")
    print(f"period champions -- uncapped: n={stats_uncapped['n']} unique={stats_uncapped['unique_tickers']} "
          f"mean={stats_uncapped['mean_return']:+.4%} median={stats_uncapped['median_return']:+.4%} "
          f"win={stats_uncapped['win_rate']:.1%}  top={stats_uncapped['top_tickers']}")
    print(f"period champions -- capped:   n={stats_capped['n']} unique={stats_capped['unique_tickers']} "
          f"mean={stats_capped['mean_return']:+.4%} median={stats_capped['median_return']:+.4%} "
          f"win={stats_capped['win_rate']:.1%}  top={stats_capped['top_tickers']}")

    diff = decompose_selection_change(champs_uncapped, champs_capped, raw, all_trading_days,
                                       config.trading.hold_days)
    diff["label_a"], diff["label_b"] = "uncapped_champion", "capped_champion"
    print(f"\ndecomposition (period champions, uncapped vs capped): common={diff['common']['n']}  "
          f"only_uncapped={diff['only_a']['n']} (blocks={diff['only_a']['distinct_blocks']})  "
          f"only_capped={diff['only_b']['n']} (blocks={diff['only_b']['distinct_blocks']})")
    if diff["significance_only_a_vs_only_b"]:
        print(f"  block-effective t: {diff['significance_only_a_vs_only_b']['block_effective_t']:+.2f}")

    # THE FULL DIAGNOSTIC SUITE, under the cap -- same rank-bucket +
    # split-sample analysis every other checkpoint here gets, run on the
    # CAPPED entries as the primary (PARA-independent) read, not the
    # uncapped ones.
    rank_bucket_stats_capped = _rank_bucket_stats(entries_capped, raw.loc[entries_capped.index])
    rank_bucket_stats_uncapped = _rank_bucket_stats(entries_uncapped, raw.loc[entries_uncapped.index])
    split_validation_capped = _split_sample_validation(
        entries_capped, raw.loc[entries_capped.index], all_trading_days, config.trading.hold_days
    )

    print("\n=== Full rank-bucket breakdown, CAPPED (the PARA-independent read) ===")
    for row in rank_bucket_stats_capped:
        if row["n"]:
            print(f"  rank {row['rank_range']:>10}: n={row['n']:>4}  mean={row['mean_return']:+.4%}  "
                  f"median={row['median_return']:+.4%}  win_rate={row['win_rate']:.1%}")
    print("\n=== Full rank-bucket breakdown, UNCAPPED (for comparison) ===")
    for row in rank_bucket_stats_uncapped:
        if row["n"]:
            print(f"  rank {row['rank_range']:>10}: n={row['n']:>4}  mean={row['mean_return']:+.4%}  "
                  f"median={row['median_return']:+.4%}  win_rate={row['win_rate']:.1%}")

    print(f"\nSplit-sample validation, CAPPED (embargoed {split_validation_capped['embargoed_entries']} entries):")
    for year, year_stats in split_validation_capped["by_year"].items():
        print(f"  {year}:")
        for row in year_stats:
            if row["n"]:
                print(f"    rank {row['rank_range']:>10}: n={row['n']:>4}  mean={row['mean_return']:+.4%}  "
                      f"win_rate={row['win_rate']:.1%}")
    print("  consistent across years:", {
        k: ("YES" if v else "no") for k, v in split_validation_capped["direction_consistent_across_years"].items()
    })

    results = {
        "ticker_cap": TICKER_CAP,
        "raw_daily_argmax_winner_para_share": float((daily_winner_tickers == "PARA").mean()),
        "raw_daily_argmax_distinct_tickers": int(daily_winner_tickers.nunique()),
        "period_champion_uncapped": stats_uncapped,
        "period_champion_capped": stats_capped,
        "decomposition_uncapped_vs_capped": diff,
        "rank_bucket_stats_capped": rank_bucket_stats_capped,
        "rank_bucket_stats_uncapped": rank_bucket_stats_uncapped,
        "split_sample_validation_capped": split_validation_capped,
    }
    RESULT_PATH.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nSaved: {RESULT_PATH}")


if __name__ == "__main__":
    main()
