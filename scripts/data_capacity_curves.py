"""Diagnostic-only data-scaling and capacity curves for the broad expert.

Purpose (explicitly diagnostic, not a search): read the SHAPE of two
curves -- IC vs. training-data span, IC vs. model capacity -- to learn
whether this project is data-limited, capacity-limited, or feature-
limited. Not to find or promote a better configuration. See the explicit
prohibitions in this file's own comments before touching anything here.

PROHIBITIONS (deliberate, not oversights):
  - No GA search. Every arm uses BEST_GENOME (scripts.backtest_broad_real)
    unmodified except for the ONE axis under test.
  - No winner selection or promotion. This script only ever prints IC and
    paired-t numbers; it never writes a new default anywhere.
  - Broad expert only. No sector experts -- their smaller per-model
    universes would confound the data curve.
  - No feature-set changes. Uses FEATURE_COLUMNS exactly as committed.

Shared evaluation block (Task 1): the most recent 12 months of the real
panel, held out from every arm, with a hold_days-trading-day embargo at
the train/eval seam (purges training rows whose forward-label window
would otherwise reach into the eval block) -- computed against the
panel's own real trading dates, not an approximated calendar.

AUTOCORRELATION-CORRECTED INFERENCE (added on the statistical-rigor
follow-up round): the naive paired SE (std(diff)/sqrt(n_dates), as used
by src.evaluation.ic_comparison.paired_ic_comparison) assumes independent
per-date observations, which is false for a hold_days-forward-return
series -- see src/evaluation/block_se.py for why, and
tests/test_block_se.py for the deliberate failing-case proof that
block_bootstrap_se (not Newey-West) is the better-matched estimator for
THIS project's specific correlation structure (a hard, known-width
hold_days block, not a smoothly-decaying process). Both are computed and
reported for every comparison; block_bootstrap_se is the adopted standard
figure ("corrected SE" / "corrected t"), Newey-West is a cross-check.
Every arm-vs-anchor comparison here is local ic_a/ic_b/diff recomputation
via per_date_cross_sectional_ic directly -- NOT a modification to
paired_ic_comparison itself, which stays untouched for its many other
call sites.

Usage:
    python -m scripts.data_capacity_curves
"""
from __future__ import annotations

import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import BROAD_EXPERT, BaselineConfig, ExpertSpec  # noqa: E402
from src.data_layer import RealDataProvider  # noqa: E402
from src.data_layer.real_provider import FEATURE_COLUMNS  # noqa: E402
from src.evaluation import keep_or_discard, paired_ic_comparison  # noqa: E402
from src.evaluation.metrics import cross_sectional_ic, per_date_cross_sectional_ic  # noqa: E402
from src.evaluation.block_se import newey_west_se, block_bootstrap_se  # noqa: E402
from src.experts.base import ExpertModel  # noqa: E402
from src.experts.registry import build_expert_data  # noqa: E402
from scripts.backtest_broad_real import BEST_GENOME  # noqa: E402

HOLD_DAYS = 5
NW_LAG = HOLD_DAYS - 1          # max possible label-window overlap between adjacent dates
BLOCK_DAYS = HOLD_DAYS          # this project's established independence unit (decompose_selection_change)
DATA_YEARS = [1, 2, 3, 4, 6, 8]
N_ESTIMATORS_SWEEP = [50, 100, 200, 400, 800, 1600]
MAX_DEPTH_SWEEP = [2, 4, 6, 8]

# Every arm-vs-anchor comparison from the original run (6 data-curve arms +
# 6 n_estimators arms + 4 max_depth arms = 16), plus Task 6's ONE new
# pre-registered arm (old-3yr vs recent-3yr) = 17. This constant exists so
# the Bonferroni threshold is computed from a single, named, checkable
# number rather than re-derived by eyeballing print output.
COMPARISON_FAMILY_SIZE = len(DATA_YEARS) + len(N_ESTIMATORS_SWEEP) + len(MAX_DEPTH_SWEEP) + 1


def _bonferroni_t(family_size: int, alpha: float = 0.05) -> float:
    """Two-sided Bonferroni-corrected critical t/z for a family of
    ``family_size`` independent-ish comparisons, via the normal
    approximation (matches how the original 16-comparison threshold,
    t~2.96, was derived)."""
    return float(norm.ppf(1 - alpha / (2 * family_size)))


def _fit_and_predict(genome, X_train, y_train, X_eval, seed: int) -> pd.Series:
    model = ExpertModel(BROAD_EXPERT, genome, random_seed=seed).fit(X_train, y_train, hold_days=HOLD_DAYS)
    return model.predict(X_eval)["predicted_return"]


def _corrected_pair(dates_common: np.ndarray, diff: np.ndarray) -> dict:
    """Both autocorrelation corrections for one paired-difference series,
    plus the naive figure for direct comparison. block_bootstrap_se is the
    adopted standard ("corrected_se"/"corrected_t") per the documented,
    deliberately-tested finding in src/evaluation/block_se.py that
    Newey-West's Bartlett taper under-recovers a hard-block correlation
    structure -- Newey-West is still reported (nw_se/nw_t) as a
    cross-check, per the request to report both where cheap."""
    n = len(diff)
    naive_se = float(np.std(diff, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    mean_diff = float(np.mean(diff)) if n else float("nan")
    nw_se = newey_west_se(diff, lag=NW_LAG)
    boot = block_bootstrap_se(diff, dates_common, block_days=BLOCK_DAYS)
    corrected_se = boot["se"]
    return {
        "n_dates": n,
        "n_blocks": boot["n_blocks"],
        "mean_diff": mean_diff,
        "naive_se": naive_se,
        "naive_t": mean_diff / naive_se if naive_se and naive_se > 0 else float("nan"),
        "nw_se": nw_se,
        "nw_t": mean_diff / nw_se if nw_se and nw_se > 0 else float("nan"),
        "corrected_se": corrected_se,
        "corrected_t": mean_diff / corrected_se if corrected_se and corrected_se > 0 else float("nan"),
    }


def _report_arm(label: str, dates_eval: np.ndarray, y_eval: np.ndarray, preds: pd.Series,
                 anchor_preds: pd.Series, anchor_ic_series: pd.Series, n_train_rows: int,
                 train_span: str, series_store: dict) -> dict:
    ic = cross_sectional_ic(dates_eval, y_eval, preds.to_numpy())
    result = paired_ic_comparison(dates_eval, y_eval, anchor_preds.to_numpy(), preds.to_numpy())
    decision = keep_or_discard(result)

    arm_ic_series = per_date_cross_sectional_ic(dates_eval, y_eval, preds.to_numpy())
    series_store[label] = arm_ic_series
    common = anchor_ic_series.index.intersection(arm_ic_series.index)
    diff = (arm_ic_series.loc[common] - anchor_ic_series.loc[common]).to_numpy()
    corrected = _corrected_pair(common.to_numpy(), diff)
    # Consistency check: our locally-recomputed naive SE must match
    # paired_ic_comparison's own naive SE (same formula, independent call
    # path) -- catches a wiring bug (wrong series diffed, wrong order)
    # immediately rather than silently reporting two different "naive" SEs.
    if result["se"] is not None and np.isfinite(result["se"]) and np.isfinite(corrected["naive_se"]):
        assert abs(result["se"] - corrected["naive_se"]) < 1e-9, (
            f"{label}: naive SE mismatch between paired_ic_comparison ({result['se']}) "
            f"and local recomputation ({corrected['naive_se']})"
        )

    print(f"  {label:<10} train_rows={n_train_rows:>9,}  span={train_span:<23}  "
          f"IC={ic:+.4f}  t(naive)={result['t']:+.2f}  t(corrected)={corrected['corrected_t']:+.2f}  [{decision}]")
    return {
        "label": label, "ic": ic, "paired_t": result["t"], "n_train_rows": n_train_rows,
        "train_span": train_span, "corrected": corrected,
    }


def main() -> None:
    config = BaselineConfig()
    seed = config.ga.random_seed
    print(f"random_seed = {seed}  (config.ga.random_seed, fixed and reported per instruction)")
    print(f"BEST_GENOME (unmodified except the one axis under test in each sweep): {BEST_GENOME}")
    print(f"FEATURE_COLUMNS ({len(FEATURE_COLUMNS)}, unmodified throughout): {FEATURE_COLUMNS}")
    print(f"NW_LAG={NW_LAG}  BLOCK_DAYS={BLOCK_DAYS}  COMPARISON_FAMILY_SIZE={COMPARISON_FAMILY_SIZE}  "
          f"Bonferroni t-threshold={_bonferroni_t(COMPARISON_FAMILY_SIZE):.3f}")

    t_start = time.perf_counter()
    provider = RealDataProvider()  # every default: full TRAINING_UNIVERSE, point-in-time filtered, labeled
    broad_spec = ExpertSpec(BROAD_EXPERT, None)

    print("\nLoading the full panel (one call, whole history)...")
    full_table = provider.get_feature_table("2018-01-01", "2025-12-31", hold_days=HOLD_DAYS)
    frame = full_table.frame
    all_dates = sorted(frame.index.get_level_values("timestamp").unique())
    print(f"  {len(frame):,} rows, {len(all_dates)} trading dates, "
          f"{frame.index.get_level_values('ticker').nunique()} tickers, "
          f"{all_dates[0].date()} .. {all_dates[-1].date()}  ({time.perf_counter()-t_start:.1f}s)")

    # === Task 1: shared evaluation block =====================================
    panel_end = all_dates[-1]
    eval_start_target = panel_end - pd.DateOffset(years=1) + pd.Timedelta(days=1)
    eval_dates = [d for d in all_dates if d >= eval_start_target]
    eval_start = eval_dates[0]
    eval_end = eval_dates[-1]

    # Embargo: step back HOLD_DAYS+1 REAL trading dates (the panel's own
    # dates, not an approximated calendar) from eval_start, so the last
    # train row's forward-hold_days label lands strictly before eval_start.
    eval_start_idx = all_dates.index(eval_start)
    train_end = all_dates[eval_start_idx - HOLD_DAYS - 1]

    print(f"\n=== Task 1: shared evaluation block ===")
    print(f"  eval block:  {eval_start.date()} .. {eval_end.date()}")
    print(f"  train_end (embargoed, {HOLD_DAYS}+1 real trading dates before eval_start): {train_end.date()}")

    eval_frame = frame[(frame.index.get_level_values("timestamp") >= eval_start) &
                        (frame.index.get_level_values("timestamp") <= eval_end)]
    eval_dates_arr = eval_frame.index.get_level_values("timestamp").to_numpy()
    y_eval = eval_frame[full_table.target_column].to_numpy()
    X_eval = eval_frame[FEATURE_COLUMNS]
    n_eval_dates = eval_frame.index.get_level_values("timestamp").nunique()
    n_eval_tickers = eval_frame.index.get_level_values("ticker").nunique()
    mean_names_per_date = len(eval_frame) / n_eval_dates
    n_blocks = n_eval_dates // HOLD_DAYS

    # Verify the embargo directly rather than trust the arithmetic: no
    # train-side row (timestamp <= train_end) may have a label date
    # (timestamp + HOLD_DAYS trading days) inside the eval block.
    train_check = frame[frame.index.get_level_values("timestamp") <= train_end]
    last_train_dates = sorted(train_check.index.get_level_values("timestamp").unique())[-HOLD_DAYS:]
    violation = any(d >= eval_start for d in last_train_dates)
    print(f"  embargo verified directly (no overlap in trading dates): {'FAIL' if violation else 'OK'}")

    print(f"  n_dates={n_eval_dates}  n_tickers={n_eval_tickers}  mean_names/date={mean_names_per_date:.1f}  "
          f"non-overlapping {HOLD_DAYS}-day blocks={n_blocks}")

    # Task 1.4: empirical noise floor -- fit ONE reference model (the
    # largest available data arm, i.e. the same model that anchors both
    # curves below) and report THREE SEs on its own per-date IC series:
    # naive (std/sqrt(n), assumes independence), Newey-West (HAC, lag=4),
    # and block bootstrap (50 non-overlapping 5-day blocks). Measured on
    # THIS block, not assumed from any figure quoted elsewhere.
    print(f"\n  Fitting the reference model (max available history through {train_end.date()}) "
          f"to measure the empirical noise floor...")
    train_all = frame[frame.index.get_level_values("timestamp") <= train_end]
    X_train_all, y_train_all = train_all[FEATURE_COLUMNS], train_all[full_table.target_column]
    t0 = time.perf_counter()
    anchor_preds = _fit_and_predict(BEST_GENOME, X_train_all, y_train_all, X_eval, seed)
    print(f"    fit+predict in {time.perf_counter()-t0:.1f}s, {len(X_train_all):,} train rows")
    anchor_ic_series = per_date_cross_sectional_ic(eval_dates_arr, y_eval, anchor_preds.to_numpy())
    anchor_values = anchor_ic_series.to_numpy()
    anchor_dates_arr = anchor_ic_series.index.to_numpy()

    naive_se = float(anchor_values.std(ddof=1) / np.sqrt(len(anchor_values)))
    nw_se_noise = newey_west_se(anchor_values, lag=NW_LAG)
    boot_noise = block_bootstrap_se(anchor_values, anchor_dates_arr, block_days=BLOCK_DAYS)
    print(f"  single-arm noise floor over {len(anchor_ic_series)} dates:")
    print(f"    naive          SE={naive_se:.4f}  (assumes {len(anchor_ic_series)} independent dates)")
    print(f"    Newey-West     SE={nw_se_noise:.4f}  (lag={NW_LAG}; cross-check -- known to under-recover a hard "
          f"block correlation structure, see block_se.py docstring)")
    print(f"    block bootstrap SE={boot_noise['se']:.4f}  (n_blocks={boot_noise['n_blocks']}; ADOPTED STANDARD "
          f"for this eval block going forward)")
    single_arm_se = boot_noise["se"]  # the adopted, corrected figure

    train_start_all = sorted(train_all.index.get_level_values("timestamp").unique())[0]

    # === Task 2: data scaling curve ===========================================
    print(f"\n=== Task 2: data scaling curve (anchor = full available history through {train_end.date()},"
          f" same model as the noise-floor fit above) ===")
    data_results = []
    arm_ic_series_by_label: dict[str, pd.Series] = {}
    for years in DATA_YEARS:
        arm_start = train_end - pd.DateOffset(years=years)
        arm_train = frame[(frame.index.get_level_values("timestamp") > arm_start) &
                           (frame.index.get_level_values("timestamp") <= train_end)]
        if arm_train.empty:
            print(f"  {years}yr: no rows before {train_end.date()} at this span -- skipped")
            continue
        actual_start = sorted(arm_train.index.get_level_values("timestamp").unique())[0]
        truncated = actual_start > arm_start
        X_tr, y_tr = arm_train[FEATURE_COLUMNS], arm_train[full_table.target_column]
        t0 = time.perf_counter()
        preds = _fit_and_predict(BEST_GENOME, X_tr, y_tr, X_eval, seed)
        span = f"{actual_start.date()}..{train_end.date()}" + (" (TRUNCATED)" if truncated else "")
        r = _report_arm(f"{years}yr", eval_dates_arr, y_eval, preds, anchor_preds, anchor_ic_series,
                         len(arm_train), span, arm_ic_series_by_label)
        r["fit_time_s"] = round(time.perf_counter() - t0, 1)
        data_results.append(r)

    # === Task 3: capacity curve (fixed at full 8yr/max-available data) ========
    print(f"\n=== Task 3: capacity curve (fixed data = full available history through {train_end.date()},"
          f" anchor = BEST_GENOME unmodified, same model as above) ===")
    print(f"  BEST_GENOME defaults on the two swept axes: n_estimators={BEST_GENOME.n_estimators}, "
          f"max_depth={BEST_GENOME.max_depth}")

    print(f"\n  -- n_estimators sweep (max_depth fixed at BEST_GENOME's {BEST_GENOME.max_depth}) --")
    n_estimators_results = []
    for n in N_ESTIMATORS_SWEEP:
        genome = replace(BEST_GENOME, n_estimators=n)
        t0 = time.perf_counter()
        preds = _fit_and_predict(genome, X_train_all, y_train_all, X_eval, seed)
        r = _report_arm(f"n_est={n}", eval_dates_arr, y_eval, preds, anchor_preds, anchor_ic_series,
                         len(X_train_all), f"{train_start_all.date()}..{train_end.date()}", arm_ic_series_by_label)
        r["fit_time_s"] = round(time.perf_counter() - t0, 1)
        n_estimators_results.append(r)

    print(f"\n  -- max_depth sweep (n_estimators fixed at BEST_GENOME's {BEST_GENOME.n_estimators}) --")
    max_depth_results = []
    for d in MAX_DEPTH_SWEEP:
        genome = replace(BEST_GENOME, max_depth=d)
        t0 = time.perf_counter()
        preds = _fit_and_predict(genome, X_train_all, y_train_all, X_eval, seed)
        r = _report_arm(f"max_depth={d}", eval_dates_arr, y_eval, preds, anchor_preds, anchor_ic_series,
                         len(X_train_all), f"{train_start_all.date()}..{train_end.date()}", arm_ic_series_by_label)
        r["fit_time_s"] = round(time.perf_counter() - t0, 1)
        max_depth_results.append(r)

    # === Task 3 (of the statistical-rigor round): corrected-SE summary tables ==
    bonf_t = _bonferroni_t(COMPARISON_FAMILY_SIZE)
    print(f"\n=== Corrected comparisons (Bonferroni threshold for {COMPARISON_FAMILY_SIZE} comparisons: "
          f"t>={bonf_t:.3f}) ===")

    def _print_corrected_table(title: str, results: list[dict]) -> None:
        print(f"\n  {title}")
        print(f"  {'label':<12} {'IC':>8} {'SE naive':>10} {'SE NW':>10} {'SE boot':>10} "
              f"{'t naive':>8} {'t corrected':>12}  survives?")
        for r in results:
            c = r["corrected"]
            survives = "YES" if abs(c["corrected_t"]) >= bonf_t else "no"
            print(f"  {r['label']:<12} {r['ic']:>+8.4f} {c['naive_se']:>10.4f} {c['nw_se']:>10.4f} "
                  f"{c['corrected_se']:>10.4f} {c['naive_t']:>+8.2f} {c['corrected_t']:>+12.2f}  {survives}")

    _print_corrected_table("Task 2 table -- data scaling curve", data_results)
    _print_corrected_table("Task 3 table -- n_estimators sweep", n_estimators_results)
    _print_corrected_table("Task 3 table -- max_depth sweep", max_depth_results)

    # === Task 3 (MDE) of the statistical-rigor round: 1yr vs 3yr, power framing =
    # NOT a new hypothesis test and NOT added to COMPARISON_FAMILY_SIZE: both
    # arms are already-fitted members of the data curve above, each already
    # individually compared to the anchor. This is a power/MDE illustration
    # only -- "is flat evidence of no effect, or of being underpowered" --
    # using the corrected SE of the pair the user named explicitly.
    print(f"\n=== MDE illustration: 1yr vs 3yr (power framing, NOT a new hypothesis test, "
          f"not added to the comparison family) ===")
    ic_1yr, ic_3yr = arm_ic_series_by_label["1yr"], arm_ic_series_by_label["3yr"]
    common_13 = ic_1yr.index.intersection(ic_3yr.index)
    diff_13 = (ic_3yr.loc[common_13] - ic_1yr.loc[common_13]).to_numpy()
    corrected_13 = _corrected_pair(common_13.to_numpy(), diff_13)
    mde_13 = 2 * corrected_13["corrected_se"]
    print(f"  point-estimate gap (3yr IC - 1yr IC) = {corrected_13['mean_diff']:+.4f}")
    print(f"  corrected SE (block bootstrap, n_blocks={corrected_13['n_blocks']}) = {corrected_13['corrected_se']:.4f}")
    print(f"  minimum detectable IC difference at 2*corrected SE = {mde_13:.4f}")
    print(f"  MDE vs largest observed gap: MDE={mde_13:.4f}  observed_gap={corrected_13['mean_diff']:+.4f}  "
          f"-> {'gap exceeds MDE (a real, detectable move)' if abs(corrected_13['mean_diff']) > mde_13 else 'gap is BELOW MDE -- flat reads as underpowered, not necessarily no-effect'}")

    # === Task 6: one new, pre-registered, non-promoting diagnostic arm ========
    # old-3yr vs recent-3yr, to separate "not data-hungry" from "old data is
    # stale". recent-3yr is the SAME "3yr" arm already fit above (no refit).
    # This is the ONE new arm/comparison this round adds to the family (16 -> 17).
    print(f"\n=== Task 6: pre-registered diagnostic arm -- old-3yr vs recent-3yr "
          f"(non-promoting; the 17th comparison) ===")
    old3_start_target, old3_end_target = pd.Timestamp("2018-01-02"), pd.Timestamp("2020-12-16")
    old3_train = frame[(frame.index.get_level_values("timestamp") >= old3_start_target) &
                        (frame.index.get_level_values("timestamp") <= old3_end_target)]
    old3_start_actual = sorted(old3_train.index.get_level_values("timestamp").unique())[0]
    old3_end_actual = sorted(old3_train.index.get_level_values("timestamp").unique())[-1]
    # No new embargo derivation needed: old3's train_end (2020-12-16) is
    # strictly earlier than the shared train_end (2024-12-16) already
    # verified embargoed above -- an earlier train_end only widens the gap
    # to eval_start, it cannot reintroduce leakage.
    X_old3, y_old3 = old3_train[FEATURE_COLUMNS], old3_train[full_table.target_column]
    t0 = time.perf_counter()
    preds_old3 = _fit_and_predict(BEST_GENOME, X_old3, y_old3, X_eval, seed)
    old3_fit_time = round(time.perf_counter() - t0, 1)
    old3_ic = cross_sectional_ic(eval_dates_arr, y_eval, preds_old3.to_numpy())
    old3_ic_series = per_date_cross_sectional_ic(eval_dates_arr, y_eval, preds_old3.to_numpy())
    recent3_ic_series = arm_ic_series_by_label["3yr"]

    print(f"  old-3yr:    train_rows={len(old3_train):>9,}  span={old3_start_actual.date()}..{old3_end_actual.date()}"
          f"  IC={old3_ic:+.4f}  fit_time_s={old3_fit_time}")
    r3 = next(r for r in data_results if r["label"] == "3yr")
    print(f"  recent-3yr: train_rows={r3['n_train_rows']:>9,}  span={r3['train_span']}  IC={r3['ic']:+.4f}  "
          f"(row-count ratio old/recent = {len(old3_train)/r3['n_train_rows']:.3f})")

    common_old = recent3_ic_series.index.intersection(old3_ic_series.index)
    diff_old = (old3_ic_series.loc[common_old] - recent3_ic_series.loc[common_old]).to_numpy()
    old_vs_recent = _corrected_pair(common_old.to_numpy(), diff_old)
    old3_survives = abs(old_vs_recent["corrected_t"]) >= bonf_t
    print(f"  diff (old3yr - recent3yr) = {old_vs_recent['mean_diff']:+.4f}  "
          f"SE naive={old_vs_recent['naive_se']:.4f}  SE NW={old_vs_recent['nw_se']:.4f}  "
          f"SE boot(adopted)={old_vs_recent['corrected_se']:.4f}  "
          f"t naive={old_vs_recent['naive_t']:+.2f}  t corrected={old_vs_recent['corrected_t']:+.2f}  "
          f"survives Bonferroni({bonf_t:.3f})? {'YES' if old3_survives else 'no'}")
    if old_vs_recent["mean_diff"] < 0 and old3_survives:
        print(f"  -> old data IS materially and significantly worse: the flat scaling curve is a net of two "
              f"opposing effects (more data helps, older data hurts), recency weighting becomes a live candidate.")
    else:
        print(f"  -> no significant old-vs-recent gap detected at the corrected threshold; no evidence the flat "
              f"scaling curve hides an opposing recency effect.")

    print(f"\nTotal wall-clock: {time.perf_counter()-t_start:.1f}s")

    import json
    out = {
        "seed": seed, "eval_start": str(eval_start.date()), "eval_end": str(eval_end.date()),
        "train_end": str(train_end.date()), "n_eval_dates": n_eval_dates, "n_eval_tickers": n_eval_tickers,
        "mean_names_per_date": mean_names_per_date, "n_blocks": n_blocks,
        "nw_lag": NW_LAG, "block_days": BLOCK_DAYS, "comparison_family_size": COMPARISON_FAMILY_SIZE,
        "bonferroni_t": bonf_t,
        "single_arm_se_naive": naive_se, "single_arm_se_nw": nw_se_noise, "single_arm_se_boot": boot_noise["se"],
        "single_arm_se_adopted": single_arm_se, "single_arm_n_blocks": boot_noise["n_blocks"],
        "anchor_ic": cross_sectional_ic(eval_dates_arr, y_eval, anchor_preds.to_numpy()),
        "anchor_ic_series": {str(k.date()): float(v) for k, v in anchor_ic_series.items()},
        "data_curve": data_results, "n_estimators_curve": n_estimators_results, "max_depth_curve": max_depth_results,
        "mde_1yr_vs_3yr": {
            "point_gap": corrected_13["mean_diff"], "corrected_se": corrected_13["corrected_se"],
            "mde_at_2se": mde_13, "corrected_t": corrected_13["corrected_t"],
        },
        "task6_old3yr_vs_recent3yr": {
            "old3_span": f"{old3_start_actual.date()}..{old3_end_actual.date()}",
            "old3_n_train_rows": len(old3_train), "old3_ic": old3_ic,
            "recent3_n_train_rows": r3["n_train_rows"], "recent3_ic": r3["ic"],
            "diff": old_vs_recent["mean_diff"], "naive_se": old_vs_recent["naive_se"],
            "nw_se": old_vs_recent["nw_se"], "corrected_se": old_vs_recent["corrected_se"],
            "naive_t": old_vs_recent["naive_t"], "corrected_t": old_vs_recent["corrected_t"],
            "survives_bonferroni": old3_survives, "fit_time_s": old3_fit_time,
        },
        "arm_ic_series": {label: {str(k.date()): float(v) for k, v in s.items()}
                           for label, s in arm_ic_series_by_label.items()},
    }
    out_path = REPO_ROOT / "data" / "processed" / "data_capacity_curves.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
