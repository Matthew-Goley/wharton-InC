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

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import BROAD_EXPERT, BaselineConfig, ExpertSpec  # noqa: E402
from src.data_layer import RealDataProvider  # noqa: E402
from src.data_layer.real_provider import FEATURE_COLUMNS  # noqa: E402
from src.evaluation import keep_or_discard, paired_ic_comparison  # noqa: E402
from src.evaluation.metrics import cross_sectional_ic, per_date_cross_sectional_ic  # noqa: E402
from src.experts.base import ExpertModel  # noqa: E402
from src.experts.registry import build_expert_data  # noqa: E402
from scripts.backtest_broad_real import BEST_GENOME  # noqa: E402

HOLD_DAYS = 5
DATA_YEARS = [1, 2, 3, 4, 6, 8]
N_ESTIMATORS_SWEEP = [50, 100, 200, 400, 800, 1600]
MAX_DEPTH_SWEEP = [2, 4, 6, 8]


def _fit_and_predict(genome, X_train, y_train, X_eval, seed: int) -> pd.Series:
    model = ExpertModel(BROAD_EXPERT, genome, random_seed=seed).fit(X_train, y_train, hold_days=HOLD_DAYS)
    return model.predict(X_eval)["predicted_return"]


def _report_arm(label: str, dates_eval: np.ndarray, y_eval: np.ndarray, preds: pd.Series,
                 anchor_preds: pd.Series, n_train_rows: int, train_span: str) -> dict:
    ic = cross_sectional_ic(dates_eval, y_eval, preds.to_numpy())
    result = paired_ic_comparison(dates_eval, y_eval, anchor_preds.to_numpy(), preds.to_numpy())
    decision = keep_or_discard(result)
    print(f"  {label:<10} train_rows={n_train_rows:>9,}  span={train_span:<23}  "
          f"IC={ic:+.4f}  paired_t(vs anchor)={result['t']:+.2f}  [{decision}]")
    return {"label": label, "ic": ic, "paired_t": result["t"], "n_train_rows": n_train_rows, "train_span": train_span}


def main() -> None:
    config = BaselineConfig()
    seed = config.ga.random_seed
    print(f"random_seed = {seed}  (config.ga.random_seed, fixed and reported per instruction)")
    print(f"BEST_GENOME (unmodified except the one axis under test in each sweep): {BEST_GENOME}")
    print(f"FEATURE_COLUMNS ({len(FEATURE_COLUMNS)}, unmodified throughout): {FEATURE_COLUMNS}")

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
    # curves below) and take std(per-date IC) / sqrt(n_dates). This is
    # measured on THIS block, not assumed from the ~0.0020 figure
    # observed on the full 497-date panel elsewhere in this project.
    print(f"\n  Fitting the reference model (max available history through {train_end.date()}) "
          f"to measure the empirical noise floor...")
    train_all = frame[frame.index.get_level_values("timestamp") <= train_end]
    X_train_all, y_train_all = train_all[FEATURE_COLUMNS], train_all[full_table.target_column]
    t0 = time.perf_counter()
    anchor_preds = _fit_and_predict(BEST_GENOME, X_train_all, y_train_all, X_eval, seed)
    print(f"    fit+predict in {time.perf_counter()-t0:.1f}s, {len(X_train_all):,} train rows")
    per_date_ic = per_date_cross_sectional_ic(eval_dates_arr, y_eval, anchor_preds.to_numpy())
    single_arm_se = per_date_ic.std(ddof=1) / np.sqrt(len(per_date_ic))
    print(f"  single-arm noise floor: std(per-date IC)={per_date_ic.std(ddof=1):.4f} over "
          f"{len(per_date_ic)} dates -> SE={single_arm_se:.4f}")
    print(f"  (the paired_ic_comparison SE used for every 'paired_t' column below is typically smaller "
          f"than this, since pairing removes shared day-to-day market-regime noise -- reported per-arm below,"
          f" not assumed.)")

    train_start_all = sorted(train_all.index.get_level_values("timestamp").unique())[0]

    # === Task 2: data scaling curve ===========================================
    print(f"\n=== Task 2: data scaling curve (anchor = full available history through {train_end.date()},"
          f" same model as the noise-floor fit above) ===")
    data_results = []
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
        r = _report_arm(f"{years}yr", eval_dates_arr, y_eval, preds, anchor_preds, len(arm_train), span)
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
        r = _report_arm(f"n_est={n}", eval_dates_arr, y_eval, preds, anchor_preds, len(X_train_all),
                         f"{train_start_all.date()}..{train_end.date()}")
        r["fit_time_s"] = round(time.perf_counter() - t0, 1)
        n_estimators_results.append(r)

    print(f"\n  -- max_depth sweep (n_estimators fixed at BEST_GENOME's {BEST_GENOME.n_estimators}) --")
    max_depth_results = []
    for d in MAX_DEPTH_SWEEP:
        genome = replace(BEST_GENOME, max_depth=d)
        t0 = time.perf_counter()
        preds = _fit_and_predict(genome, X_train_all, y_train_all, X_eval, seed)
        r = _report_arm(f"max_depth={d}", eval_dates_arr, y_eval, preds, anchor_preds, len(X_train_all),
                         f"{train_start_all.date()}..{train_end.date()}")
        r["fit_time_s"] = round(time.perf_counter() - t0, 1)
        max_depth_results.append(r)

    print(f"\nTotal wall-clock: {time.perf_counter()-t_start:.1f}s")

    import json
    out = {
        "seed": seed, "eval_start": str(eval_start.date()), "eval_end": str(eval_end.date()),
        "train_end": str(train_end.date()), "n_eval_dates": n_eval_dates, "n_eval_tickers": n_eval_tickers,
        "mean_names_per_date": mean_names_per_date, "n_blocks": n_blocks,
        "single_arm_se": single_arm_se, "anchor_ic": cross_sectional_ic(eval_dates_arr, y_eval, anchor_preds.to_numpy()),
        "data_curve": data_results, "n_estimators_curve": n_estimators_results, "max_depth_curve": max_depth_results,
    }
    out_path = REPO_ROOT / "data" / "processed" / "data_capacity_curves.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
