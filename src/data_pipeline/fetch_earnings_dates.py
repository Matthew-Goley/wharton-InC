"""Fetch earnings dates + EPS estimate/actual for the training universe --
data/processed side of baseline_system_architecture.md open question 21
(non-price data feasibility) and the has_earnings_data leakage gate
(open question 22).

Source: yfinance's Ticker.get_earnings_dates(limit=100) -- 100 is Yahoo's
own hard cap, not a choice here (a higher limit raises ValueError).
Checked directly (not assumed) against this project's TRAIN window: a
regularly-reporting large/mid-cap (AAPL, AAL) returns complete 2018-2024
coverage in that one call, both EPS estimate and actual populated.
Delisted/acquired tickers (checked: CELG, RHT, TWTR, XLNX) return nothing
at all -- expected, not a bug, and exactly the missingness pattern
open question 22 exists to test for leakage before any earnings-derived
feature gets built on top of it.

Caches every successful fetch to data/raw/reference/earnings_dates_raw.csv
so a re-run only re-fetches tickers that failed or are new to the
universe -- same convention as fetch_sector_map.py.

Usage:
    python -m src.data_pipeline.fetch_earnings_dates
    python -m src.data_pipeline.fetch_earnings_dates --force   # ignore cache
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config.training_universe import TRAINING_UNIVERSE  # noqa: E402

CACHE_PATH = REPO_ROOT / "data" / "raw" / "reference" / "earnings_dates_raw.csv"
EARNINGS_LIMIT = 100  # Yahoo's hard cap -- ValueError above this
SLEEP_BETWEEN_REQUESTS = 0.4


def load_cache() -> pd.DataFrame:
    if not CACHE_PATH.exists():
        return pd.DataFrame(columns=["ticker", "earnings_date", "eps_estimate", "eps_actual", "surprise_pct"])
    return pd.read_csv(CACHE_PATH, parse_dates=["earnings_date"])


def save_cache(rows: pd.DataFrame) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(CACHE_PATH, index=False)


def fetch_all(universe: list[str], force: bool) -> tuple[pd.DataFrame, list[str]]:
    cached = pd.DataFrame(columns=["ticker", "earnings_date", "eps_estimate", "eps_actual", "surprise_pct"]) \
        if force else load_cache()
    already_attempted = set(cached["ticker"].unique())
    to_fetch = [t for t in universe if t not in already_attempted]
    print(f"{len(already_attempted)} tickers already attempted (cached), {len(to_fetch)} to fetch")

    new_rows: list[pd.DataFrame] = []
    failures: list[str] = []
    for i, ticker in enumerate(to_fetch):
        try:
            ed = yf.Ticker(ticker).get_earnings_dates(limit=EARNINGS_LIMIT)
            if ed is None or ed.empty:
                failures.append(ticker)
                print(f"  [{i+1}/{len(to_fetch)}] {ticker}: no earnings data")
            else:
                ed = ed.reset_index().rename(columns={
                    "Earnings Date": "earnings_date", "EPS Estimate": "eps_estimate",
                    "Reported EPS": "eps_actual", "Surprise(%)": "surprise_pct",
                })
                ed["ticker"] = ticker
                new_rows.append(ed[["ticker", "earnings_date", "eps_estimate", "eps_actual", "surprise_pct"]])
        except Exception as exc:
            failures.append(ticker)
            print(f"  [{i+1}/{len(to_fetch)}] {ticker}: FAILED ({exc})")
        if (i + 1) % 25 == 0:
            combined = pd.concat([cached, *new_rows], ignore_index=True) if new_rows else cached
            print(f"  [{i+1}/{len(to_fetch)}] ...{combined['ticker'].nunique()} tickers with data so far, saving cache")
            save_cache(combined)
        time.sleep(SLEEP_BETWEEN_REQUESTS)

    # Failed tickers get a placeholder row (ticker, all-NaN) so a re-run
    # without --force doesn't keep re-attempting a name that's genuinely
    # delisted -- "already attempted and failed" is itself informative,
    # distinct from "never tried."
    if failures:
        placeholder = pd.DataFrame({
            "ticker": failures, "earnings_date": pd.NaT, "eps_estimate": pd.NA,
            "eps_actual": pd.NA, "surprise_pct": pd.NA,
        })
        new_rows.append(placeholder)

    combined = pd.concat([cached, *new_rows], ignore_index=True) if new_rows else cached
    save_cache(combined)
    return combined, failures


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--force", action="store_true", help="ignore the cache, re-fetch every ticker")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    combined, failures = fetch_all(TRAINING_UNIVERSE, args.force)

    with_data = combined.dropna(subset=["earnings_date"])["ticker"].unique()
    n_with_data = len(with_data)
    n_total = len(TRAINING_UNIVERSE)
    print(f"\n{n_with_data}/{n_total} universe tickers ({n_with_data/n_total:.1%}) have earnings data.")
    print(f"{len(failures)} newly failed this run: {failures[:20]}{'...' if len(failures) > 20 else ''}")
    all_failed = sorted(set(TRAINING_UNIVERSE) - set(with_data))
    print(f"{len(all_failed)}/{n_total} total universe tickers ({len(all_failed)/n_total:.1%}) have NO earnings "
          f"data (cumulative, across all runs): {all_failed[:20]}{'...' if len(all_failed) > 20 else ''}")
    print(f"\nSaved: {CACHE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
