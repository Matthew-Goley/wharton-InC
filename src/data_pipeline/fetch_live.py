"""Pull recent OHLCV bars for Wharton's approved live-trading list
(``config/stock_list.py``) — a RECURRING refresh job, not a one-time
historical backfill. Contrast ``src.data_pipeline.fetch_training``, which
exists to pull one long, mostly-static historical span for the ~653-name
training universe.

Why this can't just be "run fetch_training.py with a smaller ticker list":

- Every run of fetch_training.py re-derives its window from ``--start``/
  ``--end`` and treats an existing chunk CSV purely as "don't re-hit the API
  for a span already downloaded" — right for a backfill that's fetched once
  and (mostly) never touched again, wrong for a job meant to run repeatedly
  and stay current. This script instead treats the ticker-level CSV itself
  as the cache: each run fetches only the delta since the last cached date
  (or a full INITIAL_LOOKBACK_CALENDAR_DAYS window on the very first run for
  a ticker) and appends, so a daily/on-demand run costs one small request
  per ticker, not a re-derivation of the whole window.
- fetch_training.py's chunk-and-stitch machinery exists to stay inside
  Yahoo's per-request span limit on long multi-year pulls. Daily bars have
  no such limit here (``INTERVAL_LIMITS["1d"] = (None, 3650)`` in
  fetch_training.py) and this script never requests more than
  INITIAL_LOOKBACK_CALENDAR_DAYS at once — chunking doesn't apply and isn't
  imported.

What IS reused directly from fetch_training.py (no point re-deriving
tested logic): ``normalize`` (column-flattening + tz handling) and
``stitch`` (concat + boundary dedup, keeping the LAST value on any
overlapping timestamp — which is exactly the semantic this script wants
for merging cached history with a fresh delta: a freshly re-fetched bar
for a date already on file should win, e.g. an intraday-fetched bar later
corrected by the final closing print).

Output format matches fetch_training.py's exactly (``timestamp, Open,
High, Low, Close, Adj Close, Volume``), written to ``data/raw/live/``, a
separate directory from ``data/raw/training/`` — so a live-scoring
``RealDataProvider(universe=STOCK_LIST, data_dir=LIVE_DIR,
apply_point_in_time_filter=False)`` reads it with the exact same
``_build_ticker_features`` code path used for training (CLAUDE.md's "one
shared feature pipeline" rule), no second feature-computation path.

Usage
-----
    python -m src.data_pipeline.fetch_live                # refresh every approved ticker
    python -m src.data_pipeline.fetch_live --tickers AAPL MSFT
    python -m src.data_pipeline.fetch_live --force         # ignore cache, full re-backfill

Output
------
    data/raw/live/<TICKER>_1d.csv
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config.stock_list import STOCK_LIST  # noqa: E402
from src.data_pipeline.fetch_training import SLEEP_BETWEEN_REQUESTS, normalize, stitch  # noqa: E402

LIVE_DIR = REPO_ROOT / "data" / "raw" / "live"
INTERVAL = "1d"

#: How far back a ticker's FIRST fetch (no existing cache) reaches. 252
#: trading days is this project's longest rolling window
#: (pct_from_252d_high) and roughly maps to 365 calendar days; the extra
#: buffer covers weekends/holidays and this project's other windows
#: (60-day, 63-day) reaching full min_periods immediately rather than
#: leaning on their degraded-sample-size floor from day one.
INITIAL_LOOKBACK_CALENDAR_DAYS = 400


def _existing_range(path: Path) -> Optional[tuple[pd.Timestamp, pd.DataFrame]]:
    """(last cached date, cached frame), or None if nothing's cached yet."""
    if not path.exists():
        return None
    df = pd.read_csv(path, index_col="timestamp", parse_dates=["timestamp"])
    if df.empty:
        return None
    return df.index.max(), df


def _fetch_window(
    existing: Optional[tuple[pd.Timestamp, pd.DataFrame]], today: dt.date
) -> Optional[dt.date]:
    """The start date to fetch from, or None if already up to date.

    Pure (no I/O) so the incremental-vs-backfill decision is testable
    without touching the filesystem or the network.
    """
    if existing is None:
        return today - dt.timedelta(days=INITIAL_LOOKBACK_CALENDAR_DAYS)
    last_date, _cached = existing
    start = (last_date + pd.Timedelta(days=1)).date()
    return None if start > today else start


def _merge(cached: Optional[pd.DataFrame], fresh: pd.DataFrame) -> pd.DataFrame:
    """Combine cached history with a freshly-fetched window. Freshly-fetched
    values win on any overlapping date -- see the module docstring's note
    on why (a live re-fetch should be allowed to correct an already-cached
    bar, not be shadowed by it). Pure, no I/O -- reuses fetch_training's
    own tested stitch(), doesn't reimplement the dedup."""
    return stitch([cached, fresh]) if cached is not None else fresh


def fetch_one(ticker: str, today: dt.date, force: bool) -> str:
    """Refresh one ticker's cache; returns a short status string."""
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    path = LIVE_DIR / f"{ticker}_{INTERVAL}.csv"
    existing = None if force else _existing_range(path)

    start = _fetch_window(existing, today)
    if start is None:
        return "up to date"

    # yfinance's `end` is exclusive -- +1 day so a request made after
    # today's close actually gets today's bar (one made intraday won't,
    # correctly -- there is no bar yet).
    end = today + dt.timedelta(days=1)
    df = yf.download(
        ticker, start=start.isoformat(), end=end.isoformat(),
        interval=INTERVAL, auto_adjust=False, progress=False, threads=False,
    )
    time.sleep(SLEEP_BETWEEN_REQUESTS)

    cached = existing[1] if existing is not None else None
    if df is None or df.empty:
        return "no new bars" if cached is not None else "FAILED (no data)"

    fresh = normalize(df, ticker, INTERVAL)
    combined = _merge(cached, fresh)
    combined.to_csv(path, index_label="timestamp")
    added = len(combined) - (len(cached) if cached is not None else 0)
    return f"+{added} rows -> {len(combined)} total, through {combined.index.max().date()}"


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tickers", nargs="+", default=None, help="override config/stock_list.py")
    p.add_argument("--force", action="store_true",
                    help="ignore the cache, full INITIAL_LOOKBACK_CALENDAR_DAYS re-backfill")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    tickers = args.tickers or STOCK_LIST
    if not tickers:
        print(
            "config/stock_list.py is empty (0 tickers) -- nothing to fetch.\n"
            "Populate it with Wharton's approved list before running this "
            "(baseline_system_architecture.md open question 26).",
            file=sys.stderr,
        )
        return 1

    today = dt.date.today()
    print(f"fetch_live: {len(tickers)} ticker(s), through {today}")
    failures = []
    for ticker in tickers:
        status = fetch_one(ticker, today, args.force)
        print(f"  {ticker}: {status}")
        if status.startswith("FAILED"):
            failures.append(ticker)

    if failures:
        print(f"\nno data for: {', '.join(failures)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
