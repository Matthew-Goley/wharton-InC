"""Pull historical OHLCV bars from Yahoo Finance for the training universe.

Yahoo caps how far back intraday data goes, and the cap is measured from
*today*, not per request -- so you cannot walk backwards in 2-year chunks to
reach 2018. Any request whose START is older than the cap is rejected whole:

    1h data not available for startTime=... and endTime=...
    The requested range must be within the last 730 days.

Practical consequence for this project (verified 2026-08-16):

    interval  usable history          SPY rows
    1h        last ~730 days only     ~3,500
    1d        back to listing         ~2,000 for 2018-2025

So `--interval 1h` gets you roughly two years of hourly bars and nothing
older, while `--interval 1d` covers the full 2018-2025 span. If hourly bars
going back to 2018 turn out to be a hard requirement, that needs a paid
vendor (Polygon, Databento, Alpaca, Tiingo) -- yfinance cannot serve it.

The chunk-and-stitch machinery below is still doing real work: it keeps each
request inside Yahoo's per-request span limit, caches every chunk to its own
CSV so a re-run doesn't re-hit the API, and stitches the chunks into one
deduplicated, time-sorted file.

Usage
-----
    # hourly, maximum window Yahoo will serve (auto-clamped)
    python src/data_pipeline/fetch_training.py --interval 1h

    # daily, full requested span
    python src/data_pipeline/fetch_training.py --interval 1d --start 2018-01-01

    # explicit tickers, ignore cached chunks
    python src/data_pipeline/fetch_training.py --tickers SPY QQQ --force

Output
------
    data/raw/training/chunks/SPY_1h_2024-08-17_2025-08-16.csv   per-chunk
    data/raw/training/SPY_1h.csv                                stitched
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config.training_universe import TRAINING_UNIVERSE  # noqa: E402

RAW_DIR = REPO_ROOT / "data" / "raw" / "training"
CHUNK_DIR = RAW_DIR / "chunks"

# Bars are stamped in exchange-local time so a "bar" lines up with a real
# trading session across DST changes. Daily bars stay tz-naive dates.
MARKET_TZ = "America/New_York"

# interval -> (max days back from today, max span of a single request)
# None lookback == no limit. Sourced from Yahoo's documented chart limits;
# the 1h numbers are confirmed against live responses.
INTERVAL_LIMITS: dict[str, tuple[int | None, int]] = {
    "1m": (30, 7),
    "2m": (60, 60),
    "5m": (60, 60),
    "15m": (60, 60),
    "30m": (60, 60),
    "60m": (730, 730),
    "90m": (60, 60),
    "1h": (730, 730),
    "1d": (None, 3650),
    "1wk": (None, 3650),
    "1mo": (None, 3650),
}

OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]

# Yahoo throttles aggressive callers; trades are scarce but so is our IP.
SLEEP_BETWEEN_REQUESTS = 1.5


def is_intraday(interval: str) -> bool:
    return interval.endswith(("m", "h"))


def clamp_start(start: dt.date, interval: str, today: dt.date) -> tuple[dt.date, str | None]:
    """Pull `start` forward to the oldest date Yahoo will actually serve.

    Returns (clamped_start, warning_or_None).
    """
    max_lookback, _ = INTERVAL_LIMITS[interval]
    if max_lookback is None:
        return start, None

    # -1 day of headroom: the boundary is evaluated server-side in UTC and a
    # request landing exactly on it gets rejected.
    earliest = today - dt.timedelta(days=max_lookback - 1)
    if start >= earliest:
        return start, None

    lost_years = (earliest - start).days / 365.25
    warning = (
        f"Yahoo serves at most {max_lookback} days of '{interval}' bars, measured from today.\n"
        f"  requested start : {start}\n"
        f"  earliest served : {earliest}\n"
        f"  discarding      : {(earliest - start).days} days (~{lost_years:.1f} years) of history\n"
        f"  Chunking does NOT get around this -- the cap is on how old the data is,\n"
        f"  not on request size. For a longer span use --interval 1d, or a paid vendor."
    )
    return earliest, warning


def build_chunks(start: dt.date, end: dt.date, chunk_days: int) -> list[tuple[dt.date, dt.date]]:
    """Split [start, end) into consecutive half-open windows."""
    chunks: list[tuple[dt.date, dt.date]] = []
    cursor = start
    while cursor < end:
        stop = min(cursor + dt.timedelta(days=chunk_days), end)
        chunks.append((cursor, stop))
        cursor = stop
    return chunks


def normalize(df: pd.DataFrame, ticker: str, interval: str) -> pd.DataFrame:
    """Flatten yfinance's MultiIndex columns and standardize the time index."""
    if isinstance(df.columns, pd.MultiIndex):
        # Single-ticker downloads still come back as ('Close', 'SPY').
        if ticker in df.columns.get_level_values(-1):
            df = df.xs(ticker, axis=1, level=-1)
        else:
            df.columns = df.columns.droplevel(-1)

    df = df[[c for c in OHLCV_COLUMNS if c in df.columns]].copy()

    idx = pd.to_datetime(df.index)
    if is_intraday(interval):
        idx = idx.tz_localize("UTC") if idx.tz is None else idx
        idx = idx.tz_convert(MARKET_TZ)
    else:
        # Daily bars carry no meaningful intraday timestamp; keep them as
        # naive dates so they never drift a day under tz conversion.
        if idx.tz is not None:
            idx = idx.tz_convert(MARKET_TZ).tz_localize(None)
        idx = idx.normalize()

    df.index = idx
    df.index.name = "timestamp"
    return df[~df.index.duplicated(keep="last")].sort_index()


def fetch_chunk(
    ticker: str,
    start: dt.date,
    end: dt.date,
    interval: str,
    force: bool,
) -> pd.DataFrame | None:
    """Download one window, caching it to its own CSV. None if Yahoo gave nothing."""
    CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    path = CHUNK_DIR / f"{ticker}_{interval}_{start}_{end}.csv"

    if path.exists() and not force:
        print(f"    [cached] {path.name}")
        return read_csv(path, interval)

    df = yf.download(
        ticker,
        start=start.isoformat(),
        end=end.isoformat(),
        interval=interval,
        auto_adjust=False,   # keep the prices that actually printed; see note below
        progress=False,
        threads=False,
    )
    time.sleep(SLEEP_BETWEEN_REQUESTS)

    if df is None or df.empty:
        print(f"    [empty]  {start} -> {end}")
        return None

    df = normalize(df, ticker, interval)
    df.to_csv(path, index_label="timestamp")
    print(f"    [saved]  {path.name}  ({len(df)} rows)")
    return df


def read_csv(path: Path, interval: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col="timestamp", parse_dates=["timestamp"])
    idx = pd.to_datetime(df.index, utc=is_intraday(interval))
    df.index = idx.tz_convert(MARKET_TZ) if is_intraday(interval) else idx
    df.index.name = "timestamp"
    return df


def stitch(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate chunks into one series, dropping boundary duplicates."""
    combined = pd.concat(frames).sort_index()
    before = len(combined)
    combined = combined[~combined.index.duplicated(keep="last")]
    dropped = before - len(combined)
    if dropped:
        print(f"    dropped {dropped} duplicate timestamps at chunk boundaries")
    if not combined.index.is_monotonic_increasing:
        raise RuntimeError("stitched index is not monotonic -- chunk merge is wrong")
    return combined


def fetch_ticker(
    ticker: str,
    start: dt.date,
    end: dt.date,
    interval: str,
    chunk_days: int,
    force: bool,
) -> pd.DataFrame | None:
    chunks = build_chunks(start, end, chunk_days)
    print(f"  {ticker}: {len(chunks)} chunk(s) of up to {chunk_days} days")

    frames = [
        df
        for lo, hi in chunks
        if (df := fetch_chunk(ticker, lo, hi, interval, force)) is not None and not df.empty
    ]
    if not frames:
        print(f"  {ticker}: no data returned for any chunk")
        return None

    combined = stitch(frames)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out = RAW_DIR / f"{ticker}_{interval}.csv"
    combined.to_csv(out, index_label="timestamp")

    sessions = combined.index.normalize().nunique()
    print(
        f"  {ticker}: {len(combined)} rows across {sessions} sessions | "
        f"{combined.index.min()} .. {combined.index.max()}"
    )
    print(f"  {ticker}: -> {out.relative_to(REPO_ROOT)}")
    return combined


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--interval", default="1h", choices=sorted(INTERVAL_LIMITS),
                   help="bar size (default: 1h)")
    p.add_argument("--start", default="2018-01-01",
                   help="requested start date, YYYY-MM-DD (clamped to what Yahoo serves)")
    p.add_argument("--end", default=None,
                   help="exclusive end date, YYYY-MM-DD (default: today)")
    p.add_argument("--tickers", nargs="+", default=None,
                   help="override config/training_universe.py")
    p.add_argument("--chunk-days", type=int, default=730,
                   help="days per request; clamped to the interval's limit (default: 730 = 2 years)")
    p.add_argument("--force", action="store_true",
                   help="re-download chunks even if a cached CSV exists")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    today = dt.date.today()

    start = dt.date.fromisoformat(args.start)
    end = dt.date.fromisoformat(args.end) if args.end else today
    if start >= end:
        print(f"error: start ({start}) must be before end ({end})", file=sys.stderr)
        return 2

    start, warning = clamp_start(start, args.interval, today)
    if warning:
        print(f"\n!! {warning}\n", file=sys.stderr)

    _, max_span = INTERVAL_LIMITS[args.interval]
    chunk_days = max(1, min(args.chunk_days, max_span))

    tickers = args.tickers or TRAINING_UNIVERSE
    print(f"interval={args.interval}  range={start} -> {end}  tickers={tickers}")

    failures = []
    for ticker in tickers:
        if fetch_ticker(ticker, start, end, args.interval, chunk_days, args.force) is None:
            failures.append(ticker)

    if failures:
        print(f"\nno data for: {', '.join(failures)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
