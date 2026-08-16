"""Build a point-in-time S&P 500 membership table.

Why this exists
---------------
Training on *today's* S&P 500 list applied backward across 2018-2025 is
survivorship bias: names are in today's index partly because they did well,
so backtests inherit knowledge nobody had at the time. This script records
who was ACTUALLY in the index on each historical date.

Source data
-----------
Two public repositories publish the same schema -- `date,tickers` where
`tickers` is a quoted comma-separated roster. Verified 2026-08-16:

    source    rows   coverage                 in-window uniques   quality
    fja       2718   1996-01-02..2026-06-30   685                 clean
    hanshof   3482   1996-01-02..2025-08-23   660                 see below

Default is `fja` on two grounds:

  1. Coverage. hanshof stops at 2025-08-23, four months short of the
     2025-12-31 training window, with no trailing buffer.
  2. Cleanliness. hanshof contains a literal annotation as a symbol,
     "RVTY (Previously PKI)", and spells the same companies two ways in one
     file (both BF-B and BF.B, both BRK-B and BRK.B), which double-counts
     them in a naive union. fja uses dot-form consistently.

`--source hanshof` still works; the normalizer in `tickers.py` repairs both
of those defects, so the union comes out right either way.

IMPORTANT -- as-of semantics
----------------------------
These files are NOT sampled once per trading day. They are dense in 2017-2018
(~115 rows/yr) and sparse afterwards (~14 rows/yr), because a row is written
when membership CHANGES. A row therefore means:

    "this was the roster as of this date, and stayed so until the next row"

The output preserves exactly the observation dates present upstream -- it is
lossless, not interpolated. Downstream point-in-time filtering MUST resolve a
date with a backward as-of join, e.g.

    pd.merge_asof(features.sort_values("date"), membership.sort_values("date"),
                  on="date", by="ticker", direction="backward")

An exact-date equality join will silently match almost nothing. Forward-fill
to a daily grid downstream if you want one; that needs a real trading
calendar, which this script deliberately does not guess at.

Usage
-----
    python src/data_pipeline/fetch_point_in_time_universe.py
    python src/data_pipeline/fetch_point_in_time_universe.py --source hanshof
    python src/data_pipeline/fetch_point_in_time_universe.py --force --no-write-config

Outputs
-------
    data/raw/reference/<source>_sp500_components.csv   cached raw download
    data/processed/sp500_membership.csv                tidy (date, ticker)
    config/training_universe.py                        regenerated union
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
import urllib.request
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data_pipeline.tickers import normalize_tickers  # noqa: E402

REFERENCE_DIR = REPO_ROOT / "data" / "raw" / "reference"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
MEMBERSHIP_CSV = PROCESSED_DIR / "sp500_membership.csv"
UNIVERSE_PY = REPO_ROOT / "config" / "training_universe.py"

SOURCES: dict[str, str] = {
    "fja": (
        "https://raw.githubusercontent.com/fja05680/sp500/master/"
        "S%26P%20500%20Historical%20Components%20%26%20Changes%20(Updated).csv"
    ),
    "hanshof": (
        "https://raw.githubusercontent.com/hanshof/sp500_constituents/main/"
        "sp_500_historical_components.csv"
    ),
}

# Training window, per the project plan.
WINDOW_START = dt.date(2018, 1, 1)
WINDOW_END = dt.date(2025, 12, 31)

# Buffer so a name that joined or left right at a boundary is not clipped.
BUFFER_DAYS = 92  # ~3 months


def download(source: str, force: bool) -> Path:
    """Fetch the raw membership CSV, caching it under data/raw/reference/."""
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    path = REFERENCE_DIR / f"{source}_sp500_components.csv"

    if path.exists() and not force:
        print(f"  [cached] {path.relative_to(REPO_ROOT)} ({path.stat().st_size:,} bytes)")
        return path

    url = SOURCES[source]
    print(f"  downloading {source} ...")
    with urllib.request.urlopen(url, timeout=120) as resp:
        payload = resp.read()
    path.write_bytes(payload)
    print(f"  [saved] {path.relative_to(REPO_ROOT)} ({len(payload):,} bytes)")
    return path


def load_raw(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = {"date", "tickers"} - set(df.columns)
    if missing:
        raise ValueError(f"{path.name}: expected columns date,tickers -- missing {missing}")

    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.dropna(subset=["date", "tickers"])
    # A duplicated date would make the as-of join ambiguous; keep the last.
    df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    return df.reset_index(drop=True)


def select_window(df: pd.DataFrame, start: dt.date, end: dt.date) -> pd.DataFrame:
    """Rows inside the buffered window, plus the anchor row preceding it.

    The anchor matters: without a row at or before the window start, an as-of
    join cannot resolve membership on the earliest training dates.
    """
    lo = pd.Timestamp(start) - pd.Timedelta(days=BUFFER_DAYS)
    hi = pd.Timestamp(end) + pd.Timedelta(days=BUFFER_DAYS)

    inside = df[(df["date"] >= lo) & (df["date"] <= hi)]
    before = df[df["date"] < lo]

    if before.empty:
        if inside.empty:
            raise ValueError("no membership rows in or before the requested window")
        print(f"  warning: no row before {lo.date()}; earliest dates may be unresolvable")
        selected = inside
    else:
        selected = pd.concat([before.tail(1), inside])

    return selected.sort_values("date").reset_index(drop=True)


def to_long(df: pd.DataFrame) -> pd.DataFrame:
    """Explode each roster row into one row per (date, ticker)."""
    records: list[tuple[pd.Timestamp, str]] = []
    for date, roster in zip(df["date"], df["tickers"]):
        for ticker in normalize_tickers(str(roster).split(",")):
            records.append((date, ticker))

    long = pd.DataFrame.from_records(records, columns=["date", "ticker"])
    long = long.drop_duplicates().sort_values(["date", "ticker"]).reset_index(drop=True)
    return long


def write_universe(tickers: list[str], source: str, long: pd.DataFrame) -> None:
    """Regenerate config/training_universe.py from the membership table."""
    body = "\n".join(f'    "{t}",' for t in tickers)
    generated = dt.date.today().isoformat()
    span = f"{long['date'].min().date()} .. {long['date'].max().date()}"

    UNIVERSE_PY.write_text(
        f'''"""Broad training universe -- GENERATED FILE, do not hand-edit.

Regenerate with:
    python src/data_pipeline/fetch_point_in_time_universe.py --source {source}

Every ticker that was an S&P 500 member at ANY point in {span}
({len(tickers)} symbols), taken from data/processed/sp500_membership.csv.

This is deliberately a SUPERSET, not the roster for any single date. Holding
every name that was ever a member is what avoids survivorship bias; the
per-date filtering happens at feature-build time via an as-of join against
sp500_membership.csv. Do not use this list as if it were point-in-time.

Used ONLY for training. Live scoring reads config/stock_list.py (Wharton's
approved list). Generated {generated}.
"""

TRAINING_UNIVERSE = [
{body}
]
''',
        encoding="utf-8",
    )
    print(f"  wrote {UNIVERSE_PY.relative_to(REPO_ROOT)} ({len(tickers)} tickers)")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--source", default="fja", choices=sorted(SOURCES),
                   help="membership data source (default: fja -- better coverage and cleaner)")
    p.add_argument("--start", default=WINDOW_START.isoformat(), help="window start YYYY-MM-DD")
    p.add_argument("--end", default=WINDOW_END.isoformat(), help="window end YYYY-MM-DD")
    p.add_argument("--force", action="store_true", help="re-download even if cached")
    p.add_argument("--no-write-config", action="store_true",
                   help="build the membership table but leave training_universe.py alone")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    start = dt.date.fromisoformat(args.start)
    end = dt.date.fromisoformat(args.end)
    if start >= end:
        print(f"error: start ({start}) must be before end ({end})", file=sys.stderr)
        return 2

    print(f"source={args.source}  window={start} .. {end}  buffer={BUFFER_DAYS}d")

    raw_path = download(args.source, args.force)
    raw = load_raw(raw_path)
    print(f"  raw: {len(raw)} rows | {raw['date'].min().date()} .. {raw['date'].max().date()}")

    if raw["date"].max() < pd.Timestamp(end):
        print(
            f"  warning: source ends {raw['date'].max().date()}, before window end {end}. "
            "Membership after that date is unknown."
        )

    windowed = select_window(raw, start, end)
    print(f"  windowed: {len(windowed)} observation dates | "
          f"{windowed['date'].min().date()} .. {windowed['date'].max().date()}")

    long = to_long(windowed)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    long.to_csv(MEMBERSHIP_CSV, index=False, date_format="%Y-%m-%d")
    print(f"  wrote {MEMBERSHIP_CSV.relative_to(REPO_ROOT)}: {len(long):,} (date, ticker) rows")

    roster_sizes = long.groupby("date")["ticker"].size()
    print(f"  roster size per date: min={roster_sizes.min()} "
          f"median={int(roster_sizes.median())} max={roster_sizes.max()}")

    universe = sorted(long["ticker"].unique())
    print(f"\n  TRAINING_UNIVERSE would contain {len(universe)} unique tickers")
    if not 100 <= len(universe) <= 900:
        print(f"  error: {len(universe)} tickers is outside the sane range (100-900). "
              "Not writing config.", file=sys.stderr)
        return 1
    print("  sanity check passed (expected low hundreds to ~700)")

    if args.no_write_config:
        print("  --no-write-config given; leaving config/training_universe.py unchanged")
    else:
        write_universe(universe, args.source, long)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
