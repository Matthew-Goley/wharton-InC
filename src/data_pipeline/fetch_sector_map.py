"""Build config/sector_map.py -- ticker -> sector, the single source of
truth CLAUDE.md already names but that didn't exist yet.

Source: yfinance's `.info['sector']` per ticker (the same library already
used for price data, so no new dependency). Deliberately coarse (start
coarse per the plan): yfinance's own ~11 sector labels, normalized to GICS
naming, not GICS sub-industries -- more data per sector expert, less
specialization, matches "start coarse" until there's a reason to split
further.

yfinance's sector field is not fully reliable -- spot-checked and found
PARA (Paramount, a media/entertainment company) tagged "Technology", which
is wrong. Known-bad entries are corrected in MANUAL_OVERRIDES below,
checked *after* fetching so a re-run never silently loses them. This is a
coarse, "good enough to unblock sector experts" pass, not a audited GICS
mapping -- flagged, not solved.

Caches every successful lookup to data/raw/reference/sector_info_raw.csv
so a re-run only re-fetches tickers that failed or are new to the universe.

Usage:
    python -m src.data_pipeline.fetch_sector_map
    python -m src.data_pipeline.fetch_sector_map --force   # ignore cache
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

CACHE_PATH = REPO_ROOT / "data" / "raw" / "reference" / "sector_info_raw.csv"
OUTPUT_PATH = REPO_ROOT / "config" / "sector_map.py"

#: yfinance's own sector strings, normalized to GICS naming so this reads
#: the same as everywhere else in finance. Coarse by design -- this is the
#: full set of buckets, no sub-industry split.
SECTOR_NORMALIZATION: dict[str, str] = {
    "Technology": "Information Technology",
    "Financial Services": "Financials",
    "Healthcare": "Health Care",
    "Consumer Cyclical": "Consumer Discretionary",
    "Consumer Defensive": "Consumer Staples",
    "Basic Materials": "Materials",
    "Energy": "Energy",
    "Real Estate": "Real Estate",
    "Utilities": "Utilities",
    "Industrials": "Industrials",
    "Communication Services": "Communication Services",
}

#: Corrections for tickers where yfinance's sector is known wrong, checked
#: after normalization so an override always wins. Add to this list as
#: spot-checks turn up more -- don't hand-edit sector_map.py directly, it's
#: generated.
MANUAL_OVERRIDES: dict[str, str] = {
    "PARA": "Communication Services",  # media/broadcasting (CBS, Paramount
    # Pictures, MTV, Nickelodeon), not software -- yfinance tags it
    # "Technology" / "Software - Application", verified wrong 2026-08-17.
}

SLEEP_BETWEEN_REQUESTS = 0.4


def load_cache() -> dict[str, str]:
    if not CACHE_PATH.exists():
        return {}
    df = pd.read_csv(CACHE_PATH)
    return dict(zip(df["ticker"], df["sector_raw"], strict=True))


def save_cache(raw: dict[str, str]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"ticker": list(raw.keys()), "sector_raw": list(raw.values())}).to_csv(CACHE_PATH, index=False)


def fetch_all(universe: list[str], force: bool) -> tuple[dict[str, str], list[str]]:
    raw = {} if force else load_cache()
    failures: list[str] = []
    to_fetch = [t for t in universe if t not in raw]
    print(f"{len(raw)} cached, {len(to_fetch)} to fetch")

    for i, ticker in enumerate(to_fetch):
        try:
            info = yf.Ticker(ticker).info
            sector = info.get("sector")
            if sector:
                raw[ticker] = sector
            else:
                failures.append(ticker)
                print(f"  [{i+1}/{len(to_fetch)}] {ticker}: no sector field")
        except Exception as exc:
            failures.append(ticker)
            print(f"  [{i+1}/{len(to_fetch)}] {ticker}: FAILED ({exc})")
        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{len(to_fetch)}] ...{len(raw)} fetched so far, saving cache")
            save_cache(raw)
        time.sleep(SLEEP_BETWEEN_REQUESTS)

    save_cache(raw)
    return raw, failures


def write_sector_map(universe: list[str], raw: dict[str, str]) -> dict[str, str]:
    resolved: dict[str, str] = {}
    unmapped_raw: set[str] = set()
    for ticker in universe:
        if ticker in MANUAL_OVERRIDES:
            resolved[ticker] = MANUAL_OVERRIDES[ticker]
            continue
        sector_raw = raw.get(ticker)
        if sector_raw is None:
            continue
        sector = SECTOR_NORMALIZATION.get(sector_raw)
        if sector is None:
            unmapped_raw.add(sector_raw)
            continue
        resolved[ticker] = sector

    if unmapped_raw:
        print(f"WARNING: unnormalized raw sector strings, not in SECTOR_NORMALIZATION: {sorted(unmapped_raw)}")

    counts = pd.Series(list(resolved.values())).value_counts()
    print("\nSector counts:")
    print(counts.to_string())

    body = ",\n".join(f'    "{t}": "{s}"' for t, s in sorted(resolved.items()))
    generated = pd.Timestamp.today().date().isoformat()
    OUTPUT_PATH.write_text(
        f'''"""Ticker -> sector -- GENERATED FILE, do not hand-edit.

Regenerate with:
    python -m src.data_pipeline.fetch_sector_map

Source: yfinance `.info['sector']`, normalized to GICS naming, with manual
corrections for known-bad entries (see MANUAL_OVERRIDES in the fetch
script). Coarse by design ({len(counts)} sectors, not GICS sub-industries)
-- more data per sector expert, less specialization; start here, split
further only if a sector expert's own data volume becomes the binding
constraint. Not a point-in-time mapping: a ticker's *current* sector is
applied across its whole price history, same simplification the broad
model already makes implicitly by not having a sector-history mapping.

{len(resolved)} of {len(universe)} universe tickers resolved. Generated {generated}.
"""

SECTOR_MAP: dict[str, str] = {{
{body}
}}
''',
        encoding="utf-8",
    )
    print(f"\nWrote {OUTPUT_PATH.relative_to(REPO_ROOT)}: {len(resolved)}/{len(universe)} tickers")
    return resolved


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--force", action="store_true", help="ignore the cache, re-fetch every ticker")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    raw, failures = fetch_all(TRAINING_UNIVERSE, args.force)
    write_sector_map(TRAINING_UNIVERSE, raw)
    if failures:
        print(f"\n{len(failures)} tickers had no sector data: {failures[:20]}{'...' if len(failures) > 20 else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
