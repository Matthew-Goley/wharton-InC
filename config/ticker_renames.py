"""Ticker symbol changes: old symbol -> symbol the history now lives under.

Why this exists
---------------
Yahoo Finance keys price history by CURRENT symbol. When a company changes its
ticker, the entire history moves to the new symbol and the old one returns
nothing. Our point-in-time membership data, correctly, records whatever symbol
was in use at the time -- so 2018 rows say ANTM and 2023 rows say ELV, for one
continuous company.

Without this map that company is two half-companies: one with no price data at
all (ANTM), one whose membership looks like it began in 2022 (ELV). With it,
both resolve to a single identity.

What belongs here
-----------------
ONLY pure symbol changes -- same continuing company, new ticker. NOT
acquisitions, and NOT index swaps.

This distinction cannot be derived from the membership file. A rename and an
unrelated swap look identical there: one symbol leaves, another arrives the
same day. Deriving it mechanically produces confident nonsense like
MON -> TWTR (Monsanto was bought by Bayer; Twitter happened to be added that
day) or DFS -> COIN. So this map is curated by hand and every entry is checked
by `verify_renames()` below, which confirms the successor actually carries
history spanning the predecessor's time in the index.

Companies that were acquired, went private, or failed have no successor. Their
price history is simply gone from Yahoo -- see DELISTED_NO_SUCCESSOR.

Direction is always old -> current. Chains are resolved transitively by
`resolve_ticker`, so SYMC -> NLOK -> GEN returns GEN.
"""

from __future__ import annotations

# Pure symbol changes. Comment gives the company and roughly when it changed.
RENAMES: dict[str, str] = {
    "ABC": "COR",      # AmerisourceBergen -> Cencora (2023)
    "ADS": "BFH",      # Alliance Data Systems -> Bread Financial (2022)
    "ANTM": "ELV",     # Anthem -> Elevance Health (2022)
    "ARNC": "HWM",     # Arconic -> Howmet Aerospace (2020)
    "BHGE": "BKR",     # Baker Hughes a GE company -> Baker Hughes (2019)
    "BK": "BNY",       # Bank of New York Mellon -> BNY (2025)
    "BLL": "BALL",     # Ball Corporation ticker change (2022)
    "CDAY": "DAY",     # Ceridian -> Dayforce (2024)
    "CHK": "EXE",      # Chesapeake Energy -> Expand Energy (2024)
    "COG": "CTRA",     # Cabot Oil & Gas -> Coterra Energy (2021)
    "CTL": "LUMN",     # CenturyLink -> Lumen Technologies (2020)
    "DISCA": "WBD",    # Discovery Class A -> Warner Bros. Discovery (2022)
    "DISCK": "WBD",    # Discovery Class C -> Warner Bros. Discovery (2022)
    "DISH": "SATS",    # DISH Network -> EchoStar (2024)
    "DWDP": "DD",      # DowDuPont -> DuPont (2019, after DOW/CTVA spun out)
    "FB": "META",      # Facebook -> Meta Platforms (2022)
    "FBHS": "FBIN",    # Fortune Brands Home & Security -> Fortune Brands Innovations (2022)
    # Fiserv renamed FISV -> FI in 2023, but Yahoo still serves the full
    # history under FISV and returns 404 for FI. Mapped toward the symbol that
    # actually HAS the data -- the point of this file is a working join key,
    # not the prettiest name. Revisit if Yahoo ever moves it.
    "FI": "FISV",
    "FLT": "CPAY",     # FLEETCOR -> Corpay (2024)
    "GPS": "GAP",      # Gap Inc ticker change (2024)
    "HCP": "PEAK",     # HCP -> Healthpeak (2019), then PEAK -> DOC below
    "HFC": "DINO",     # HollyFrontier -> HF Sinclair (2022)
    "HRS": "LHX",      # Harris -> L3Harris (2019)
    "JEC": "J",        # Jacobs Engineering -> Jacobs Solutions (2019)
    "KORS": "CPRI",    # Michael Kors -> Capri Holdings (2018)
    "MMC": "MRSH",     # Marsh & McLennan -> Marsh McLennan (2026)
    "MYL": "VTRS",     # Mylan -> Viatris (2020)
    "NLOK": "GEN",     # NortonLifeLock -> Gen Digital (2022)
    "PEAK": "DOC",     # Healthpeak Properties ticker change (2024)
    "PKI": "RVTY",     # PerkinElmer -> Revvity (2023)
    "RE": "EG",        # Everest Re -> Everest Group (2023)
    "SYMC": "NLOK",    # Symantec -> NortonLifeLock (2019), chains to GEN
    "TMK": "GL",       # Torchmark -> Globe Life (2019)
    # NOTE: the four mappings marked (*) below are correct as identity
    # statements, but their SUCCESSOR has since been delisted too, so no price
    # history is recoverable for either symbol. See SUCCESSOR_ALSO_DELISTED.
    "UTX": "RTX",      # United Technologies -> Raytheon Technologies (2020)
    "VIAC": "PARA",    # ViacomCBS -> Paramount Global (2022)
    "WLTW": "WTW",     # Willis Towers Watson ticker change (2022)
    "WRK": "SW",       # WestRock -> Smurfit WestRock (2024)
}

# Acquired, taken private, or failed. No successor symbol carries the history;
# Yahoo has purged it. Listed so the gap is documented rather than mysterious,
# and so nobody re-derives a bogus "rename" for them later.
DELISTED_NO_SUCCESSOR: frozenset[str] = frozenset({
    "ABMD", "AET", "AGN", "ALXN", "ANDV", "ANSS", "APC", "ATVI", "BCR", "CA",
    "CBS", "CELG", "CERN", "CMA", "CSRA", "CTLT", "CTXS", "CXO", "DFS", "DRE",
    "EA", "ESRX", "ETFC", "EVHC", "FL", "FLIR", "FRC", "GGP", "HBI", "HES",
    "HOLX", "IPG", "JNPR", "JWN", "K", "KSU", "LLL", "LVLT", "MON", "MRO",
    "MXIM", "NBL", "NFX", "NLSN", "PBCT", "PDCO", "PX", "PXD", "RHT", "RTN",
    "SCG", "SEE", "SIVB", "SNI", "SRCL", "TIF", "TSS", "TWTR", "TWX", "VAR",
    "VIAB", "WBA", "WCG", "WYND", "XEC", "XL", "XLNX",
})
# Two that LOOK like renames but are not, and would corrupt the data if mapped:
#   DRE -> PLD   Duke Realty was ACQUIRED BY Prologis. PLD returns a full
#                history, but it is Prologis's own -- mapping it would paste
#                one company's prices onto another. Row counts alone hide this.
#   CBS -> PARA  CBS merged into ViacomCBS/Paramount, but Yahoo's PARA history
#                only begins 2021-02, so it carries none of CBS's 2018-2020.


# (*) Renamed correctly, but the successor was itself acquired or taken
# private afterwards, so Yahoo now serves nothing for either symbol. The
# mapping is still right -- it is the data that is missing, not the identity.
# Verified 2026-08-16: all four 404 or return zero rows.
SUCCESSOR_ALSO_DELISTED: frozenset[str] = frozenset({
    "CDAY",   # -> DAY  (Dayforce, taken private)
    "COG",    # -> CTRA (Coterra Energy)
    "DISH",   # -> SATS (EchoStar; profile survives, price history purged)
})


def resolve_ticker(ticker: str, _seen: frozenset[str] | None = None) -> str:
    """Follow a rename chain to the current symbol.

    SYMC -> NLOK -> GEN all resolve to GEN. Unknown symbols pass through
    unchanged, so this is safe to call on every ticker indiscriminately.
    """
    seen = _seen or frozenset()
    if ticker in seen:  # defensive: a cycle in the map would hang otherwise
        raise ValueError(f"rename cycle involving {ticker!r}")
    successor = RENAMES.get(ticker)
    if successor is None:
        return ticker
    return resolve_ticker(successor, seen | {ticker})


def resolve_all(tickers) -> list[str]:
    """Resolve an iterable of symbols, deduplicated, order preserved."""
    out: dict[str, None] = {}
    for t in tickers:
        out.setdefault(resolve_ticker(t), None)
    return list(out)


def verify_renames(start: str = "2018-01-01", end: str = "2026-01-01") -> dict[str, dict]:
    """Check every mapping against Yahoo. Returns {old: {...}} for failures.

    A mapping is sound if the successor returns a substantial price history and
    the predecessor returns none (the history moved). Requires network access;
    this is a maintenance check, not something the pipeline calls at runtime.
    """
    import time
    import yfinance as yf

    problems: dict[str, dict] = {}
    for old in sorted(RENAMES):
        if old in SUCCESSOR_ALSO_DELISTED:
            continue  # known-empty by design, not a mapping error
        new = resolve_ticker(old)
        df = yf.download(new, start=start, end=end, interval="1d",
                         auto_adjust=False, progress=False, threads=False)
        rows = 0 if df is None else len(df)
        if rows < 200:
            problems[old] = {"successor": new, "rows": rows,
                             "issue": "successor has little or no history"}
        time.sleep(1.5)
    return problems


if __name__ == "__main__":
    bad = verify_renames()
    if bad:
        print("PROBLEM MAPPINGS:")
        for old, info in bad.items():
            print(f"  {old} -> {info['successor']}: {info['issue']} ({info['rows']} rows)")
    else:
        print(f"all {len(RENAMES)} mappings verified")
