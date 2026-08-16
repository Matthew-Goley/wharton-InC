"""Ticker symbol normalization.

Shared by the point-in-time universe builder and (later) live scoring, so a
symbol resolves to exactly one spelling everywhere in the system. Index
membership sources and Wharton's approved list do not agree on share-class
punctuation; Yahoo Finance wants hyphens.

    BRK.B  -> BRK-B
    BF.B   -> BF-B
"""

from __future__ import annotations


def normalize_ticker(raw: str) -> str | None:
    """Coerce one symbol into Yahoo Finance's format.

    Returns None for tokens that are not usable symbols (blanks, NaN
    leftovers), so callers can filter them out rather than fetching junk.
    """
    if raw is None:
        return None

    ticker = str(raw).strip().upper()
    if not ticker or ticker in {"NAN", "NONE", "-"}:
        return None

    # Some sources annotate renames inline, e.g. "RVTY (Previously PKI)".
    # Keep the live symbol, drop the commentary.
    if "(" in ticker:
        ticker = ticker.split("(", 1)[0].strip()

    # Yahoo spells share classes with a hyphen: BRK.B -> BRK-B.
    ticker = ticker.replace(".", "-")

    # Collapse any interior whitespace that survived a ragged split.
    ticker = "".join(ticker.split())

    return ticker or None


def normalize_tickers(raw_tickers) -> list[str]:
    """Normalize an iterable of symbols, dropping junk and preserving order.

    Deduplicates, because sources sometimes spell one company two ways within
    the same file (BF-B and BF.B both appear in hanshof's data).
    """
    seen: dict[str, None] = {}
    for raw in raw_tickers:
        ticker = normalize_ticker(raw)
        if ticker is not None:
            seen.setdefault(ticker, None)
    return list(seen)
