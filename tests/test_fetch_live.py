"""fetch_live's pure logic (no network, no filesystem beyond tmp_path):
the incremental-vs-backfill window decision, the cached/fresh merge, and
the empty-approved-list guard. The actual yfinance call (fetch_one's
middle section) isn't unit-tested here -- same convention as every other
fetch_*.py script in this project (fetch_training, fetch_sector_map,
fetch_earnings_dates), verified by running them against real data, not
mocked.
"""
from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from src.data_pipeline import fetch_live
from src.data_pipeline.fetch_live import INITIAL_LOOKBACK_CALENDAR_DAYS, _existing_range, _fetch_window, _merge


def _frame(dates: list[str]) -> pd.DataFrame:
    idx = pd.DatetimeIndex([pd.Timestamp(d) for d in dates], name="timestamp")
    return pd.DataFrame({"Close": range(len(dates))}, index=idx)


# --- _fetch_window: the incremental-vs-backfill decision --------------------

def test_fetch_window_is_a_full_backfill_when_nothing_is_cached():
    today = dt.date(2026, 8, 18)
    start = _fetch_window(None, today)
    assert start == today - dt.timedelta(days=INITIAL_LOOKBACK_CALENDAR_DAYS)


def test_fetch_window_is_incremental_from_the_day_after_the_last_cached_date():
    today = dt.date(2026, 8, 18)
    existing = (pd.Timestamp("2026-08-10"), _frame(["2026-08-10"]))
    start = _fetch_window(existing, today)
    assert start == dt.date(2026, 8, 11)


def test_fetch_window_is_none_when_already_up_to_date():
    """Deliberate failing-case check: an off-by-one here would either
    re-fetch today redundantly every run or skip a real new day -- this is
    exactly the kind of pacing/off-by-one class of bug this project has
    hit before elsewhere, worth a direct check rather than trusting the
    arithmetic by eye."""
    today = dt.date(2026, 8, 18)
    existing = (pd.Timestamp("2026-08-18"), _frame(["2026-08-18"]))
    assert _fetch_window(existing, today) is None

    # One day further behind must NOT also read as "up to date".
    existing_stale = (pd.Timestamp("2026-08-17"), _frame(["2026-08-17"]))
    assert _fetch_window(existing_stale, today) == dt.date(2026, 8, 18)


# --- _merge: fresh must win on any overlapping date -------------------------

def test_merge_with_no_cache_returns_fresh_unchanged():
    fresh = _frame(["2026-08-17", "2026-08-18"])
    out = _merge(None, fresh)
    pd.testing.assert_frame_equal(out, fresh)


def test_merge_appends_non_overlapping_fresh_rows():
    cached = _frame(["2026-08-10", "2026-08-11"])
    fresh = _frame(["2026-08-12", "2026-08-13"])
    out = _merge(cached, fresh)
    assert list(out.index.strftime("%Y-%m-%d")) == ["2026-08-10", "2026-08-11", "2026-08-12", "2026-08-13"]


def test_merge_fresh_value_wins_on_an_overlapping_date():
    """Deliberate failing-case check on the design decision itself, not
    just the plumbing: a live re-fetch of a date already on file (e.g. an
    intraday-fetched bar later corrected by the final closing print) must
    REPLACE the cached value, not be silently shadowed by it. Distinct
    Close values on the same date make a same-value merge pass by
    accident impossible."""
    cached = _frame(["2026-08-17", "2026-08-18"])  # Close = 0, 1
    fresh = pd.DataFrame({"Close": [999.0]}, index=pd.DatetimeIndex([pd.Timestamp("2026-08-18")], name="timestamp"))
    out = _merge(cached, fresh)
    assert len(out) == 2  # not 3 -- the overlap collapses, doesn't duplicate
    assert out.loc[pd.Timestamp("2026-08-18"), "Close"] == 999.0


# --- _existing_range ---------------------------------------------------------

def test_existing_range_is_none_when_the_file_does_not_exist(tmp_path):
    assert _existing_range(tmp_path / "NOPE_1d.csv") is None


def test_existing_range_reads_the_last_cached_date(tmp_path):
    path = tmp_path / "AAPL_1d.csv"
    _frame(["2026-08-10", "2026-08-12", "2026-08-11"]).to_csv(path, index_label="timestamp")
    last_date, cached = _existing_range(path)
    assert last_date == pd.Timestamp("2026-08-12")  # max, not last-written
    assert len(cached) == 3


# --- main(): the empty-approved-list guard -----------------------------------

def test_main_fails_loudly_when_the_approved_list_is_empty(monkeypatch):
    """Explicitly monkeypatched rather than relying on config/stock_list.py
    actually being empty right now -- this must keep testing the guard
    even after the team populates the real list, not silently stop
    meaning anything once that happens."""
    monkeypatch.setattr(fetch_live, "STOCK_LIST", [])
    assert fetch_live.main([]) == 1


def test_main_uses_explicit_tickers_override_even_with_an_empty_stock_list(monkeypatch):
    """--tickers must be able to bypass an empty config/stock_list.py (e.g.
    for a manual one-off test fetch) -- confirm the guard checks the
    RESOLVED ticker list, not STOCK_LIST directly, without actually
    hitting the network (fetch_one is stubbed)."""
    monkeypatch.setattr(fetch_live, "STOCK_LIST", [])
    called = []
    monkeypatch.setattr(fetch_live, "fetch_one", lambda ticker, today, force: called.append(ticker) or "up to date")
    assert fetch_live.main(["--tickers", "AAPL"]) == 0
    assert called == ["AAPL"]
