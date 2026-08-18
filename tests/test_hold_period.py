"""period_number_for: the calendar-anchored period-number derivation
TradeSelector's live-mode path depends on (baseline_system_architecture.md
open question 27)."""
from __future__ import annotations

import pytest

from config import hold_period
from config.hold_period import period_number_for


def test_window_start_date_itself_is_period_1():
    assert period_number_for("2026-08-17", window_start_date="2026-08-17") == 1


def test_counts_trading_days_forward():
    # Mon 2026-08-17 .. Fri 2026-08-21 is 5 trading days, no weekend in between.
    assert period_number_for("2026-08-21", window_start_date="2026-08-17") == 5


def test_skips_weekends_not_just_calendar_days():
    # Mon 2026-08-17 -> Mon 2026-08-24 spans a weekend; still period 6
    # (5 weekdays after the start), not 8 (raw calendar days).
    assert period_number_for("2026-08-24", window_start_date="2026-08-17") == 6


def test_raises_when_window_start_date_is_not_set(monkeypatch):
    monkeypatch.setattr(hold_period, "WINDOW_START_DATE", None)
    with pytest.raises(ValueError, match="WINDOW_START_DATE"):
        period_number_for("2026-08-17")


def test_falls_back_to_the_module_constant_when_not_passed_explicitly(monkeypatch):
    monkeypatch.setattr(hold_period, "WINDOW_START_DATE", "2026-08-17")
    assert period_number_for("2026-08-21") == 5
