"""monitor_score_live.py (B4): the daily log-and-diff wrapper around
score_live.py. score_live.py itself is never invoked here for real --
_run_score_live is monkeypatched, same convention as the rest of this
project's real-data scripts (verified by running the real thing, not
mocked; only the pure wiring is unit-tested).
"""
from __future__ import annotations

from datetime import datetime

import pytest

from scripts import monitor_score_live as m


@pytest.fixture(autouse=True)
def _isolated_log_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "LOG_DIR", tmp_path)
    return tmp_path


# --- _prior_log ------------------------------------------------------------

def test_prior_log_returns_none_with_no_logs(_isolated_log_dir):
    assert m._prior_log("2026-08-19") is None


def test_prior_log_picks_the_most_recent_date_strictly_before(_isolated_log_dir):
    for d in ("2026-08-15", "2026-08-17", "2026-08-18"):
        (_isolated_log_dir / f"{d}.log").write_text("x")
    assert m._prior_log("2026-08-19").stem == "2026-08-18"


def test_prior_log_never_returns_todays_own_file(_isolated_log_dir):
    """Deliberate failing-case: if today's own log already exists (e.g. a
    re-run) it must not be returned as its own 'prior' -- a diff of a file
    against itself would trivially always read 'no change'."""
    (_isolated_log_dir / "2026-08-19.log").write_text("today")
    (_isolated_log_dir / "2026-08-18.log").write_text("yesterday")
    assert m._prior_log("2026-08-19").stem == "2026-08-18"


# --- main(): weekend skip ----------------------------------------------------

def test_weekend_is_skipped_without_calling_score_live(monkeypatch, capsys):
    """Deliberate failing-case: _run_score_live is a mock that raises if
    called at all -- proves the weekend skip is a real short-circuit, not
    just quiet output on a path that still runs the subprocess."""
    saturday = datetime(2026, 8, 15, 12, 0, 0)  # a real Saturday

    class _FakeDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return saturday

    monkeypatch.setattr(m, "datetime", _FakeDatetime)
    monkeypatch.setattr(m, "_run_score_live", lambda: (_ for _ in ()).throw(AssertionError("should not run on a weekend")))

    assert m.main([]) == 0
    assert "weekend" in capsys.readouterr().out.lower()


def test_weekend_force_flag_still_runs(monkeypatch):
    saturday = datetime(2026, 8, 15, 12, 0, 0)

    class _FakeDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return saturday

    monkeypatch.setattr(m, "datetime", _FakeDatetime)
    called = []
    monkeypatch.setattr(m, "_run_score_live", lambda: called.append(True) or (0, "ok"))

    assert m.main(["--force"]) == 0
    assert called == [True]


# --- main(): logging + diffing ----------------------------------------------

def test_identical_output_two_days_reports_no_change(_isolated_log_dir, monkeypatch, capsys):
    monday = datetime(2026, 8, 17, 12, 0, 0)

    class _FakeDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return monday

    monkeypatch.setattr(m, "datetime", _FakeDatetime)
    monkeypatch.setattr(m, "_run_score_live", lambda: (1, "same failure every day"))
    assert m.main(["--force"]) == 0
    capsys.readouterr()

    tuesday = datetime(2026, 8, 18, 12, 0, 0)

    class _FakeDatetime2(datetime):
        @classmethod
        def now(cls, tz=None):
            return tuesday

    monkeypatch.setattr(m, "datetime", _FakeDatetime2)
    monkeypatch.setattr(m, "_run_score_live", lambda: (1, "same failure every day"))
    assert m.main(["--force"]) == 0
    out = capsys.readouterr().out
    assert "no change" in out.lower()


def test_changed_output_is_flagged_not_silently_absorbed(_isolated_log_dir, monkeypatch, capsys):
    """Deliberate failing-case: a real change (e.g. preflight starts
    passing) must produce visible diff output, not just a quiet log write."""
    monday = datetime(2026, 8, 17, 12, 0, 0)

    class _FakeDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return monday

    monkeypatch.setattr(m, "datetime", _FakeDatetime)
    monkeypatch.setattr(m, "_run_score_live", lambda: (1, "stock_list.py is empty"))
    assert m.main(["--force"]) == 0
    capsys.readouterr()

    tuesday = datetime(2026, 8, 18, 12, 0, 0)

    class _FakeDatetime2(datetime):
        @classmethod
        def now(cls, tz=None):
            return tuesday

    monkeypatch.setattr(m, "datetime", _FakeDatetime2)
    monkeypatch.setattr(m, "_run_score_live", lambda: (0, "3 recommendations: AAPL, MSFT, NVDA"))
    assert m.main(["--force"]) == 0
    out = capsys.readouterr().out
    assert "changed" in out.lower()
    assert "AAPL" in out
