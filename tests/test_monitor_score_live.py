"""monitor_score_live.py (B4): the daily log-and-diff wrapper around
score_live.py. score_live.py itself is never invoked here for real --
_run_score_live is monkeypatched, same convention as the rest of this
project's real-data scripts (verified by running the real thing, not
mocked; only the pure wiring is unit-tested).
"""
from __future__ import annotations

from datetime import date, datetime

import pytest

from scripts import monitor_score_live as m


@pytest.fixture(autouse=True)
def _isolated_paths(tmp_path, monkeypatch):
    """Every protected/observed path this script touches gets its own
    tmp_path fixture -- LOG_DIR, and the two non-mutation-guard targets
    (PORTFOLIO_STATE_PATH, CONFIG_DIR) -- so no test can ever read or
    write anything under the real project tree."""
    log_dir = tmp_path / "logs"
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    portfolio_path = tmp_path / "portfolio_state.json"
    portfolio_path.write_text('{"trade_log": []}')

    monkeypatch.setattr(m, "LOG_DIR", log_dir)
    monkeypatch.setattr(m, "CONFIG_DIR", config_dir)
    monkeypatch.setattr(m, "PORTFOLIO_STATE_PATH", portfolio_path)
    return {"log_dir": log_dir, "config_dir": config_dir, "portfolio_path": portfolio_path}


def _frozen(when: datetime):
    class _FakeDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return when
    return _FakeDatetime


# --- _prior_log --------------------------------------------------------------

def test_prior_log_returns_none_with_no_logs(_isolated_paths):
    assert m._prior_log("2026-08-19") is None


def test_prior_log_picks_the_most_recent_date_strictly_before(_isolated_paths):
    log_dir = _isolated_paths["log_dir"]
    log_dir.mkdir(exist_ok=True)
    for d in ("2026-08-15", "2026-08-17", "2026-08-18"):
        (log_dir / f"{d}.log").write_text("x")
    assert m._prior_log("2026-08-19").stem == "2026-08-18"


def test_prior_log_never_returns_todays_own_file(_isolated_paths):
    """Deliberate failing-case: if today's own log already exists (e.g. a
    re-run) it must not be returned as its own 'prior' -- a diff of a file
    against itself would trivially always read 'no change'."""
    log_dir = _isolated_paths["log_dir"]
    log_dir.mkdir(exist_ok=True)
    (log_dir / "2026-08-19.log").write_text("today")
    (log_dir / "2026-08-18.log").write_text("yesterday")
    assert m._prior_log("2026-08-19").stem == "2026-08-18"


# --- trading-day detection ----------------------------------------------------

def test_weekday_non_holiday_is_a_trading_day(_isolated_paths):
    assert m._is_trading_day(date(2026, 8, 17)) is True  # a real Monday


def test_weekend_is_not_a_trading_day(_isolated_paths):
    assert m._is_trading_day(date(2026, 8, 15)) is False  # a real Saturday


def test_nyse_holiday_is_not_a_trading_day(_isolated_paths):
    assert m._is_trading_day(date(2026, 9, 7)) is False  # Labor Day 2026, a Monday


def test_a_weekday_that_is_not_actually_a_holiday_is_not_misflagged(_isolated_paths):
    """Deliberate failing-case: the day right after a holiday must NOT be
    swept up as a holiday too -- proves the check is exact-date, not a
    date-range or fuzzy match."""
    assert m._is_trading_day(date(2026, 9, 8)) is True  # the Tuesday after Labor Day


# --- main(): non-trading-day skip ---------------------------------------------

def test_weekend_is_skipped_without_calling_score_live(monkeypatch, capsys, _isolated_paths):
    """Deliberate failing-case: _run_score_live is a mock that raises if
    called at all -- proves the skip is a real short-circuit."""
    monkeypatch.setattr(m, "datetime", _frozen(datetime(2026, 8, 15, 12, 0, 0)))  # Saturday
    monkeypatch.setattr(m, "_run_score_live", lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not run")))
    assert m.main([]) == 0
    assert "weekend" in capsys.readouterr().out.lower()


def test_holiday_is_skipped_and_named_as_a_holiday_not_a_weekend(monkeypatch, capsys, _isolated_paths):
    monkeypatch.setattr(m, "datetime", _frozen(datetime(2026, 9, 7, 12, 0, 0)))  # Labor Day, a Monday
    monkeypatch.setattr(m, "_run_score_live", lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not run")))
    assert m.main([]) == 0
    out = capsys.readouterr().out.lower()
    assert "holiday" in out
    assert "weekend" not in out.split("holiday")[0][-20:]  # not misreported as a weekend


def test_force_flag_still_runs_on_a_non_trading_day(monkeypatch, _isolated_paths):
    monkeypatch.setattr(m, "datetime", _frozen(datetime(2026, 8, 15, 12, 0, 0)))
    called = []
    monkeypatch.setattr(m, "_run_score_live", lambda *a, **k: called.append(a) or (0, "ok"))
    assert m.main(["--force"]) == 0
    assert called


# --- main(): --tickers passthrough --------------------------------------------

def test_tickers_flag_is_passed_through_to_score_live(monkeypatch, _isolated_paths):
    monkeypatch.setattr(m, "datetime", _frozen(datetime(2026, 8, 17, 12, 0, 0)))  # a Monday
    received = {}
    monkeypatch.setattr(m, "_run_score_live", lambda tickers=None: received.setdefault("tickers", tickers) or (0, "ok"))
    assert m.main(["--tickers", "AAPL", "MSFT"]) == 0
    assert received["tickers"] == ["AAPL", "MSFT"]


# --- main(): logging + diffing -------------------------------------------------

def test_identical_output_two_days_reports_no_change(_isolated_paths, monkeypatch, capsys):
    monkeypatch.setattr(m, "datetime", _frozen(datetime(2026, 8, 17, 12, 0, 0)))
    monkeypatch.setattr(m, "_run_score_live", lambda *a, **k: (1, "same failure every day"))
    assert m.main([]) == 0
    capsys.readouterr()

    monkeypatch.setattr(m, "datetime", _frozen(datetime(2026, 8, 18, 12, 0, 0)))
    monkeypatch.setattr(m, "_run_score_live", lambda *a, **k: (1, "same failure every day"))
    assert m.main([]) == 0
    out = capsys.readouterr().out
    assert "no change" in out.lower()


def test_changed_output_is_flagged_not_silently_absorbed(_isolated_paths, monkeypatch, capsys):
    """Deliberate failing-case: a real change (e.g. preflight starts
    passing) must produce visible diff output, not just a quiet log write."""
    monkeypatch.setattr(m, "datetime", _frozen(datetime(2026, 8, 17, 12, 0, 0)))
    monkeypatch.setattr(m, "_run_score_live", lambda *a, **k: (1, "stock_list.py is empty"))
    assert m.main([]) == 0
    capsys.readouterr()

    monkeypatch.setattr(m, "datetime", _frozen(datetime(2026, 8, 18, 12, 0, 0)))
    monkeypatch.setattr(m, "_run_score_live", lambda *a, **k: (0, "3 recommendations: AAPL, MSFT, NVDA"))
    assert m.main([]) == 0
    out = capsys.readouterr().out
    assert "changed" in out.lower()
    assert "AAPL" in out


# --- non-mutation guard --------------------------------------------------------

def test_untouched_protected_paths_leave_run_clean(_isolated_paths, monkeypatch):
    """Positive control: portfolio_state.json and config/ are left
    byte-identical by a run that doesn't touch them -- the common,
    correct case."""
    portfolio_path = _isolated_paths["portfolio_path"]
    (_isolated_paths["config_dir"] / "cost_model.py").write_text("X = 1\n")
    before_portfolio = portfolio_path.read_bytes()

    monkeypatch.setattr(m, "datetime", _frozen(datetime(2026, 8, 17, 12, 0, 0)))
    monkeypatch.setattr(m, "_run_score_live", lambda *a, **k: (0, "ok, touched nothing"))
    assert m.main([]) == 0
    assert portfolio_path.read_bytes() == before_portfolio


def test_a_run_that_mutates_portfolio_state_is_caught_not_silently_allowed(_isolated_paths, monkeypatch):
    """The actual point of the guard: score_live.py must NEVER write
    portfolio_state.json (a human does, by hand, after a real trade). A
    fake _run_score_live that mutates it mid-run must cause main() to
    raise LiveStateMutationError, proving the guard is a real check
    against the file's actual bytes, not a no-op."""
    portfolio_path = _isolated_paths["portfolio_path"]

    def _mutating_run(*a, **k):
        portfolio_path.write_text('{"trade_log": [{"ticker": "AAPL"}]}')  # simulated violation
        return (0, "ok")

    monkeypatch.setattr(m, "datetime", _frozen(datetime(2026, 8, 17, 12, 0, 0)))
    monkeypatch.setattr(m, "_run_score_live", _mutating_run)
    with pytest.raises(m.LiveStateMutationError):
        m.main([])


def test_a_run_that_creates_a_new_config_file_is_also_caught(_isolated_paths, monkeypatch):
    """Deliberate failing-case, distinct from the mutation case above: a
    protected path going from not-existing to existing is exactly as much
    a violation as one changing -- config/ must never gain a file either."""
    config_dir = _isolated_paths["config_dir"]

    def _config_writing_run(*a, **k):
        (config_dir / "new_setting.py").write_text("SNEAKY = True\n")
        return (0, "ok")

    monkeypatch.setattr(m, "datetime", _frozen(datetime(2026, 8, 17, 12, 0, 0)))
    monkeypatch.setattr(m, "_run_score_live", _config_writing_run)
    with pytest.raises(m.LiveStateMutationError):
        m.main([])


# --- gap check ------------------------------------------------------------------

def test_no_gaps_when_every_trading_day_has_a_log(_isolated_paths, monkeypatch, capsys):
    log_dir = _isolated_paths["log_dir"]
    log_dir.mkdir(exist_ok=True)
    for d in ("2026-08-17", "2026-08-18", "2026-08-19"):  # Mon, Tue, Wed -- consecutive trading days
        (log_dir / f"{d}.log").write_text("x")
    assert m._find_gaps("2026-08-19") == []


def test_a_missing_weekday_between_logs_is_flagged_as_a_gap(_isolated_paths):
    """Deliberate failing-case: 08-17 and 08-19 exist but 08-18 (a real
    Tuesday, not a holiday) doesn't -- the monitor apparently didn't run
    that day, and that must be visible, not silently skipped over."""
    log_dir = _isolated_paths["log_dir"]
    log_dir.mkdir(exist_ok=True)
    (log_dir / "2026-08-17.log").write_text("x")
    (log_dir / "2026-08-19.log").write_text("x")
    assert m._find_gaps("2026-08-19") == ["2026-08-18"]


def test_a_missing_weekend_or_holiday_is_not_a_gap(_isolated_paths):
    """A day with no log file at all is only a gap if it was actually an
    expected trading day -- a weekend with no log (e.g. the monitor was
    only ever run --force on trading days, never touched on weekends)
    must not be flagged."""
    log_dir = _isolated_paths["log_dir"]
    log_dir.mkdir(exist_ok=True)
    (log_dir / "2026-08-14.log").write_text("x")  # Friday
    (log_dir / "2026-08-17.log").write_text("x")  # Monday -- Sat/Sun between, no logs, not a gap
    assert m._find_gaps("2026-08-17") == []


def test_gap_check_is_printed_during_a_normal_run(_isolated_paths, monkeypatch, capsys):
    log_dir = _isolated_paths["log_dir"]
    log_dir.mkdir(exist_ok=True)
    (log_dir / "2026-08-17.log").write_text(f"[x] HEARTBEAT exit_code=1\n{m._SEPARATOR}\nold\n")
    # 08-18 (Tuesday) has no log -- a real gap once 08-19 runs.

    monkeypatch.setattr(m, "datetime", _frozen(datetime(2026, 8, 19, 12, 0, 0)))
    monkeypatch.setattr(m, "_run_score_live", lambda *a, **k: (1, "old"))
    m.main([])
    out = capsys.readouterr().out.lower()
    assert "gap check" in out
    assert "2026-08-18" in out
