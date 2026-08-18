"""score_live.py's pure logic: the preflight checks and the reasoning-
report join (baseline_system_architecture.md open question 26/38). The
network/real-data pipeline itself (fetch -> feature build -> predict ->
rank -> select) isn't unit-tested here -- same convention as every other
real-data script in this project, verified by running it, not mocked.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from config import hold_period
from scripts import score_live
from scripts.score_live import _build_reasoning_report, _preflight
from src.config import BaselineConfig


# --- _preflight ----------------------------------------------------------------

def test_preflight_passes_when_everything_is_ready(tmp_path, monkeypatch):
    monkeypatch.setattr(hold_period, "WINDOW_START_DATE", "2026-08-17")
    fake_model_path = tmp_path / "models.joblib"
    fake_model_path.write_text("placeholder")
    monkeypatch.setattr(score_live, "MODEL_PATH", fake_model_path)
    assert _preflight(["AAPL"]) is None


def test_preflight_names_an_empty_ticker_list(monkeypatch):
    monkeypatch.setattr(hold_period, "WINDOW_START_DATE", "2026-08-17")
    monkeypatch.setattr(score_live, "MODEL_PATH", Path(__file__))  # any existing path
    error = _preflight([])
    assert error is not None
    assert "stock_list.py is empty" in error


def test_preflight_names_a_missing_window_start_date(monkeypatch):
    monkeypatch.setattr(hold_period, "WINDOW_START_DATE", None)
    monkeypatch.setattr(score_live, "MODEL_PATH", Path(__file__))
    error = _preflight(["AAPL"])
    assert error is not None
    assert "WINDOW_START_DATE" in error


def test_preflight_names_a_missing_model_bundle(tmp_path, monkeypatch):
    monkeypatch.setattr(hold_period, "WINDOW_START_DATE", "2026-08-17")
    monkeypatch.setattr(score_live, "MODEL_PATH", tmp_path / "does_not_exist.joblib")
    error = _preflight(["AAPL"])
    assert error is not None
    assert "does not exist" in error


def test_preflight_reports_every_problem_at_once_not_just_the_first():
    """A real run with multiple missing preconditions should name all of
    them in one message, not fail-fast on the first and hide the rest."""
    import config.hold_period as hp
    original = hp.WINDOW_START_DATE
    try:
        hp.WINDOW_START_DATE = None
        error = _preflight([])
        assert "stock_list.py is empty" in error
        assert "WINDOW_START_DATE" in error
    finally:
        hp.WINDOW_START_DATE = original


# --- _build_reasoning_report / _reason_for_row ----------------------------------

def _ranked_row(**overrides):
    defaults = dict(
        broad_return=0.005, broad_confidence=0.6, sector_return=0.006, sector_confidence=0.5,
        sector_expert="tech", combined_score=0.0055, n_experts=2, ranking_score=1.2, rank=1,
    )
    defaults.update(overrides)
    return defaults


def _make_ranked(rows: dict) -> pd.DataFrame:
    index = pd.MultiIndex.from_tuples(rows.keys(), names=["ticker", "timestamp"])
    return pd.DataFrame(list(rows.values()), index=index)


def _make_trades(rows: list[dict]) -> pd.DataFrame:
    index = pd.MultiIndex.from_tuples([(r["ticker"], r["timestamp"]) for r in rows], names=["ticker", "timestamp"])
    frame = pd.DataFrame(rows, index=index)
    return frame.drop(columns=["ticker", "timestamp"])


T0 = pd.Timestamp("2026-08-18")


def test_reasoning_report_is_empty_but_correctly_shaped_for_no_trades():
    empty_trades = pd.DataFrame(columns=["action", "period", "forced"])
    empty_trades.index = pd.MultiIndex.from_tuples([], names=["ticker", "timestamp"])
    ranked = _make_ranked({("AAPL", T0): _ranked_row()})
    report = _build_reasoning_report(empty_trades, ranked, BaselineConfig())
    assert report.empty
    assert "reasoning" in report.columns


def test_reasoning_report_joins_the_dropped_fields_back_for_an_entry():
    """The actual point of gap #8: sector_expert/broad_return/sector_return/
    ranking_score exist in `ranked` but not in TradeSelector's own output --
    confirm the join actually recovers them."""
    trades = _make_trades([
        {"ticker": "AAPL", "timestamp": T0, "action": "entry", "period": 5, "entry_period": 5,
         "position_size_pct": 0.1, "forced": False},
    ])
    ranked = _make_ranked({("AAPL", T0): _ranked_row(sector_expert="tech", broad_return=0.01, sector_return=0.012)})
    report = _build_reasoning_report(trades, ranked, BaselineConfig())
    row = report.loc[("AAPL", T0)]
    assert row["sector_expert"] == "tech"
    assert row["broad_return"] == pytest.approx(0.01)
    assert row["sector_return"] == pytest.approx(0.012)


def test_reasoning_names_the_sector_model_and_agreement_for_an_entry():
    config = BaselineConfig()
    row = pd.Series(_ranked_row(sector_expert="tech", broad_return=0.01, sector_return=0.012))
    row["action"] = "entry"
    row["position_size_pct"] = 0.1
    reasoning = score_live._reason_for_row(row, config)
    assert "tech" in reasoning
    assert "agree" in reasoning
    assert "+1.00%" in reasoning  # broad_return formatted as a predicted-edge percentage


def test_reasoning_flags_disagreement_between_broad_and_sector():
    config = BaselineConfig()
    row = pd.Series(_ranked_row(sector_expert="tech", broad_return=0.01, sector_return=-0.02))
    row["action"] = "entry"
    row["position_size_pct"] = 0.1
    reasoning = score_live._reason_for_row(row, config)
    assert "disagree" in reasoning


def test_reasoning_notes_no_sector_coverage_when_broad_only():
    """Deliberate failing-case check: sector_expert=NaN (n_experts==1,
    broad-only) must not be mistaken for a real sector name in the
    reasoning text."""
    config = BaselineConfig()
    row = pd.Series(_ranked_row(sector_expert=float("nan"), n_experts=1))
    row["action"] = "entry"
    row["position_size_pct"] = 0.1
    reasoning = score_live._reason_for_row(row, config)
    assert "no sector coverage" in reasoning


def test_reasoning_for_a_scheduled_exit_names_the_entry_period_not_a_prediction():
    config = BaselineConfig()
    row = pd.Series({"action": "exit", "entry_period": 3, "period": 8, "forced": False})
    reasoning = score_live._reason_for_row(row, config)
    assert "scheduled exit" in reasoning
    assert "entered period 3" in reasoning
    assert "predict" not in reasoning  # an exit is mechanical, not a model decision


def test_reasoning_for_a_forced_close_says_so_explicitly():
    config = BaselineConfig()
    row = pd.Series({"action": "exit", "entry_period": 3, "period": 50, "forced": True})
    reasoning = score_live._reason_for_row(row, config)
    assert "forced close" in reasoning
