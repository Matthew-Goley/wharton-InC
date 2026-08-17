import pandas as pd
import pytest

from src.combiner.overall_model import OverallModel
from src.config import ExpertSpec

BROAD = ExpertSpec(name="broad", sector=None)
TECH = ExpertSpec(name="tech", sector="Technology")


def _preds(rows: dict) -> pd.DataFrame:
    """rows: {(ticker, timestamp): (predicted_return, confidence)}"""
    index = pd.MultiIndex.from_tuples(rows.keys(), names=["ticker", "timestamp"])
    return pd.DataFrame(
        {"predicted_return": [v[0] for v in rows.values()], "confidence": [v[1] for v in rows.values()]},
        index=index,
    )


T0 = pd.Timestamp("2020-01-01")


def test_agreeing_sector_and_broad_are_confidence_weighted_plus_bonus():
    model = OverallModel((BROAD, TECH), agreement_bonus=0.1)
    predictions = {
        "broad": _preds({("AAA", T0): (0.02, 0.8), ("BBB", T0): (-0.01, 0.6)}),
        "tech": _preds({("AAA", T0): (0.04, 0.4)}),
    }
    combined = model.combine(predictions)

    aaa = combined.loc[("AAA", T0)]
    expected_weighted = (0.02 * 0.8 + 0.04 * 0.4) / (0.8 + 0.4)
    expected_multiplier = 1.0 + 0.1 * min(0.8, 0.4)
    assert aaa["combined_score"] == pytest.approx(expected_weighted * expected_multiplier)
    assert aaa["n_experts"] == 2

    bbb = combined.loc[("BBB", T0)]
    assert bbb["combined_score"] == pytest.approx(-0.01)
    assert bbb["n_experts"] == 1


def test_bearish_agreement_pushes_more_negative_not_toward_positive():
    """Agreement should reinforce whichever direction both experts share —
    including a shared bearish view. A multiplier >1 on a negative
    weighted-average makes it more negative; an unsigned/additive bonus
    could instead nudge a bearish-agreement ticker toward positive, risking
    a false buy signal."""
    model = OverallModel((BROAD, TECH), agreement_bonus=0.1)
    predictions = {
        "broad": _preds({("AAA", T0): (-0.02, 0.8)}),
        "tech": _preds({("AAA", T0): (-0.04, 0.4)}),
    }
    combined = model.combine(predictions)
    expected_weighted = (-0.02 * 0.8 + -0.04 * 0.4) / (0.8 + 0.4)
    expected_multiplier = 1.0 + 0.1 * min(0.8, 0.4)  # >1: reinforces the bearish agreement, in proportion
    assert combined.loc[("AAA", T0), "combined_score"] == pytest.approx(expected_weighted * expected_multiplier)
    assert combined.loc[("AAA", T0), "combined_score"] < expected_weighted


def test_disagreeing_experts_get_no_bonus():
    model = OverallModel((BROAD, TECH), agreement_bonus=0.1)
    predictions = {
        "broad": _preds({("AAA", T0): (0.02, 0.5)}),
        "tech": _preds({("AAA", T0): (-0.03, 0.5)}),
    }
    combined = model.combine(predictions)
    expected_weighted = (0.02 * 0.5 + -0.03 * 0.5) / (0.5 + 0.5)
    assert combined.loc[("AAA", T0), "combined_score"] == pytest.approx(expected_weighted)


def test_ticker_with_no_sector_expert_falls_back_to_broad_only():
    model = OverallModel((BROAD, TECH))
    predictions = {
        "broad": _preds({("ZZZ", T0): (0.05, 0.9)}),
        "tech": _preds({}),
    }
    combined = model.combine(predictions)
    assert combined.loc[("ZZZ", T0), "combined_score"] == pytest.approx(0.05)
    assert combined.loc[("ZZZ", T0), "n_experts"] == 1


def test_rank_orders_within_timestamp_independently():
    model = OverallModel((BROAD, TECH))
    t1 = pd.Timestamp("2020-01-02")
    predictions = {
        "broad": _preds(
            {("AAA", T0): (0.02, 0.5), ("BBB", T0): (0.09, 0.5), ("AAA", t1): (0.5, 0.5), ("BBB", t1): (0.01, 0.5)}
        ),
        "tech": _preds({}),
    }
    ranked = model.rank(model.combine(predictions))
    assert ranked.loc[("BBB", T0), "rank"] == 1
    assert ranked.loc[("AAA", T0), "rank"] == 2
    assert ranked.loc[("AAA", t1), "rank"] == 1
    assert ranked.loc[("BBB", t1), "rank"] == 2


def test_rejects_missing_or_duplicate_broad_expert():
    with pytest.raises(ValueError):
        OverallModel((TECH,))  # no broad expert at all


def test_agreement_bonus_stays_proportional_to_prediction_size():
    """Regression for a real bug: an *additive* bonus in raw-return units
    dominated small real predictions (agreement_bonus * confidence ~3-5% vs.
    typical 5-day predictions well under 2%), so agreement alone -- not
    predicted magnitude -- decided rank-1 on the first real has_sector=True
    run, and combined_score (which TradeSelector's cost-clearing filter
    reads as real % units) was inflated on bonus points rather than edge.
    The multiplicative fix must keep the bonus's *ratio* to the prediction
    constant regardless of the prediction's absolute size."""
    model = OverallModel((BROAD, TECH), agreement_bonus=0.1)
    small = model.combine({
        "broad": _preds({("AAA", T0): (0.002, 0.5)}),
        "tech": _preds({("AAA", T0): (0.002, 0.5)}),
    }).loc[("AAA", T0), "combined_score"]
    large = model.combine({
        "broad": _preds({("AAA", T0): (0.20, 0.5)}),
        "tech": _preds({("AAA", T0): (0.20, 0.5)}),
    }).loc[("AAA", T0), "combined_score"]
    # Same confidences, same relative agreement -- the bonus-inflated score
    # should scale with the prediction, not add a fixed absolute amount.
    assert small == pytest.approx(0.002 * 1.05)
    assert large == pytest.approx(0.20 * 1.05)
    assert large / small == pytest.approx(0.20 / 0.002)


def test_vol_adjusted_ranking_demotes_high_vol_large_prediction():
    """A bigger raw prediction on a much more volatile name should be
    demoted below a smaller prediction on a calmer one, once risk-adjusted
    -- the whole point of the mode."""
    model = OverallModel((BROAD,))
    predictions = {"broad": _preds({("LOUD", T0): (0.10, 0.9), ("CALM", T0): (0.02, 0.9)})}
    combined = model.combine(predictions)
    risk = pd.Series({("LOUD", T0): 0.20, ("CALM", T0): 0.01})
    risk.index.names = ["ticker", "timestamp"]

    raw_ranked = model.rank(combined)
    assert raw_ranked.loc[("LOUD", T0), "rank"] == 1  # bigger raw prediction wins by default

    vol_ranked = model.rank(combined, ranking_score="vol_adjusted", risk=risk)
    assert vol_ranked.loc[("CALM", T0), "rank"] == 1  # demoted once divided by its own volatility
    assert vol_ranked.loc[("LOUD", T0), "ranking_score"] == pytest.approx(0.10 / 0.20)
    assert vol_ranked.loc[("CALM", T0), "ranking_score"] == pytest.approx(0.02 / 0.01)
    # combined_score is untouched -- TradeSelector's cost-clearing filter
    # still sees real return units, not a risk-adjusted ratio.
    assert vol_ranked.loc[("LOUD", T0), "combined_score"] == pytest.approx(0.10)
    assert vol_ranked.loc[("CALM", T0), "combined_score"] == pytest.approx(0.02)


def test_vol_adjusted_ranking_requires_risk():
    model = OverallModel((BROAD,))
    combined = model.combine({"broad": _preds({("AAA", T0): (0.05, 0.9)})})
    with pytest.raises(ValueError, match="requires a `risk`"):
        model.rank(combined, ranking_score="vol_adjusted")


def test_unknown_ranking_score_rejected():
    model = OverallModel((BROAD,))
    combined = model.combine({"broad": _preds({("AAA", T0): (0.05, 0.9)})})
    with pytest.raises(ValueError, match="unknown ranking_score"):
        model.rank(combined, ranking_score="nonsense")


def test_broad_only_with_zero_sector_experts_configured():
    """No sector experts in the roster at all (not just no data for one
    ticker) — the real-world case until sector data exists. Regression test:
    the empty-sector fallback frame used to have a bare RangeIndex with no
    named levels, which broke the join against broad's MultiIndex."""
    model = OverallModel((BROAD,))
    predictions = {"broad": _preds({("AAA", T0): (0.05, 0.9), ("BBB", T0): (-0.02, 0.5)})}
    combined = model.combine(predictions)
    ranked = model.rank(combined)

    assert combined.loc[("AAA", T0), "combined_score"] == pytest.approx(0.05)
    assert combined.loc[("AAA", T0), "n_experts"] == 1
    assert ranked.loc[("AAA", T0), "rank"] == 1
    assert ranked.loc[("BBB", T0), "rank"] == 2
