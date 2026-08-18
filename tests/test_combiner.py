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
    # Explicit "raw" -- this test is about rank()'s per-timestamp grouping
    # in general, not about ranking_score; rank()'s own resolved default
    # is "vol_adjusted" (TradingConfig.ranking_score), which needs a
    # `risk` series this test doesn't supply.
    ranked = model.rank(model.combine(predictions), ranking_score="raw")
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

    raw_ranked = model.rank(combined, ranking_score="raw")
    assert raw_ranked.loc[("LOUD", T0), "rank"] == 1  # bigger raw prediction wins under "raw"

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


# --- rank()'s default resolves from TradingConfig, not a hardcoded literal
# (baseline_system_architecture.md open question 33: gap #5's actual fix) --

def test_bare_rank_call_resolves_to_the_declared_standing_default():
    """Deliberate failing-case check on the fix itself: a truly bare call
    (no ranking_score, no risk) must raise vol_adjusted's own "requires
    risk" error, not silently succeed under "raw" -- proof the default
    genuinely resolved to TradingConfig.ranking_score ("vol_adjusted"),
    not that this is a no-op signature change. If TradingConfig.
    ranking_score is ever changed back to "raw", this test would need
    updating too -- it enforces "the default IS whatever config
    declares", not the literal string "vol_adjusted" per se."""
    from src.config import TradingConfig
    assert TradingConfig.ranking_score == "vol_adjusted"  # the premise this test depends on
    model = OverallModel((BROAD,))
    combined = model.combine({"broad": _preds({("AAA", T0): (0.05, 0.9)})})
    with pytest.raises(ValueError, match="requires a `risk`"):
        model.rank(combined)  # no ranking_score, no risk -- bare, exactly like the 5 real call sites


def test_bare_rank_call_with_risk_actually_computes_vol_adjusted():
    """The positive counterpart: supply risk and nothing else, confirm the
    OUTPUT is genuinely vol-adjusted (score / risk), not raw combined_score
    -- a behavioral proof, not just "it didn't raise"."""
    model = OverallModel((BROAD,))
    predictions = {"broad": _preds({("LOUD", T0): (0.10, 0.9), ("CALM", T0): (0.02, 0.9)})}
    combined = model.combine(predictions)
    risk = pd.Series({("LOUD", T0): 0.20, ("CALM", T0): 0.01})
    risk.index.names = ["ticker", "timestamp"]

    bare_ranked = model.rank(combined, risk=risk)  # no ranking_score -- must behave like "vol_adjusted"
    explicit_ranked = model.rank(combined, ranking_score="vol_adjusted", risk=risk)
    pd.testing.assert_series_equal(bare_ranked["ranking_score"], explicit_ranked["ranking_score"])
    assert bare_ranked.loc[("CALM", T0), "rank"] == 1  # demoted LOUD, same as the explicit-mode test above


def test_explicit_ranking_score_still_overrides_the_default():
    """A caller that explicitly wants "raw" (or any other mode) must still
    be able to say so -- the default changing doesn't remove the ability
    to opt out of it."""
    model = OverallModel((BROAD,))
    predictions = {"broad": _preds({("LOUD", T0): (0.10, 0.9), ("CALM", T0): (0.02, 0.9)})}
    combined = model.combine(predictions)
    ranked = model.rank(combined, ranking_score="raw")
    assert ranked.loc[("LOUD", T0), "rank"] == 1  # bigger raw prediction wins, vol_adjusted would have demoted it


def test_uncertainty_adjusted_demotes_high_disagreement():
    """Two covered tickers with the same raw combined_score, but one has
    broad/sector experts agreeing closely and the other disagreeing a
    lot -- the low-disagreement one should rank higher once penalized."""
    model = OverallModel((BROAD, TECH), agreement_bonus=0.0)  # bonus off: isolate the disagreement effect
    predictions = {
        "broad": _preds({("AGREE", T0): (0.05, 0.5), ("DISAGREE", T0): (0.05, 0.5)}),
        "tech": _preds({("AGREE", T0): (0.052, 0.5), ("DISAGREE", T0): (0.09, 0.5)}),
    }
    combined = model.combine(predictions)
    # Both have the same weighted-average combined_score by construction? Not
    # required -- what matters is disagreement, not identical combined_score.
    ranked = model.rank(combined, ranking_score="uncertainty_adjusted")
    assert ranked.loc[("AGREE", T0), "rank"] == 1
    disagree_uncertainty = abs(0.05 - 0.09)
    agree_uncertainty = abs(0.05 - 0.052)
    assert ranked.loc[("DISAGREE", T0), "ranking_score"] == pytest.approx(
        combined.loc[("DISAGREE", T0), "combined_score"] - disagree_uncertainty
    )
    assert ranked.loc[("AGREE", T0), "ranking_score"] == pytest.approx(
        combined.loc[("AGREE", T0), "combined_score"] - agree_uncertainty
    )


def test_uncertainty_adjusted_penalizes_uncovered_ticker_not_promotes_it():
    """The trap this mode has to avoid: a ticker with no sector expert has
    no disagreement to measure. Defaulting that to zero uncertainty would
    make 'no second opinion' read as 'maximally confident' and promote it
    over a covered ticker with a smaller but well-agreed-upon prediction.

    Three tickers, so the fill value (that date's MAX covered disagreement)
    differs from any one covered ticker's own disagreement -- a uniform
    penalty can't flip a ranking, so this needs a real spread to prove the
    mechanism does anything:
      - UNCOVERED: no tech prediction at all, raw score 0.06 -- the single
        largest raw score in the batch, would win outright under 'raw'.
      - COVERED: broad/tech agree closely (disagreement 0.002) -- small
        combined_score, but should barely be penalized.
      - NOISY: broad/tech disagree a lot (disagreement 0.06) -- sets that
        date's max, which UNCOVERED must inherit as its own penalty.
    """
    model = OverallModel((BROAD, TECH), agreement_bonus=0.0)
    predictions = {
        "broad": _preds({("UNCOVERED", T0): (0.06, 0.5), ("COVERED", T0): (0.05, 0.5), ("NOISY", T0): (0.03, 0.5)}),
        "tech": _preds({("COVERED", T0): (0.052, 0.5), ("NOISY", T0): (0.09, 0.5)}),  # no UNCOVERED row
    }
    combined = model.combine(predictions)
    raw_ranked = model.rank(combined, ranking_score="raw")
    assert raw_ranked.loc[("UNCOVERED", T0), "rank"] == 1  # wins on raw score alone

    ranked = model.rank(combined, ranking_score="uncertainty_adjusted")
    # UNCOVERED's fill uncertainty must equal that date's MAX observed
    # disagreement among covered rows -- NOISY's 0.06, not COVERED's 0.002
    # and not zero.
    max_disagreement = abs(0.03 - 0.09)
    assert max_disagreement == pytest.approx(0.06)
    assert ranked.loc[("UNCOVERED", T0), "ranking_score"] == pytest.approx(0.06 - max_disagreement)
    # With the correct (large) fill applied, UNCOVERED's advantage is wiped
    # out and COVERED -- barely penalized for its own small, real
    # disagreement -- must now outrank it.
    assert ranked.loc[("COVERED", T0), "rank"] == 1
    assert ranked.loc[("UNCOVERED", T0), "rank"] > 1


def test_uncertainty_adjusted_falls_back_to_global_max_for_a_fully_uncovered_date():
    """A date with zero covered rows has no per-date max to fill from --
    must fall back to the global max across the whole frame, not 0 or NaN."""
    model = OverallModel((BROAD, TECH), agreement_bonus=0.0)
    t1 = pd.Timestamp("2020-01-02")
    predictions = {
        "broad": _preds({("AAA", T0): (0.05, 0.5), ("BBB", t1): (0.05, 0.5)}),
        "tech": _preds({("AAA", T0): (0.09, 0.5)}),  # only T0 has any coverage; t1 is fully uncovered
    }
    combined = model.combine(predictions)
    ranked = model.rank(combined, ranking_score="uncertainty_adjusted")
    global_max_disagreement = abs(0.05 - 0.09)
    assert ranked.loc[("BBB", t1), "ranking_score"] == pytest.approx(0.05 - global_max_disagreement)


def test_uncertainty_adjusted_degenerates_to_no_penalty_with_zero_coverage_anywhere():
    """No sector expert ever matches anything in this frame -- there's no
    max disagreement to fall back to at all. Must not crash or produce
    NaN; penalty is 0 (equivalent to 'raw') in this fully-degenerate case."""
    model = OverallModel((BROAD,))
    combined = model.combine({"broad": _preds({("AAA", T0): (0.05, 0.9), ("BBB", T0): (0.02, 0.9)})})
    ranked = model.rank(combined, ranking_score="uncertainty_adjusted")
    assert ranked.loc[("AAA", T0), "ranking_score"] == pytest.approx(0.05)
    assert ranked.loc[("BBB", T0), "ranking_score"] == pytest.approx(0.02)
    assert ranked.loc[("AAA", T0), "rank"] == 1


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
    # Explicit "raw" -- this test is about combine()'s zero-sector-experts
    # join behavior, not ranking_score; rank() is only used to get a rank
    # column for the assertions below.
    ranked = model.rank(combined, ranking_score="raw")

    assert combined.loc[("AAA", T0), "combined_score"] == pytest.approx(0.05)
    assert combined.loc[("AAA", T0), "n_experts"] == 1
    assert ranked.loc[("AAA", T0), "rank"] == 1
    assert ranked.loc[("BBB", T0), "rank"] == 2
