"""Feature-work: market-relative features, tested as a family -- the LAST
feature family tested in this project's tight loop (baseline_system_
architecture.md open question 25), pre-registered as such regardless of
outcome: if the family clears, keep it and stop; if it nulls, stop with a
null. No further feature candidates queued after this one.

Three candidates, added together, all against a point-in-time-safe DAILY
market return series (cross-sectional mean of each ticker's own 1-day
return -- built fresh in _add_market_relative_features, deliberately NOT
MARKET_TARGET_COLUMN, which is the forward-looking demeaning label and
would be a severe lookahead bug used as a feature ingredient):

    beta_60        -- systematic risk exposure. Market-neutral strategy
        literature (Frazzini & Pedersen 2014, "Betting Against Beta")
        documents low-beta stocks outperforming risk-adjusted; whether that
        shows up as raw predictive IC on this universe/horizon is untested.
    idio_vol_60    -- idiosyncratic volatility. Ang, Hodrick, Xing & Zhang
        2006 document a robust NEGATIVE relationship between idiosyncratic
        vol and subsequent returns (the "idio vol puzzle") -- one of the
        more replicated cross-sectional anomalies in the literature.
    market_corr_60 -- how much of a name's variance is systematic vs.
        idiosyncratic; related to but distinct from beta (beta is scaled by
        relative volatility, correlation isn't) -- diversification/factor-
        loading information beta alone doesn't fully capture.

PRE-REGISTERED RULE, before running: report the FAMILY result first (all
three features together, one paired_ic_comparison, one keep_or_discard
decision). Decompose into the three individual features ONLY if the family
clears 2 SE -- same discipline as the earnings family and the 5-day cohort
swap: a null on the family doesn't get chased into three separate
underpowered single-feature tests looking for whichever one might be
"close."

Usage:
    python -m scripts.feature_candidate_market_relative
"""
from __future__ import annotations

from src.data_layer.real_provider import FEATURE_COLUMNS
from scripts.feature_candidate_common import test_candidate_columns

FAMILY = ["beta_60", "idio_vol_60", "market_corr_60"]


def main() -> None:
    print("Pre-registered rule: report the family result first; decompose into individual")
    print("features only if the family clears 2 SE. This is the LAST feature family tested")
    print("in this project's tight loop, regardless of outcome.\n")
    result = test_candidate_columns(FEATURE_COLUMNS + FAMILY, label=f"market-relative family {FAMILY}")

    print("\n" + "=" * 72)
    if result["decision"] == "KEEP":
        print("Family CLEARED 2 SE -- proceeding to decompose into individual features.")
        for feature in FAMILY:
            test_candidate_columns(FEATURE_COLUMNS + [feature], label=feature)
    else:
        print(f"Family did not clear ({result['decision']}) -- per the pre-registered rule, NOT")
        print("decomposing into individual single-feature tests.")
        print("Per the pre-registered stop: this was the last feature family tested. Stopping")
        print("feature-work iteration.")


if __name__ == "__main__":
    main()
