"""Leakage gate (baseline_system_architecture.md open question 22): does
has_earnings_data -- a pure missingness INDICATOR, no real earnings
content at all -- move IC on its own?

Delisted/acquired tickers (53/653, 8.1% of the universe) have zero
earnings data and are systematically volatile, event-driven names.
XGBoost routes NaN to a learned default branch, so a later earnings-
derived feature that's NaN for exactly this same set could let the model
learn "missing -> behaves differently" -- survivorship correlation, not
earnings economics. This test isolates that missingness pattern alone,
BEFORE any real earnings feature is built on top of it, so a later
"earnings surprise moved IC" result can be told apart from "missingness
alone moved IC."

STOP CONDITION: if this clears 2 SE on its own, that's survivorship
leakage -- any subsequent earnings-derived IC result would be
contaminated and shouldn't be trusted without addressing this first.

Usage:
    python -m scripts.feature_candidate_has_earnings_data
"""
from __future__ import annotations

from src.data_layer.real_provider import FEATURE_COLUMNS
from scripts.feature_candidate_common import test_candidate_columns

CANDIDATE_FEATURE = "has_earnings_data"


def main() -> None:
    result = test_candidate_columns(FEATURE_COLUMNS + [CANDIDATE_FEATURE], label=CANDIDATE_FEATURE)
    print("\n" + "=" * 72)
    if abs(result["t"]) > 2.0:
        print("STOP: has_earnings_data alone moved IC past 2 SE -- this is survivorship")
        print("leakage. Do not proceed to test any earnings-derived feature until this")
        print("is understood and addressed.")
    else:
        print("CLEAR: has_earnings_data alone shows no detectable IC effect. Proceeding")
        print("to test real earnings-derived features is not pre-contaminated by this.")


if __name__ == "__main__":
    main()
