"""Feature-work: earnings features, tested as a family (baseline_system_
architecture.md open question 22), only after the has_earnings_data
leakage gate cleared.

Two candidates, added together:
    days_to_next_earnings -- likely an event-risk flag more than a return
        predictor (heightened volatility/gap risk when an earnings call
        falls inside the hold_days window), not obviously signed.
    earnings_surprise -- post-earnings-announcement-drift (PEAD) theory
        (Bernard & Thomas 1989/1990) says returns continue drifting in
        the direction of a surprise for weeks afterward; this is the one
        expected to carry any real IC effect, if either does.

PRE-REGISTERED RULE, before running: report the FAMILY result first
(both features together, one paired_ic_comparison, one keep_or_discard
decision). Decompose into the two individual features ONLY if the family
clears 2 SE -- same discipline as the 5-day cohort swap (open question
19): a null on the pair doesn't get chased into two separate underpowered
single-feature tests looking for whichever one might be "close."

Usage:
    python -m scripts.feature_candidate_earnings
"""
from __future__ import annotations

from src.data_layer.real_provider import FEATURE_COLUMNS
from scripts.feature_candidate_common import test_candidate_columns

FAMILY = ["days_to_next_earnings", "earnings_surprise"]


def main() -> None:
    print("Pre-registered rule: report the family result first; decompose into individual")
    print("features only if the family clears 2 SE.\n")
    result = test_candidate_columns(FEATURE_COLUMNS + FAMILY, label=f"earnings family {FAMILY}")

    print("\n" + "=" * 72)
    if result["decision"] == "KEEP":
        print("Family CLEARED 2 SE -- proceeding to decompose into individual features.")
        for feature in FAMILY:
            test_candidate_columns(FEATURE_COLUMNS + [feature], label=feature)
    else:
        print(f"Family did not clear ({result['decision']}) -- per the pre-registered rule, NOT")
        print("decomposing into individual single-feature tests.")


if __name__ == "__main__":
    main()
