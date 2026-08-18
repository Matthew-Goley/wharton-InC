"""Feature-work tight loop, iteration 2: swap px_to_ma_20 -> px_to_ma_10
(short-term reversal, Jegadeesh 1990).

Motivated by the horizon audit (baseline_system_architecture.md open
question 14): short-term reversal is validated at roughly a 1-week
horizon in the literature, but px_to_ma_20's reference point is a full
trading month -- a genuine, low-risk mismatch to test, since shortening a
moving-average reference (unlike shortening a skew/kurtosis window)
doesn't trade horizon-match for pure estimation noise. Different family
from iteration 1 (reversal, not momentum) and closer to the 5-day label,
per the plan.

This is a SWAP, not an addition: the candidate set is the baseline 15
features with px_to_ma_20 replaced by px_to_ma_10, isolating the window-
length question directly rather than testing whether a 16th column helps.

Usage:
    python -m scripts.feature_candidate_px_to_ma_10
"""
from __future__ import annotations

from src.data_layer.real_provider import FEATURE_COLUMNS
from scripts.feature_candidate_common import test_candidate_columns

CANDIDATE_COLUMNS = [c if c != "px_to_ma_20" else "px_to_ma_10" for c in FEATURE_COLUMNS]


def main() -> None:
    assert "px_to_ma_20" not in CANDIDATE_COLUMNS
    assert "px_to_ma_10" in CANDIDATE_COLUMNS
    assert len(CANDIDATE_COLUMNS) == len(FEATURE_COLUMNS)  # swap, not an addition
    test_candidate_columns(CANDIDATE_COLUMNS, label="px_to_ma_20 -> px_to_ma_10 (swap)")


if __name__ == "__main__":
    main()
