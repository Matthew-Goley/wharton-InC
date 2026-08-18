"""Feature-work tight loop, iteration 1: sector_momentum_60_skip_10
(Moskowitz & Grinblatt 1999, industry momentum -- see
RealDataProvider._add_sector_momentum's docstring).

RESULT (already run, recorded in baseline_system_architecture.md open
question 14): REJECTED. IC +0.0209 -> +0.0095, paired t=-2.69 -- a real
move in the wrong direction. Kept runnable for reproducibility.

Usage:
    python -m scripts.feature_candidate_sector_momentum
"""
from __future__ import annotations

from src.data_layer.real_provider import FEATURE_COLUMNS
from scripts.feature_candidate_common import test_candidate_columns

CANDIDATE_FEATURE = "sector_momentum_60_skip_10"


def main() -> None:
    test_candidate_columns(FEATURE_COLUMNS + [CANDIDATE_FEATURE], label=CANDIDATE_FEATURE)


if __name__ == "__main__":
    main()
