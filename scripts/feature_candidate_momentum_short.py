"""Feature-work tight loop, iteration 4: swap momentum_60_skip_10 ->
momentum_10_skip_2 (Jegadeesh & Titman 1993, momentum -- see
baseline_system_architecture.md open question 16's horizon audit).

Tested separately from the 5-day cohort swap (iteration 3) because it's
the single largest horizon mismatch in the feature set on its own -- a
60-day formation window against a 5-day target -- large enough that the
power argument for a family test (iteration 3's docstring) doesn't apply
here: if this one feature carries a real effect, it may be big enough to
clear 2 SE by itself, unlike the smaller individual contributions the
cohort swap was built to detect in aggregate.

Usage:
    python -m scripts.feature_candidate_momentum_short
"""
from __future__ import annotations

from src.data_layer.real_provider import FEATURE_COLUMNS
from scripts.feature_candidate_common import test_candidate_columns

CANDIDATE_COLUMNS = [c if c != "momentum_60_skip_10" else "momentum_10_skip_2" for c in FEATURE_COLUMNS]


def main() -> None:
    assert "momentum_60_skip_10" not in CANDIDATE_COLUMNS
    assert "momentum_10_skip_2" in CANDIDATE_COLUMNS
    assert len(CANDIDATE_COLUMNS) == len(FEATURE_COLUMNS)
    test_candidate_columns(CANDIDATE_COLUMNS, label="momentum_60_skip_10 -> momentum_10_skip_2 (swap)")


if __name__ == "__main__":
    main()
