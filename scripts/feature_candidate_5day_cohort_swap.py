"""Feature-work tight loop, iteration 3: swap the ENTIRE 20-day cohort to
5-day windows in one go, not one feature at a time.

Power argument (why this is one test, not eight): at this project's
measured paired-IC SE (~0.0032, see the vol_adjusted IC audit and
iteration 2's px_to_ma_10 result), a single feature contributing a real
but modest 0.002 to IC shows up at ~0.6 SE -- invisible against the 2 SE
bar. Five or more such features swapped together, if they share a
direction, show up around 3 SE against roughly the same noise floor (the
per-date IC dispersion is dominated by the whole model's day-to-day
variance, not by how many features changed). Iteration 2's null
(px_to_ma_10, t=-1.22) is equally consistent with "no horizon effect" and
"a real effect too small to see alone" -- no number of further one-at-a-
time swaps resolves that ambiguity. This test does: if the aggregate
clears 2 SE, the hypothesis is live and worth decomposing feature by
feature; if it doesn't, the whole audit finding (open question 16) is
dead in one run instead of eight underpowered ones.

Swapped (all 20-day-windowed features except the two the audit itself
excluded -- amihud_20/log_dollar_volume, slow-moving characteristics
where the window question doesn't really apply, and momentum_60_skip_10,
tested separately next per its own, larger mismatch):
    px_to_ma_20        -> px_to_ma_5
    volatility_20      -> volatility_5
    semivol_skew       -> semivol_skew_5
    max_ret_20         -> max_ret_5
    min_ret_20         -> min_ret_5
    realized_skew_20   -> realized_skew_5
    realized_kurt_20   -> realized_kurt_5
    ret_vol_corr_20    -> ret_vol_corr_5

Usage:
    python -m scripts.feature_candidate_5day_cohort_swap
"""
from __future__ import annotations

from src.data_layer.real_provider import FEATURE_COLUMNS
from scripts.feature_candidate_common import test_candidate_columns

SWAP_MAP = {
    "px_to_ma_20": "px_to_ma_5",
    "volatility_20": "volatility_5",
    "semivol_skew": "semivol_skew_5",
    "max_ret_20": "max_ret_5",
    "min_ret_20": "min_ret_5",
    "realized_skew_20": "realized_skew_5",
    "realized_kurt_20": "realized_kurt_5",
    "ret_vol_corr_20": "ret_vol_corr_5",
}

CANDIDATE_COLUMNS = [SWAP_MAP.get(c, c) for c in FEATURE_COLUMNS]


def main() -> None:
    assert len(CANDIDATE_COLUMNS) == len(FEATURE_COLUMNS)  # swap, not an addition
    assert not (set(SWAP_MAP) & set(CANDIDATE_COLUMNS))  # none of the old names survive
    assert set(SWAP_MAP.values()) <= set(CANDIDATE_COLUMNS)  # all new names present
    test_candidate_columns(CANDIDATE_COLUMNS, label=f"5-day cohort swap ({len(SWAP_MAP)} features)")


if __name__ == "__main__":
    main()
