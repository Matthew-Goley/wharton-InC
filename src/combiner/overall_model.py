"""Combination layer (architecture §4): experts -> one ranked score per ticker."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import BROAD_EXPERT, ExpertSpec

_JOIN_COLUMNS = ["broad_return", "broad_confidence", "sector_return", "sector_confidence", "sector_expert"]


class OverallModel:
    """Merges per-expert predictions into a single combined_score per ticker.

    Per architecture §4: the broad model always contributes; the sector
    expert contributes too when the ticker belongs to that sector. The
    combined score is a confidence-weighted average of predicted returns,
    plus a bonus when broad and sector agree on direction.
    """

    def __init__(self, experts: tuple[ExpertSpec, ...], agreement_bonus: float = 0.1):
        broad_specs = [spec for spec in experts if spec.sector is None]
        if len(broad_specs) != 1 or broad_specs[0].name != BROAD_EXPERT:
            raise ValueError(f"Expected exactly one broad expert named '{BROAD_EXPERT}'")
        self.experts = experts
        self.agreement_bonus = agreement_bonus

    def combine(self, expert_predictions: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """``expert_predictions``: expert name -> DataFrame(predicted_return, confidence)
        indexed by (ticker, timestamp), as produced by ``ExpertModel.predict``.
        """
        broad = expert_predictions[BROAD_EXPERT].rename(
            columns={"predicted_return": "broad_return", "confidence": "broad_confidence"}
        )

        sector_specs = [spec for spec in self.experts if spec.sector is not None]
        sector_frames = [
            expert_predictions[spec.name]
            .rename(columns={"predicted_return": "sector_return", "confidence": "sector_confidence"})
            .assign(sector_expert=spec.name)
            for spec in sector_specs
            if spec.name in expert_predictions
        ]
        # Sector membership is disjoint (a ticker belongs to one sector), so
        # concatenating sector predictions never produces overlapping index
        # entries — a plain outer join against `broad` is safe.
        if sector_frames:
            sector_combined = pd.concat(sector_frames)
        else:
            # No sector experts configured at all (e.g. sector data isn't
            # available yet, broad-only run) — an empty frame with a bare
            # RangeIndex has no named levels to join broad's MultiIndex
            # against ("cannot join with no overlapping index names"), so
            # it needs the same (ticker, timestamp) index shape, just empty.
            sector_combined = pd.DataFrame(
                columns=["sector_return", "sector_confidence", "sector_expert"],
                index=broad.index[:0],
            )

        merged = broad.join(sector_combined, how="left")
        has_sector = merged["sector_return"].notna()

        sector_return = merged["sector_return"].fillna(0.0)
        sector_confidence = merged["sector_confidence"].fillna(0.0)
        total_confidence = merged["broad_confidence"] + sector_confidence

        weighted_return = np.where(
            has_sector,
            (merged["broad_return"] * merged["broad_confidence"] + sector_return * sector_confidence)
            / total_confidence.replace(0.0, np.nan),
            merged["broad_return"],
        )

        agree = has_sector & (np.sign(merged["broad_return"]) == np.sign(sector_return)) & (
            np.sign(merged["broad_return"]) != 0
        )
        # Signed: agreement reinforces whichever direction both experts point
        # (more positive when both are bullish, more negative when both are
        # bearish). An unsigned bonus would nudge a bearish-agreement ticker
        # *toward* positive, which could flip it into a false buy signal.
        bonus = np.where(
            agree,
            self.agreement_bonus * np.minimum(merged["broad_confidence"], sector_confidence) * np.sign(merged["broad_return"]),
            0.0,
        )

        merged["combined_score"] = weighted_return + bonus
        merged["n_experts"] = np.where(has_sector, 2, 1)
        return merged[_JOIN_COLUMNS + ["combined_score", "n_experts"]]

    @staticmethod
    def rank(combined: pd.DataFrame) -> pd.DataFrame:
        """Add a per-timestamp ``rank`` column (1 = best combined_score)."""
        out = combined.copy()
        out["rank"] = (
            out.groupby(level="timestamp")["combined_score"]
            .rank(ascending=False, method="first")
            .astype(int)
        )
        return out.sort_index(level="timestamp")
