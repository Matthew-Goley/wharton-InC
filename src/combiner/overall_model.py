"""Combination layer (architecture §4): experts -> one ranked score per ticker."""
from __future__ import annotations

from typing import Literal, Optional

import numpy as np
import pandas as pd

from src.config import BROAD_EXPERT, ExpertSpec

#: Below this, a risk denominator is treated as unreliable-small rather than
#: divided by, so a near-zero volatility reading can't produce a runaway score.
_DEFAULT_RISK_FLOOR = 0.005

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
    def rank(
        combined: pd.DataFrame,
        ranking_score: Literal["raw", "vol_adjusted"] = "raw",
        risk: Optional[pd.Series] = None,
        risk_floor: float = _DEFAULT_RISK_FLOOR,
    ) -> pd.DataFrame:
        """Add a per-timestamp ``rank`` column (1 = best score).

        ``ranking_score`` selects what gets *sorted* — it never touches
        ``combined_score`` itself, which stays in real return units for
        ``TradeSelector``'s long-only check and cost-clearing filter (§5).
        A risk-adjusted ratio isn't comparable to a fixed % cost threshold,
        so ranking and eligibility deliberately read two different columns:

        - ``"raw"`` (default): rank by ``combined_score`` directly.
        - ``"vol_adjusted"``: rank by ``combined_score / risk``. Diagnosed
          on the real backtest: taking the argmax of predicted return
          across ~580 tickers a day disproportionately promotes whichever
          name's raw prediction was largest *in magnitude* — which
          correlates with recent realized volatility, not with prediction
          reliability (measured: rank-1 picks averaged 4-5x the |predicted
          return| of every other rank, concentrated in a handful of
          high-volatility names). Dividing by recent volatility demotes a
          big prediction that's just a big vol reading. Requires ``risk``
          — a series shaped like the ``volatility_20`` feature, aligned to
          the same (ticker, timestamp) index.

        A third mode belongs in this same slot once it's buildable:
        ranking by a lower-confidence-bound derived from broad/sector-
        expert *disagreement* addresses a second, distinct effect that
        vol-adjustment structurally can't reach — the argmax of many noisy
        estimates is upward-biased by selection alone, even under
        homoskedastic noise, independent of volatility. That needs a
        second opinion to detect (ensemble disagreement as an uncertainty
        estimate), which needs sector experts to exist. Not implemented;
        tracked as an ablation row ("Selection score") in
        baseline_system_architecture.md.
        """
        out = combined.copy()
        if ranking_score == "raw":
            score = out["combined_score"]
        elif ranking_score == "vol_adjusted":
            if risk is None:
                raise ValueError("ranking_score='vol_adjusted' requires a `risk` series (e.g. volatility_20)")
            denom = risk.reindex(out.index)
            denom = denom.fillna(denom.max()).clip(lower=risk_floor)
            score = out["combined_score"] / denom
        else:
            raise ValueError(f"unknown ranking_score {ranking_score!r} (expected 'raw' or 'vol_adjusted')")

        out["ranking_score"] = score
        out["rank"] = (
            out.groupby(level="timestamp")["ranking_score"]
            .rank(ascending=False, method="first")
            .astype(int)
        )
        return out.sort_index(level="timestamp")
