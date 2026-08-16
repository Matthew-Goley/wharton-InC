"""Scoring functions used as GA fitness and for reporting."""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def information_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Spearman rank correlation between predicted and realized returns.

    The standard "IC" metric for return-prediction quality: what matters for
    ranking trade candidates is getting the *order* right, not the exact
    magnitude, so rank correlation is a better fitness signal than RMSE.
    Returns 0.0 (neutral, not rewarded) if either side is constant.
    """
    if len(y_true) < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return 0.0
    ic, _ = spearmanr(y_true, y_pred)
    return float(ic) if np.isfinite(ic) else 0.0


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))
