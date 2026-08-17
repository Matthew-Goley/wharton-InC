"""Scoring functions used as GA fitness and for reporting."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

#: A date needs at least this many names before its cross-sectional IC means
#: anything. Ranking three stocks against each other is mostly noise.
MIN_NAMES_PER_DATE = 10


def information_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Pooled Spearman rank correlation, over every row at once.

    Kept for reporting and comparison. NOT used as GA fitness -- see
    :func:`cross_sectional_ic` for why pooling is the wrong question.

    Returns 0.0 (neutral, not rewarded) if either side is constant.
    """
    if len(y_true) < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return 0.0
    ic, _ = spearmanr(y_true, y_pred)
    return float(ic) if np.isfinite(ic) else 0.0


def cross_sectional_ic(
    dates: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    min_names: int = MIN_NAMES_PER_DATE,
) -> float:
    """Mean per-date Spearman IC -- the metric that matches how we trade.

    Computes the rank correlation *within each date*, then averages across
    dates. This asks "on a given day, did the model rank that day's stocks in
    the order they actually performed?", which is exactly what the selector
    needs: it picks the top N names on one date, and never compares a name on
    Tuesday against a different name on Friday.

    Pooling every (ticker, date) row into one correlation instead — what
    :func:`information_coefficient` does — mixes that question with a second,
    untradeable one: "which DAYS were up days?". About a third of the forward
    return's variance is the whole market moving together on a given day, and
    a model cannot forecast that. Measured on the real universe the two
    metrics disagree even in sign: `volatility_20` scores +0.0386 pooled but
    +0.0027 per-date, so a search optimizing the pooled number chases
    market-regime structure that produces no stock-picking edge.

    Dates with fewer than ``min_names`` names, or with no variation on either
    side, are skipped rather than scored 0 — they carry no information, and
    counting them as zeros would drag a real signal toward nothing.

    Returns 0.0 if no date qualifies.
    """
    frame = pd.DataFrame({"date": dates, "y": y_true, "p": y_pred})
    frame = frame[np.isfinite(frame["y"]) & np.isfinite(frame["p"])]
    if frame.empty:
        return 0.0

    counts = frame.groupby("date", sort=False)["y"].transform("size")
    frame = frame[counts >= min_names]
    if frame.empty:
        return 0.0

    grouped = frame.groupby("date", sort=False)
    # Spearman == Pearson on ranks, and ranking per date lets the whole thing
    # run as grouped sums instead of one scipy call per date. At ~900 genomes
    # x 5 folds that difference is minutes, not microseconds.
    frame = frame.assign(ry=grouped["y"].rank(), rp=grouped["p"].rank())
    frame["ry_rp"] = frame["ry"] * frame["rp"]
    frame["ry_sq"] = frame["ry"] ** 2
    frame["rp_sq"] = frame["rp"] ** 2

    agg = frame.groupby("date", sort=False).agg(
        n=("ry", "size"),
        sum_x=("ry", "sum"),
        sum_y=("rp", "sum"),
        sum_xy=("ry_rp", "sum"),
        sum_xx=("ry_sq", "sum"),
        sum_yy=("rp_sq", "sum"),
    )

    cov = agg["sum_xy"] - agg["sum_x"] * agg["sum_y"] / agg["n"]
    var_x = agg["sum_xx"] - agg["sum_x"] ** 2 / agg["n"]
    var_y = agg["sum_yy"] - agg["sum_y"] ** 2 / agg["n"]

    denominator = np.sqrt(var_x * var_y)
    # denominator == 0 means one side was constant that date (every name tied,
    # or a collapsed model predicting the same value for everything).
    per_date = (cov / denominator).replace([np.inf, -np.inf], np.nan).dropna()
    if per_date.empty:
        return 0.0
    return float(per_date.mean())


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))
