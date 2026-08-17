"""Comparing two TradeSelector outputs — e.g. before/after a scoring change
— as a 3-way decomposition (common / only-in-a / only-in-b), not a headline
mean-return delta.

Motivated by a real miscall: comparing the agreement-bonus scale fix's
before/after backtests by aggregate mean return/trade suggested a real
improvement (+0.25% -> +0.43%/trade). Decomposing showed only 33% of entries
were common to both scorings — the fix reshuffled two-thirds of the entry
list, not a narrow rank-1 substitution — and the mean difference between the
two *differing* subsets wasn't distinguishable from noise at any plausible
effective sample size (block-effective t~0.2, t~0.8 even in the maximally
generous raw-n/no-overlap-correction case). A scoring change is far more
disruptive to what actually gets traded than a "ranking improved" headline
implies, and the aggregate delta conflates a large, low-power comparison
(the differing trades) with a large amount of literally-identical data (the
common trades, which by definition can't move either scoring's mean).

Every future scoring-change comparison (the full sector roster, the
uncertainty_adjusted ranking mode, the agreement-bonus on/off ablation)
should run through this rather than a headline mean.

Significance uses *block-effective* n, not raw trade count: entries within
one hold_days-wide window overlap in time and aren't independent draws, so
their count overstates precision. Counting distinct non-overlapping
hold_days-wide trading-day blocks each subset actually touches is cruder
than a proper long-run-variance estimate, but unlike a flat
``raw_n / hold_days`` deflation it reflects how concentrated or spread out
each subset's trades really are in calendar time — which turned out to
matter on its own: two subsets of equal raw n can span very different block
counts, and *that* difference is a real, exactly-countable finding
(a composition fact, not a noisy return estimate) independent of whether
either subset's mean return is statistically meaningful.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _group_stats(idx: pd.MultiIndex, raw_returns: pd.Series) -> dict:
    if len(idx) == 0:
        return {"n": 0, "mean_return": None, "median_return": None, "win_rate": None}
    r = raw_returns.loc[idx]
    return {
        "n": int(len(idx)),
        "mean_return": float(r.mean()),
        "median_return": float(r.median()),
        "win_rate": float((r > 0).mean()),
    }


def _distinct_blocks(idx: pd.MultiIndex, day_block: dict) -> int:
    if len(idx) == 0:
        return 0
    return len({day_block[t] for t in idx.get_level_values("timestamp")})


def decompose_selection_change(
    entries_a: pd.DataFrame,
    entries_b: pd.DataFrame,
    raw_returns: pd.Series,
    all_trading_days: np.ndarray,
    hold_days: int,
) -> dict:
    """3-way decomposition of two :class:`TradeSelector` entry sets: trades
    common to both, only in ``entries_a``, only in ``entries_b``.

    ``entries_a``/``entries_b`` are entry-only rows (``action == "entry"``)
    from two runs of :meth:`TradeSelector.run`, indexed by
    ``(ticker, timestamp)``. ``raw_returns`` must be indexed the same way and
    cover every entry in either set — typically
    ``FeatureTable.frame[RAW_TARGET_COLUMN]`` for the window both runs were
    scored on.

    Returns a dict with ``common``/``only_a``/``only_b`` stats (n, mean,
    median, win_rate; ``only_a``/``only_b`` additionally get
    ``distinct_blocks`` and ``trades_per_active_block``, the composition
    facts that survive even when the return comparison doesn't), plus
    ``significance_only_a_vs_only_b`` — block-effective SE and t-stat for
    whether the two differing subsets' mean returns are actually
    distinguishable, or ``None`` if either subset is empty.
    """
    all_days = np.sort(np.asarray(all_trading_days))
    day_block = {d: i // hold_days for i, d in enumerate(all_days)}
    n_blocks_total = int(np.ceil(len(all_days) / hold_days))

    idx_a, idx_b = entries_a.index, entries_b.index
    common = idx_a[idx_a.isin(idx_b)]
    only_a = idx_a[~idx_a.isin(idx_b)]
    only_b = idx_b[~idx_b.isin(idx_a)]

    stats_only_a = _group_stats(only_a, raw_returns)
    stats_only_b = _group_stats(only_b, raw_returns)
    blocks_a = _distinct_blocks(only_a, day_block)
    blocks_b = _distinct_blocks(only_b, day_block)
    stats_only_a["distinct_blocks"] = blocks_a
    stats_only_a["trades_per_active_block"] = (len(only_a) / blocks_a) if blocks_a else None
    stats_only_b["distinct_blocks"] = blocks_b
    stats_only_b["trades_per_active_block"] = (len(only_b) / blocks_b) if blocks_b else None

    significance = None
    if len(only_a) and len(only_b):
        r_a, r_b = raw_returns.loc[only_a], raw_returns.loc[only_b]
        se_a = r_a.std() / np.sqrt(max(blocks_a, 1))
        se_b = r_b.std() / np.sqrt(max(blocks_b, 1))
        se_diff = float(np.sqrt(se_a**2 + se_b**2))
        diff = float(r_b.mean() - r_a.mean())
        significance = {
            "mean_diff_b_minus_a": diff,
            "block_effective_se": se_diff,
            "block_effective_t": (diff / se_diff) if se_diff > 0 else None,
        }

    return {
        "n_blocks_total": n_blocks_total,
        "common": _group_stats(common, raw_returns),
        "only_a": stats_only_a,
        "only_b": stats_only_b,
        "significance_only_a_vs_only_b": significance,
    }
