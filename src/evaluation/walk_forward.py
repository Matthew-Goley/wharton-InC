"""Walk-forward train/validation splitting over panel (ticker, timestamp) data.

Splits are drawn on the *timestamp* axis only, then applied across every
ticker at once — this is what keeps the GA fitness function (architecture
§3: "train on past, test on unseen future") free of look-ahead leakage.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd

from src.config import WalkForwardConfig


@dataclass(frozen=True)
class WalkForwardSplit:
    train: pd.DataFrame
    val: pd.DataFrame


def generate_walk_forward_splits(
    frame: pd.DataFrame, config: WalkForwardConfig
) -> Iterator[WalkForwardSplit]:
    """Yield up to ``config.n_splits`` (train, val) folds, oldest-first.

    ``frame`` must carry a MultiIndex with a ``timestamp`` level (see
    :class:`src.data_layer.interface.FeatureTable`). Each fold's validation
    window is strictly after its training window; with ``expanding=True``
    each successive training window grows to include prior validation
    windows, otherwise it stays a fixed-size rolling window.

    Yields nothing if there isn't enough history for even one fold — callers
    should treat "no folds" as "fitness undefined" rather than crashing.
    """
    timestamps = frame.index.get_level_values("timestamp")
    unique_ts = np.sort(pd.unique(timestamps))
    n = len(unique_ts)

    min_train = config.min_train_periods
    val = config.val_periods
    available = n - min_train
    if available < val:
        return  # not enough history for a single fold

    max_folds = 1 + (available - val) // val
    n_folds = max(1, min(config.n_splits, max_folds))
    step = (available - val) // n_folds if n_folds > 1 else 0
    step = max(step, val)  # non-overlapping validation windows

    ts_level = frame.index.get_level_values("timestamp")
    for i in range(n_folds):
        train_end = min_train + i * step
        val_start = train_end
        val_end = val_start + val
        if val_end > n:
            break

        train_start = 0 if config.expanding else max(0, train_end - min_train)
        train_ts = unique_ts[train_start:train_end]
        val_ts = unique_ts[val_start:val_end]

        train_mask = ts_level.isin(train_ts)
        val_mask = ts_level.isin(val_ts)
        yield WalkForwardSplit(train=frame.loc[train_mask], val=frame.loc[val_mask])
