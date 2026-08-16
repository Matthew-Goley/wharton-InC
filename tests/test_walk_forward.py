import numpy as np
import pandas as pd
import pytest

from src.config import WalkForwardConfig
from src.evaluation.walk_forward import generate_walk_forward_splits


def _panel(n_periods: int, n_tickers: int = 3) -> pd.DataFrame:
    timestamps = pd.date_range("2020-01-01", periods=n_periods, freq="h")
    tickers = [f"T{i}" for i in range(n_tickers)]
    index = pd.MultiIndex.from_product([tickers, timestamps], names=["ticker", "timestamp"])
    return pd.DataFrame({"value": np.arange(len(index))}, index=index)


def test_val_strictly_after_train():
    frame = _panel(n_periods=500)
    config = WalkForwardConfig(n_splits=3, min_train_periods=200, val_periods=50)
    for split in generate_walk_forward_splits(frame, config):
        train_max = split.train.index.get_level_values("timestamp").max()
        val_min = split.val.index.get_level_values("timestamp").min()
        assert train_max < val_min


def test_expanding_window_grows():
    frame = _panel(n_periods=500)
    config = WalkForwardConfig(n_splits=3, min_train_periods=200, val_periods=50, expanding=True)
    splits = list(generate_walk_forward_splits(frame, config))
    sizes = [s.train.index.get_level_values("timestamp").nunique() for s in splits]
    assert sizes == sorted(sizes)
    assert len(set(sizes)) > 1  # actually grows, not just non-decreasing


def test_rolling_window_stays_fixed_size():
    frame = _panel(n_periods=500)
    config = WalkForwardConfig(n_splits=3, min_train_periods=200, val_periods=50, expanding=False)
    splits = list(generate_walk_forward_splits(frame, config))
    sizes = {s.train.index.get_level_values("timestamp").nunique() for s in splits}
    assert sizes == {200}


def test_every_ticker_present_in_every_fold():
    frame = _panel(n_periods=400, n_tickers=4)
    config = WalkForwardConfig(n_splits=2, min_train_periods=150, val_periods=40)
    for split in generate_walk_forward_splits(frame, config):
        assert set(split.train.index.get_level_values("ticker")) == {"T0", "T1", "T2", "T3"}
        assert set(split.val.index.get_level_values("ticker")) == {"T0", "T1", "T2", "T3"}


def test_not_enough_history_yields_nothing():
    frame = _panel(n_periods=50)
    config = WalkForwardConfig(n_splits=3, min_train_periods=200, val_periods=50)
    assert list(generate_walk_forward_splits(frame, config)) == []


def test_respects_requested_n_splits_upper_bound():
    frame = _panel(n_periods=5000)
    config = WalkForwardConfig(n_splits=3, min_train_periods=200, val_periods=50)
    splits = list(generate_walk_forward_splits(frame, config))
    assert len(splits) <= 3
