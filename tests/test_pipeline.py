"""Covers exit_rule_spec.md §8 test case 9: label horizon must match hold_days."""
import pandas as pd
import pytest

from src.config import BaselineConfig
from src.data_layer.interface import DataProvider, FeatureTable
from src.pipeline.train_baseline import train_baseline


class _MismatchedHorizonProvider(DataProvider):
    """Ignores the requested hold_days and always claims a different horizon
    — stands in for a stale cached feature table (the scenario the assert
    in train_baseline exists to catch)."""

    def get_feature_table(self, start: str, end: str, hold_days: int) -> FeatureTable:
        index = pd.MultiIndex.from_tuples([("AAA", pd.Timestamp("2020-01-01"))], names=["ticker", "timestamp"])
        frame = pd.DataFrame({"f1": [0.0], "fwd_return": [0.0], "sector": ["Technology"]}, index=index)
        return FeatureTable(
            frame=frame,
            feature_columns=["f1"],
            target_column="fwd_return",
            target_horizon_days=hold_days + 1,  # deliberately wrong
        )


def test_label_horizon_mismatch_raises_loudly():
    config = BaselineConfig()
    with pytest.raises(ValueError, match="label horizon mismatch"):
        train_baseline(_MismatchedHorizonProvider(), config, start="2020-01-01", end="2020-02-01")
