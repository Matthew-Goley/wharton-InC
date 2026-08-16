"""The contract the rest of the system is built against.

Architecture §1 (Data Layer) is explicitly out of scope for this pass. This
module defines the *shape* everything downstream (experts, GA, combiner,
trade selection) depends on, so that a real implementation — Yahoo Finance
pull, Parquet cache, cleaning, feature engineering — can be dropped in later
by writing one class that satisfies ``DataProvider``. Nothing outside
``src/data_layer`` should need to change when that happens.

Use ``src.data_layer.MockDataProvider`` in the meantime: it satisfies the
same interface with synthetic data so the training pipeline is runnable and
testable today.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class FeatureTable:
    """One unified feature table plus the metadata every expert needs.

    ``frame`` is indexed by a (ticker, timestamp) MultiIndex and holds:
      - the feature columns listed in ``feature_columns``
      - the training target in ``target_column`` (an N-day forward return,
        N = ``target_horizon_days``)
      - a ``sector`` column (ticker's sector tag, per architecture §1)
    """

    frame: pd.DataFrame
    feature_columns: list[str]
    target_column: str
    #: N in "N-day forward return" — what the target actually represents.
    #: Must equal the selector's ``hold_days`` (exit_rule_spec.md §2, §9);
    #: ``train_baseline`` asserts this rather than assuming it, since a
    #: cached/precomputed feature table could silently drift out of sync
    #: with a changed config.
    target_horizon_days: int
    sector_column: str = "sector"

    def sector_map(self) -> dict[str, str]:
        """ticker -> sector, derived from the frame."""
        tickers = self.frame.index.get_level_values(0)
        sectors = self.frame[self.sector_column]
        return dict(zip(tickers, sectors, strict=True))

    def slice_for(self, sector: str | None) -> pd.DataFrame:
        """Rows for one sector, or the full universe when ``sector`` is None."""
        if sector is None:
            return self.frame
        return self.frame[self.frame[self.sector_column] == sector]


class DataProvider(ABC):
    """Abstract source of clean, feature-engineered, ticker+timestamp data.

    A concrete implementation owns everything in architecture §1: sourcing
    bars, adjusting for splits/dividends/delistings, computing features, and
    caching the result in a columnar format. Callers never see any of that —
    they just get a :class:`FeatureTable` back.
    """

    @abstractmethod
    def get_feature_table(self, start: str, end: str, hold_days: int) -> FeatureTable:
        """Return the unified feature table for the given date range.

        ``hold_days`` sets the training label's horizon (N-day forward
        return) — it is not a post-hoc filter, it changes what the target
        column *means*, per exit_rule_spec.md §2. The returned table's
        ``target_horizon_days`` must equal it.
        """
        raise NotImplementedError
