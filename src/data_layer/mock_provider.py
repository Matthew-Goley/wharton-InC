"""Synthetic stand-in for the real Data Layer.

Generates a plausible-looking feature table (moving averages, volatility,
momentum, sector tags — matching the shape architecture §1 promises) from a
random walk, purely so experts/GA/combiner/selection can be built, run, and
tested before the real Yahoo Finance pipeline exists. Swap
``MockDataProvider`` for a real ``DataProvider`` implementation without
touching any other module.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_layer.interface import DataProvider, FeatureTable

FEATURE_COLUMNS: list[str] = [
    "ma_5",
    "ma_20",
    "ma_60",
    "volatility_20",
    "atr_14",
    "momentum_10",
    "momentum_30",
]
TARGET_COLUMN = "fwd_return"


class MockDataProvider(DataProvider):
    """Random-walk synthetic universe with sector tags.

    Not a model of real markets — just internally consistent enough
    (features genuinely correlated with the forward-return target) that GA
    fitness and expert training exercise real learning dynamics rather than
    pure noise.
    """

    def __init__(
        self,
        n_tickers_per_sector: int = 15,
        n_periods: int = 2000,
        seed: int = 0,
    ) -> None:
        self.n_tickers_per_sector = n_tickers_per_sector
        self.n_periods = n_periods
        self.seed = seed
        self._sectors = ["Technology", "Financials", "Real Estate"]

    def get_feature_table(self, start: str, end: str, hold_days: int) -> FeatureTable:
        rng = np.random.default_rng(self.seed)
        timestamps = pd.date_range(start=start, end=end, periods=self.n_periods)

        rows = []
        ticker_id = 0
        for sector in self._sectors:
            # Each sector gets its own latent factor so sector experts have
            # something real to pick up on.
            sector_factor = rng.normal(scale=0.15, size=self.n_periods)
            for _ in range(self.n_tickers_per_sector):
                ticker = f"{sector[:3].upper()}{ticker_id:03d}"
                ticker_id += 1
                rows.append(
                    self._simulate_ticker(rng, ticker, sector, sector_factor, timestamps, hold_days)
                )

        frame = pd.concat(rows).sort_index()
        return FeatureTable(
            frame=frame,
            feature_columns=FEATURE_COLUMNS,
            target_column=TARGET_COLUMN,
            target_horizon_days=hold_days,
        )

    def _simulate_ticker(
        self,
        rng: np.random.Generator,
        ticker: str,
        sector: str,
        sector_factor: np.ndarray,
        timestamps: pd.DatetimeIndex,
        hold_days: int,
    ) -> pd.DataFrame:
        n = len(timestamps)
        idio = rng.normal(scale=0.2, size=n)
        log_returns = 0.4 * sector_factor + idio
        price = 100 * np.exp(np.cumsum(log_returns) * 0.01)

        price_s = pd.Series(price, index=timestamps)
        ma_5 = price_s.rolling(5, min_periods=1).mean()
        ma_20 = price_s.rolling(20, min_periods=1).mean()
        ma_60 = price_s.rolling(60, min_periods=1).mean()
        volatility_20 = price_s.pct_change().rolling(20, min_periods=2).std().fillna(0.0)
        atr_14 = price_s.diff().abs().rolling(14, min_periods=1).mean()
        momentum_10 = price_s.pct_change(10).fillna(0.0)
        momentum_30 = price_s.pct_change(30).fillna(0.0)

        # Forward return is genuinely (weakly) predictable from momentum +
        # the sector factor, so a model has a real signal to find. Shifted
        # by hold_days: the label is the N-day forward return, per
        # exit_rule_spec.md §2, not always next-period. Noise scaled by
        # sqrt(hold_days) as a simple stand-in for variance accumulating
        # over a longer holding window.
        noise = rng.normal(scale=0.02 * np.sqrt(hold_days), size=n)
        fwd_return = (
            0.05 * momentum_10.to_numpy()
            + 0.10 * sector_factor
            - 0.01 * volatility_20.to_numpy()
            + noise
        )
        fwd_return = pd.Series(fwd_return, index=timestamps).shift(-hold_days).fillna(0.0)

        df = pd.DataFrame(
            {
                "ma_5": ma_5,
                "ma_20": ma_20,
                "ma_60": ma_60,
                "volatility_20": volatility_20,
                "atr_14": atr_14,
                "momentum_10": momentum_10,
                "momentum_30": momentum_30,
                TARGET_COLUMN: fwd_return,
                "sector": sector,
            }
        )
        df.index = pd.MultiIndex.from_product(
            [[ticker], timestamps], names=["ticker", "timestamp"]
        )
        return df
