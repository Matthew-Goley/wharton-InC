"""Real Data Layer: reads the fetched wharton-InC price CSVs into a FeatureTable.

Feature engineering mirrors MockDataProvider's shape exactly (same 7
columns) so nothing downstream needs to know which provider is in use.

Two correctness rules taken directly from the wharton-InC README's
"Gotchas" and CLAUDE.md's "no lookahead" rule:

- Uses ``Close``, never ``Adj Close``, for every feature and the label.
  ``Adj Close`` is retroactively rewritten on every dividend event, so a
  2018 row's value is not what was actually knowable in 2018 — training on
  it leaks future information into the past.
- Point-in-time universe: each ticker's rows are filtered to dates it was
  actually an S&P 500 member (an as-of join against
  ``data/processed/sp500_membership.csv``), so training doesn't apply
  today's roster backward across history (survivorship bias).

Sector split is NOT available yet — no sector classification data has been
sourced (``config/sector_map.py`` doesn't exist, per CLAUDE.md's plan).
Every row gets ``sector="Unknown"``, which means a sector expert
(tech/financials/real_estate) would train on zero rows. Build a
``BaselineConfig`` with ``experts=(ExpertSpec("broad", None),)`` when using
this provider until sector data exists.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.data_layer.interface import DataProvider, FeatureTable

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAINING_DIR = REPO_ROOT / "data" / "raw" / "training"
MEMBERSHIP_CSV = REPO_ROOT / "data" / "processed" / "sp500_membership.csv"

FEATURE_COLUMNS: list[str] = ["ma_5", "ma_20", "ma_60", "volatility_20", "atr_14", "momentum_10", "momentum_30"]
TARGET_COLUMN = "fwd_return"
UNKNOWN_SECTOR = "Unknown"

#: Rolling windows go up to 60 days; shorter histories would be nothing but
#: min_periods=1 fallback for their entire life, not real features.
_MIN_TICKER_ROWS = 60


class RealDataProvider(DataProvider):
    def __init__(
        self,
        universe: Optional[list[str]] = None,
        data_dir: Path = TRAINING_DIR,
        membership_csv: Path = MEMBERSHIP_CSV,
    ):
        if universe is None:
            from config.training_universe import TRAINING_UNIVERSE  # local import: avoid a hard dependency at module load
            universe = TRAINING_UNIVERSE
        self.universe = universe
        self.data_dir = Path(data_dir)
        self.membership_csv = Path(membership_csv)
        self._membership = self._load_membership()
        #: Sorted observation dates — the dates the membership file actually
        #: records. Used to resolve an arbitrary date to the roster in force.
        self._observation_dates = np.sort(self._membership["date"].unique())
        #: Exact (date, ticker) membership pairs, for O(1) lookup.
        self._member_index = pd.MultiIndex.from_frame(self._membership[["date", "ticker"]])
        #: Populated by the most recent get_feature_table() call — tickers
        #: whose price file was missing (delisted, acquired, no successor;
        #: see wharton-InC README "Gotchas" — ~8% of the universe by design).
        self.skipped_tickers: list[str] = []

    def _load_membership(self) -> pd.DataFrame:
        df = pd.read_csv(self.membership_csv, parse_dates=["date"])
        # `ticker` is the current symbol and matches the price filenames;
        # `ticker_asof` (the symbol in use at the time) is the historical
        # record and deliberately not used as a join key here.
        df = df[["date", "ticker"]].drop_duplicates()
        return df.sort_values(["date", "ticker"]).reset_index(drop=True)

    def get_feature_table(self, start: str, end: str, hold_days: int) -> FeatureTable:
        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
        self.skipped_tickers = []

        rows = []
        for ticker in self.universe:
            path = self.data_dir / f"{ticker}_1d.csv"
            if not path.exists():
                self.skipped_tickers.append(ticker)
                continue
            ticker_frame = self._build_ticker_features(ticker, path, hold_days)
            if ticker_frame is not None:
                rows.append(ticker_frame)

        if not rows:
            raise ValueError(f"No usable ticker data found under {self.data_dir}")

        frame = pd.concat(rows)
        timestamps = frame.index.get_level_values("timestamp")
        frame = frame[(timestamps >= start_ts) & (timestamps <= end_ts)]
        frame = self._filter_point_in_time(frame)
        frame = frame.sort_index()

        return FeatureTable(
            frame=frame,
            feature_columns=FEATURE_COLUMNS,
            target_column=TARGET_COLUMN,
            target_horizon_days=hold_days,
        )

    def _build_ticker_features(self, ticker: str, path: Path, hold_days: int) -> Optional[pd.DataFrame]:
        raw = pd.read_csv(path, index_col="timestamp", parse_dates=["timestamp"]).sort_index()
        if len(raw) < _MIN_TICKER_ROWS:
            self.skipped_tickers.append(ticker)
            return None

        close = raw["Close"]
        prev_close = close.shift(1)

        ma_5 = close.rolling(5, min_periods=1).mean()
        ma_20 = close.rolling(20, min_periods=1).mean()
        ma_60 = close.rolling(60, min_periods=1).mean()
        volatility_20 = close.pct_change().rolling(20, min_periods=2).std().fillna(0.0)

        # True Range (standard definition, now that real O/H/L/C are available):
        # max(high-low, |high-prev_close|, |low-prev_close|).
        true_range = pd.concat(
            [raw["High"] - raw["Low"], (raw["High"] - prev_close).abs(), (raw["Low"] - prev_close).abs()],
            axis=1,
        ).max(axis=1)
        atr_14 = true_range.rolling(14, min_periods=1).mean()

        momentum_10 = close.pct_change(10).fillna(0.0)
        momentum_30 = close.pct_change(30).fillna(0.0)

        # N-day forward return, N = hold_days (exit_rule_spec.md §2) — the
        # label a fixed-hold position's exit realizes, not next-bar return.
        fwd_return = close.pct_change(hold_days).shift(-hold_days)

        df = pd.DataFrame(
            {
                "ma_5": ma_5, "ma_20": ma_20, "ma_60": ma_60,
                "volatility_20": volatility_20, "atr_14": atr_14,
                "momentum_10": momentum_10, "momentum_30": momentum_30,
                TARGET_COLUMN: fwd_return,
                "sector": UNKNOWN_SECTOR,
            }
        )
        # The last hold_days rows have no realized forward return yet — not
        # a feature gap to fill, a label that genuinely doesn't exist.
        df = df.dropna(subset=[TARGET_COLUMN])
        if df.empty:
            return None
        df.index = pd.MultiIndex.from_product([[ticker], df.index], names=["ticker", "timestamp"])
        return df

    def _filter_point_in_time(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Keep only rows where the ticker was an S&P 500 member on that date.

        sp500_membership.csv records a row only when membership *changes*, so
        a row means "this was the roster as of this date, and stayed so until
        the next row." Resolving an arbitrary date therefore takes two steps,
        and the order matters:

          1. Resolve the date to the latest observation at or before it. This
             is the as-of part, and it runs on the DATE AXIS ALONE.
          2. Require an exact (observation_date, ticker) membership row.

        Doing the as-of join per ticker instead — ``merge_asof(...,
        by="ticker", direction="backward")`` — is subtly wrong, and was the
        original bug here. When a company leaves the index it simply stops
        having rows, so a per-ticker backward join keeps matching its FINAL
        row forever and reports it as a member for the rest of history. Such
        a join can only fail *before* a ticker's first row, never *after* its
        last: it strips pre-entry rows but no post-exit ones. Measured on the
        real data, that leaked thousands of rows per departed company (AMG
        left the index in 2019 and kept 1,517 rows through 2025).

        Resolving the date globally first and then demanding an exact pair
        closes the hole: once a company is dropped, the observation in force
        no longer lists it, so the pair lookup fails as it should.
        """
        idx = frame.index.to_frame(index=False)
        dates = idx["timestamp"].to_numpy(dtype="datetime64[ns]")

        # Latest observation at or before each row's date. pos == -1 means the
        # row predates every observation, so membership is simply unknown —
        # treated as "not a member" rather than guessed at.
        pos = np.searchsorted(self._observation_dates, dates, side="right") - 1
        resolved = np.full(len(dates), np.datetime64("NaT"), dtype="datetime64[ns]")
        known = pos >= 0
        resolved[known] = self._observation_dates[pos[known]]

        probe = pd.MultiIndex.from_arrays(
            [resolved, idx["ticker"].to_numpy()], names=["date", "ticker"]
        )
        # NaT never matches, so unknown-membership rows drop out here too.
        return frame[probe.isin(self._member_index)]
