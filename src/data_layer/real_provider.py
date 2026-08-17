"""Real Data Layer: reads the fetched wharton-InC price CSVs into a FeatureTable.

Every feature here is **scale-free** and the target is **cross-sectionally
demeaned**. Both are deliberate, and both replaced earlier versions that
measured as pure noise:

- The first cut used raw ``ma_5``/``ma_20``/``ma_60``/``atr_14``, which are
  dollar amounts. Pooled across the universe, ``ma_20`` ran from $12 to
  $96,938 on a single day, so those columns mostly encoded "how expensive is
  one share" — which predicts nothing. Per-date IC t-statistics were +0.92,
  +1.10, +1.27, +1.13: indistinguishable from zero. Re-expressed as ratios,
  the same underlying information becomes usable (``ma_5/ma_20 - 1`` went
  from t=+0.92 to t=-2.84). Trees are invariant to monotone transforms of a
  *single* feature, but not to pooling different price regimes into one
  column, which is what was happening.
- The target is each name's forward return minus that date's universe mean.
  About a third of the raw forward return's variance is the whole market
  moving together, which nothing here can forecast; removing it lets the
  model spend its capacity on relative performance, which is what the
  selector actually acts on. ``fwd_return_raw`` and ``market_fwd_return`` are
  kept alongside for reporting and cost checks.

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

FEATURE_COLUMNS: list[str] = [
    "px_to_ma_20",     # close vs its 20d average — distance from trend
    "ma_5_over_20",    # short vs medium trend
    "ma_20_over_60",   # medium vs long trend
    "volatility_20",   # already unitless (std of returns)
    "atr_pct",         # true range as a fraction of price
    "momentum_10",     # already unitless (pct change)
    "momentum_30",
    "jump_ratio",      # today's move in jump-robust local sigma (see below)
]
TARGET_COLUMN = "fwd_return"
#: Kept in the frame but never fed to a model — the un-demeaned label and the
#: market component removed from it, for reporting and cost-model checks.
RAW_TARGET_COLUMN = "fwd_return_raw"
MARKET_TARGET_COLUMN = "market_fwd_return"
UNKNOWN_SECTOR = "Unknown"

# --- jump ratio (Khashanah, Chen, Buckle & Hawkes, JRSS-C 2025) -------------
#: Window of consecutive absolute returns the running median is taken over.
#: The paper's Med_k estimator generalizes Andersen, Dobrev & Schaumburg's
#: MedRV (k=3) to k = 3, 5, 7, 9; a block of k can absorb up to (k-1)//2 large
#: returns, so k=9 survives a cluster of four. The paper settles on k=9 for
#: the S&P 500, which is the same index this universe is drawn from.
JUMP_MED_K = 9
#: Trailing window the local medians are averaged over to get a daily sigma.
#: The paper averages across one trading day of 2-min bars; on daily bars the
#: analogous "recent regime" span is a quarter.
JUMP_VOL_WINDOW = 63
#: c_k, chosen so Med_k is an unbiased estimator of realized volatility when
#: returns are Gaussian (paper §3). c_3 is the classic MedRV constant
#: pi/(6-4*sqrt(3)+pi).
MED_K_SCALE: dict[int, float] = {3: 1.41865, 5: 1.62360, 7: 1.74332, 9: 1.82184}

#: Rolling windows go up to 60 days and the jump sigma needs JUMP_VOL_WINDOW
#: more on top; shorter histories would be nothing but min_periods fallback
#: for their entire life, not real features.
_MIN_TICKER_ROWS = 60 + JUMP_VOL_WINDOW

#: Floor for the jump-ratio denominator, so a dead-flat stretch (sigma == 0)
#: cannot produce an infinite ratio.
_MIN_JUMP_SIGMA = 1e-8


def jump_ratio(
    close: pd.Series,
    k: int = JUMP_MED_K,
    window: int = JUMP_VOL_WINDOW,
) -> pd.Series:
    """Today's log return in units of jump-robust local volatility.

    Implements the Med_k standardized return of Khashanah, Chen, Buckle &
    Hawkes (JRSS-C 2025, §3), adapted from their intraday setting to daily
    bars. Their estimator of local volatility is a running median of ``k``
    consecutive absolute returns, scaled by ``c_k`` so it is unbiased for
    Gaussian returns:

        Med_k(t) = c_k * median(|r_{t-k+1}|, ..., |r_t|)^2      (variance)
        sigma(t) = sqrt( mean of Med_k over the trailing window )
        ratio(t) = r_t / sigma(t-1)

    Why a median rather than the usual bipower variation: bipower multiplies
    *adjacent* absolute returns, so a cluster of large moves inflates the very
    denominator meant to reveal them, and the jump masks itself. That is the
    paper's central complaint about the RV-BV, LM and ABD tests — on the 2010
    flash crash, six consecutive large returns pushed bipower variation above
    realized variance, reporting *no jumps* on the most jump-heavy day in the
    sample. A median over k=9 absorbs up to four large returns in the block
    and keeps pointing at the real local volatility.

    Two deliberate departures from the paper, both because this is a
    *predictive feature* rather than an *ex-post detector*:

    1. The window is trailing, not centred. The paper's Med_k(t,j) is centred
       on interval j and its daily sigma averages every return in the day,
       including ones after j. Here that would read the future, which
       CLAUDE.md forbids outright.
    2. The denominator is lagged one period (``sigma.shift(1)``), so a jump
       never contributes to the volatility it is measured against. The median
       already makes this contamination small; shifting removes it entirely
       and costs nothing.

    The result is kept **signed**, not absolute. The paper thresholds |ratio|
    because it only asks "was this a jump?"; a model also wants to know which
    way, since a -4 sigma move and a +4 sigma move imply opposite trades.

    Leading values are NaN until enough history accumulates. They are left as
    NaN rather than filled with 0.0, because "no jump" and "unknown" are
    different claims and XGBoost routes missing values natively.
    """
    if k not in MED_K_SCALE:
        raise ValueError(f"no Med_k scale constant for k={k}; have {sorted(MED_K_SCALE)}")

    log_return = np.log(close).diff()
    local_median = log_return.abs().rolling(k).median()
    local_variance = MED_K_SCALE[k] * local_median.pow(2)
    # min_periods at half the window so a ticker isn't blind for a full
    # quarter; a sigma from ~30 observations is noisier but usable.
    sigma = np.sqrt(local_variance.rolling(window, min_periods=window // 2).mean())
    return log_return / sigma.shift(1).clip(lower=_MIN_JUMP_SIGMA)


class RealDataProvider(DataProvider):
    def __init__(
        self,
        universe: Optional[list[str]] = None,
        data_dir: Path = TRAINING_DIR,
        membership_csv: Path = MEMBERSHIP_CSV,
        demean_target: bool = True,
    ):
        #: When True (the default) the label is excess return over that date's
        #: universe mean. See _demean_target for what that changes downstream.
        self.demean_target = demean_target
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
        # After the point-in-time filter, so the "market" is the roster that
        # was actually investable that day rather than every name on file.
        frame = self._demean_target(frame)

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

        # Everything below is a ratio. The moving averages and ATR above are
        # intermediates in dollars and never leave this method — pooled across
        # a universe spanning $12 to $96,938 a share they carry no signal (see
        # the module docstring).
        df = pd.DataFrame(
            {
                "px_to_ma_20": close / ma_20 - 1.0,
                "ma_5_over_20": ma_5 / ma_20 - 1.0,
                "ma_20_over_60": ma_20 / ma_60 - 1.0,
                "volatility_20": volatility_20,
                "atr_pct": atr_14 / close,
                "momentum_10": momentum_10,
                "momentum_30": momentum_30,
                "jump_ratio": jump_ratio(close),
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

    def _demean_target(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Turn the label into excess return over that date's universe mean.

        Roughly a third of the raw forward return's variance is simply the
        whole market moving on a given day. No model built from one stock's
        own price history can forecast that, so leaving it in the label spends
        model capacity on an unwinnable problem and adds noise to every
        fitness measurement. Subtracting the cross-sectional mean leaves the
        part that is actually a stock-picking question: did this name beat its
        peers?

        This also lines the label up with how the system is scored and traded
        — ``cross_sectional_ic`` ranks within a date, and ``TradeSelector``
        takes the top N names on a date.

        DOWNSTREAM CONSEQUENCE, worth being explicit about: predictions are now
        *relative*. A name predicted to beat the universe by 0.5% can still
        lose money outright in a falling market, and this system is long-only
        with no hedge. The selection layer's cost filter
        (``cost_model.clears_cost``) consequently reads as "expected
        outperformance must clear round-trip costs", not "expected profit
        must". ``fwd_return_raw`` and ``market_fwd_return`` are carried in the
        frame so that check can be revisited against absolute returns without
        rebuilding the table. Pass ``demean_target=False`` to opt out.
        """
        if not self.demean_target or frame.empty:
            return frame
        raw = frame[TARGET_COLUMN]
        market = raw.groupby(level="timestamp").transform("mean")
        return frame.assign(
            **{
                RAW_TARGET_COLUMN: raw,
                MARKET_TARGET_COLUMN: market,
                TARGET_COLUMN: raw - market,
            }
        )

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
