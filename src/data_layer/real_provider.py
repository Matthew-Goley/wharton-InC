"""Real Data Layer: reads the fetched wharton-InC price CSVs into a FeatureTable.

Features are chosen by documented effect, not by measured strength. Selecting
on t-statistics from this data was tried and measured *worse* than no
selection at all: a set picked by |dev t| > 2.5 scored S/N 0.75 out of
sample, while a set assembled purely from the anomalies literature scored
3.78 on the same holdout. The strongest in-sample features were largely the
ones that flipped sign out of sample. Add a feature here because a paper says
it should work, then let the walk-forward tell you if it did.

Every feature is **scale-free** — invariant to share price — except
``log_dollar_volume`` and ``amihud_20``, where the dollar amount is the
economic quantity (size, and price impact per dollar traded) rather than a
unit artefact. The target is **cross-sectionally demeaned**. Both rules
replaced earlier versions that measured as pure noise:

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
  today's roster backward across history (survivorship bias). This is a
  TRAINING-universe concept only -- it says nothing about Wharton's
  approved list, which was never part of the S&P 500's history to begin
  with. Live scoring must construct this provider with
  ``apply_point_in_time_filter=False`` (also skips loading
  ``sp500_membership.csv`` entirely -- a live-scoring run has no reason to
  need that file on disk), not feed it a small ``universe`` and expect the
  filter to pass everything through by accident.

Sector comes from ``config/sector_map.py`` (``SECTOR_MAP``, ticker -> GICS-
normalized sector name, built by ``src.data_pipeline.fetch_sector_map`` from
yfinance's ``.info['sector']``). Not point-in-time — a ticker's *current*
sector is applied across its whole price history, the same simplification
already made for the universe list itself. 70 of 653 universe tickers have
no resolved sector (mostly delisted/acquired names with no live yfinance
metadata) and fall back to ``UNKNOWN_SECTOR``, which never matches any
sector expert's filter, so those rows are broad-only — the same behavior
every ticker had before the map existed.
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
EARNINGS_DATES_CACHE = REPO_ROOT / "data" / "raw" / "reference" / "earnings_dates_raw.csv"


def _load_earnings_events() -> dict[str, pd.DataFrame]:
    """ticker -> sorted DataFrame(earnings_date, surprise_pct) of real
    earnings events from the cached fetch (src.data_pipeline.
    fetch_earnings_dates). Empty dict if that fetch hasn't run yet --
    every earnings-keyed feature is then just NaN/0 for every ticker, not
    an error.

    earnings_date is parsed by slicing the first 10 characters of the raw
    "YYYY-MM-DD HH:MM:SS+-HH:MM" string, NOT via pd.to_datetime(...,
    utc=True). Checked empirically before picking this: converting to UTC
    first shifts the calendar date for evening announcements that cross
    the UTC day boundary -- 460 of ~51,700 real rows in the actual cached
    fetch differ between the two methods. The string slice preserves the
    LOCAL (Eastern) calendar date exactly as yfinance reported it, with no
    timezone arithmetic at all.
    """
    if not EARNINGS_DATES_CACHE.exists():
        return {}
    df = pd.read_csv(EARNINGS_DATES_CACHE)
    df["earnings_date"] = pd.to_datetime(df["earnings_date"].astype(str).str[:10], errors="coerce")
    df = df.dropna(subset=["earnings_date"])
    events: dict[str, pd.DataFrame] = {}
    for ticker, group in df.groupby("ticker"):
        events[ticker] = group[["earnings_date", "surprise_pct"]].sort_values("earnings_date").reset_index(drop=True)
    return events

#: One or two representatives per documented cross-sectional effect. The set
#: is fixed by *concept*, chosen from the literature before scoring anything —
#: see the module docstring for why selecting on measured t-statistics turned
#: out to be actively worse than not selecting at all.
FEATURE_COLUMNS: list[str] = [
    # short-term reversal (Jegadeesh 1990)
    "px_to_ma_20",
    "momentum_5",
    # momentum, skipping the most recent stretch (Jegadeesh & Titman 1993)
    "momentum_60_skip_10",
    # proximity to the 52-week high (George & Hwang 2004)
    "pct_from_252d_high",
    # idiosyncratic volatility (Ang, Hodrick, Xing & Zhang 2006)
    "volatility_20",
    # good vs bad volatility (Bollerslev et al.)
    "semivol_skew",
    # lottery demand — MAX and MIN (Bali, Cakici & Whitelaw 2011)
    "max_ret_20",
    "min_ret_20",
    # higher moments of the return distribution (Amaya et al.)
    "realized_skew_20",
    "realized_kurt_20",
    # illiquidity (Amihud 2002) and size
    "amihud_20",
    "log_dollar_volume",
    # volume dynamics
    "volume_shock",
    "ret_vol_corr_20",
    # jump-robust standardized return (Khashanah et al. 2025)
    "jump_ratio",
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

#: Guard for ratio denominators that can legitimately reach zero.
_EPS = 1e-12


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
        normalize_features: bool = False,
        sector_map: Optional[dict[str, str]] = None,
        earnings_events: Optional[dict[str, pd.DataFrame]] = None,
        apply_point_in_time_filter: bool = True,
    ):
        #: ticker -> sector, defaulting to the real generated map. Injectable
        #: (matching universe/data_dir/membership_csv's existing pattern) so
        #: tests can exercise sector-keyed logic (sector_momentum_60_skip_10)
        #: against a small synthetic map instead of needing real tickers to
        #: happen to appear in config/sector_map.py.
        if sector_map is None:
            from config.sector_map import SECTOR_MAP  # local import: avoid a hard dependency at module load
            sector_map = SECTOR_MAP
        self.sector_map = sector_map
        #: ticker -> sorted DataFrame(earnings_date, surprise_pct), defaulting
        #: to the real cache (src.data_pipeline.fetch_earnings_dates).
        #: Injectable for the same reason sector_map is. Feeds three
        #: candidate features: has_earnings_data (a pure missingness
        #: indicator, tested alone first as a leakage gate -- open question
        #: 22), days_to_next_earnings, and earnings_surprise (see
        #: _earnings_features_for_ticker for the point-in-time contract
        #: each one has to satisfy).
        self.earnings_events = _load_earnings_events() if earnings_events is None else earnings_events
        #: When True (the default) the label is excess return over that date's
        #: universe mean. See _demean_target for what that changes downstream.
        self.demean_target = demean_target
        #: When True, every feature is replaced by its cross-sectional
        #: percentile rank that day (see _normalize_features_cross_sectionally).
        #:
        #: TESTED AND REJECTED as a fix for argmax concentration (one name
        #: dominating the daily #1-ranked slot) -- kept here, off by default,
        #: as a negative finding so it doesn't get re-tried without reading
        #: this. Gate result: PARA's daily-argmax share barely moved (49.3%
        #: -> 45.1%, ~9% of the way to what fair rotation implies) while
        #: out-of-sample IC dropped a real-looking +0.0209 -> +0.0141 (~4.6
        #: SE from zero down to ~3.1 SE). It CAN'T work, not just didn't
        #: this time: percentile rank is order-preserving within each
        #: feature -- a name that's the cross-section's most extreme value
        #: every day is still the most extreme value after rank-transforming
        #: (1.0 instead of a raw reading), the same corner of feature space,
        #: mapped to the same extreme prediction. Winsorized clipping fails
        #: the identical way (order-preserving in the interior, and an
        #: outlier past the clip bound still lands ON the bound, still
        #: top-ranked) -- don't try that either. No monotonic per-feature
        #: transform can fix persistent argmax dominance by a name that's a
        #: genuine, consistent extreme on the features that matter; the
        #: actual fix is TradingConfig.max_ticker_share_pct, which doesn't
        #: touch the model or cost any IC. Full writeup: baseline_system_
        #: architecture.md's ablation table, "Cross-sectional feature
        #: normalization" row.
        #:
        #: Off by default. Every feature here is already scale-free *within
        #: a ticker's own history* (a ratio, not a raw price level), but
        #: never made comparable *across tickers on the same day* -- a tree
        #: model sees PARA's raw feature magnitude, not its relative standing
        #: that day, and trees don't extrapolate past the range of values
        #: seen in training. Diagnosed from PARA being the single
        #: highest-scoring name in the whole universe on 245/497 test days
        #: (49.3%) — see baseline_system_architecture.md's ablation table.
        self.normalize_features = normalize_features
        if universe is None:
            from config.training_universe import TRAINING_UNIVERSE  # local import: avoid a hard dependency at module load
            universe = TRAINING_UNIVERSE
        self.universe = universe
        self.data_dir = Path(data_dir)
        self.membership_csv = Path(membership_csv)
        #: When True (the default), get_feature_table applies
        #: _filter_point_in_time (S&P 500 point-in-time membership) --
        #: correct, and required, for training on the historical universe
        #: (avoids survivorship bias). Set False for LIVE scoring against
        #: config/stock_list.py (Wharton's approved list): that list has no
        #: relationship to S&P 500 membership at all, so the filter isn't
        #: "off by default and rarely needed" there, it's simply the wrong
        #: question to ask of today's approved-list roster. Guards the
        #: membership-file load too (below), not just the filter call in
        #: get_feature_table -- a live-scoring provider has no reason to
        #: require sp500_membership.csv to even exist on disk.
        self.apply_point_in_time_filter = apply_point_in_time_filter
        if self.apply_point_in_time_filter:
            self._membership = self._load_membership()
            #: Sorted observation dates — the dates the membership file
            #: actually records. Used to resolve an arbitrary date to the
            #: roster in force.
            self._observation_dates = np.sort(self._membership["date"].unique())
            #: Exact (date, ticker) membership pairs, for O(1) lookup.
            self._member_index = pd.MultiIndex.from_frame(self._membership[["date", "ticker"]])
        else:
            self._membership = None
            self._observation_dates = np.array([], dtype="datetime64[ns]")
            self._member_index = pd.MultiIndex.from_arrays([[], []], names=["date", "ticker"])
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
        if self.apply_point_in_time_filter:
            frame = self._filter_point_in_time(frame)
        frame = frame.sort_index()
        # After the point-in-time filter (when applied), so every
        # cross-sectional operation below is computed against the roster
        # that was actually investable that day, not every name on file.
        # For a live-scoring provider (apply_point_in_time_filter=False)
        # every row on file for the requested window is investable by
        # construction -- it's already Wharton's approved list, not the
        # broad training universe -- so skipping the filter is correct,
        # not a shortcut.
        frame = self._add_sector_momentum(frame)
        frame = self._add_market_relative_features(frame)
        frame = self._normalize_features_cross_sectionally(frame)
        frame = self._demean_target(frame)

        return FeatureTable(
            frame=frame,
            feature_columns=FEATURE_COLUMNS,
            target_column=TARGET_COLUMN,
            target_horizon_days=hold_days,
        )

    def _earnings_features_for_ticker(
        self, ticker: str, dates: pd.DatetimeIndex
    ) -> tuple[pd.Series, pd.Series]:
        """days_to_next_earnings and earnings_surprise for one ticker's
        price dates -- CANDIDATE features (not in FEATURE_COLUMNS), via
        two asymmetric as-of joins against that ticker's known earnings
        events, each with a deliberately different point-in-time contract:

        - days_to_next_earnings: FORWARD as-of, same-day match ALLOWED --
          the next scheduled earnings date on or after this row's date.
          Not a lookahead: announcement dates are public knowledge well
          in advance of the event itself, so knowing "today is earnings
          day" as of today is legitimate -- the calendar fact was already
          known before today arrived. Also usable downstream as an
          event-risk flag (heightened volatility/gap risk), not just a
          return predictor, when this falls inside the hold_days window.
        - earnings_surprise: BACKWARD as-of, same-day match FORBIDDEN
          (allow_exact_matches=False) -- the most recent PRIOR earnings
          event's surprise. THE highest-risk lookahead surface in this
          feature set: forbidding the same-day match means a row can
          never see its own event's outcome, only a strictly earlier one,
          regardless of whether that day's announcement happened before
          or after market hours. See
          test_earnings_surprise_is_not_visible_before_its_own_
          announcement_date (tests/test_real_provider.py) for the
          explicit, deliberately adversarial regression test.

        NaN/NaT (not zero) for a ticker with no earnings data at all, or
        for dates outside the known earnings calendar's range (before the
        first known event, or after the last one on file) -- the same
        "unknown, not zero" convention as every other feature here.
        """
        events = self.earnings_events.get(ticker)
        if events is None or events.empty:
            nan = pd.Series(np.nan, index=dates)
            return nan.copy(), nan.copy()

        # merge_asof requires sorted keys -- `dates` comes from close.index,
        # already sorted ascending by raw.sort_index() upstream.
        left = pd.DataFrame({"timestamp": dates})
        next_event = pd.merge_asof(
            left, events[["earnings_date"]], left_on="timestamp", right_on="earnings_date",
            direction="forward", allow_exact_matches=True,
        )
        days_to_next = (next_event["earnings_date"] - next_event["timestamp"]).dt.days
        days_to_next.index = dates

        prior_event = pd.merge_asof(
            left, events, left_on="timestamp", right_on="earnings_date",
            direction="backward", allow_exact_matches=False,
        )
        surprise = prior_event["surprise_pct"]
        surprise.index = dates

        return surprise, days_to_next

    def _build_ticker_features(self, ticker: str, path: Path, hold_days: int) -> Optional[pd.DataFrame]:
        raw = pd.read_csv(path, index_col="timestamp", parse_dates=["timestamp"]).sort_index()
        if len(raw) < _MIN_TICKER_ROWS:
            self.skipped_tickers.append(ticker)
            return None

        close = raw["Close"]
        volume = raw["Volume"]
        returns = close.pct_change()
        earnings_surprise, days_to_next_earnings = self._earnings_features_for_ticker(ticker, close.index)
        ma_20 = close.rolling(20, min_periods=1).mean()
        #: CANDIDATE feature ingredients (not in FEATURE_COLUMNS) -- see
        #: px_to_ma_10's own note below FEATURE_COLUMNS for the horizon-
        #: match reasoning, and baseline_system_architecture.md open
        #: question 18 for the 5-day family-swap test these (and the
        #: other *_5 candidates below) feed.
        ma_10 = close.rolling(10, min_periods=1).mean()
        ma_5 = close.rolling(5, min_periods=1).mean()

        # Semivariance split — downside and upside volatility are priced
        # differently (Bollerslev et al., "good" vs "bad" volatility).
        downside = returns.where(returns < 0, 0.0).pow(2).rolling(20, min_periods=5).mean().pow(0.5)
        upside = returns.where(returns > 0, 0.0).pow(2).rolling(20, min_periods=5).mean().pow(0.5)
        # 5-day analogs -- require the full window (min_periods=5) since
        # there's no room to go lower and still mean anything.
        downside_5 = returns.where(returns < 0, 0.0).pow(2).rolling(5, min_periods=5).mean().pow(0.5)
        upside_5 = returns.where(returns > 0, 0.0).pow(2).rolling(5, min_periods=5).mean().pow(0.5)

        dollar_volume = close * volume

        # N-day forward return, N = hold_days (exit_rule_spec.md §2) — the
        # label a fixed-hold position's exit realizes, not next-bar return.
        fwd_return = close.pct_change(hold_days).shift(-hold_days)

        # Every column below is scale-free (invariant to share price) except
        # log_dollar_volume, which is deliberately a level — size is itself the
        # documented predictor there, not a unit artefact.
        df = pd.DataFrame(
            {
                "px_to_ma_20": close / ma_20 - 1.0,
                # CANDIDATE feature (not in FEATURE_COLUMNS) -- short-term
                # reversal (Jegadeesh 1990) is validated at roughly a
                # 1-week horizon, but px_to_ma_20's reference is a full
                # month; px_to_ma_10 tests whether a reference closer to
                # hold_days resolves that mismatch (baseline_system_
                # architecture.md open question 14's feature-audit).
                "px_to_ma_10": close / ma_10 - 1.0,
                "momentum_5": close.pct_change(5),
                "momentum_60_skip_10": close.shift(10) / close.shift(60) - 1.0,
                # CANDIDATE feature (not in FEATURE_COLUMNS) -- Jegadeesh &
                # Titman 1993 validates 3-12 month formation predicting
                # 3-12 months ahead; a 60-day formation against a 5-day
                # target is the single largest horizon mismatch in the
                # feature set (baseline_system_architecture.md open
                # question 16). Scaled down ~6x, preserving the original's
                # skip-the-most-recent-stretch structure (60/10 -> 10/2).
                "momentum_10_skip_2": close.shift(2) / close.shift(10) - 1.0,
                "pct_from_252d_high": close / close.rolling(252, min_periods=60).max() - 1.0,
                "volatility_20": returns.rolling(20, min_periods=5).std(),
                "semivol_skew": (upside - downside) / (upside + downside + _EPS),
                "max_ret_20": returns.rolling(20, min_periods=5).max(),
                "min_ret_20": returns.rolling(20, min_periods=5).min(),
                "realized_skew_20": returns.rolling(20, min_periods=10).skew(),
                "realized_kurt_20": returns.rolling(20, min_periods=10).kurt(),
                # |return| per dollar traded. The +1 guards the thin-volume
                # tail; 1e9 keeps log1p in a numerically useful range.
                "amihud_20": np.log1p(
                    (returns.abs() / (dollar_volume + 1.0)).rolling(20, min_periods=5).mean() * 1e9
                ),
                "log_dollar_volume": np.log1p(dollar_volume.rolling(20, min_periods=5).mean()),
                "volume_shock": volume / (volume.rolling(20, min_periods=5).mean() + _EPS),
                "ret_vol_corr_20": returns.rolling(20, min_periods=10).corr(volume.pct_change()),
                "jump_ratio": jump_ratio(close),
                # CANDIDATE features (not in FEATURE_COLUMNS) -- the 5-day
                # cohort for baseline_system_architecture.md open question
                # 18's family swap: same formulas as their 20-day
                # counterparts above, windowed to match hold_days=5
                # instead of the ~monthly horizon those features'
                # literature actually validates. min_periods=5 (the full
                # window) throughout -- no room to go lower and still mean
                # anything, especially for skew/kurtosis.
                "px_to_ma_5": close / ma_5 - 1.0,
                "volatility_5": returns.rolling(5, min_periods=5).std(),
                "semivol_skew_5": (upside_5 - downside_5) / (upside_5 + downside_5 + _EPS),
                "max_ret_5": returns.rolling(5, min_periods=5).max(),
                "min_ret_5": returns.rolling(5, min_periods=5).min(),
                "realized_skew_5": returns.rolling(5, min_periods=5).skew(),
                "realized_kurt_5": returns.rolling(5, min_periods=5).kurt(),
                "ret_vol_corr_5": returns.rolling(5, min_periods=5).corr(volume.pct_change()),
                # CANDIDATE feature (not in FEATURE_COLUMNS) -- a
                # missingness INDICATOR, deliberately tested alone before
                # any real earnings-derived feature (baseline_system_
                # architecture.md open question 22): delisted/acquired
                # tickers have zero earnings data (checked: CELG, RHT,
                # TWTR, XLNX all return nothing) and are systematically
                # volatile, event-driven names. XGBoost routes NaN to a
                # learned default branch, so a later earnings feature that's
                # NaN for exactly this same set could let the model learn
                # "missing -> behaves differently" -- survivorship
                # correlation, not earnings economics. This constant
                # (1.0 if the ticker has any real earnings_date on file,
                # else 0.0) isolates and tests that missingness pattern by
                # itself, with no real earnings content attached, so any
                # later "earnings surprise moved IC" result can be told
                # apart from "missingness alone moved IC."
                "has_earnings_data": (
                    1.0 if ticker in self.earnings_events and not self.earnings_events[ticker].empty else 0.0
                ),
                # CANDIDATE features (not in FEATURE_COLUMNS) -- see
                # _earnings_features_for_ticker for the point-in-time
                # contract each one has to satisfy, and open question 22
                # (baseline_system_architecture.md) for the leakage-gate
                # test has_earnings_data (above) passed before these were
                # built.
                "days_to_next_earnings": days_to_next_earnings,
                "earnings_surprise": earnings_surprise,
                # INTERNAL ONLY -- not a feature, not in FEATURE_COLUMNS,
                # dropped by _add_market_relative_features once consumed.
                # Exposed here (rather than recomputed cross-sectionally)
                # so that step can build a point-in-time-safe DAILY market
                # return series from it. Deliberately NOT
                # MARKET_TARGET_COLUMN, which is the cross-sectional mean
                # of the FORWARD hold_days-ahead return -- a label, not a
                # feature ingredient; using it here would be a severe
                # lookahead bug.
                "_daily_return": returns,
                TARGET_COLUMN: fwd_return,
                "sector": self.sector_map.get(ticker, UNKNOWN_SECTOR),
            }
        )
        # A zero denominator anywhere above yields +/-inf, which XGBoost would
        # treat as a real extreme rather than a gap. NaN is the honest encoding.
        df = df.replace([np.inf, -np.inf], np.nan)
        # The last hold_days rows have no realized forward return yet — not
        # a feature gap to fill, a label that genuinely doesn't exist.
        df = df.dropna(subset=[TARGET_COLUMN])
        if df.empty:
            return None
        df.index = pd.MultiIndex.from_product([[ticker], df.index], names=["ticker", "timestamp"])
        return df

    def _add_sector_momentum(self, frame: pd.DataFrame) -> pd.DataFrame:
        """CANDIDATE feature -- TESTED AND REJECTED, kept computed (but
        still not in FEATURE_COLUMNS) as a negative finding, same posture
        as normalize_features: don't re-add this to the model without
        reading why it was removed.

        Result (scripts/feature_candidate_sector_momentum.py, paired
        against broad's established 15-feature baseline, same genome/seed,
        held-out TEST window): out-of-sample IC +0.0209 -> +0.0095, paired
        t=-2.69 -- a real, decisive move in the WRONG direction, not a
        non-result. Adding this feature roughly halved broad's only
        clearly-nonzero measurement. Best guess at why, not confirmed: a
        sector-wide average is structurally closer to a regime/market-level
        signal than this project's other features, which are deliberately
        idiosyncratic and scale-free within a single ticker's own history
        (see the module docstring's note on why the target is cross-
        sectionally demeaned) -- exactly the kind of common, per-date
        co-movement that doesn't survive out of sample. Caveat on the test
        itself: reused broad's existing genome rather than re-running the
        GA for the enlarged feature set, per this project's "GA last, only
        if IC moves" rule -- a properly re-tuned model might behave
        differently, untested.

        A ticker's sector's own momentum, leave-one-out so this isn't
        circular (a name's sector-momentum feature should describe its
        peers, not partly itself). Motivated directly by Moskowitz &
        Grinblatt 1999 ("Do Industries Explain Momentum?") — industry-level
        momentum is a real, separately-documented effect, in some
        specifications largely subsuming individual-stock momentum. Built
        from ``momentum_60_skip_10`` (this project's own Jegadeesh &
        Titman-style momentum feature) rather than the noisier 5-day one,
        matching the multi-month formation periods the industry-momentum
        literature actually studies.

        Requires ``self.sector_map`` — NaN for the ~70 universe tickers
        with no resolved sector (and for any ticker on a day it's the only
        member of its sector present, where "leave one out" would divide
        by zero) — same fallback shape as every other sector-keyed
        computation in this project.
        """
        if frame.empty:
            return frame
        timestamps = frame.index.get_level_values("timestamp")
        grouped = frame.groupby([timestamps, "sector"])["momentum_60_skip_10"]
        sector_sum = grouped.transform("sum")
        sector_count = grouped.transform("count")
        own = frame["momentum_60_skip_10"]
        loo_count = sector_count - 1
        loo_mean = (sector_sum - own) / loo_count.replace(0, np.nan)
        # "Unknown" isn't a real sector -- every unmapped ticker sharing it
        # would otherwise get averaged together as if they were peers.
        loo_mean = loo_mean.where(frame["sector"] != UNKNOWN_SECTOR)
        return frame.assign(sector_momentum_60_skip_10=loo_mean)

    def _add_market_relative_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        """CANDIDATE features (not in FEATURE_COLUMNS) -- beta, idiosyncratic
        volatility, and index correlation against a point-in-time-safe DAILY
        market return series.

        NOT the same "market series" used for target demeaning
        (MARKET_TARGET_COLUMN, the cross-sectional mean of the FORWARD
        hold_days-ahead return) -- that's a forward-looking label; using it
        as a feature ingredient would be a severe lookahead bug. This
        instead builds a SEPARATE, backward-looking daily market return: the
        cross-sectional mean of each ticker's own 1-day return, known as of
        the close of each trading day -- same timing as every other feature
        in this pipeline.

        beta_60 / idio_vol_60 / market_corr_60: 60-day rolling window
        (matching the existing momentum_60_skip_10 precedent already used in
        this project), min_periods=30 (half the window, this project's
        general convention for requiring meaningful sample size before
        trusting a rolling statistic).

        idio_vol_60 is computed via the closed-form OLS variance identity
        (var(stock) = beta^2 * var(market) + var(residual), exact for a
        single-regressor OLS since the residual is orthogonal to the
        regressor by construction) rather than an actual per-window
        regression loop -- cheaper and numerically equivalent.

        Consumes and drops the internal-only ``_daily_return`` column (see
        _build_ticker_features) plus the ``_market_return`` column computed
        here -- neither is a real feature, only the three named outputs are.
        """
        if frame.empty or "_daily_return" not in frame.columns:
            return frame
        out = frame.copy()
        out["_market_return"] = out.groupby(level="timestamp")["_daily_return"].transform("mean")

        window, min_periods = 60, 30

        def _per_ticker(g: pd.DataFrame) -> pd.DataFrame:
            stock = g["_daily_return"]
            mkt = g["_market_return"]
            cov = stock.rolling(window, min_periods=min_periods).cov(mkt)
            var_mkt = mkt.rolling(window, min_periods=min_periods).var()
            var_stock = stock.rolling(window, min_periods=min_periods).var()
            beta = cov / var_mkt.replace(0, np.nan)
            # Numerical guard: floating-point error can push this fractionally
            # negative when idiosyncratic variance is near zero.
            idio_var = (var_stock - beta**2 * var_mkt).clip(lower=0)
            idio_vol = np.sqrt(idio_var)
            corr = stock.rolling(window, min_periods=min_periods).corr(mkt)
            return pd.DataFrame(
                {"beta_60": beta, "idio_vol_60": idio_vol, "market_corr_60": corr},
                index=g.index,
            )

        computed = out.groupby(level="ticker", group_keys=False).apply(_per_ticker)
        out = out.join(computed)
        out = out.drop(columns=["_daily_return", "_market_return"])
        return out

    def _normalize_features_cross_sectionally(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Replace every feature with its cross-sectional percentile rank
        that day, centered to roughly (-0.5, 0.5].

        Rank, not z-score: a z-score still lets one name's raw extremity
        dominate (an outlier just gets a bigger |z|, preserving exactly the
        magnitude-domination problem this exists to remove). Percentile rank
        is bounded regardless of how extreme the raw value is — the single
        most extreme name that day gets *rank* 1, the same rank any other
        day's most-extreme name would get, whether its raw reading was mildly
        or wildly outside normal range. That's the point: it caps how much
        one chaotic name's *magnitude* can dominate the model's input, without
        touching its *ordinal* position (still identifiably the most extreme
        name that day).

        Applied uniformly to every column in FEATURE_COLUMNS, including
        log_dollar_volume — its raw level is deliberately meaningful (size
        effects are real), but "small relative to peers on this day" is
        already what a size effect cross-sectionally means, and ranking it
        removes 8 years of nominal market growth as a confound for free
        (a "small" 2018 dollar-volume figure isn't a "small" 2025 one).

        NaN passes through unchanged (pandas' rank(pct=True) excludes NaN
        from the ranking and returns NaN for it) — a feature that's missing
        because a ticker's own rolling window hasn't filled yet should stay
        missing, not be assigned a fabricated rank.
        """
        if not self.normalize_features or frame.empty:
            return frame
        normalized = frame.copy()
        grouped = frame.groupby(level="timestamp")
        for col in FEATURE_COLUMNS:
            normalized[col] = grouped[col].rank(pct=True) - 0.5
        return normalized

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
