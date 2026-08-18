"""Point-in-time correctness of RealDataProvider.

Built on a tiny synthetic fixture rather than the real 126MB dataset, so
these stay fast and don't depend on what happens to be fetched locally.
"""
import numpy as np
import pandas as pd
import pytest

from src.data_layer.real_provider import (
    FEATURE_COLUMNS,
    MARKET_TARGET_COLUMN,
    RAW_TARGET_COLUMN,
    TARGET_COLUMN,
    RealDataProvider,
    jump_ratio,
)

# Observation dates for the fixture. Membership only changes on these, so a
# date in between resolves back to the most recent one.
OBSERVATIONS = ["2020-01-01", "2020-03-01", "2020-05-01", "2020-07-01"]

# ALWAYS is a member throughout; LEAVER is dropped as of 2020-05-01;
# JOINER is added as of 2020-03-01.
ROSTERS = {
    "2020-01-01": ["ALWAYS", "LEAVER"],
    "2020-03-01": ["ALWAYS", "LEAVER", "JOINER"],
    "2020-05-01": ["ALWAYS", "JOINER"],
    "2020-07-01": ["ALWAYS", "JOINER"],
}


@pytest.fixture
def fixture_dirs(tmp_path):
    """Write a miniature membership file + price CSVs; return (dir, csv)."""
    rows = [
        {"date": date, "ticker": ticker, "ticker_asof": ticker}
        for date, tickers in ROSTERS.items()
        for ticker in tickers
    ]
    membership_csv = tmp_path / "sp500_membership.csv"
    pd.DataFrame(rows).to_csv(membership_csv, index=False)

    data_dir = tmp_path / "training"
    data_dir.mkdir()
    dates = pd.bdate_range("2019-12-01", "2020-08-31")
    for i, ticker in enumerate(["ALWAYS", "LEAVER", "JOINER"]):
        # Distinct wiggly series per ticker so features/labels aren't constant.
        close = 100 + np.arange(len(dates)) * 0.1 + np.sin(np.arange(len(dates)) / (3 + i)) * 2
        # Volume also wiggly, on a DIFFERENT period/phase than close, so
        # volume-vs-return correlation features (ret_vol_corr_*) have real
        # variance to work with instead of correlating against a constant
        # (undefined, all-NaN) -- a constant Volume silently starved every
        # volume-dependent feature of real test coverage until this was
        # caught by the 5-day cohort's own not-all-NaN check.
        volume = 1_000_000 * (1 + 0.3 * np.sin(np.arange(len(dates)) / (4 + i) + i))
        pd.DataFrame(
            {
                "timestamp": dates,
                "Open": close - 0.5,
                "High": close + 1.0,
                "Low": close - 1.0,
                "Close": close,
                # Deliberately wrong: if anything reads Adj Close instead of
                # Close, the features collapse to this constant.
                "Adj Close": 1.0,
                "Volume": volume,
            }
        ).to_csv(data_dir / f"{ticker}_1d.csv", index=False)

    return data_dir, membership_csv


def _kept(fixture_dirs, hold_days=5):
    data_dir, membership_csv = fixture_dirs
    provider = RealDataProvider(
        universe=["ALWAYS", "LEAVER", "JOINER"],
        data_dir=data_dir,
        membership_csv=membership_csv,
    )
    table = provider.get_feature_table("2019-12-01", "2020-08-31", hold_days=hold_days)
    return table.frame.index.to_frame(index=False)


def test_drops_rows_before_a_ticker_joins(fixture_dirs):
    kept = _kept(fixture_dirs)
    joiner = kept[kept.ticker == "JOINER"]
    assert not joiner.empty
    # JOINER is absent from the 2020-01-01 roster, so nothing before the
    # 2020-03-01 observation should survive.
    assert joiner.timestamp.min() >= pd.Timestamp("2020-03-01")


def test_drops_rows_after_a_ticker_leaves(fixture_dirs):
    """Regression: merge_asof(by='ticker') kept a departed member forever.

    A departed ticker simply stops having rows, so a per-ticker backward
    as-of join goes on matching its final row for the rest of history.
    """
    kept = _kept(fixture_dirs)
    leaver = kept[kept.ticker == "LEAVER"]
    assert not leaver.empty
    # Dropped as of the 2020-05-01 observation — nothing from that date on.
    assert leaver.timestamp.max() < pd.Timestamp("2020-05-01")


def test_keeps_rows_in_the_gap_before_the_next_observation(fixture_dirs):
    """Membership persists until the NEXT observation, not until its last row.

    LEAVER last appears in the 2020-03-01 roster, so April is still valid —
    trimming at 2020-03-01 would be over-correcting.
    """
    kept = _kept(fixture_dirs)
    leaver = kept[kept.ticker == "LEAVER"]
    assert (leaver.timestamp > pd.Timestamp("2020-04-01")).any()


def test_continuous_member_is_never_dropped(fixture_dirs):
    kept = _kept(fixture_dirs)
    always = kept[kept.ticker == "ALWAYS"]
    assert always.timestamp.min() >= pd.Timestamp("2020-01-01")
    assert always.timestamp.max() >= pd.Timestamp("2020-08-01")
    # No interior holes: every observation window is represented.
    for lo, hi in zip(OBSERVATIONS, OBSERVATIONS[1:]):
        window = always[(always.timestamp >= lo) & (always.timestamp < hi)]
        assert not window.empty, f"ALWAYS has no rows in {lo}..{hi}"


def test_every_kept_row_is_a_member_under_the_in_force_observation(fixture_dirs):
    """The actual contract, checked directly rather than via a proxy."""
    _, membership_csv = fixture_dirs
    membership = pd.read_csv(membership_csv, parse_dates=["date"])
    observations = np.sort(membership.date.unique())
    pairs = set(zip(membership.date.to_numpy(), membership.ticker.to_numpy()))

    kept = _kept(fixture_dirs)
    dates = kept.timestamp.to_numpy("datetime64[ns]")
    pos = np.searchsorted(observations, dates, side="right") - 1
    assert (pos >= 0).all(), "a kept row predates every observation"
    assert all(
        (observations[p], ticker) in pairs
        for p, ticker in zip(pos, kept.ticker)
    )


def test_features_come_from_close_not_adj_close(fixture_dirs):
    data_dir, membership_csv = fixture_dirs
    provider = RealDataProvider(
        universe=["ALWAYS"], data_dir=data_dir, membership_csv=membership_csv
    )
    frame = provider.get_feature_table("2019-12-01", "2020-08-31", hold_days=5).frame
    # Adj Close is a constant 1.0 in the fixture, so anything derived from it
    # would be identically zero. Real Close moves.
    assert frame["momentum_5"].abs().sum() > 0
    assert frame["px_to_ma_20"].abs().sum() > 0


# --- scale-free features ---------------------------------------------------

def _write_prices(data_dir, ticker, close):
    dates = pd.bdate_range("2019-06-03", periods=len(close))
    pd.DataFrame(
        {
            "timestamp": dates, "Open": close * 0.999, "High": close * 1.01,
            "Low": close * 0.99, "Close": close, "Adj Close": close,
            "Volume": 1_000_000,
        }
    ).to_csv(data_dir / f"{ticker}_1d.csv", index=False)
    return dates


#: Deliberately price-dependent: both measure trading activity in dollars,
#: where the dollar amount IS the economic quantity (size, and price impact
#: per dollar traded) rather than a unit artefact.
PRICE_DEPENDENT_BY_DESIGN = ["log_dollar_volume", "amihud_20"]


def test_features_are_invariant_to_share_price(tmp_path):
    """A $5 stock and a $500 stock with the same shape must look identical.

    This is the whole point of the scale-free rewrite: raw moving averages
    pooled across the universe encoded share price, not behaviour. The two
    dollar-volume features are exempted and checked separately below.
    """
    data_dir = tmp_path / "training"
    data_dir.mkdir()
    rng = np.random.default_rng(0)
    base = 50 * np.exp(np.cumsum(rng.normal(0, 0.01, 400)))
    dates = _write_prices(data_dir, "CHEAP", base)
    _write_prices(data_dir, "PRICEY", base * 100.0)  # identical shape, 100x

    membership_csv = tmp_path / "m.csv"
    pd.DataFrame(
        [{"date": dates[0], "ticker": t, "ticker_asof": t} for t in ("CHEAP", "PRICEY")]
    ).to_csv(membership_csv, index=False)

    provider = RealDataProvider(
        universe=["CHEAP", "PRICEY"], data_dir=data_dir,
        membership_csv=membership_csv, demean_target=False,
    )
    frame = provider.get_feature_table("2019-01-01", "2030-01-01", hold_days=5).frame
    scale_free = [c for c in FEATURE_COLUMNS if c not in PRICE_DEPENDENT_BY_DESIGN]
    cheap = frame.xs("CHEAP", level="ticker")
    pricey = frame.xs("PRICEY", level="ticker")
    pd.testing.assert_frame_equal(
        cheap[scale_free], pricey[scale_free], atol=1e-9, check_exact=False
    )
    # ...and the exempted pair really does move with price, so the exemption
    # is a documented choice rather than an untested assumption.
    for col in PRICE_DEPENDENT_BY_DESIGN:
        assert not np.allclose(
            cheap[col].dropna().to_numpy(), pricey[col].dropna().to_numpy()
        ), f"{col} was expected to depend on price level"


# --- cross-sectionally demeaned target -------------------------------------

def test_demeaned_target_sums_to_zero_each_date(fixture_dirs):
    data_dir, membership_csv = fixture_dirs
    provider = RealDataProvider(
        universe=["ALWAYS", "LEAVER", "JOINER"], data_dir=data_dir,
        membership_csv=membership_csv, demean_target=True,
    )
    frame = provider.get_feature_table("2019-12-01", "2020-08-31", hold_days=5).frame
    per_date = frame.groupby(level="timestamp")[TARGET_COLUMN].mean()
    assert np.allclose(per_date.to_numpy(), 0.0, atol=1e-12)
    # the raw label and the market leg it was split into are both retained
    assert RAW_TARGET_COLUMN in frame.columns
    assert MARKET_TARGET_COLUMN in frame.columns
    assert np.allclose(
        (frame[RAW_TARGET_COLUMN] - frame[MARKET_TARGET_COLUMN]).to_numpy(),
        frame[TARGET_COLUMN].to_numpy(),
    )


def test_demeaning_can_be_disabled(fixture_dirs):
    data_dir, membership_csv = fixture_dirs
    provider = RealDataProvider(
        universe=["ALWAYS", "LEAVER", "JOINER"], data_dir=data_dir,
        membership_csv=membership_csv, demean_target=False,
    )
    frame = provider.get_feature_table("2019-12-01", "2020-08-31", hold_days=5).frame
    assert RAW_TARGET_COLUMN not in frame.columns
    per_date = frame.groupby(level="timestamp")[TARGET_COLUMN].mean()
    assert not np.allclose(per_date.to_numpy(), 0.0, atol=1e-12)


def test_feature_normalization_is_off_by_default(fixture_dirs):
    data_dir, membership_csv = fixture_dirs
    provider = RealDataProvider(
        universe=["ALWAYS", "LEAVER", "JOINER"], data_dir=data_dir, membership_csv=membership_csv,
    )
    frame = provider.get_feature_table("2019-12-01", "2020-08-31", hold_days=5).frame
    # Raw ratio features aren't bounded to (-0.5, 0.5] -- with normalization
    # off, at least one column should have values outside that range.
    assert (frame[FEATURE_COLUMNS].abs() > 0.5).to_numpy().any()


def test_normalized_features_are_bounded_percentile_ranks(fixture_dirs):
    data_dir, membership_csv = fixture_dirs
    provider = RealDataProvider(
        universe=["ALWAYS", "LEAVER", "JOINER"], data_dir=data_dir, membership_csv=membership_csv,
        normalize_features=True,
    )
    frame = provider.get_feature_table("2019-12-01", "2020-08-31", hold_days=5).frame
    values = frame[FEATURE_COLUMNS].to_numpy()
    finite = values[np.isfinite(values)]
    assert finite.min() >= -0.5 - 1e-9
    assert finite.max() <= 0.5 + 1e-9


def test_normalized_feature_ranks_are_consistent_within_a_date(fixture_dirs):
    data_dir, membership_csv = fixture_dirs
    provider = RealDataProvider(
        universe=["ALWAYS", "LEAVER", "JOINER"], data_dir=data_dir, membership_csv=membership_csv,
        normalize_features=True,
    )
    frame = provider.get_feature_table("2019-12-01", "2020-08-31", hold_days=5).frame
    # Same-date rows for a feature with no ties should be a permutation of
    # the same rank set (evenly spaced through (-0.5, 0.5]) every date that
    # has the same number of names present.
    col = "px_to_ma_20"
    by_date = frame.groupby(level="timestamp")[col].apply(lambda s: tuple(sorted(s.dropna().to_numpy())))
    sizes = frame.groupby(level="timestamp")[col].count()
    for date, ranks in by_date.items():
        n = sizes.loc[date]
        if n == 0:
            continue
        expected = tuple(sorted((i + 1) / n - 0.5 for i in range(n)))
        assert np.allclose(ranks, expected, atol=1e-9), f"{date}: {ranks} != {expected}"


def test_normalization_leaves_nan_as_nan_not_a_fabricated_rank(fixture_dirs):
    data_dir, membership_csv = fixture_dirs
    provider = RealDataProvider(
        universe=["ALWAYS", "LEAVER", "JOINER"], data_dir=data_dir, membership_csv=membership_csv,
        normalize_features=True,
    )
    frame = provider.get_feature_table("2019-12-01", "2020-08-31", hold_days=5).frame
    # realized_kurt_20 needs a 10-period warmup -- rows before that are NaN
    # pre-normalization and must stay NaN, not get assigned a rank.
    raw_provider = RealDataProvider(
        universe=["ALWAYS", "LEAVER", "JOINER"], data_dir=data_dir, membership_csv=membership_csv,
    )
    raw_frame = raw_provider.get_feature_table("2019-12-01", "2020-08-31", hold_days=5).frame
    assert (frame["realized_kurt_20"].isna() == raw_frame["realized_kurt_20"].isna()).all()


# --- sector_momentum_60_skip_10 (candidate feature, Moskowitz & Grinblatt 1999) ---

def test_sector_momentum_is_not_yet_a_promoted_feature(fixture_dirs):
    """Computed unconditionally, but must not silently start feeding models
    until a paired IC test promotes it -- see baseline_system_architecture.md."""
    assert "sector_momentum_60_skip_10" not in FEATURE_COLUMNS
    data_dir, membership_csv = fixture_dirs
    provider = RealDataProvider(
        universe=["ALWAYS", "LEAVER", "JOINER"], data_dir=data_dir, membership_csv=membership_csv,
        sector_map={"ALWAYS": "Tech", "LEAVER": "Tech", "JOINER": "Health Care"},
    )
    frame = provider.get_feature_table("2019-12-01", "2020-08-31", hold_days=5).frame
    assert "sector_momentum_60_skip_10" in frame.columns


def test_sector_momentum_is_leave_one_out_average_of_peers(fixture_dirs):
    """With exactly two members in a sector, each one's sector-momentum
    feature must equal the OTHER member's own momentum_60_skip_10 exactly
    -- the leave-one-out average of a single peer is just that peer."""
    data_dir, membership_csv = fixture_dirs
    provider = RealDataProvider(
        universe=["ALWAYS", "LEAVER", "JOINER"], data_dir=data_dir, membership_csv=membership_csv,
        sector_map={"ALWAYS": "Tech", "LEAVER": "Tech", "JOINER": "Health Care"},
    )
    frame = provider.get_feature_table("2019-12-01", "2020-08-31", hold_days=5).frame
    # ALWAYS and LEAVER coexist (both members) in [2020-03-01, 2020-05-01).
    date = pd.Timestamp("2020-03-16")
    always_mom = frame.loc[("ALWAYS", date), "momentum_60_skip_10"]
    leaver_mom = frame.loc[("LEAVER", date), "momentum_60_skip_10"]
    assert np.isfinite(always_mom) and np.isfinite(leaver_mom)
    assert frame.loc[("ALWAYS", date), "sector_momentum_60_skip_10"] == pytest.approx(leaver_mom)
    assert frame.loc[("LEAVER", date), "sector_momentum_60_skip_10"] == pytest.approx(always_mom)


def test_sector_momentum_is_nan_for_a_lone_sector_member(fixture_dirs):
    """JOINER is the only Health Care member (a real, mapped sector) --
    leave-one-out has zero peers left, must be NaN, not a divide-by-zero
    crash -- distinct from the unmapped-ticker case below."""
    data_dir, membership_csv = fixture_dirs
    provider = RealDataProvider(
        universe=["ALWAYS", "LEAVER", "JOINER"], data_dir=data_dir, membership_csv=membership_csv,
        sector_map={"ALWAYS": "Tech", "LEAVER": "Tech", "JOINER": "Health Care"},
    )
    frame = provider.get_feature_table("2019-12-01", "2020-08-31", hold_days=5).frame
    date = pd.Timestamp("2020-03-16")
    assert pd.isna(frame.loc[("JOINER", date), "sector_momentum_60_skip_10"])


def test_sector_momentum_is_nan_when_unmapped(fixture_dirs):
    """A ticker not in the sector map (or sharing the 'Unknown' fallback
    with unrelated other unmapped tickers) must get NaN, not a spurious
    average computed across tickers that aren't actually peers."""
    data_dir, membership_csv = fixture_dirs
    provider = RealDataProvider(
        universe=["ALWAYS", "LEAVER", "JOINER"], data_dir=data_dir, membership_csv=membership_csv,
        sector_map={"ALWAYS": "Tech"},  # LEAVER and JOINER both unmapped -> "Unknown"
    )
    frame = provider.get_feature_table("2019-12-01", "2020-08-31", hold_days=5).frame
    date = pd.Timestamp("2020-03-16")
    assert pd.isna(frame.loc[("LEAVER", date), "sector_momentum_60_skip_10"])
    assert pd.isna(frame.loc[("JOINER", date), "sector_momentum_60_skip_10"])
    # ALWAYS is alone in "Tech" (its only real-sector peer, LEAVER, is
    # unmapped here) -- zero real peers, also NaN.
    assert pd.isna(frame.loc[("ALWAYS", date), "sector_momentum_60_skip_10"])


def test_sector_map_defaults_to_the_real_generated_map():
    """Constructor injection (sector_map=None) must fall back to
    config.sector_map.SECTOR_MAP, not silently produce an empty map."""
    provider = RealDataProvider()
    assert provider.sector_map.get("AAPL") == "Information Technology"


# --- px_to_ma_10 (candidate feature, horizon-match audit) -------------------

def test_px_to_ma_10_is_not_yet_a_promoted_feature(fixture_dirs):
    assert "px_to_ma_10" not in FEATURE_COLUMNS
    data_dir, membership_csv = fixture_dirs
    provider = RealDataProvider(universe=["ALWAYS"], data_dir=data_dir, membership_csv=membership_csv)
    frame = provider.get_feature_table("2019-12-01", "2020-08-31", hold_days=5).frame
    assert "px_to_ma_10" in frame.columns
    assert frame["px_to_ma_10"].abs().sum() > 0  # real price data moves it off zero


# --- 5-day cohort (candidate features, open question 18's family swap) -----

_FIVE_DAY_COHORT = [
    "px_to_ma_5", "volatility_5", "semivol_skew_5", "max_ret_5", "min_ret_5",
    "realized_skew_5", "realized_kurt_5", "ret_vol_corr_5",
]


def test_five_day_cohort_is_computed_but_not_yet_promoted(fixture_dirs):
    for col in _FIVE_DAY_COHORT:
        assert col not in FEATURE_COLUMNS
    data_dir, membership_csv = fixture_dirs
    provider = RealDataProvider(universe=["ALWAYS"], data_dir=data_dir, membership_csv=membership_csv)
    frame = provider.get_feature_table("2019-12-01", "2020-08-31", hold_days=5).frame
    for col in _FIVE_DAY_COHORT:
        assert col in frame.columns
        assert frame[col].notna().any(), f"{col} is all-NaN -- window/min_periods likely misconfigured"


def test_momentum_10_skip_2_is_not_yet_a_promoted_feature(fixture_dirs):
    assert "momentum_10_skip_2" not in FEATURE_COLUMNS
    data_dir, membership_csv = fixture_dirs
    provider = RealDataProvider(universe=["ALWAYS"], data_dir=data_dir, membership_csv=membership_csv)
    frame = provider.get_feature_table("2019-12-01", "2020-08-31", hold_days=5).frame
    assert "momentum_10_skip_2" in frame.columns
    assert frame["momentum_10_skip_2"].notna().any()


# --- has_earnings_data (candidate leakage-gate feature, open question 22) --

def _earnings_events_for(ticker: str, dates: list, surprises: list) -> dict:
    return {ticker: pd.DataFrame({
        "earnings_date": [pd.Timestamp(d) for d in dates], "surprise_pct": surprises,
    })}


def test_has_earnings_data_is_not_yet_a_promoted_feature(fixture_dirs):
    assert "has_earnings_data" not in FEATURE_COLUMNS
    data_dir, membership_csv = fixture_dirs
    provider = RealDataProvider(
        universe=["ALWAYS", "LEAVER", "JOINER"], data_dir=data_dir, membership_csv=membership_csv,
        earnings_events=_earnings_events_for("ALWAYS", ["2020-02-03"], [10.0]),
    )
    frame = provider.get_feature_table("2019-12-01", "2020-08-31", hold_days=5).frame
    assert "has_earnings_data" in frame.columns


def test_has_earnings_data_is_1_for_covered_tickers_0_for_others(fixture_dirs):
    """The whole point is a clean, constant-per-ticker indicator -- exactly
    1.0 for a ticker with real earnings data on file, exactly 0.0 (never
    NaN -- this must never itself be missing) for one without."""
    data_dir, membership_csv = fixture_dirs
    provider = RealDataProvider(
        universe=["ALWAYS", "LEAVER", "JOINER"], data_dir=data_dir, membership_csv=membership_csv,
        earnings_events=_earnings_events_for("ALWAYS", ["2020-02-03"], [10.0]),  # only ALWAYS "has" data
    )
    frame = provider.get_feature_table("2019-12-01", "2020-08-31", hold_days=5).frame
    always_vals = frame.xs("ALWAYS", level="ticker")["has_earnings_data"]
    leaver_vals = frame.xs("LEAVER", level="ticker")["has_earnings_data"]
    assert (always_vals == 1.0).all()
    assert (leaver_vals == 0.0).all()
    assert not always_vals.isna().any() and not leaver_vals.isna().any()


def test_earnings_data_defaults_to_empty_when_fetch_has_not_run(tmp_path, monkeypatch):
    """Before src.data_pipeline.fetch_earnings_dates has ever run, the real
    cache file doesn't exist -- must default to "nobody has earnings data"
    (empty dict), not crash."""
    import src.data_layer.real_provider as rp
    monkeypatch.setattr(rp, "EARNINGS_DATES_CACHE", tmp_path / "does_not_exist.csv")
    assert rp._load_earnings_events() == {}


# --- days_to_next_earnings / earnings_surprise (candidate features) --------

def test_days_and_surprise_are_not_yet_promoted_features(fixture_dirs):
    assert "days_to_next_earnings" not in FEATURE_COLUMNS
    assert "earnings_surprise" not in FEATURE_COLUMNS


def test_earnings_surprise_is_not_visible_before_its_own_announcement_date(fixture_dirs):
    """THE regression test for the highest-risk lookahead surface in the
    feature set. Two known events for ALWAYS: 2020-02-03 (surprise=10.0)
    and 2020-05-01 (surprise=-20.0). A naive as-of join that ALLOWS
    same-day matches would leak each event's own surprise on its own
    announcement date -- this asserts that does NOT happen."""
    data_dir, membership_csv = fixture_dirs
    provider = RealDataProvider(
        universe=["ALWAYS"], data_dir=data_dir, membership_csv=membership_csv,
        earnings_events=_earnings_events_for("ALWAYS", ["2020-02-03", "2020-05-01"], [10.0, -20.0]),
    )
    frame = provider.get_feature_table("2019-12-01", "2020-08-31", hold_days=5).frame
    s = frame.xs("ALWAYS", level="ticker")["earnings_surprise"]

    assert pd.isna(s.loc[pd.Timestamp("2020-01-31")])  # strictly before the first event: no prior surprise yet
    # THE adversarial assertion: ON the first event's own date, the
    # surprise must NOT yet be visible -- a buggy allow_exact_matches=True
    # implementation would leak 10.0 here instead of NaN.
    assert pd.isna(s.loc[pd.Timestamp("2020-02-03")])
    assert s.loc[pd.Timestamp("2020-02-04")] == pytest.approx(10.0)  # visible starting the next trading day
    assert s.loc[pd.Timestamp("2020-03-16")] == pytest.approx(10.0)  # still the most recent prior event
    # ON the second event's own date: still the FIRST event's surprise,
    # not the second's -- same adversarial check, second occurrence.
    assert s.loc[pd.Timestamp("2020-05-01")] == pytest.approx(10.0)
    assert s.loc[pd.Timestamp("2020-05-04")] == pytest.approx(-20.0)  # visible starting the next trading day


def test_days_to_next_earnings_allows_same_day_and_counts_down(fixture_dirs):
    """Not a lookahead risk the way earnings_surprise is -- the calendar
    date itself is public knowledge in advance, so same-day (0 days) is a
    legitimate value, and NaN once the known calendar runs out."""
    data_dir, membership_csv = fixture_dirs
    provider = RealDataProvider(
        universe=["ALWAYS"], data_dir=data_dir, membership_csv=membership_csv,
        earnings_events=_earnings_events_for("ALWAYS", ["2020-02-03", "2020-05-01"], [10.0, -20.0]),
    )
    frame = provider.get_feature_table("2019-12-01", "2020-08-31", hold_days=5).frame
    d = frame.xs("ALWAYS", level="ticker")["days_to_next_earnings"]

    assert d.loc[pd.Timestamp("2020-01-31")] == 3  # Jan 31 -> Feb 3
    assert d.loc[pd.Timestamp("2020-02-03")] == 0  # same-day IS allowed here
    assert d.loc[pd.Timestamp("2020-02-04")] == (pd.Timestamp("2020-05-01") - pd.Timestamp("2020-02-04")).days
    assert pd.isna(d.loc[pd.Timestamp("2020-06-01")])  # past the last known event -- no further date on file


def test_earnings_features_are_nan_for_a_ticker_with_no_earnings_data(fixture_dirs):
    data_dir, membership_csv = fixture_dirs
    provider = RealDataProvider(
        universe=["ALWAYS", "LEAVER"], data_dir=data_dir, membership_csv=membership_csv,
        earnings_events=_earnings_events_for("ALWAYS", ["2020-02-03"], [10.0]),  # LEAVER gets nothing
    )
    frame = provider.get_feature_table("2019-12-01", "2020-08-31", hold_days=5).frame
    leaver = frame.xs("LEAVER", level="ticker")
    assert leaver["earnings_surprise"].isna().all()
    assert leaver["days_to_next_earnings"].isna().all()


# --- jump ratio (Khashanah et al., JRSS-C 2025) -----------------------------

def _quiet_returns(n=400, scale=0.01, seed=0):
    return np.random.default_rng(seed).normal(0, scale, n)


def _series_from_returns(returns):
    return pd.Series(100 * np.exp(np.cumsum(returns)))


def test_jump_ratio_is_scale_invariant():
    close = _series_from_returns(_quiet_returns())
    pd.testing.assert_series_equal(jump_ratio(close), jump_ratio(close * 1000.0))


def test_jump_ratio_flags_a_planted_jump():
    returns = _quiet_returns()
    returns[300] = 0.12  # ~12 sigma against a 1% daily vol
    ratio = jump_ratio(_series_from_returns(returns))
    assert abs(ratio.iloc[300]) > 6
    # a normal day nearby stays small
    assert abs(ratio.iloc[290]) < 4


def test_jump_ratio_survives_clustered_jumps():
    """The paper's central claim, and the reason for a median denominator.

    Bipower-style estimators multiply adjacent absolute returns, so a run of
    large moves inflates the denominator and the cluster masks itself — on the
    2010 flash crash that pushed BV above RV and reported no jumps at all. A
    running median over k=9 absorbs up to four large returns in the block.
    """
    returns = _quiet_returns()
    returns[300:305] = [0.10, -0.09, 0.11, -0.08, 0.10]  # five in a row
    close = _series_from_returns(returns)
    ratio = jump_ratio(close)
    assert (ratio.iloc[300:305].abs() > 5).all(), "clustered jumps were masked"

    # A plain rolling-std denominator gets inflated by the cluster and mutes
    # the later members — the failure mode Med_k exists to avoid.
    log_r = np.log(close).diff()
    naive = log_r / log_r.rolling(63, min_periods=31).std().shift(1)
    assert naive.iloc[304:305].abs().item() < ratio.iloc[304:305].abs().item()


def test_jump_ratio_leading_values_are_nan_not_zero():
    ratio = jump_ratio(_series_from_returns(_quiet_returns()))
    assert ratio.iloc[:9].isna().all()  # "unknown", not "no jump"


def test_jump_ratio_rejects_unsupported_k():
    with pytest.raises(ValueError, match="Med_k scale constant"):
        jump_ratio(_series_from_returns(_quiet_returns()), k=4)
