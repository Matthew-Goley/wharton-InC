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
                "Volume": 1_000_000,
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
