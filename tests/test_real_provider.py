"""Point-in-time correctness of RealDataProvider.

Built on a tiny synthetic fixture rather than the real 126MB dataset, so
these stay fast and don't depend on what happens to be fetched locally.
"""
import numpy as np
import pandas as pd
import pytest

from src.data_layer.real_provider import RealDataProvider

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
    # Adj Close is a constant 1.0 in the fixture; real Close is ~100+.
    assert (frame["ma_5"] > 50).all()
