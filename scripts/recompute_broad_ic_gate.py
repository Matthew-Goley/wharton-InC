"""Recompute item 12's foundational single-arm gate figure -- broad's own
IC being different from zero ("+0.0209, ~4.6 SE from zero", cited as "the
only unambiguously real number this project has measured") -- with the
corrected block-bootstrap SE. This is a single-arm significance test
(IC vs. 0), not a paired comparison, so it uses the same treatment as
item 45's single-arm noise floor: block_bootstrap_se applied directly to
the per-date IC series itself, no diffing.

Usage:
    python -m scripts.recompute_broad_ic_gate
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import BROAD_EXPERT, BaselineConfig  # noqa: E402
from src.data_layer import RealDataProvider  # noqa: E402
from src.evaluation.block_se import block_bootstrap_se, newey_west_se  # noqa: E402
from src.evaluation.metrics import per_date_cross_sectional_ic  # noqa: E402
from src.experts.base import ExpertModel  # noqa: E402
from scripts.backtest_broad_real import BEST_GENOME, TRAIN_START, TRAIN_END, TEST_START, TEST_END  # noqa: E402

HOLD_DAYS = 5


def main() -> None:
    config = BaselineConfig()
    provider = RealDataProvider()

    print(f"Loading TRAIN window ({TRAIN_START}..{TRAIN_END})...")
    train_table = provider.get_feature_table(TRAIN_START, TRAIN_END, hold_days=HOLD_DAYS)
    train_frame = train_table.frame
    feature_cols = train_table.feature_columns
    model = ExpertModel(BROAD_EXPERT, BEST_GENOME, random_seed=config.ga.random_seed).fit(
        train_frame[feature_cols], train_frame[train_table.target_column]
    )

    print(f"Loading TEST window ({TEST_START}..{TEST_END})...")
    test_table = provider.get_feature_table(TEST_START, TEST_END, hold_days=HOLD_DAYS)
    test_frame = test_table.frame
    y_test = test_frame[test_table.target_column]
    preds = model.predict(test_frame[feature_cols])["predicted_return"]
    dates = test_frame.index.get_level_values("timestamp").to_numpy()

    ic_series = per_date_cross_sectional_ic(dates, y_test.to_numpy(), preds.to_numpy())
    values = ic_series.to_numpy()
    n = len(values)
    mean_ic = float(values.mean())

    naive_se = float(values.std(ddof=1) / np.sqrt(n))
    nw_se = newey_west_se(values, lag=HOLD_DAYS - 1)
    boot = block_bootstrap_se(values, ic_series.index.to_numpy(), block_days=HOLD_DAYS)

    print(f"\n=== broad's own IC vs. zero, n_dates={n} ===")
    print(f"  mean IC: {mean_ic:+.4f}")
    print(f"  SE naive:     {naive_se:.4f}  t naive:     {mean_ic/naive_se:+.2f}")
    print(f"  SE NW:        {nw_se:.4f}  t NW:        {mean_ic/nw_se:+.2f}")
    print(f"  SE corrected: {boot['se']:.4f}  t corrected: {mean_ic/boot['se']:+.2f}  "
          f"(block bootstrap, n_blocks={boot['n_blocks']})")


if __name__ == "__main__":
    main()
