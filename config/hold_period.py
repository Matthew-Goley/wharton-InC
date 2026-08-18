"""Fixed-holding-period exit rule parameters. See exit_rule_spec.md (§1-2).

`hold_days` (N) is not an independent knob — it defines the training label
(N-day forward return), so changing it invalidates trained experts and
requires a full retrain + GA re-run per value (the outer sweep loop in
`src.pipeline.sweep`). This is the single source of truth for both the
label horizon and the selector's exit timing; a mismatch between the two is
a silent correctness bug, so it's asserted at pipeline start
(`train_baseline`) rather than assumed.
"""
from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd

#: N — a position entered on day d is closed on day d + N, unconditionally.
HOLD_DAYS: int = 5

#: Coarse outer-loop sweep candidates (spec §2). Each value is a full
#: retrain of every expert, so keep this short.
HOLD_DAYS_SWEEP: tuple[int, ...] = (1, 3, 5, 10)

#: Competition window: 10 weeks * 5 trading days/week (spec §5).
WINDOW_TRADING_DAYS: int = 50

#: Calendar date the competition's live trading window begins (period 1,
#: exit_rule_spec.md's period convention). `None` until the team confirms
#: it from the real competition schedule — same not-yet-confirmed pattern
#: as `config/cost_model.py`'s `POSITION_CAP_PCT`. Needed by
#: `period_number_for` below, and by `TradeSelector.run`'s live-mode path
#: (baseline_system_architecture.md open question 27) to know which
#: window-day "today" is without depending on whatever happens to be in a
#: given day's scored DataFrame.
WINDOW_START_DATE: Optional[str] = None


def period_number_for(
    date: Union[str, pd.Timestamp], window_start_date: Optional[str] = None
) -> int:
    """1-indexed trading-day period number for `date` (period 1 =
    WINDOW_START_DATE, matching TradeSelector.run's existing
    enumerate(..., start=1) convention for a full-window backtest).

    Calendar-anchored deliberately: derived from the date alone, not from
    whatever timestamps happen to be in a given day's scored `ranked`
    frame -- that's exactly what breaks for a single-day live call (a
    1-row frame always reads as "period 1" under the old, frame-relative
    approach). See open question 27.

    Uses `numpy.busday_count` (weekends only) rather than a real NYSE
    trading calendar -- a documented approximation, not exactness. A
    handful of unmodeled market holidays across a 10-week/50-trading-day
    window is a small, bounded error, not a correctness guarantee; good
    enough to unblock live pacing (self-correcting every run, per
    TradeSelector._entries_allowed), not good enough to trust for an
    exact-equality check against it.
    """
    start = window_start_date or WINDOW_START_DATE
    if start is None:
        raise ValueError(
            "WINDOW_START_DATE is not set (config/hold_period.py) -- confirm the "
            "competition's real start date before computing a live period number."
        )
    start_ts, date_ts = pd.Timestamp(start), pd.Timestamp(date)
    return int(np.busday_count(start_ts.date(), date_ts.date())) + 1
