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

#: True once the team has confirmed WINDOW_START_DATE against the real
#: competition schedule. `False` means the date below is a PLACEHOLDER —
#: every consumer of WINDOW_START_DATE must check THIS flag, not merely
#: whether the date is set, before treating a computed period_number as
#: real. Exists because a previous placeholder value silently put the
#: system at period 18 of 50 with an empty portfolio and a doubled
#: derived entry rate (baseline_system_architecture.md item 42) — a
#: plausible-looking wrong number that was nearly read as real. Flip to
#: `True` in the SAME commit that sets the real, confirmed date; never
#: set independently of it.
WINDOW_START_DATE_CONFIRMED: bool = True

#: Calendar date the competition's live trading window begins (period 1,
#: exit_rule_spec.md's period convention). CONFIRMED by the team
#: 2026-10-11: the season started Monday 2026-09-28 (replacing the earlier
#: 2026-08-19 placeholder). Needed by `period_number_for` below, and by
#: `TradeSelector.run`'s live-mode path (baseline_system_architecture.md
#: open question 27) to know which window-day "today" is without depending
#: on whatever happens to be in a given day's scored DataFrame.
WINDOW_START_DATE: Optional[str] = "2026-09-28"


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

    This function does NOT check `WINDOW_START_DATE_CONFIRMED` -- it's a
    pure function of whatever start date it's handed (explicit arg, or
    the module default), including a placeholder one. Callers that read
    the module default (score_live.py) are responsible for checking
    `WINDOW_START_DATE_CONFIRMED` themselves and surfacing an unmistakable
    warning when it's `False` -- see score_live.py's own preflight/banner.
    """
    start = window_start_date or WINDOW_START_DATE
    if start is None:
        raise ValueError(
            "WINDOW_START_DATE is not set (config/hold_period.py) -- confirm the "
            "competition's real start date before computing a live period number."
        )
    start_ts, date_ts = pd.Timestamp(start), pd.Timestamp(date)
    return int(np.busday_count(start_ts.date(), date_ts.date())) + 1
