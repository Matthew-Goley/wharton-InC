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

#: N — a position entered on day d is closed on day d + N, unconditionally.
HOLD_DAYS: int = 5

#: Coarse outer-loop sweep candidates (spec §2). Each value is a full
#: retrain of every expert, so keep this short.
HOLD_DAYS_SWEEP: tuple[int, ...] = (1, 3, 5, 10)

#: Competition window: 10 weeks * 5 trading days/week (spec §5).
WINDOW_TRADING_DAYS: int = 50
