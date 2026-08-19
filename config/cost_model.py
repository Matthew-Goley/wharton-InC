"""Trading cost assumptions used by the selection layer's cost-clearing filter.

PLACEHOLDER VALUES. Commission rate and spread are not yet confirmed against
WInS's actual execution model — hardcode real numbers here once known (see
`baseline_system_architecture.md` §4, open question 6). Until then the
selection layer runs on these conservative-ish guesses, not competition fact.
"""
from __future__ import annotations

#: Round-trip commission as a fraction of trade value (one-way; a full
#: entry+exit pays this twice). WInS may not charge commission at all —
#: unconfirmed, so this is a nonzero placeholder rather than assuming free.
COMMISSION_RATE: float = 0.0005  # 5 bps one-way

#: Assumed bid-ask spread cost, in basis points of trade value, paid once
#: per side (so also effectively doubled over a full entry+exit).
ASSUMED_SPREAD_BPS: float = 5.0

#: A flagged trade's predicted edge must exceed round-trip cost by at least
#: this multiple before the selection layer will act on it (architecture §5,
#: "Selection threshold" ablation). 1.0 = breakeven after costs; > 1.0 is a
#: real margin of safety.
MIN_EDGE_MARGIN: float = 2.0

#: Per-stock position limit, as a fraction of total portfolio value.
#: Competition rule, not yet CONFIRMED against WInS Portfolio Summary ->
#: Rules -- but no longer `None`. `None` here is not a neutral "disabled"
#: state: TradeSelector._size_positions requires a real cap to size
#: anything at all (a live correctness bug otherwise -- baseline_system_
#: architecture.md open question 41 -- overlapping fixed-hold positions
#: compound exposure past 100% under any per-period-recomputed formula).
#: 0.10 is a deliberately chosen WORKING DEFAULT, not a guess at the real
#: competition rule: max_concurrent_positions = floor(1/POSITION_CAP_PCT)
#: = 10, chosen to match this project's own standing hold_days=5 (config/
#: hold_period.py) against its own standing pacing (trade_budget=200 over
#: window_trading_days=50 implies roughly 2 entries/period at steady
#: state) -- 5 * 2 = 10, so the cap and the pacing it has to coexist with
#: are mutually consistent by construction, not independently guessed
#: numbers that might starve entries or blow the budget. Replace with the
#: real competition value the moment it's confirmed.
POSITION_CAP_PCT: float | None = 0.10


def round_trip_cost_fraction() -> float:
    """Total cost (commission + spread, both sides) as a fraction of trade value."""
    return 2 * COMMISSION_RATE + 2 * (ASSUMED_SPREAD_BPS / 10_000)


def clears_cost(predicted_return: float, margin: float = MIN_EDGE_MARGIN) -> bool:
    """Whether a predicted edge is worth acting on after round-trip costs."""
    return abs(predicted_return) > margin * round_trip_cost_fraction()
