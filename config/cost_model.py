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
#: Competition rule, not yet confirmed — see WInS Portfolio Summary → Rules.
#: `None` disables the cap until a real value is hardcoded here.
POSITION_CAP_PCT: float | None = None


def round_trip_cost_fraction() -> float:
    """Total cost (commission + spread, both sides) as a fraction of trade value."""
    return 2 * COMMISSION_RATE + 2 * (ASSUMED_SPREAD_BPS / 10_000)


def clears_cost(predicted_return: float, margin: float = MIN_EDGE_MARGIN) -> bool:
    """Whether a predicted edge is worth acting on after round-trip costs."""
    return abs(predicted_return) > margin * round_trip_cost_fraction()
