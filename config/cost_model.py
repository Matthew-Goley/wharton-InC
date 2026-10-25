"""Trading cost and position-limit values used by the selection layer.

Confirmed by the team 2026-10-11: $5 flat commission per trade, 20%
per-stock position cap, $300,000 starting cash. The spread figure is still
an assumption (WInS's fill model is unknown) -- see ASSUMED_SPREAD_BPS.
"""
from __future__ import annotations

#: WInS commission: a flat dollar fee per trade (each side -- an entry and
#: its exit are two trades), NOT a percentage of trade value. Confirmed.
COMMISSION_PER_TRADE_USD: float = 5.0

#: Starting portfolio value. Confirmed. Used only to express the flat
#: commission as a fraction of a typical position (COMMISSION_RATE below);
#: the live cash balance itself lives in data/processed/portfolio_state.json.
STARTING_CAPITAL_USD: float = 300_000.0

#: Assumed bid-ask spread cost, in basis points of trade value, paid once
#: per side (so also effectively doubled over a full entry+exit). Still an
#: ASSUMPTION -- WInS may simply fill at the last price, in which case this
#: is conservative. 5 bps is generous for S&P 500 large caps.
ASSUMED_SPREAD_BPS: float = 5.0

#: A flagged trade's predicted edge must exceed round-trip cost by at least
#: this multiple before the selection layer will act on it (architecture §5,
#: "Selection threshold" ablation). 1.0 = breakeven after costs; > 1.0 is a
#: real margin of safety.
MIN_EDGE_MARGIN: float = 2.0

#: Per-stock position limit, as a fraction of total portfolio value -- the
#: real competition rule (WInS Portfolio Summary -> Rules), confirmed. A
#: CEILING, not the size this system trades at: see POSITION_SIZE_PCT.
POSITION_CAP_PCT: float = 0.20

#: Fixed size of every entry, as a fraction of portfolio value -- a strategy
#: choice, which must stay <= POSITION_CAP_PCT (score_live.py refuses
#: otherwise). It also sets max concurrent positions = floor(1/size), which
#: is why it isn't simply the 20% cap: at 0.20 the selector could hold only 5
#: names, and with hold_days=5 that admits ~1 entry/day, so roughly half the
#: 200-trade budget would go unspent (baseline_system_architecture.md item
#: 51's simulation already showed 10 slots slightly binding at hold_days=5).
#: 0.10 = 10 concurrent names, matched to hold_days=5 x ~2 entries/day of
#: pacing (item 41), and more diversified for a signal this weak (IC ~0.02).
#: Raise toward POSITION_CAP_PCT only as a deliberate concentration decision.
POSITION_SIZE_PCT: float = 0.10

#: The flat fee expressed as a one-way fraction of a typical position's
#: value ($5 / $30,000 = ~1.7 bps at the defaults above). An approximation:
#: the real fraction drifts as portfolio value moves, but a few-percent
#: drift on a ~2 bps cost is immaterial next to the spread assumption.
COMMISSION_RATE: float = COMMISSION_PER_TRADE_USD / (POSITION_SIZE_PCT * STARTING_CAPITAL_USD)


def round_trip_cost_fraction() -> float:
    """Total cost (commission + spread, both sides) as a fraction of trade value."""
    return 2 * COMMISSION_RATE + 2 * (ASSUMED_SPREAD_BPS / 10_000)


def clears_cost(predicted_return: float, margin: float = MIN_EDGE_MARGIN) -> bool:
    """Whether a predicted edge is worth acting on after round-trip costs."""
    return abs(predicted_return) > margin * round_trip_cost_fraction()
