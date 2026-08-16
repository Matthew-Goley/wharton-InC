# Exit Rule — Fixed Holding Period (Baseline)

Status: spec, not yet implemented.
Scope: baseline variant only. Alternative exit rules are deferred to the ablation table (§7).

---

## 1. Rule

A position entered on trading day `d` is closed on trading day `d + N`, unconditionally.

No stop-loss, no take-profit, no signal-reversal exit, no early close. The only exception is the end-of-window force-close (§5).

`N` is a config parameter (`hold_days`), swept as described in §2.

Rationale: fixed hold is the only exit rule where realized holding period matches the model's prediction horizon by construction. Every variable-horizon alternative reintroduces a mismatch between what the experts were trained to predict and what the backtest scores.

---

## 2. Coupling to Training Target

`N` is not an independent knob. It defines the label.

| `hold_days` | Expert training target |
|---|---|
| N | N-day forward return |

Constraint: **changing `N` invalidates trained experts.** The label changes, so experts must be retrained and the GA search re-run.

Implication for the search: `N` is an **outer loop** around the existing per-expert GA search, not a member of the GA population. Total cost multiplies by the number of `N` values swept.

Suggested initial sweep: `N ∈ {1, 3, 5, 10}`. Keep it coarse — each value is a full retrain of every expert.

Config: `hold_days` in `config/`, read by both the feature/label builder and the selector. Single source of truth; a mismatch between label horizon and selector hold is a silent correctness bug, so it should be asserted at pipeline start, not assumed.

---

## 3. Budget Accounting

Every entry obligates a future exit. The 200-trade cap is therefore **100 round trips**, not 200 opportunities.

Definitions at any point in the backtest:

| Symbol | Meaning |
|---|---|
| `E` | trades executed so far (entries + exits) |
| `P` | currently open positions |
| `R` | `TRADE_BUDGET - E` — remaining trades |

Exits for currently open positions are already committed, so they are reserved:

```
free = R - P
max_new_entries = floor(free / 2)
```

The selector must never propose more entries than `max_new_entries` in a period. This replaces raw budget-remaining checks in the current `TradeSelector`.

Invariant to assert: `E + P <= TRADE_BUDGET` at all times.

---

## 4. Concurrent Position Ceiling

Entry rate is bounded by capital as well as budget. With equal-weight sizing capped at `position_cap_pct`:

```
max_concurrent_positions = floor(1 / position_cap_pct)
```

Under fixed hold, positions persist for `N` days, so steady-state concurrency is approximately `entries_per_day * N`. The binding constraint on entries per period is:

```
min(max_new_entries, max_concurrent_positions - P)
```

Note: `position_cap_pct` and `N` interact. A tight cap with a long hold will starve the entry rate and leave budget unspent; a loose cap with a short hold will exhaust budget early. Both values are currently placeholders (see open questions) — this interaction should be checked once real values are set.

---

## 5. End-of-Window Handling

Window: 50 trading days (10 weeks).

**Entry cutoff.** A position entered on day `d` requires `d + N <= 50`. No entries are permitted after day `50 - N`. Entries taken later would be force-closed early, breaking the horizon match that §1 exists to preserve.

**Force-close.** Any position still open on day 50 is closed on day 50. Under the entry cutoff this should never trigger; it exists as a safety net. If it fires, log it — it indicates the cutoff was not enforced.

Consequence: the effective entry window is `50 - N` days, not 50. Pacing logic must use the shorter window.

---

## 6. Interaction With Existing Constraints

| Constraint | Effect |
|---|---|
| No day trading | Satisfied for all `N >= 1`; entry and exit always fall on different days. |
| Max 2 trades/stock/day | Non-binding under fixed hold, with one exception: a ticker exiting on day `d` must not be re-entered on day `d`. Disallow same-day re-entry. |
| Long-only | Entries are long only; exits are sells closing an existing long. No change to §1. |
| Cost-clearing filter | Applied at entry only. Predicted N-day return must clear round-trip cost (both legs), not one-way — the exit is already committed. |
| `portfolio_state.json` | Must persist `entry_date` and `scheduled_exit_date` per position, since the exit is determined at entry time and must survive across manual execution sessions. |

---

## 7. Deferred to Ablation

Not in the baseline. Each makes holding period variable, which reintroduces the horizon mismatch and makes budget consumption stochastic. All should be measured against fixed-hold as control.

| Variant | Description |
|---|---|
| Stop-loss / take-profit | Early exit on price threshold |
| Signal-reversal exit | Exit when expert consensus flips |
| Ranked replacement | A higher-ranked candidate evicts a weaker holding |
| Variable hold by confidence | `N` scaled per-position by prediction confidence |

---

## 8. Test Cases

1. Every entry has exactly one matching exit, `N` trading days later.
2. Total executed trades never exceeds `TRADE_BUDGET`.
3. `E + P <= TRADE_BUDGET` holds at every period.
4. No entries proposed after day `50 - N`.
5. No positions open after day 50.
6. No ticker entered while already held.
7. No ticker re-entered on its own exit day.
8. `N = 1` produces no same-day round trip.
9. Label horizon in the feature builder equals `hold_days` in the selector — assert at startup, fail loudly on mismatch.
10. With `position_cap_pct` set such that `max_concurrent_positions < max_new_entries`, entries are correctly throttled by the capital ceiling rather than the budget.

---

## 9. Open Questions

- Optimal `N` is unknown; §2 sweep is the way to find it. Expect it to interact with sector — a single global `N` may be suboptimal, but per-sector `N` multiplies the retrain cost and is out of scope for the baseline.
- Whether unspent budget at window close indicates a too-conservative selector or a correctly selective one is not determinable from the backtest alone; needs comparison against a forced-full-budget variant.
