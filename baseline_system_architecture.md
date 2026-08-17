# Trading System Architecture: XGBoost + Genetic Algorithm

## Competition Constraints
- No day trading — never buy and sell the same symbol same-day (WInS rule)
- No short selling, no margin — long-only
- Position limit: a cap on % of portfolio in any single stock (value TBD — confirm in WInS Portfolio Summary → Rules once available, then hardcode in `config/cost_model.py`)
- Maximum 200 trades over a 10-week period — the actual binding constraint (see below)
- Maximum 2 trades per stock per day — effectively non-binding: hitting it on a given stock would require the same-day entry+exit that "no day trading" separately prohibits. Kept here because it's a real rule, not because the system is expected to approach it.
- All execution is manual via WInS — the system's only output is a ranked recommendation; a human places every order
- These constraints mean the system must optimize for **selectivity**, not raw predictive accuracy

---

## 1. System Layers

### Layer 1: Data
- Source: Yahoo Finance (`yfinance`), **daily bars**
- Universe: ~685 tickers that were ever S&P 500 members across 2018–2026, survivorship-bias-free via point-in-time membership (`data/processed/sp500_membership.csv`) rather than today's roster applied backward
- Store in a unified, columnar format for fast loading
- Cleaning and feature engineering (moving averages, volatility, sector tags) happens here, mostly CPU/disk-bound

**Bar frequency — decided 2026-08-16, daily not hourly:**
- Originally assumed hourly, from a rows-per-ticker estimate that was never checked against what Yahoo actually serves. Verified: `yfinance` caps intraday (1h) data at ~730 days *measured from today*, not per request — chunking cannot walk further back than that. Hourly bars covering the full 2018–2025 window are not obtainable without a paid vendor (Polygon, Databento, Alpaca, Tiingo); that's a deferred option, not a pursued one.
- Even if it were available, intraday resolution buys nothing here: **no day trading is allowed**, so a signal can't be acted on and exited within the same session. Hourly's only structural advantage — more decision points per day — is unusable by rule.
- Row count was the wrong tradeoff to optimize. Hourly-capped-at-730-days actually has *more total rows* than 8 years of daily (~9.7M vs. ~6M at ~3,000 tickers) — what it costs is **calendar coverage**, 2 years vs. 8. A model that needs to survive regime shifts needs the 8 years of distinct regimes daily bars provide, not the extra row density hourly would add from inside a single regime.
- Net: daily bars over the full ~8yr history is both the only feasible option and the correct one under this competition's rules.

### Layer 2: Sector Experts
- Universe split by sector (tech, financials, real estate, etc.)
- One XGBoost model trained per sector, on that sector's data only
- Each sector expert gets its own genetic algorithm search (population of hyperparameter candidates: tree depth, learning rate, number of trees, etc.), evolved against its own validation slice

### Layer 3: Broad Model
- One XGBoost model trained across the full universe, all tickers pooled
- Captures general, market-wide patterns (momentum, volatility clustering) that aren't sector-specific
- Gets its own separate genetic algorithm search

### Layer 4: Combination
- **Starting approach: weighted blending.** Broad model and relevant sector expert predictions are combined via a weighted average, weights tuned during the genetic algorithm search.
- **Future exploration: stacking.** Sector expert + broad model predictions become input features to a small final model that learns how much to trust each expert. More powerful but needs its own training data to learn on, which is expensive given the short competition window — revisit once more historical backtest seasons are banked.

### Layer 5: Trade Selection (built on top of layers 2-4)
- Not just "will this go up" but "how confident is the model" — via prediction intervals or agreement between broad model and sector expert
- **Long-only**: no short selling, no margin — only act on positive predicted-edge signals
- **Cost-clearing filter**: only flag a trade if predicted edge clears commission + estimated spread by a margin (`config/cost_model.py`) — an "opportunity" that doesn't survive costs isn't one
- **Position sizing**: a parametrized rule — equal-weight to start, confidence-weighted or vol-scaled are candidates — capped at the per-stock position limit. A calibrated rule tuned by backtest grid search, not a second trained model; trade counts are far too small to fit one without overfitting.
- **Reads current portfolio state** (`data/processed/portfolio_state.json`, updated by hand after every real trade — WInS has no API) before sizing new trades, since the system has no other way to know what's already held
- **Exit rule: fixed holding period** (`exit_rule_spec.md`). A position entered on day `d` closes on `d + hold_days` (N), unconditionally — no stop-loss, no take-profit, no signal-reversal exit. `hold_days` isn't a knob independent of the model: it *is* the training label (N-day forward return), so it's swept as a full retrain per value (§3), not tuned by the GA. The 200-trade budget is therefore ~100 round trips, not 200 opportunities — every entry reserves a unit for its own future exit.
- Trade budget tracker: counts down from 200 (as entry+exit pairs), paces spending across the 10 weeks rather than front-loading
- Expected value ranking: combines predicted return size with confidence to pick the top slice of opportunities, since only ~100 round trips total can be made across the whole universe and window
- No day trading, satisfied by construction: an entry and its exit always fall on different days (hold_days ≥ 1), and a ticker exiting on day `d` cannot be re-entered on that same `d`

---

## 2. Training & Validation Plan

### Training Data
- Train on the full ~8 years of history, not just a window matching the 10-week competition length — more history means the model learns real patterns instead of noise from one short period
- Weight recent months more heavily in the loss/sampling, so the model stays tuned to current market regimes rather than treating 2019 the same as this year

### Validation Approach
- Use walk-forward validation throughout: train only on past data, test only on future data, never let the model see ahead
- Standard accuracy-based validation is not sufficient given the trade cap. Add a **"top-200 selection quality" backtest**: instead of scoring every prediction the model makes, simulate the model only acting on its top-ranked ~200 opportunities across a historical window, and score performance on that slice specifically. A model can have mediocre average accuracy but strong top-slice accuracy — that's what actually matters under this constraint.

### Trade Definition Under the Competition's Rules
- No day trading means a trade is a same-symbol entry on one day and an exit on a later day — never both within one session
- On daily bars this is the natural unit of decision anyway: one ranked score per ticker per day, one action (or no action) per ticker per day
- The 2-trades/stock/day cap from the rules is real but isn't expected to bind under this design — hitting it would require exactly the same-day round trip that's separately prohibited

### Compute: CPU, not GPU — decided 2026-08-16
- The Arc B70 isn't usable here at all: XGBoost's GPU acceleration (`device="cuda"`) is NVIDIA-only — confirmed via `xgboost.build_info()` on the installed build (`USE_CUDA: True`, `USE_NCCL: True`, no SYCL/oneAPI backend). `theta-core`'s GPU usage (`torch.xpu`) works because PyTorch ships a native Intel GPU backend; XGBoost has no equivalent.
- Even setting hardware aside, GPU is the wrong lever for this workload's shape. The GA search's cost is thousands of small, independent fits (7 features, ≤~1.4M rows for the broad expert), not one large model to accelerate — that's what GPUs are good at, not what this is.
- Benchmarked on the dev machine (i5-12600K, 16 threads) at realistic scale (400k rows, 7 features): a single fit goes from 3.25s (1 thread) to 0.97s (16 threads) — XGBoost's own CPU histogram parallelism already captures most of the available speedup. A first-pass test of running GA candidates across a multiprocess worker pool instead showed no further gain on top of that (~1.02x) — inconclusive, not a demonstrated win, so not pursued further without a real need.
- If GA search throughput becomes an actual bottleneck at full production scale (population 50 × generations 20 × 4 experts × up to 4 folds × 3 models/candidate), the productive next steps are early stopping (cut wasted boosting rounds past convergence) and a properly-tuned worker pool — not GPU acquisition, which wouldn't work with this library regardless of tuning.
- The entry+exit pairing is now literal, not just conceptual: the fixed-hold exit rule (`exit_rule_spec.md`) schedules every entry's exit at entry time, `hold_days` trading days later

---

## 3. Ablation Plan — Variations to Test

| Variation | Description |
|---|---|
| Baseline | Model predictions only, trades taken purely on rank order, fixed-holding-period exit (`hold_days` swept as its own outer loop, not a GA gene) |
| Weighted blend vs. stacking | Compare simple blending against a stacked meta-model once enough backtest history is banked |
| Fixed-weight vs. dynamic blend | Compare static blend weights against a genetic-algorithm-evolved blend that adjusts sector-expert trust based on that sector's current volatility |
| Even pacing vs. front-loaded budget | Compare spreading the 200-trade budget evenly across 10 weeks vs. spending more early when historical model confidence tends to be higher |
| Selection threshold | Compare how strict the cost-clearing margin should be — too loose spends the trade budget on marginal edges that barely survive costs, too strict leaves budget unspent |
| Selection score | `OverallModel.rank(ranking_score=...)`: raw predicted return vs. `vol_adjusted` (divide by recent realized volatility before ranking) vs. `uncertainty_adjusted` (planned — rank by a lower confidence bound derived from broad/sector-expert disagreement, once sector experts exist to disagree). First real-data backtest found the single top-ranked pick each day is a reliable loser (mean −4.1%, 62 trades, 60% concentrated in one ticker mid-merger) driven by argmax-over-noisy-predictions selecting outsized, high-volatility names; `vol_adjusted` demotes that real but only partially (mean improves to −2.8%, still the worst-performing rank) — residual is plausibly selection bias in the argmax itself, which volatility can't see and only a second, independent estimate (disagreement) can. |
| Stop-loss / take-profit exit | Early exit on a price threshold, vs. fixed-hold |
| Signal-reversal exit | Exit when expert consensus flips, vs. fixed-hold |
| Ranked replacement | A higher-ranked candidate evicts a weaker existing holding, vs. never evicting |
| Variable hold by confidence | `hold_days` scaled per-position by prediction confidence, vs. one global N |

Run all variations across the same walk-forward, top-200-style backtest windows so results are comparable and it's clear what's actually contributing to performance. Fixed-hold is the control for the last four rows: every alternative makes holding period variable, which reintroduces a mismatch between what the experts were trained to predict (an N-day forward return) and what the backtest actually scores — see `exit_rule_spec.md` §1, §7.

*Cut from the original plan: "Fixed schedule," "Event-triggered," and "Combined" timing/trigger variations. All three assumed intraday decision points — evolving a checking schedule or a trigger threshold only means something if there's more than one legal decision point per ticker per day. Under daily bars and the no-day-trading rule there is exactly one, so the timing layer they were testing doesn't exist. See the Bar Frequency decision in Layer 1.*

---

## 4. Open Questions / Where More Testing Is Needed

This is the ambiguity that current reasoning hasn't resolved — flagging explicitly so it gets tested rather than assumed. (Three questions from the original list are removed as resolved: hourly-vs-intraday checking frequency and GA interaction between model quality and timing strategy — both were about the timing/trigger layer, which doesn't exist under daily bars + no day trading, see the Bar Frequency decision in Layer 1 and the cut ablation rows above — and GPU vs. CPU training, see the Compute decision in §2.)

1. **What should the selection threshold actually be built from?** Candidates include prediction intervals, ensemble agreement between broad and sector models, and volatility — but which combination best predicts a genuinely good trade opportunity, and how strict the cost-clearing margin should be, is unverified.

2. **Weighted blending vs. stacking performance gap** — untested. Need enough banked backtest history to know whether stacking's added complexity is actually worth it, or whether blending captures most of the value.

3. **How much should recent data be up-weighted?** The right decay rate/recency weighting for the 8 years of training data against current market regime is unknown and needs to be swept as its own parameter.

4. **Pacing of the 200-trade budget** — even spend vs. front-loaded vs. confidence-adaptive pacing needs a direct backtest comparison; there's no a priori reason to assume one is better without testing on real historical windows.

5. **Sector boundary definitions** — how sectors are split (tech, financials, real estate, etc.) will affect how much each sector expert actually specializes vs. overlaps with the broad model. Different sector groupings should be tested for whether they meaningfully change expert quality.

6. **Cost model and position limit are placeholders.** Commission rate, assumed spread, and the per-stock position-limit percentage aren't confirmed yet (see `config/cost_model.py`, WInS Portfolio Summary → Rules) — the selection layer runs on placeholder values until the real numbers are filled in.

7. **Optimal `hold_days` (N) is unknown** — the sweep in `exit_rule_spec.md` §2 is how to find it, but expect it to interact with sector (a single global N may be suboptimal, though per-sector N multiplies retrain cost and is out of scope for the baseline). Whether unspent budget at window close means the selector was too conservative or correctly selective also isn't determinable from a backtest alone — needs comparison against a forced-full-budget variant.
