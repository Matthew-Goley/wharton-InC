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
| Agreement bonus on vs. off | `OverallModel(agreement_bonus=...)`: 0 disables it. First real `has_sector=True` run (Information Technology checkpoint) found agreement and accuracy aren't the same thing — two experts can share a blind spot on a chaotic name (broad model and a sector model can both read PARA's momentum features as extreme) as easily as they can confirm a real signal, and the bonus *amplifies* whichever direction they agree on. That's the opposite operation from `uncertainty_adjusted` (which *demotes* broad/sector disagreement) reading the same broad-vs-sector gap — potentially complementary, potentially redundant, potentially fighting each other. Not assumed to belong in the final combiner; run both ways once `uncertainty_adjusted` exists to find out. (A prior version of this bonus was additive in raw-return units and was found to dominate real predictions — fixed to multiplicative, which keeps it proportional to the prediction it reinforces; see `OverallModel.combine`'s comment.) Decomposed the fix's effect on the IT checkpoint rather than trusting the aggregate (`src.evaluation.decompose_selection_change`, now the standard tool for any scoring-change comparison — see `data/processed/agreement_bonus_fix_decomposition.json`): rank-1 is untouched by the fix (−4.11% broad-only → −3.62% with IT covered — that's step 4's job, not the blend's). The fix reshuffles 66.7% of all entries (663/994 dropped, 663/994 added, only 331 common to both scorings), and the apparent aggregate improvement (+0.25% → +0.43% mean/trade) is *not* statistically distinguishable from noise on that differing subset — block-effective SE (non-overlapping `hold_days`-wide trading blocks, not raw trade count, as the independence unit) gives t≈0.19, and even the maximally generous raw-n-no-overlap-correction case only reaches t≈0.77. Real mechanical fix (verified: bonus share of `combined_score` dropped from median 76% to ~4.6%), but no aggregate performance claim survives at this sample size — only training-path-validation and rank-1-untouched are load-bearing return findings from this checkpoint. One thing *is* a measured fact, not an estimate, and survives the effective-n problem entirely because it's a count: the dropped set concentrates into 30/100 available trading blocks (22.1 trades/active block) vs. the added set's 48/100 (13.8 trades/active block) — the fix spread trading activity across 60% more distinct time periods, a real reduction in temporal concentration risk regardless of whether mean return moved. See open question 8 for the pre-registered prediction this episode's common-set pattern (double-selected trades realizing worse than either side's unique picks) motivates for step 5. |
| Per-ticker frequency cap | `TradingConfig.max_ticker_share_pct` (None by default, off): caps any one ticker's share of the window's total entries, enforced in `TradeSelector._eligible`. Added because concentration, not any one ticker, turned out to be the invariant: the single most extreme-scoring name's share of the rank-1 *entry* bucket was 60%, 46%, 62%, 81%, 75% across five completely different scoring architectures (broad-only, buggy bonus, fixed bonus, full sector coverage, full coverage minus the two worst experts) — the architecture changed four times and the concentration never moved, which is what makes it the thing to fix, not PARA specifically (a different chaotic ticker would take its place in a different period). Deliberately ticker-agnostic — no hardcoded exclusion list — both because no corporate-actions data exists to build one from (only Yahoo daily bars) and because a fitted list would be exactly the kind of sample-specific rule this project has already refused three times (rank 3–5, the bucket boundaries, skip-rank-1). Calibrating it surfaced two things worth recording on their own: (1) PARA's *total* entry count (16/994, 1.6%) isn't unusual — TSLA (14) and ALB (13) are close behind — its pathology is concentrated in *which rank* those entries win (15/16 are rank-1), not overall volume, so a cap has to sit below ~1% share to bind at all; a 5% cap did nothing. (2) The underlying dominance is starker than the entry-level numbers suggested: PARA is the raw daily argmax winner (rank==1 by score, before any trading mechanics) on 245/497 test days — 49.3%, not far from half the window. Capping at 1% and comparing the "period champion" (the best-ranked entry actually taken each day — the PARA-independent analog of rank-1, since the static rank label isn't recomputed after a capped ticker is excluded) shows a real-direction shift: mean −0.80% (uncapped, n=101, PARA 15 champion-days) → +0.25% (capped, n=103, PARA 8 champion-days), unique tickers 49→61. Decomposed rather than trusted as a headline delta: block-effective t=+1.41 — real direction, not conventionally significant, the same verdict as almost everything else measured this way in this project. See `scripts/rerun_full_sector_with_ticker_cap.py` and `data/processed/backtest_full_sector_capped_real.json`. |
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

8. **Pre-registered prediction for the agreement-bonus ablation (step 5).** Decomposing the bonus scale-fix's IT checkpoint found the *common* set — trades both the buggy and fixed scoring picked — realized worse (mean −0.23%, n=331) than either scoring's unique picks (+0.48% / +0.77%, n=663 each), though not significantly so (t≈0.4–1.1 depending on plausible block count and dispersion — same effective-n caveat as everything else here). Mechanistic account: surviving both scorings means scoring high on two different extremes at once (high agreement-bonus under the old formula, high predicted-return-share under the new one) — a stronger, compounded version of the same argmax/winner's-curse selection bias documented in the "Selection score" row, not a separate effect. If that account is right, it makes a falsifiable, checkable prediction once `uncertainty_adjusted` exists: **`agreement_bonus=off` should outperform `agreement_bonus=on`, and the common-set penalty should shrink or vanish under `agreement_bonus=off`.** Written down now, before step 5 runs it, so the result gets checked against this rather than fitted to whatever comes out. It can fail.

9. **PARA-class names may not be an ensemble problem at all — tested directly, not fixed by more experts.** Full sector coverage (all 11 experts, `scripts/train_all_sector_experts_real.py`) made rank-1 *worse*, not better: n=21, mean −3.97%, median −7.29%, win 42.9% (the median more extreme than the mean means most rank-1 picks were far worse than −7%, offset by a few large positive outliers). Decomposed: 17/21 (81%) are PARA, via the `communication_services` expert — which itself measured negative out-of-sample IC (−0.0164) — agreeing with broad on a bullish read the market kept punishing (11 of those trades: −7% to −41%). Tested the obvious fix: re-ran with `communication_services` and `materials` (the two clearly negative-IC experts, −0.0164 and −0.0422) excluded from the roster entirely. **Rank-1 did not recover** — n=20, mean −4.68%, median −5.68%, win 30%, still 75% PARA — because without sector coverage PARA just reverts to broad-only, which was the *original* problem this whole investigation started from. Broad and every sector model trained on the same 15-feature set read PARA's merger-driven price action as genuine bullish momentum, the same wrong way, independent of which model or how many. That means the fix is more likely at the feature or eligibility layer — an M&A/corporate-action flag, or excluding tickers under a pending deal from trading entirely — not the ensemble architecture. It also means `uncertainty_adjusted` (step 4), which demotes broad/sector *disagreement*, structurally cannot catch this case: PARA's failure mode is agreement with a shared blind spot, not disagreement. Step 5 can still validate `uncertainty_adjusted` on its own pre-registered terms (item 8), but should not be expected to fix PARA, and a positive step-5 result shouldn't be read as having done so.

   One other finding from the same full-coverage run: **measured, not estimated** — trades unique to full coverage (vs. broad-only) concentrated into just 21/100 trading blocks (44.2 trades/active block) vs. broad-only's unique trades spanning 63/100 (14.7/block) — a real deterioration in temporal spread, the opposite direction from the agreement-bonus fix's improvement (item 8's episode moved 30→48 blocks).

   **The 4 weak sector experts, resolved precisely rather than by an assumed formula.** Computed the actual per-date IC series for every expert (not just the aggregate) and its empirical SE (std across dates / √n_dates) — the honest version of "is this distinguishable from zero," same standard as everywhere else here:

   | expert | median names/day | mean IC | t |
   |---|---|---|---|
   | broad | 492 | +0.0209 | +4.27 |
   | tech | 78 | +0.0240 | +3.40 |
   | consumer_discretionary | 56 | +0.0167 | +2.55 |
   | health_care | 60 | +0.0160 | +2.53 |
   | real_estate | 31 | +0.0193 | +2.34 |
   | energy | 20 | +0.0231 | +2.13 |
   | utilities | 31 | +0.0167 | +1.76 |
   | financials | 67 | +0.0049 | +0.61 |
   | consumer_staples | 36 | −0.0010 | −0.14 |
   | industrials | 71 | +0.0003 | +0.04 |
   | communication_services | 20 | −0.0164 | −1.43 |
   | materials | 22 | −0.0422 | **−3.52** |

   Only 2 of the 4 originally-flagged sectors are actually explained by small-universe noise: consumer_staples (t=−0.14, genuinely flat) and communication_services (t=−1.43, matches the ballpark estimate, not conventionally significant). The other 2 don't fit that story: **materials is not noise** — t=−3.52 is a real, statistically distinguishable negative signal, not sampling imprecision, so folding it into a merged "other" bucket isn't obviously a fix (diluting a real anti-signal with more data might help or might just spread it) — exclusion from the roster is the better-supported move until shown otherwise. **industrials is unexplained by sector size** — 71 names/day, the second-largest sample in the table (bigger than tech's 78-name sector, which scored a clean +3.40), yet its IC sits at exactly zero. Sector size isn't the deciding factor at all: energy and communication_services have the *same* median sample (20/day) and land on opposite sides of significance (+2.13 vs −1.43); real_estate and utilities (31/day, same bracket as consumer_staples' 36) both score fine. Whatever explains materials and industrials, it isn't "the universe was too small to measure" — that explanation only holds for the 2 sectors it was checked against. Revise the plan: exclude materials from the roster on the strength of its measured negative IC; leave industrials as a genuinely open question (large-enough sample, flat result, no candidate explanation yet) rather than merging it away; consumer_staples and communication_services can reasonably fold into a broader bucket if desired, since their case for "not enough signal to measure" actually holds.

   Separately, communication_services' aggregate insignificance (t=−1.43) doesn't reduce its demonstrated, mechanistic harm on PARA specifically (17/21 rank-1 entries in the full-coverage run, traced directly to this expert agreeing with broad's shared misread) — an expert can be indistinguishable-from-zero on average while still being actively wrong, and confidently so, on a specific chaotic name. These are different claims computed from different evidence; neither overturns the other.
