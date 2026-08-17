# Baseline Training System

Implementation of [`baseline_system_architecture.md`](baseline_system_architecture.md)
§§2–5 (Sector/Broad Experts, Genetic Algorithm, Combination Layer, Trade
Selection). §1 (Data Layer) has a real fetch pipeline (merged in from
`wharton-InC`, see below) but no `DataProvider` wired up to it yet — see
"Wiring in the real Data Layer".

**Bar frequency is daily, not hourly** — decided 2026-08-16 after finding
`yfinance` can't serve hourly bars back to 2018, and that intraday
resolution isn't usable anyway given the competition's no-day-trading rule.
Full reasoning in the architecture doc §1 and `CLAUDE.md`.

## Layout

```
config/
  stock_list.py       Wharton's approved (tradeable) tickers — empty, TBD.
  training_universe.py Broad training universe (generated, 685 tickers).
  cost_model.py         Commission/spread/position-cap assumptions used by
                        the selection layer's cost-clearing filter — all
                        placeholder values pending confirmed competition rules.
  hold_period.py         hold_days (N) and the competition window length —
                        single source of truth for the fixed-hold exit rule
                        (exit_rule_spec.md), read by both the label horizon
                        and the selector.

data/
  raw/training/         ~685 tickers, daily OHLCV, 2018-2026 (from wharton-InC).
  raw/reference/         Point-in-time S&P 500 membership source.
  processed/             sp500_membership.csv, portfolio_state.json
                        (hand-maintained — WInS has no API).

src/
  data_pipeline/       Fetch scripts (yfinance pull, point-in-time universe
                      builder), merged in from wharton-InC. Not yet wired
                      into a DataProvider — see below.
  config.py          Central config: expert roster, GA settings, hyperparameter
                      search space, walk-forward settings, trading rules
                      (budget, long-only, cost margin, position cap).
  data_layer/         §1. interface.py defines DataProvider/FeatureTable;
                      mock_provider.py is a synthetic stand-in for fast
                      iteration; real_provider.py reads the actual price
                      CSVs (broad expert only until sector data exists).
  experts/            §2. ExpertModel wraps three XGBoost models (mean-return
                      regressor + two quantile regressors) into a single
                      predict_return + confidence output. registry.py slices
                      the shared feature table per expert (broad vs. sector).
  ga/                 §3. genome.py: genome = XGBConfig, plus sample/mutate/
                      crossover operators. population.py: selection + elitism.
                      search.py: the generational loop, fitness = mean
                      walk-forward Information Coefficient.
  evaluation/          Walk-forward (train-on-past/test-on-future) splitting,
                      plus the IC / RMSE scoring functions used as fitness.
  combiner/            §4. OverallModel: confidence-weighted combination of
                      broad + sector expert predictions, with a *signed*
                      agreement bonus (reinforces whichever direction both
                      experts share, bullish or bearish), then per-period ranking.
  selection/           §5. TradeSelector implements the fixed-holding-period
                      exit rule (exit_rule_spec.md): a position entered on
                      period d closes on d + hold_days, unconditionally.
                      Enforces long-only, cost-clearing (round-trip, charged
                      once at entry), equal-weight sizing capped at
                      position_cap_pct, an entry cutoff so nothing is
                      entered too late to exit inside the window, and a
                      logged force-close safety net. Budget accounting
                      treats every entry as obligating a future exit (200
                      trades = ~100 round trips), and the concurrent-
                      position ceiling (1/position_cap_pct) throttles
                      entries alongside the budget. No timing/trigger logic
                      beyond that, by design — see the doc's Ablation Plan.
                      portfolio_state.py: reads/writes the hand-maintained
                      data/processed/portfolio_state.json, including each
                      position's entry_date/scheduled_exit_date.
  pipeline/            train_baseline.py orchestrates: DataProvider (asserts
                      its label horizon matches hold_days) -> per-expert GA
                      search -> train experts -> combine -> rank -> select
                      trades (portfolio-aware, restrictable to a live
                      trading window via trading_window_periods).
                      sweep.py: the hold_days outer loop (spec §2) — N
                      defines the training label, so each value swept is a
                      full retrain, not a GA gene.

scripts/run_baseline_demo.py   Runs the full pipeline against
                                MockDataProvider with small/fast GA settings,
                                to prove the wiring works end to end.
tests/                          Unit tests for walk-forward splitting, GA
                                operators/selection, the combiner's scoring
                                math, the trade selector's rule enforcement,
                                and ExpertModel's fit/predict contract.
```

## Running

```
pip install -r requirements.txt
python -m pytest                    # unit tests (~2s)
python -m scripts.run_baseline_demo # full pipeline on synthetic data (~30-60s)
```

## The real Data Layer

`src.data_layer.RealDataProvider` reads the actual `data/raw/training/*.csv`
price files: feature engineering, an N-day forward return target (N =
`hold_days`), and point-in-time universe filtering (as-of join against
`data/processed/sp500_membership.csv`, avoiding survivorship bias). Uses
`Close`, never `Adj Close` — the latter is retroactively rewritten on every
dividend, which leaks future information into the past.

Validated before trusting it with a long run: a negative control (pure
noise target — mean IC ≈ 0, no leakage) and a positive control (planted
signal recovered near its theoretical ceiling) confirm the pipeline itself
is correct.

**Features are scale-free, and chosen by documented effect rather than by
measured strength.** The first version used raw `ma_5`/`ma_20`/`ma_60`/
`atr_14`, which are dollar amounts — pooled across a universe spanning \$12
to \$96,938 a share they encoded "how expensive is one share", and all four
measured as noise. Everything is now a ratio, except `log_dollar_volume` and
`amihud_20` where the dollar amount *is* the economic quantity (size, and
price impact per dollar traded).

The current 15 are one or two representatives per documented cross-sectional
effect: short-term reversal (Jegadeesh 1990), momentum skipping the recent
stretch (Jegadeesh & Titman 1993), 52-week-high proximity (George & Hwang
2004), idiosyncratic volatility (Ang et al. 2006), good-vs-bad volatility
(Bollerslev et al.), lottery demand (Bali, Cakici & Whitelaw 2011), realized
higher moments (Amaya et al.), illiquidity (Amihud 2002), size, volume
dynamics, and Med_k jumps.

**Selecting features on their t-statistics was tried and measured worse than
not selecting at all.** On a 2018–2023 / 2024–2025 split, a set picked by
|dev t| > 2.5 scored S/N 0.75 out of sample; the literature-assembled set
scored 3.78 on the same holdout; throwing in all 32 candidates scored 3.01.
Seven of the strongest in-sample features flipped sign out of sample. Add a
feature because a paper says it should work, then let the walk-forward judge.

The target is each name's forward return minus that date's universe mean,
removing the ~33% of variance that is just the whole market moving.
`fwd_return_raw` and `market_fwd_return` stay in the frame for reporting.

**Jump ratio.** `jump_ratio` implements the Med_k standardized return of
Khashanah, Chen, Buckle & Hawkes (*JRSS-C* 2025) — today's log return over a
jump-robust local volatility built from a running median of k = 9 absolute
returns. A median denominator, unlike bipower variation, is not inflated by
a *cluster* of large moves, so consecutive jumps don't mask each other (on
the 2010 flash crash, bipower variation exceeded realized variance and
reported no jumps at all). Adapted for prediction rather than detection: the
window is trailing rather than centred, and the denominator is lagged one
period so a jump never contributes to the volatility it is measured against.
On its own it is not yet significant (t = −1.74).

**Fitness is mean per-date IC, not pooled.** `cross_sectional_ic` computes
rank correlation *within* each date, then averages. Pooling every
(ticker, date) row into one correlation mixes stock-picking skill with
"which days were up days", and the two disagree even in sign —
`volatility_20` scores +0.0386 pooled and +0.0027 per-date. Fold geometry is
sized against that metric's noise: per-date IC has a standard deviation of
~0.19, so the old 100-period validation windows had a standard error of
0.019 against a signal of ~0.013, and the search was ranking candidates on
luck. The current 4 × 315-period folds give ~0.0054 averaged — a
signal-to-noise ratio of 2.4x, up from 1.4x — and now reach the final day of
the data instead of stopping 17 months short.

**Where the signal actually stands.** Walk-forward, out of sample, with a
fixed reference model: mean IC **+0.0114**, positive in 3 of 4 folds
(−0.0116 / +0.0203 / +0.0147 / +0.0223). The negative fold is the
2020-12→2022-03 meme-stock window, where reversal signals inverted. The old
8-feature set scored +0.0020 on the same folds, so this is roughly a 6x
improvement — but the level is still small, and regime-dependent.

Two S/N figures, because the second is the honest one. Naive (`sd/√n`):
**3.32**. Newey-West corrected: **1.98**. The label is a 5-day forward
return computed daily, so adjacent dates share 4/5 of their target window;
that autocorrelation inflates naive standard errors by ~1.65x here. Quote
the Newey-West number.

**Known gap: no sector data yet.** No sector classification has been
sourced (`config/sector_map.py` doesn't exist per CLAUDE.md's plan), so
every row gets `sector="Unknown"` — a sector expert (tech/financials/
real_estate) would train on zero rows. Build `BaselineConfig` with
`experts=(ExpertSpec("broad", None),)` when using this provider until
sector data exists — see `scripts/train_broad_real.py`.

## Exit rule: fixed holding period

Resolved per [`exit_rule_spec.md`](exit_rule_spec.md) (was previously an
open gap — `TradeSelector` only proposed entries, never exits). A position
entered on period `d` closes on `d + hold_days`, unconditionally — no
stop-loss, no signal-reversal, no early close. That means:

- `hold_days` isn't a GA gene — it defines the training label (N-day
  forward return), so sweeping it (`src.pipeline.sweep.sweep_hold_days`)
  means a full retrain per value, not a population member.
- The 200-trade budget is really ~100 round trips: every entry reserves a
  budget unit for its own future exit.
- Demo output: 200/200 budget spent as 100 entries + 100 exits, 0 positions
  left open, 0 force-closes needed (`python -m scripts.run_baseline_demo`).

Stop-loss/take-profit, signal-reversal exits, ranked replacement, and
confidence-scaled holding periods are deferred to the ablation table (spec
§7) — each makes holding period variable, which reintroduces the mismatch
between what the experts were trained to predict and what gets backtested,
so fixed-hold is the control.

## A calibration note worth keeping in mind

`ExpertModel` standardizes the target (zero mean, unit variance) before
fitting and un-standardizes predictions afterward. Without this,
`gamma`/`reg_alpha`/`reg_lambda` — which operate in the target's own units —
silently prune every split on a small-magnitude return target (e.g. daily
return std ~0.01–0.05), collapsing every expert to a constant predictor
regardless of hyperparameters. Standardizing keeps the GA's search space
meaningful no matter the return scale of a given sector or bar frequency.
See `tests/test_expert_model.py::test_works_regardless_of_target_scale`.

## Scale note

`src.config.BaselineConfig()` defaults (population 50, 20 generations) are
sized for a real run, not for quick iteration — a single expert's search
alone is O(1,000) XGBoost fits. `scripts/run_baseline_demo.py` overrides
these to something fast; do the same for local iteration and dial back up
for the real thing.

`GeneticSearch(..., max_workers=N)` parallelizes population evaluation
across a process pool (each worker single-threaded, so N processes don't
also each grab every core) — `src.ga.search.default_max_workers()` picks a
safe default (cores − 2). Measured on a 16-thread machine: ~3.9x speedup
over sequential, identical results. At real universe scale (~580 tickers,
~1M rows), one generation of the broad expert alone takes ~13 minutes even
parallelized — `scripts/train_broad_real.py` is the real, long-running
entry point; budget hours, not minutes, for a full search.
