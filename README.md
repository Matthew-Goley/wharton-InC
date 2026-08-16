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
  data_layer/         §1 contract. interface.py defines DataProvider/
                      FeatureTable; mock_provider.py is a synthetic stand-in
                      so everything below is runnable without real data wired up.
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

## Wiring in the real Data Layer

Everything downstream depends only on `src.data_layer.DataProvider`:

```python
class DataProvider(ABC):
    def get_feature_table(self, start: str, end: str, hold_days: int) -> FeatureTable: ...
```

`FeatureTable` is a `(ticker, timestamp)`-indexed frame with named feature
columns, a target column (an N-day forward return, N = `hold_days`,
recorded on `target_horizon_days` so `train_baseline` can assert the label
actually matches the selector's exit timing), and a sector column. The raw
data and fetch
scripts already exist (`data/raw/training/*.csv`, `src/data_pipeline/`) —
what's missing is the feature-engineering step (`build_features.py` in
CLAUDE.md's plan: moving averages, volatility, momentum, sector tags,
computed once from the raw CSVs) and a `DataProvider` that reads it into a
`FeatureTable`. Once that exists, pass an instance of it to `train_baseline()`
instead of `MockDataProvider`. Nothing in `experts/`, `ga/`, `combiner/`, or
`selection/` needs to change.

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
