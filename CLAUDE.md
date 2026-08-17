# CLAUDE.md

## Project

`wharton-InC` — an algorithmic/technical trading strategy for the Wharton Global High School Investment Competition. Built by Matthew and JR.

## What this system does

Generates trading signals for a small set of approved stocks/ETFs, then hands flagged trade recommendations to a human for manual execution on WInS (the Wharton Investment Simulator). Nothing in this codebase places live orders — there is no WInS API, so the system's job ends at "here's what to do, and how much."

## Competition constraints (hard rules — do not violate in any logic)

- Only stocks/ETFs on Wharton's approved list are tradeable live (`config/stock_list.py`)
- No day trading — never suggest buying and selling the same symbol same-day
- No short selling, no margin
- Position limit: a cap on % of portfolio in any single stock (value set by the competition — confirm in WInS Portfolio Summary → Rules once available, then hardcode in `config/cost_model.py`)
- Fixed cap on total number of trades for the season — 200 trades over 10 weeks — treat trades as a scarce resource, not something to spend casually
- All execution is manual. The system's only output is a recommendation; a team member places every order in WInS by hand.

## Architecture

**Bar frequency: daily, not hourly** (decided 2026-08-16). `yfinance` only serves ~730 days of hourly bars measured from today, so hourly back to 2018 isn't obtainable without a paid vendor — and it wouldn't matter if it were, since "no day trading" means a signal can't be acted on and exited same-session anyway. Daily bars over the full ~8yr history is both what's actually fetchable and what the rules make usable. Full reasoning: `baseline_system_architecture.md` §1, Layer 1.

**Two data universes, one shared feature pipeline:**
- `config/training_universe.py` — broad universe, ~8yr history (2018–2026), survivorship-bias-free via point-in-time S&P 500 membership, used only for training
- `config/stock_list.py` — Wharton's approved list, recent/live data, used only for scoring
- `src/features/build_features.py` is the ONE function both training and live scoring call. Never write a second, separate feature computation path — that's how train/live drift happens.

**Per-sector models:**
- One XGBoost model per sector (`config/sector_map.py` is the single source of truth for ticker → sector; training and live scoring must use the exact same mapping)
- Trained with walk-forward validation (`TimeSeriesSplit` or equivalent) — never a random train/test split, the data is time-ordered and autocorrelated
- Every feature for date `t` must only use information available as of `t`. No lookahead, no using restated/revised data as if it were known at the time.
- Trained model artifacts live in `src/models/saved/` and are gitignored. Only `train.py` is committed — models must always be reproducible from the script + data pipeline, never from a binary blob in git history.

**Allocation & filter layer (sits after the sector models, before output):**
- Decides trade/no-trade: only act on a signal if predicted edge clears commission + estimated spread cost by some margin
- Decides position size via a parametrized rule (equal-weight, confidence-weighted, or vol-scaled — not yet finalized)
- This is a calibrated rule tuned by backtest grid search (or `optuna`), NOT a second trained ML model — trade counts are far too small to fit something like XGBoost on this layer without overfitting
- Needs current portfolio state to size correctly (see below)

**Portfolio state:**
- WInS has no API, so cash/positions are not automatically knowable
- Tracked manually in `data/processed/portfolio_state.json`, updated by hand after every real trade (ticker, shares, price, date)
- The allocation layer reads this file before generating new flags

## Repo structure

This file originated in the standalone `wharton-InC` data-pipeline repo and was
merged into the `gen alg + XGB` project, which already had a working
GA/XGBoost training system under different module names. The table maps the
concepts above onto where they actually live now; `[ ]` = not built yet.

| Concept above | Lives at |
|---|---|
| Data fetch (`fetch_training.py`) | `src/data_pipeline/fetch_training.py` ✅ |
| Point-in-time universe builder | `src/data_pipeline/fetch_point_in_time_universe.py` ✅ |
| Live-data puller (`fetch_live.py`) | `[ ]` not built — only the training puller exists so far |
| `build_features.py` | `[ ]` not built — `src/data_layer/interface.py` defines the `DataProvider`/`FeatureTable` contract this should implement; only a synthetic `MockDataProvider` exists today |
| Per-sector model training (`models/train.py`) | `src/experts/base.py` (`ExpertModel`) + `src/ga/search.py` (`GeneticSearch`, walk-forward CV) ✅ |
| `score_live.py` | `[ ]` not built |
| Allocation & filter layer | `src/selection/trade_selector.py` (`TradeSelector`) ✅ |
| `sector_map.py` | `[ ]` not built — sector membership is currently a field on the feature table, not a standalone config module |
| `cost_model.py` | `config/cost_model.py` ✅ |
| `portfolio_state.json` | `data/processed/portfolio_state.json` ✅ (now also carries `entry_date`/`scheduled_exit_date` per position, per `exit_rule_spec.md`) |

Not in the original `wharton-InC` plan, added since: `config/hold_period.py` (`hold_days` N + the competition window length — single source of truth for the fixed-holding-period exit rule, `exit_rule_spec.md`) and `src/pipeline/sweep.py` (the `hold_days` outer-loop sweep — N defines the training label, so each value swept is a full retrain, not a GA gene).

Actual current layout:

```
gen alg + XGB/
├── baseline_system_architecture.md   # design doc — read this alongside this file
├── exit_rule_spec.md                 # fixed-holding-period exit rule spec
├── config/
│   ├── stock_list.py            # approved (tradeable) tickers — currently empty, TBD
│   ├── training_universe.py     # broader training universe (generated)
│   ├── cost_model.py            # commission rate, assumed spread — placeholder values
│   └── hold_period.py           # hold_days (N), competition window length
├── data/
│   ├── raw/training/            # broad universe, ~8yr daily bars
│   ├── raw/reference/           # point-in-time S&P 500 membership source
│   └── processed/               # sp500_membership.csv, portfolio_state.json
├── src/
│   ├── data_pipeline/            # fetch_training.py, fetch_point_in_time_universe.py, tickers.py
│   ├── data_layer/               # DataProvider interface (§1) — RealDataProvider (broad only, no sector data yet) + MockDataProvider
│   ├── experts/                  # ExpertModel — XGBoost + confidence (§2)
│   ├── ga/                       # genetic hyperparameter search (§3)
│   ├── combiner/                 # OverallModel — blend + rank (§4)
│   ├── selection/                # TradeSelector — fixed-hold exit rule, cost filter, sizing, budget (§5)
│   └── pipeline/                 # train_baseline.py wires all of the above; sweep.py sweeps hold_days
├── tests/
├── scripts/run_baseline_demo.py
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## Environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Conventions

- Ticker lists live only in `config/` — never hardcode a symbol elsewhere
- Any new data source or feature must be checked for point-in-time correctness before it's trusted
- Prefer editing `build_features.py` over adding a parallel, slightly-different feature function
- Trade recommendations must always show the reasoning (which sector model fired, predicted edge, why the allocation layer sized it the way it did) — this feeds directly into the team's final competition report