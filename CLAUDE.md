# CLAUDE.md

## Project

`wharton-InC` — an algorithmic/technical trading strategy for the Wharton Global High School Investment Competition. Built by Matthew and JR.

## What this system does

Generates trading signals for a small set of approved stocks/ETFs, then hands flagged trade recommendations to a human for manual execution on WInS (the Wharton Investment Simulator). Nothing in this codebase places live orders — there is no WInS API, so the system's job ends at "here's what to do, and how much."

## Competition constraints (hard rules — do not violate in any logic)

- Only stocks/ETFs on Wharton's approved list are tradeable live (`config/stock_list.py`)
- No day trading — never suggest buying and selling the same symbol same-day
- No short selling, no margin
- Position limit: a cap on % of portfolio in any single stock (value set by the competition — confirm in WInS Portfolio Summary → Rules once available, then hardcode in `config/`)
- Fixed cap on total number of trades for the season — treat trades as a scarce resource, not something to spend casually
- All execution is manual. The system's only output is a recommendation; a team member places every order in WInS by hand.

## Architecture

**Two data universes, one shared feature pipeline:**
- `config/training_universe.py` — broad universe, ~10yr history, used only for training
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

```
wharton-InC/
├── data/
│   ├── raw/
│   │   ├── training/          # broad universe, 10yr historical
│   │   └── live/               # approved list only, recent/live
│   └── processed/
│       └── portfolio_state.json
├── src/
│   ├── data_pipeline/
│   │   ├── fetch_training.py   # broad universe puller
│   │   └── fetch_live.py       # approved-list-only puller
│   ├── features/
│   │   └── build_features.py   # shared by train + live, no exceptions
│   ├── models/
│   │   ├── train.py            # walk-forward CV, saves one model per sector
│   │   └── saved/               # gitignored model artifacts
│   ├── signals/
│   │   └── score_live.py        # loads saved models, scores approved list
│   ├── allocation/
│   │   └── filter_and_size.py   # trade/no-trade + position sizing rule
│   ├── backtest/
│   └── reporting/
├── config/
│   ├── stock_list.py            # approved (tradeable) tickers
│   ├── training_universe.py     # broader training universe
│   ├── sector_map.py            # ticker -> sector, single source of truth
│   └── cost_model.py            # commission rate, assumed spread
├── venv/                         # gitignored
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