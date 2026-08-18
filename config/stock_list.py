"""Wharton's approved (tradeable) live list — the only symbols this system
may actually flag a trade on (CLAUDE.md "Competition constraints").

Distinct from ``config/training_universe.py`` (the ~653-name broad universe,
training only, survivorship-bias-free via point-in-time S&P 500 membership).
This list has no relationship to S&P 500 membership at all — it's whatever
Wharton's WInS rules designate, confirm there once available (see
``baseline_system_architecture.md`` open question 26). Empty until then; a
declared empty list (rather than no file / no variable at all) so
``src.data_pipeline.fetch_live`` and ``RealDataProvider(universe=STOCK_LIST,
apply_point_in_time_filter=False)`` can import this unconditionally and fail
loudly with a clear message ("nothing to fetch") instead of an ImportError.
"""
from __future__ import annotations

STOCK_LIST: list[str] = []
