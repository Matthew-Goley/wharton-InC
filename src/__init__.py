"""Baseline trading system: GA-tuned XGBoost experts + combination + trade selection.

See ``baseline_system_architecture.md`` at the repo root for the design this
package implements. The Data Layer (architecture §1) is intentionally out of
scope for now — ``src.data_layer`` exposes only the interface the rest of the
system is built against, plus a synthetic stand-in so everything downstream
is runnable and testable today.
"""
