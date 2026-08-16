"""Wires the expert roster (config.EXPERTS) up to concrete data slices."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.config import ExpertSpec
from src.data_layer.interface import FeatureTable


@dataclass
class ExpertData:
    """Everything one expert needs, sliced from the shared feature table."""

    spec: ExpertSpec
    frame: pd.DataFrame  # full slice, incl. target/sector — for walk-forward splitting
    X: pd.DataFrame
    y: pd.Series


def build_expert_data(table: FeatureTable, spec: ExpertSpec) -> ExpertData:
    frame = table.slice_for(spec.sector)
    return ExpertData(
        spec=spec,
        frame=frame,
        X=frame[table.feature_columns],
        y=frame[table.target_column],
    )


def build_all_expert_data(
    table: FeatureTable, experts: tuple[ExpertSpec, ...]
) -> dict[str, ExpertData]:
    """One :class:`ExpertData` per row in ``experts``, keyed by expert name."""
    return {spec.name: build_expert_data(table, spec) for spec in experts}
