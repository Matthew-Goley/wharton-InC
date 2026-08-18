"""The expert model wrapper (architecture §2).

Each expert is one ``ExpertModel``: an XGBoost regressor predicting
next-period return, plus two auxiliary quantile regressors used to derive a
confidence score. Genetic search (architecture §3) operates entirely on
``XGBConfig`` — sample one, train an ``ExpertModel`` from it, score it on a
walk-forward validation fold, repeat.

Persistence (baseline_system_architecture.md open question 36): every real
training script used to persist models with a bare ``joblib.dump(models,
path)`` — a pickle of weights with no record of what roster, label horizon,
or feature set produced it. That gap is exactly how the stale-``materials``-
key artifact (open question 26 gap #4) happened: nothing about the file
itself could be checked against the current config, only inferred by
loading it and inspecting dict keys by hand. ``ExpertModel.save``/``.load``
and the module-level ``save_experts``/``load_experts`` fix this — every
saved artifact carries ``feature_columns`` and ``hold_days`` (and,
optionally, ``sector``) as first-class metadata, and ``src.config.
assert_config_consistency`` gained an ``artifact=`` parameter that checks
an ``ExpertBundle`` against a live config in one call: roster, label
horizon, and feature columns, not just weights that happen to load.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

#: Below this, a predicted [lo, hi] interval is treated as degenerate rather
#: than divided by, so confidence never blows up when a model is (correctly
#: or incorrectly) very sure.
_MIN_INTERVAL_EPS = 1e-6

#: Floor for the target's training-fold std, so a near-constant target
#: (or a single-row fold) can't blow up the standardization step below.
_MIN_TARGET_STD = 1e-8

#: Bumped whenever the saved-payload shape changes, so a mismatched old
#: file fails loudly and specifically at load() rather than however joblib
#: happens to fail (a KeyError deep in a payload dict, or nothing at all if
#: the shape coincidentally still fits).
_ARTIFACT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class XGBConfig:
    """One XGBoost hyperparameter configuration — the GA's genome.

    Field set matches :class:`src.config.HyperparamSpace` one-to-one: every
    range in that space has exactly one concrete value here.
    """

    max_depth: int
    learning_rate: float
    n_estimators: int
    subsample: float
    colsample_bytree: float
    min_child_weight: float
    reg_alpha: float
    reg_lambda: float
    gamma: float

    def as_xgb_params(self) -> dict:
        return asdict(self)


class ExpertModel:
    """A trained (or trainable) expert: predicts return + confidence."""

    def __init__(
        self,
        name: str,
        config: XGBConfig,
        random_seed: int = 0,
        n_jobs: int | None = None,
        sector: Optional[str] = None,
    ):
        self.name = name
        self.config = config
        self.random_seed = random_seed
        #: Threads per fit. Leave None for XGBoost's default (all cores) —
        #: right for a single one-off fit. Set to 1 when many fits run
        #: concurrently across worker processes (GA search), so N processes
        #: don't each also grab all cores and oversubscribe the machine.
        self.n_jobs = n_jobs
        #: The sector this expert was trained for (None for the broad
        #: expert, or if not supplied). Optional and not used by fit/
        #: predict at all -- pure self-description, checked on save/load
        #: as one more independently-verifiable field against
        #: config.EXPERTS, the same class of check as the roster/hold_days/
        #: feature_columns ones (open question 36). Belt-and-suspenders:
        #: the "Technology" vs "Information Technology" sector-name
        #: mismatch (open question 30's bug-class tally) is exactly the
        #: kind of thing a self-describing model could have caught sooner.
        self.sector = sector
        self._mean_model: XGBRegressor | None = None
        self._lo_model: XGBRegressor | None = None
        self._hi_model: XGBRegressor | None = None
        self._target_mean: float = 0.0
        self._target_std: float = 1.0
        #: Set by fit() (feature_columns automatically, from X.columns --
        #: derived, not hand-typed, so it can't drift from what was
        #: actually fit on) or by load() (from a saved artifact's own
        #: metadata). hold_days must be passed explicitly to fit() to be
        #: set -- there's no way to derive a label's horizon from X/y
        #: alone, and a model that doesn't know it can still be used
        #: in-memory for the rest of this session (a one-off IC comparison
        #: never needs to save), just not saved (see save(), which
        #: requires it).
        self.feature_columns: Optional[list[str]] = None
        self.hold_days: Optional[int] = None

    def fit(self, X: pd.DataFrame, y: pd.Series, hold_days: Optional[int] = None) -> "ExpertModel":
        self.feature_columns = list(X.columns)
        self.hold_days = hold_days
        params = self.config.as_xgb_params()
        if self.n_jobs is not None:
            params["n_jobs"] = self.n_jobs

        # Returns are typically small in magnitude (e.g. std ~0.01-0.05 for
        # daily bars). gamma/reg_alpha/reg_lambda are loss-reduction /
        # weight-magnitude penalties in the *target's own units*, so a
        # hyperparameter space sized for a "normal" target (values ~O(1))
        # would swamp a raw return target and silently prune every split,
        # collapsing the model to a constant predictor. Standardizing the
        # target to zero mean / unit variance before fitting — and undoing
        # it in predict() — keeps regularization strength independent of
        # whatever return scale (or sector, or bar frequency) is in play.
        self._target_mean = float(np.mean(y))
        self._target_std = max(float(np.std(y)), _MIN_TARGET_STD)
        y_scaled = (y - self._target_mean) / self._target_std

        self._mean_model = XGBRegressor(
            objective="reg:squarederror",
            random_state=self.random_seed,
            **params,
        ).fit(X, y_scaled)

        # Quantile pair brackets a ~80% predictive interval; its width is
        # the confidence signal (architecture §2: "a confidence measure").
        self._lo_model = XGBRegressor(
            objective="reg:quantileerror",
            quantile_alpha=0.1,
            random_state=self.random_seed,
            **params,
        ).fit(X, y_scaled)
        self._hi_model = XGBRegressor(
            objective="reg:quantileerror",
            quantile_alpha=0.9,
            random_state=self.random_seed,
            **params,
        ).fit(X, y_scaled)
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return a frame indexed like ``X`` with predicted_return + confidence."""
        if self._mean_model is None:
            raise RuntimeError(f"Expert '{self.name}' must be fit() before predict()")
        # Self-description put to use, not just recorded: refuse to predict
        # on a feature set different from the one this model was actually
        # fit on, rather than letting XGBoost silently reorder/misalign or
        # fail with an unrelated-looking error deeper in its own code.
        # Exact order, not just set equality -- matches how this project
        # always constructs X (frame[FEATURE_COLUMNS], a fixed order), so a
        # differently-ordered X is itself suspicious, not an innocent
        # equivalent.
        if self.feature_columns is not None and list(X.columns) != self.feature_columns:
            raise ValueError(
                f"Expert '{self.name}' was fit on feature_columns={self.feature_columns}, but "
                f"predict() was called with columns={list(X.columns)} -- refusing to predict on "
                f"a mismatched feature set."
            )

        predicted_return = self._mean_model.predict(X) * self._target_std + self._target_mean
        lo = self._lo_model.predict(X) * self._target_std + self._target_mean
        hi = self._hi_model.predict(X) * self._target_std + self._target_mean
        interval_width = np.maximum(hi - lo, _MIN_INTERVAL_EPS)

        # Tight interval relative to the target's typical scale -> high
        # confidence; wide interval -> low confidence. Bounded in (0, 1].
        normalized_width = interval_width / self._target_std
        confidence = 1.0 / (1.0 + normalized_width)

        return pd.DataFrame(
            {"predicted_return": predicted_return, "confidence": confidence},
            index=X.index,
        )

    # -- persistence (open question 36) --------------------------------------

    def _to_payload(self) -> dict:
        if self._mean_model is None:
            raise RuntimeError(f"Expert '{self.name}' must be fit() before it can be saved")
        if self.hold_days is None:
            raise RuntimeError(
                f"Expert '{self.name}' was fit() without hold_days -- refusing to save a model "
                f"that can't self-describe its label horizon (baseline_system_architecture.md "
                f"open question 26 gap #4: a model file that can't self-describe is exactly how "
                f"the stale-materials-key problem happened). Pass hold_days= to fit() before "
                f"saving."
            )
        return {
            "schema_version": _ARTIFACT_SCHEMA_VERSION,
            "name": self.name,
            "sector": self.sector,
            "config": self.config,
            "random_seed": self.random_seed,
            "feature_columns": list(self.feature_columns),
            "hold_days": self.hold_days,
            "target_mean": self._target_mean,
            "target_std": self._target_std,
            "mean_model": self._mean_model,
            "lo_model": self._lo_model,
            "hi_model": self._hi_model,
        }

    @classmethod
    def _from_payload(cls, payload: dict) -> "ExpertModel":
        if payload.get("schema_version") != _ARTIFACT_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported ExpertModel artifact schema_version={payload.get('schema_version')!r} "
                f"(expected {_ARTIFACT_SCHEMA_VERSION}) -- fails loudly here rather than however "
                f"joblib happens to fail on a payload shape this code doesn't recognize."
            )
        model = cls(
            payload["name"], payload["config"], random_seed=payload["random_seed"],
            sector=payload.get("sector"),
        )
        model.feature_columns = payload["feature_columns"]
        model.hold_days = payload["hold_days"]
        model._target_mean = payload["target_mean"]
        model._target_std = payload["target_std"]
        model._mean_model = payload["mean_model"]
        model._lo_model = payload["lo_model"]
        model._hi_model = payload["hi_model"]
        return model

    def save(self, path: Path) -> None:
        """Persist this one model, self-describing (feature_columns,
        hold_days, sector) -- not a bare pickle of weights. Raises if
        unfit or if hold_days was never declared (see _to_payload)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._to_payload(), path)

    @classmethod
    def load(cls, path: Path) -> "ExpertModel":
        """Restore a single model saved via save()."""
        return cls._from_payload(joblib.load(Path(path)))


@dataclass(frozen=True)
class ExpertBundle:
    """A whole trained roster, self-describing at the bundle level --
    everything src.config.assert_config_consistency's ``artifact=``
    parameter needs to validate it against a live config in one call:
    roster (``models``' own keys), label horizon, and feature columns.
    Returned by load_experts(); build one implicitly via save_experts()."""

    models: dict[str, "ExpertModel"]
    hold_days: int
    feature_columns: tuple[str, ...]
    saved_at: str  # ISO-8601 UTC timestamp -- "is this stale" at a glance


def save_experts(models: dict[str, "ExpertModel"], path: Path) -> None:
    """Save a whole trained roster as one file (the shape every real
    training script already produces -- one joblib file holding a dict).

    Validates INTERNAL bundle consistency before writing: every model must
    agree on hold_days and feature_columns, since ExpertBundle's own
    fields assume the bundle speaks with one voice. Catching a disagreeing
    model here, at save time, is cheaper than a confusing prediction-time
    error (or a silently-wrong assert_config_consistency check) after the
    fact.
    """
    if not models:
        raise ValueError("cannot save an empty roster")

    payloads = {name: model._to_payload() for name, model in models.items()}  # raises per-model if unfit/no hold_days

    hold_days_values = {p["hold_days"] for p in payloads.values()}
    if len(hold_days_values) != 1:
        raise ValueError(
            f"models in this bundle disagree on hold_days, can't save one self-consistent "
            f"bundle: { {name: p['hold_days'] for name, p in payloads.items()} }"
        )
    feature_sets = {tuple(p["feature_columns"]) for p in payloads.values()}
    if len(feature_sets) != 1:
        raise ValueError(
            f"models in this bundle disagree on feature_columns, can't save one self-consistent "
            f"bundle: { {name: p['feature_columns'] for name, p in payloads.items()} }"
        )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "hold_days": next(iter(hold_days_values)),
            "feature_columns": list(next(iter(feature_sets))),
            "models": payloads,
        },
        path,
    )


def load_experts(path: Path) -> ExpertBundle:
    """Restore a whole trained roster saved via save_experts()."""
    raw = joblib.load(Path(path))
    models = {name: ExpertModel._from_payload(payload) for name, payload in raw["models"].items()}
    return ExpertBundle(
        models=models,
        hold_days=raw["hold_days"],
        feature_columns=tuple(raw["feature_columns"]),
        saved_at=raw["saved_at"],
    )
