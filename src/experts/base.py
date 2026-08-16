"""The expert model wrapper (architecture §2).

Each expert is one ``ExpertModel``: an XGBoost regressor predicting
next-period return, plus two auxiliary quantile regressors used to derive a
confidence score. Genetic search (architecture §3) operates entirely on
``XGBConfig`` — sample one, train an ``ExpertModel`` from it, score it on a
walk-forward validation fold, repeat.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

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

    def __init__(self, name: str, config: XGBConfig, random_seed: int = 0):
        self.name = name
        self.config = config
        self.random_seed = random_seed
        self._mean_model: XGBRegressor | None = None
        self._lo_model: XGBRegressor | None = None
        self._hi_model: XGBRegressor | None = None
        self._target_mean: float = 0.0
        self._target_std: float = 1.0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ExpertModel":
        params = self.config.as_xgb_params()

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
