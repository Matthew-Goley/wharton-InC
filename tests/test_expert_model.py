import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import information_coefficient
from src.experts.base import ExpertModel, XGBConfig

SMALL_CONFIG = XGBConfig(
    max_depth=3,
    learning_rate=0.2,
    n_estimators=30,
    subsample=1.0,
    colsample_bytree=1.0,
    min_child_weight=1.0,
    reg_alpha=0.0,
    reg_lambda=1.0,
    gamma=0.0,
)


def _toy_data(n=300, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"f1": rng.normal(size=n), "f2": rng.normal(size=n)})
    y = pd.Series(0.03 * X["f1"] + 0.005 * rng.normal(size=n))
    return X, y


def test_predict_returns_expected_columns_and_index():
    X, y = _toy_data()
    model = ExpertModel("test", SMALL_CONFIG).fit(X, y)
    preds = model.predict(X)
    assert list(preds.columns) == ["predicted_return", "confidence"]
    assert list(preds.index) == list(X.index)


def test_confidence_is_bounded():
    X, y = _toy_data()
    model = ExpertModel("test", SMALL_CONFIG).fit(X, y)
    preds = model.predict(X)
    assert (preds["confidence"] > 0).all()
    assert (preds["confidence"] <= 1).all()


def test_predict_before_fit_raises():
    X, _ = _toy_data()
    model = ExpertModel("test", SMALL_CONFIG)
    with pytest.raises(RuntimeError):
        model.predict(X)


def test_learns_real_signal_on_held_out_data():
    X_train, y_train = _toy_data(n=400, seed=1)
    X_val, y_val = _toy_data(n=200, seed=2)
    model = ExpertModel("test", SMALL_CONFIG).fit(X_train, y_train)
    preds = model.predict(X_val)
    ic = information_coefficient(y_val.to_numpy(), preds["predicted_return"].to_numpy())
    assert ic > 0.2  # f1 is a strong, easy-to-find linear driver of y


def test_works_regardless_of_target_scale():
    """Predictions/IC shouldn't degrade just because the raw target is tiny
    (e.g. daily returns ~1e-2) — this is what target standardization in
    ExpertModel.fit/predict is for."""
    X, y = _toy_data(n=400, seed=3)
    tiny_y = y * 1e-3

    model_normal = ExpertModel("normal", SMALL_CONFIG).fit(X, y)
    model_tiny = ExpertModel("tiny", SMALL_CONFIG).fit(X, tiny_y)

    ic_normal = information_coefficient(y.to_numpy(), model_normal.predict(X)["predicted_return"].to_numpy())
    ic_tiny = information_coefficient(tiny_y.to_numpy(), model_tiny.predict(X)["predicted_return"].to_numpy())
    assert ic_tiny == pytest.approx(ic_normal, abs=0.05)
