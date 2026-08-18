import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import information_coefficient
from src.experts.base import ExpertBundle, ExpertModel, XGBConfig, load_experts, save_experts

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


# --- self-description (feature_columns, hold_days, sector) -----------------
# baseline_system_architecture.md open question 36: a model file that
# can't self-describe is exactly how the stale-materials-key problem
# happened. These are the fields assert_config_consistency's artifact=
# parameter validates.

def test_fit_captures_feature_columns_automatically():
    """Derived from X.columns, not hand-typed -- can't drift from what was
    actually fit on."""
    X, y = _toy_data()
    model = ExpertModel("test", SMALL_CONFIG).fit(X, y)
    assert model.feature_columns == ["f1", "f2"]


def test_fit_without_hold_days_leaves_it_none():
    """A model only used in-memory this session (e.g. a one-off IC
    comparison) never needs hold_days -- must not be forced to supply one
    just to fit()."""
    X, y = _toy_data()
    model = ExpertModel("test", SMALL_CONFIG).fit(X, y)
    assert model.hold_days is None


def test_fit_with_hold_days_stores_it():
    X, y = _toy_data()
    model = ExpertModel("test", SMALL_CONFIG).fit(X, y, hold_days=5)
    assert model.hold_days == 5


def test_predict_rejects_a_mismatched_feature_set():
    """Deliberate failing-case check: refuses to predict on a feature set
    different from the one this model was actually fit on, rather than
    trusting XGBoost's own column handling silently."""
    X, y = _toy_data()
    model = ExpertModel("test", SMALL_CONFIG).fit(X, y)
    wrong_order = X[["f2", "f1"]]
    with pytest.raises(ValueError, match="mismatched feature set"):
        model.predict(wrong_order)


def test_predict_still_works_when_feature_columns_matches():
    """Positive counterpart -- the check above isn't accidentally always
    firing."""
    X, y = _toy_data()
    model = ExpertModel("test", SMALL_CONFIG).fit(X, y)
    model.predict(X)  # must not raise


# --- save/load: a single model ------------------------------------------------

def test_save_requires_the_model_to_be_fit(tmp_path):
    model = ExpertModel("test", SMALL_CONFIG)
    with pytest.raises(RuntimeError, match="must be fit"):
        model.save(tmp_path / "model.joblib")


def test_save_requires_hold_days_to_have_been_declared(tmp_path):
    """Deliberate failing-case check, the actual point of this whole
    feature: a model fit without hold_days can't be saved at all -- this
    is where 'can't self-describe' is refused, not silently allowed
    through to become the next stale artifact."""
    X, y = _toy_data()
    model = ExpertModel("test", SMALL_CONFIG).fit(X, y)  # no hold_days
    with pytest.raises(RuntimeError, match="hold_days"):
        model.save(tmp_path / "model.joblib")


def test_save_and_load_round_trips_predictions(tmp_path):
    X, y = _toy_data()
    model = ExpertModel("test", SMALL_CONFIG, sector="Technology").fit(X, y, hold_days=5)
    original_preds = model.predict(X)

    path = tmp_path / "model.joblib"
    model.save(path)
    loaded = ExpertModel.load(path)
    loaded_preds = loaded.predict(X)

    pd.testing.assert_frame_equal(original_preds, loaded_preds)


def test_save_and_load_round_trips_metadata(tmp_path):
    X, y = _toy_data()
    model = ExpertModel("test", SMALL_CONFIG, sector="Technology").fit(X, y, hold_days=5)
    path = tmp_path / "model.joblib"
    model.save(path)
    loaded = ExpertModel.load(path)

    assert loaded.name == "test"
    assert loaded.sector == "Technology"
    assert loaded.hold_days == 5
    assert loaded.feature_columns == ["f1", "f2"]
    assert loaded.config == SMALL_CONFIG


def test_load_rejects_an_unsupported_schema_version(tmp_path):
    """Deliberate failing-case check: a payload shape this code doesn't
    recognize must fail loudly and specifically here, not however joblib/
    a KeyError deep in _from_payload happens to fail."""
    import joblib
    path = tmp_path / "bad.joblib"
    joblib.dump({"schema_version": 999}, path)
    with pytest.raises(ValueError, match="schema_version"):
        ExpertModel.load(path)


# --- save/load: a whole bundle -------------------------------------------------

def _fit(name: str, seed: int, hold_days: int = 5, sector=None) -> ExpertModel:
    X, y = _toy_data(seed=seed)
    return ExpertModel(name, SMALL_CONFIG, sector=sector).fit(X, y, hold_days=hold_days)


def test_save_experts_rejects_an_empty_bundle(tmp_path):
    with pytest.raises(ValueError, match="empty"):
        save_experts({}, tmp_path / "bundle.joblib")


def test_save_experts_rejects_disagreeing_hold_days(tmp_path):
    """Deliberate failing-case check: catching this at save time, not
    letting an internally-inconsistent bundle be written and discovered
    later via a confusing prediction/assert_config_consistency error."""
    models = {"broad": _fit("broad", seed=0, hold_days=5), "tech": _fit("tech", seed=1, hold_days=10)}
    with pytest.raises(ValueError, match="hold_days"):
        save_experts(models, tmp_path / "bundle.joblib")


def test_save_experts_rejects_disagreeing_feature_columns(tmp_path):
    broad = _fit("broad", seed=0)
    X2, y2 = _toy_data(seed=1)
    X2 = X2.rename(columns={"f2": "f3"})
    tech = ExpertModel("tech", SMALL_CONFIG).fit(X2, y2, hold_days=5)
    with pytest.raises(ValueError, match="feature_columns"):
        save_experts({"broad": broad, "tech": tech}, tmp_path / "bundle.joblib")


def test_save_and_load_experts_round_trips_the_bundle(tmp_path):
    models = {
        "broad": _fit("broad", seed=0),
        "tech": _fit("tech", seed=1, sector="Information Technology"),
    }
    path = tmp_path / "bundle.joblib"
    save_experts(models, path)
    bundle = load_experts(path)

    assert isinstance(bundle, ExpertBundle)
    assert set(bundle.models.keys()) == {"broad", "tech"}
    assert bundle.hold_days == 5
    assert set(bundle.feature_columns) == {"f1", "f2"}
    assert bundle.models["tech"].sector == "Information Technology"
    # A real ISO-8601 timestamp, not a placeholder -- "is this stale" at a glance.
    from datetime import datetime
    datetime.fromisoformat(bundle.saved_at)  # raises if malformed


def test_loaded_bundle_models_still_predict_correctly(tmp_path):
    X, y = _toy_data(seed=0)
    original = _fit("broad", seed=0)
    path = tmp_path / "bundle.joblib"
    save_experts({"broad": original}, path)
    bundle = load_experts(path)

    pd.testing.assert_frame_equal(original.predict(X), bundle.models["broad"].predict(X))
