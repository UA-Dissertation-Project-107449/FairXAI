"""Model-family parameterization of MitigationEngine."""

import numpy as np
import pandas as pd
import pytest

from fairxai.fairness.mitigation import MitigationEngine
from fairxai.models import BaselineLogisticRegression, RandomForestModel


@pytest.fixture
def tiny_split():
    """60-row separable-ish binary problem with one sensitive column."""
    rng = np.random.default_rng(7)
    n = 60
    X = pd.DataFrame(
        {
            "f1": rng.normal(size=n),
            "f2": rng.normal(size=n),
            "f3": rng.normal(size=n),
        }
    )
    y = pd.Series((X["f1"] + rng.normal(scale=0.3, size=n) > 0).astype(int), name="target")
    sensitive = pd.DataFrame({"sex": rng.integers(0, 2, size=n)})
    return X, y, sensitive


def test_default_engine_is_still_logistic_regression(tiny_split):
    """Back-compat guard: no-arg construction must not change."""
    engine = MitigationEngine()
    assert engine.model_type == "logistic_regression"
    assert engine.model_params == {}
    assert isinstance(engine._new_model(), BaselineLogisticRegression)


def test_preprocessing_trains_the_requested_family(tiny_split):
    X, y, sensitive = tiny_split
    engine = MitigationEngine(
        model_type="random_forest", model_params={"n_estimators": 10, "max_depth": 3}
    )
    result = engine.apply_technique(
        technique_name="smote",
        stage="pre-processing",
        X_train=X,
        y_train=y,
        X_test=X,
        y_test=y,
        sensitive_train=sensitive,
        sensitive_test=sensitive,
        sensitive_attr="sex",
    )
    assert isinstance(result["model"], RandomForestModel)
    assert result["metadata"]["model_type"] == "random_forest"


def test_reweighting_passes_sample_weights_to_the_family(tiny_split):
    X, y, sensitive = tiny_split
    engine = MitigationEngine(
        model_type="random_forest", model_params={"n_estimators": 10, "max_depth": 3}
    )
    result = engine.apply_technique(
        technique_name="reweighting",
        stage="pre-processing",
        X_train=X,
        y_train=y,
        X_test=X,
        y_test=y,
        sensitive_train=sensitive,
        sensitive_test=sensitive,
        sensitive_attr="sex",
    )
    assert isinstance(result["model"], RandomForestModel)
    assert result["model"].feature_names == list(X.columns)


def test_unknown_model_type_raises(tiny_split):
    engine = MitigationEngine(model_type="not_a_model")
    with pytest.raises(ValueError, match="Unknown model_type"):
        engine._new_model()


def test_inprocessing_wraps_the_requested_family(tiny_split):
    from sklearn.ensemble import RandomForestClassifier

    X, y, sensitive = tiny_split
    engine = MitigationEngine(
        model_type="random_forest", model_params={"n_estimators": 5, "max_depth": 2}
    )
    result = engine.apply_technique(
        technique_name="exponentiated_gradient",
        stage="in-processing",
        X_train=X,
        y_train=y,
        X_test=X,
        y_test=y,
        sensitive_train=sensitive,
        sensitive_test=sensitive,
        sensitive_attr="sex",
        max_iter=2,
    )
    assert isinstance(result["model"].estimator, RandomForestClassifier)
    assert result["metadata"]["model_type"] == "random_forest"


def test_inprocessing_default_is_still_logistic_regression(tiny_split):
    from sklearn.linear_model import LogisticRegression

    X, y, sensitive = tiny_split
    engine = MitigationEngine()
    result = engine.apply_technique(
        technique_name="exponentiated_gradient",
        stage="in-processing",
        X_train=X,
        y_train=y,
        X_test=X,
        y_test=y,
        sensitive_train=sensitive,
        sensitive_test=sensitive,
        sensitive_attr="sex",
        max_iter=2,
    )
    assert isinstance(result["model"].estimator, LogisticRegression)


def test_combo_trains_the_requested_family(tiny_split):
    X, y, sensitive = tiny_split
    engine = MitigationEngine(
        model_type="random_forest", model_params={"n_estimators": 5, "max_depth": 2}
    )
    result = engine.apply_combo(
        techniques=["smote", "threshold_optimizer"],
        X_train=X,
        y_train=y,
        X_test=X,
        y_test=y,
        sensitive_train=sensitive,
        sensitive_test=sensitive,
        sensitive_attr="sex",
    )
    assert result["metadata"]["model_type"] == "random_forest"


def test_postprocessing_records_model_type(tiny_split):
    X, y, sensitive = tiny_split
    engine = MitigationEngine(
        model_type="random_forest", model_params={"n_estimators": 5, "max_depth": 2}
    )
    base = engine._new_model()
    base.train(X, y)
    result = engine.apply_technique(
        technique_name="threshold_optimizer",
        stage="post-processing",
        X_train=X,
        y_train=y,
        X_test=X,
        y_test=y,
        sensitive_train=sensitive,
        sensitive_test=sensitive,
        sensitive_attr="sex",
        base_model=base,
    )
    assert result["metadata"]["model_type"] == "random_forest"
