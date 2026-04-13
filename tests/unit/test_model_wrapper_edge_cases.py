"""Unit tests for model wrapper edge cases seen in pipeline logs."""

import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from fairxai.models.random_forest import RandomForestModel
from fairxai.models.xgboost_model import XGBoostModel


def test_rf_balanced_subsample_maps_to_balanced_for_sample_weights():
    assert RandomForestModel._resolve_sample_weight_strategy("balanced_subsample") == "balanced"
    assert RandomForestModel._resolve_sample_weight_strategy("balanced") == "balanced"
    assert RandomForestModel._resolve_sample_weight_strategy(None) == "balanced"


def test_xgboost_cuda_predict_proba_uses_dmatrix_path():
    pytest.importorskip("xgboost")

    class DummyBooster:
        def predict(self, _dmatrix):
            return np.array([0.2, 0.8], dtype=float)

    class DummyEstimator:
        def get_booster(self):
            return DummyBooster()

        def predict_proba(self, _X):
            raise AssertionError("sklearn predict_proba path should not be used")

    model = XGBoostModel.__new__(XGBoostModel)
    model._device = "cuda"
    model.feature_names = ["f1", "f2"]
    model.model = DummyEstimator()

    X = pd.DataFrame({"f1": [1.0, 2.0], "f2": [3.0, 4.0]})
    proba = XGBoostModel.predict_proba(model, X)

    assert np.allclose(proba, np.array([0.2, 0.8]))


def test_xgboost_cuda_predict_proba_falls_back_when_dmatrix_path_fails():
    pytest.importorskip("xgboost")

    class BrokenBooster:
        def predict(self, _dmatrix):
            raise RuntimeError("boom")

    class DummyEstimator:
        def get_booster(self):
            return BrokenBooster()

        def predict_proba(self, _X):
            return np.array([[0.9, 0.1], [0.1, 0.9]], dtype=float)

    model = XGBoostModel.__new__(XGBoostModel)
    model._device = "cuda"
    model.feature_names = ["f1", "f2"]
    model.model = DummyEstimator()

    X = pd.DataFrame({"f1": [1.0, 2.0], "f2": [3.0, 4.0]})
    proba = XGBoostModel.predict_proba(model, X)

    assert np.allclose(proba, np.array([0.1, 0.9]))


def test_rf_cuml_train_retries_without_sample_weight_when_unsupported():
    class DummyEstimator:
        def __init__(self):
            self.fit_calls = []

        def fit(self, X, y, sample_weight=None):
            if sample_weight is not None:
                self.fit_calls.append("with_sample_weight")
                raise TypeError("got an unexpected keyword argument 'sample_weight'")
            self.fit_calls.append("without_sample_weight")

        def predict_proba(self, X):
            n = len(X)
            # Return 2D proba matrix to exercise wrapper selection of positive class.
            return np.column_stack([np.full(n, 0.4), np.full(n, 0.6)])

    model = RandomForestModel.__new__(RandomForestModel)
    model._use_gpu = True
    model._class_weight = "balanced_subsample"
    model.model = DummyEstimator()
    model.model_name = "RandomForest"
    model.feature_names = None
    model.training_metrics = {}

    X = pd.DataFrame({"f1": [1.0, 2.0, 3.0, 4.0], "f2": [0.1, 0.2, 0.3, 0.4]})
    y = pd.Series([0, 1, 0, 1])

    metrics = RandomForestModel.train(model, X, y)

    assert model.model.fit_calls == ["with_sample_weight", "without_sample_weight"]
    assert isinstance(metrics, dict)
    assert "accuracy" in metrics


def test_xgboost_train_uses_wrapper_predict_path_not_estimator_predict():
    class DummyEstimator:
        def fit(self, X, y):
            return None

        def predict(self, X):
            raise AssertionError("direct estimator.predict should not be called")

    model = XGBoostModel.__new__(XGBoostModel)
    model._device = "cpu"
    model.model = DummyEstimator()
    model.model_name = "XGBoost"
    model.feature_names = None
    model.training_metrics = {}

    def _predict_proba(_self, X):
        return np.full(len(X), 0.8, dtype=float)

    model.predict_proba = types.MethodType(_predict_proba, model)

    X = pd.DataFrame({"f1": [1.0, 2.0, 3.0, 4.0], "f2": [0.5, 0.6, 0.7, 0.8]})
    y = pd.Series([1, 1, 1, 1])

    metrics = XGBoostModel.train(model, X, y)
    assert isinstance(metrics, dict)
    assert "accuracy" in metrics
