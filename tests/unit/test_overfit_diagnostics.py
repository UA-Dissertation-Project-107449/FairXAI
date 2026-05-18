"""Unit tests for overfit diagnostics: gap table and training-time overfit warning."""

import json
import logging
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add src and scripts/common to path so both fairxai and assess_predictions are importable.
_FAIRXAI_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_FAIRXAI_ROOT / "src"))
sys.path.insert(0, str(_FAIRXAI_ROOT / "scripts" / "common"))

from assess_predictions import write_overfit_gap_table  # noqa: E402

from fairxai.models.sklearn_wrapper import SklearnClassifierWrapper  # noqa: E402
from fairxai.models.xgboost_model import XGBoostModel  # noqa: E402

# ---------------------------------------------------------------------------
# write_overfit_gap_table tests
# ---------------------------------------------------------------------------


def _make_training_results(train_f1, test_f1, train_auc=0.95, test_auc=0.82):
    return {
        "cleveland": {
            "logistic_regression": {
                "train_metrics": {
                    "f1_score": train_f1,
                    "recall": train_f1 - 0.02,
                    "auc_roc": train_auc,
                },
                "test_metrics": {
                    "f1_score": test_f1,
                    "recall": test_f1 - 0.01,
                    "auc_roc": test_auc,
                },
            }
        }
    }


def test_gap_table_low_risk(tmp_path):
    data = _make_training_results(train_f1=0.82, test_f1=0.79)
    results_path = tmp_path / "training_results.json"
    results_path.write_text(json.dumps(data))

    write_overfit_gap_table(results_path, tmp_path)

    out = pd.read_csv(tmp_path / "overfit_gap_table.csv")
    assert len(out) == 1
    row = out.iloc[0]
    assert row["dataset"] == "cleveland"
    assert row["model"] == "logistic_regression"
    assert abs(row["f1_gap"] - (0.82 - 0.79)) < 1e-6
    assert row["overfit_risk"] == "low"


def test_gap_table_medium_risk(tmp_path):
    data = _make_training_results(train_f1=0.90, test_f1=0.81)
    results_path = tmp_path / "training_results.json"
    results_path.write_text(json.dumps(data))

    write_overfit_gap_table(results_path, tmp_path)

    out = pd.read_csv(tmp_path / "overfit_gap_table.csv")
    assert out.iloc[0]["overfit_risk"] == "medium"


def test_gap_table_high_risk_due_to_train_f1(tmp_path):
    data = _make_training_results(train_f1=0.99, test_f1=0.85)
    results_path = tmp_path / "training_results.json"
    results_path.write_text(json.dumps(data))

    write_overfit_gap_table(results_path, tmp_path)

    out = pd.read_csv(tmp_path / "overfit_gap_table.csv")
    assert out.iloc[0]["overfit_risk"] == "high"


def test_gap_table_high_risk_due_to_large_gap(tmp_path):
    data = _make_training_results(train_f1=0.97, test_f1=0.79)
    results_path = tmp_path / "training_results.json"
    results_path.write_text(json.dumps(data))

    write_overfit_gap_table(results_path, tmp_path)

    out = pd.read_csv(tmp_path / "overfit_gap_table.csv")
    assert out.iloc[0]["overfit_risk"] == "high"


def test_gap_table_columns_present(tmp_path):
    data = _make_training_results(train_f1=0.85, test_f1=0.80)
    results_path = tmp_path / "training_results.json"
    results_path.write_text(json.dumps(data))

    write_overfit_gap_table(results_path, tmp_path)

    out = pd.read_csv(tmp_path / "overfit_gap_table.csv")
    expected_cols = {
        "dataset",
        "model",
        "train_f1",
        "test_f1",
        "f1_gap",
        "train_recall",
        "test_recall",
        "recall_gap",
        "train_auc",
        "test_auc",
        "auc_gap",
        "train_accuracy",
        "test_accuracy",
        "accuracy_gap",
        "overfit_risk",
    }
    assert expected_cols.issubset(set(out.columns))


def test_gap_table_missing_file_logs_warning(tmp_path, caplog):
    missing = tmp_path / "does_not_exist.json"
    with caplog.at_level(logging.WARNING):
        write_overfit_gap_table(missing, tmp_path)
    assert not (tmp_path / "overfit_gap_table.csv").exists()
    assert any("not found" in r.message for r in caplog.records)


def test_gap_table_multiple_models(tmp_path):
    data = {
        "cleveland": {
            "logistic_regression": {
                "train_metrics": {"f1_score": 0.83, "recall": 0.80, "auc_roc": 0.88},
                "test_metrics": {"f1_score": 0.80, "recall": 0.78, "auc_roc": 0.85},
            },
            "random_forest": {
                "train_metrics": {"f1_score": 0.99, "recall": 0.99, "auc_roc": 1.00},
                "test_metrics": {"f1_score": 0.82, "recall": 0.79, "auc_roc": 0.87},
            },
        }
    }
    results_path = tmp_path / "training_results.json"
    results_path.write_text(json.dumps(data))

    write_overfit_gap_table(results_path, tmp_path)

    out = pd.read_csv(tmp_path / "overfit_gap_table.csv")
    assert len(out) == 2
    risks = dict(zip(out["model"], out["overfit_risk"]))
    assert risks["logistic_regression"] == "low"
    assert risks["random_forest"] == "high"


# ---------------------------------------------------------------------------
# Overfit warning in SklearnClassifierWrapper
# ---------------------------------------------------------------------------


class _AlwaysHighEstimator:
    """Estimator that always predicts 1 and returns proba=1.0 (simulates memorization)."""

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.ones(len(X), dtype=int)

    def predict_proba(self, X):
        return np.column_stack([np.zeros(len(X)), np.ones(len(X))])


def test_sklearn_wrapper_emits_overfit_warning_when_train_metrics_high(caplog):
    wrapper = SklearnClassifierWrapper(estimator=_AlwaysHighEstimator(), model_name="TestModel")
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [0.1, 0.2, 0.3, 0.4]})
    y = pd.Series([1, 1, 1, 1])

    with caplog.at_level(logging.WARNING):
        wrapper.train(X, y)

    overfit_warnings = [r for r in caplog.records if "OVERFIT-RISK" in r.message]
    assert overfit_warnings, "Expected [OVERFIT-RISK] warning when train metrics >= 0.98"


def test_sklearn_wrapper_no_overfit_warning_when_metrics_low(caplog):
    class _LowMetricEstimator:
        def fit(self, X, y):
            return self

        def predict(self, X):
            return np.zeros(len(X), dtype=int)

        def predict_proba(self, X):
            return np.column_stack([np.ones(len(X)) * 0.6, np.ones(len(X)) * 0.4])

    wrapper = SklearnClassifierWrapper(estimator=_LowMetricEstimator(), model_name="TestModel")
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [0.1, 0.2, 0.3, 0.4]})
    y = pd.Series([0, 1, 0, 1])

    with caplog.at_level(logging.WARNING):
        wrapper.train(X, y)

    overfit_warnings = [r for r in caplog.records if "OVERFIT-RISK" in r.message]
    assert not overfit_warnings


# ---------------------------------------------------------------------------
# Overfit warning in XGBoostModel.train()
# ---------------------------------------------------------------------------


def test_xgboost_train_emits_overfit_warning_when_metrics_high(caplog):
    pytest.importorskip("xgboost")

    class _HighMetricEstimator:
        def fit(self, X, y):
            return self

        def predict(self, X):
            return np.ones(len(X), dtype=int)

        def predict_proba(self, X):
            return np.column_stack([np.zeros(len(X)), np.ones(len(X))])

    model = XGBoostModel.__new__(XGBoostModel)
    model._device = "cpu"
    model.model = _HighMetricEstimator()
    model.model_name = "XGBoost"
    model.feature_names = None
    model.training_metrics = {}

    X = pd.DataFrame({"f1": [1.0, 2.0, 3.0, 4.0], "f2": [0.5, 0.6, 0.7, 0.8]})
    y = pd.Series([1, 1, 1, 1])

    def _predict_proba_high(_self, X):
        return np.ones(len(X), dtype=float)

    model.predict_proba = types.MethodType(_predict_proba_high, model)

    with caplog.at_level(logging.WARNING):
        XGBoostModel.train(model, X, y)

    overfit_warnings = [r for r in caplog.records if "OVERFIT-RISK" in r.message]
    assert overfit_warnings, "Expected [OVERFIT-RISK] warning from XGBoostModel.train()"
