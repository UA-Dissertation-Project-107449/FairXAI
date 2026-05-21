"""Regression tests for mitigation AUC handling."""

import numpy as np
import pandas as pd

from fairxai.fairness.mitigation import MitigationEngine


def test_missing_probability_auc_is_nan_not_zero():
    engine = MitigationEngine()
    metrics = engine._compute_metrics(
        pd.Series([0, 1, 0, 1]),
        np.array([0, 1, 0, 1]),
        y_proba=None,
    )

    assert np.isnan(metrics["auc_roc"])


def test_postprocessing_after_fairlearn_predictor_uses_underlying_scores():
    class _Predictor:
        def predict_proba(self, X):
            scores = np.linspace(0.1, 0.9, len(X))
            return np.column_stack([1 - scores, scores])

    class _FairlearnLikeModel:
        predictors_ = [_Predictor()]

        def predict(self, X):
            return np.array([0, 0, 1, 1])

    class _Postprocessor:
        def predict(self, X, sensitive_features=None):
            return np.array([0, 0, 1, 1])

    engine = MitigationEngine()
    engine.postprocessing.apply_threshold_optimizer = lambda *args, **kwargs: _Postprocessor()

    X = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]})
    y = pd.Series([0, 0, 1, 1])
    sensitive = pd.DataFrame({"sex": [0, 1, 0, 1]})

    result = engine._apply_postprocessing(
        "threshold_optimizer",
        _FairlearnLikeModel(),
        X,
        y,
        X,
        y,
        sensitive,
        sensitive,
        "sex",
    )

    assert result["test_metrics"]["auc_roc"] == 1.0
    assert result["predictions"]["y_proba"] is not None
