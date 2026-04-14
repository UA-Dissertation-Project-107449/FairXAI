"""Generic sklearn-like classifier wrapper utilities."""

from __future__ import annotations

import logging
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


class SklearnClassifierWrapper:
    """Reusable wrapper that normalizes train/eval/predict API across models."""

    def __init__(self, estimator, model_name: str):
        self.model = estimator
        self.model_name = model_name
        self.feature_names = None
        self.training_metrics = {}

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict:
        self.feature_names = list(X_train.columns)

        logging.info(f"Training {self.model_name}...")
        logging.info(f"  Features: {len(self.feature_names)}")
        logging.info(f"  Training samples: {len(X_train)}")
        logging.info(f"  Positive class: {y_train.sum()} ({y_train.mean():.2%})")

        self.model.fit(X_train, y_train)

        y_train_pred = self.model.predict(X_train)
        y_train_proba = self.predict_proba(X_train)

        self.training_metrics = self._calculate_metrics(y_train, y_train_pred, y_train_proba)

        logging.info("[SUCCESS] Training complete")
        logging.info(f"  Train Accuracy: {self.training_metrics['accuracy']:.4f}")
        logging.info(f"  Train AUC-ROC: {self.training_metrics['auc_roc']:.4f}")

        return self.training_metrics

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X)
            if np.asarray(proba).ndim == 2 and np.asarray(proba).shape[1] > 1:
                return np.asarray(proba)[:, 1]
            return np.asarray(proba).reshape(-1)

        if hasattr(self.model, "decision_function"):
            scores = np.asarray(self.model.decision_function(X)).reshape(-1)
            return 1.0 / (1.0 + np.exp(-scores))

        # Last-resort fallback for models without calibrated scores.
        return np.asarray(self.model.predict(X)).reshape(-1).astype(float)

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series, threshold: float = 0.5) -> Dict:
        y_pred = self.predict(X_test, threshold=threshold)
        y_proba = self.predict_proba(X_test)

        metrics = self._calculate_metrics(y_test, y_pred, y_proba)
        metrics["threshold"] = threshold

        cm = confusion_matrix(y_test, y_pred)
        metrics["confusion_matrix"] = {
            "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]),
            "tp": int(cm[1, 1]),
        }
        return metrics

    def _calculate_metrics(
        self, y_true: pd.Series, y_pred: np.ndarray, y_proba: np.ndarray
    ) -> Dict:
        auc = 0.0
        try:
            if len(np.unique(y_proba)) > 1:
                auc = float(roc_auc_score(y_true, y_proba))
        except ValueError:
            auc = 0.0

        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
            "auc_roc": auc,
            "n_samples": len(y_true),
            "n_positive": int(y_true.sum()),
            "n_negative": int((y_true == 0).sum()),
        }

    def get_feature_importance(self) -> pd.DataFrame:
        if self.feature_names is None:
            raise ValueError("Model must be trained first")

        if hasattr(self.model, "feature_importances_"):
            raw = np.asarray(self.model.feature_importances_).reshape(-1)
            return pd.DataFrame(
                {
                    "feature": self.feature_names,
                    "importance": raw,
                    "abs_importance": np.abs(raw),
                }
            ).sort_values("abs_importance", ascending=False)

        if hasattr(self.model, "coef_"):
            coeff = np.asarray(self.model.coef_)
            if coeff.ndim > 1:
                coeff = coeff[0]
            coeff = coeff.reshape(-1)
            return pd.DataFrame(
                {
                    "feature": self.feature_names,
                    "coefficient": coeff,
                    "abs_coefficient": np.abs(coeff),
                }
            ).sort_values("abs_coefficient", ascending=False)

        logging.info(
            "%s does not expose direct feature importance coefficients; "
            "returning NaN importances.",
            self.model_name,
        )
        return pd.DataFrame(
            {
                "feature": self.feature_names,
                "importance": [np.nan] * len(self.feature_names),
            }
        )

    def save(self, filepath: str):
        joblib.dump(
            {
                "model": self.model,
                "feature_names": self.feature_names,
                "training_metrics": self.training_metrics,
                "model_name": self.model_name,
            },
            filepath,
        )
        logging.info(f"[SUCCESS] Model saved to: {filepath}")

    def load(self, filepath: str):
        data = joblib.load(filepath)
        self.model = data["model"]
        self.feature_names = data.get("feature_names")
        self.training_metrics = data.get("training_metrics", {})
        self.model_name = data.get("model_name", self.model_name)
        logging.info(f"[SUCCESS] Model loaded from: {filepath}")
