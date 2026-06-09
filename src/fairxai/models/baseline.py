"""Baseline model implementations for cardiac disease prediction."""

import logging
from typing import Dict, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


class BaselineLogisticRegression:
    """Logistic Regression baseline for cardiac disease prediction."""

    def __init__(
        self,
        C: float = 1.0,
        penalty: str = "l2",
        solver: str = "lbfgs",
        tol: float = 1e-4,
        l1_ratio: Optional[float] = None,
        max_iter: int = 1000,
        random_state: int = 42,
        class_weight: Optional[str] = None,
    ):
        """
        Initialize Logistic Regression model.

        Args:
            C: Inverse of regularization strength
            penalty: Regularization penalty (e.g. 'l1', 'l2', 'elasticnet', 'none')
            solver: Optimization solver (e.g. 'lbfgs', 'liblinear', 'saga')
            tol: Tolerance for stopping criteria
            l1_ratio: Elastic-net mixing parameter (only used if penalty='elasticnet')
            max_iter: Maximum iterations for convergence
            random_state: Random seed
            class_weight: 'balanced' to handle class imbalance, or None
        """
        self.model = LogisticRegression(
            C=C,
            penalty=penalty,
            solver=solver,
            tol=tol,
            l1_ratio=l1_ratio,
            max_iter=max_iter,
            random_state=random_state,
            class_weight=class_weight,
        )
        self.feature_names = None
        self.training_metrics = {}

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict:
        """
        Train the logistic regression model.

        Args:
            X_train: Training features
            y_train: Training labels

        Returns:
            Dictionary with training metrics
        """
        self.feature_names = list(X_train.columns)

        logging.info("Training Logistic Regression...")
        logging.info(f"  Features: {len(self.feature_names)}")
        logging.info(f"  Training samples: {len(X_train)}")
        logging.info(f"  Positive class: {y_train.sum()} ({y_train.mean():.2%})")

        self.model.fit(X_train, y_train)

        # Calculate training metrics
        y_train_pred = self.model.predict(X_train)
        y_train_proba = self.model.predict_proba(X_train)[:, 1]

        self.training_metrics = self._calculate_metrics(y_train, y_train_pred, y_train_proba)

        train_auc = self.training_metrics.get("auc_roc", 0.0)
        train_f1 = self.training_metrics.get("f1_score", 0.0)
        train_acc = self.training_metrics.get("accuracy", 0.0)
        if train_auc >= 0.98 or train_f1 >= 0.98 or train_acc >= 0.99:
            logging.warning(
                f"[OVERFIT-RISK] LogisticRegression: train_auc_roc={train_auc:.4f} "
                f"train_f1={train_f1:.4f} train_accuracy={train_acc:.4f} "
                f"— check test/CV metrics for memorization."
            )

        logging.info(
            "[SUCCESS] LogisticRegression training complete: "
            f"train_accuracy={self.training_metrics['accuracy']:.4f} "
            f"train_auc_roc={self.training_metrics['auc_roc']:.4f}"
        )

        return self.training_metrics

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """
        Predict binary labels.

        Args:
            X: Features
            threshold: Decision threshold

        Returns:
            Binary predictions
        """
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probabilities for positive class.

        Args:
            X: Features

        Returns:
            Probabilities for positive class
        """
        return self.model.predict_proba(X)[:, 1]

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series, threshold: float = 0.5) -> Dict:
        """
        Evaluate model on test set.

        Args:
            X_test: Test features
            y_test: Test labels
            threshold: Decision threshold

        Returns:
            Dictionary with evaluation metrics
        """
        y_pred = self.predict(X_test, threshold=threshold)
        y_proba = self.predict_proba(X_test)

        metrics = self._calculate_metrics(y_test, y_pred, y_proba)
        metrics["threshold"] = threshold

        # Add confusion matrix
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
        """Calculate standard classification metrics."""
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
            "auc_roc": float(roc_auc_score(y_true, y_proba)),
            "n_samples": len(y_true),
            "n_positive": int(y_true.sum()),
            "n_negative": int((y_true == 0).sum()),
        }

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance based on coefficients.

        Returns:
            DataFrame with features and their coefficients
        """
        if self.feature_names is None:
            raise ValueError("Model must be trained first")

        coeffs = self.model.coef_[0]
        importance_df = pd.DataFrame(
            {
                "feature": self.feature_names,
                "coefficient": coeffs,
                "abs_coefficient": np.abs(coeffs),
            }
        ).sort_values("abs_coefficient", ascending=False)

        return importance_df

    def save(self, filepath: str):
        """Save model to disk."""
        joblib.dump(
            {
                "model": self.model,
                "feature_names": self.feature_names,
                "training_metrics": self.training_metrics,
            },
            filepath,
        )
        logging.info(f"Model saved to: {filepath}")

    def load(self, filepath: str):
        """Load model from disk."""
        data = joblib.load(filepath)
        self.model = data["model"]
        self.feature_names = data["feature_names"]
        self.training_metrics = data["training_metrics"]
        logging.info(f"Model loaded from: {filepath}")


def generate_predictions_with_metadata(
    model: BaselineLogisticRegression,
    X: pd.DataFrame,
    y: pd.Series,
    sensitive_attrs: pd.DataFrame,
    threshold: float = 0.5,
    extra_meta: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Generate predictions with probabilities and metadata.

    Args:
        model: Trained model
        X: Features
        y: True labels
        sensitive_attrs: DataFrame with sensitive attributes (age_group, sex)
        threshold: Decision threshold
        extra_meta: Optional analysis-only metadata carried alongside predictions
            (e.g. continuous ``age_raw`` for post-hoc re-binning). Unlike
            ``sensitive_attrs`` these are NOT fairness-grouping keys and NOT
            model features — they are passengers for downstream analysis.

    Returns:
        DataFrame with predictions, probabilities, metadata, and features
    """
    predictions = pd.DataFrame(
        {
            "y_true": y.values,
            "y_pred": model.predict(X, threshold=threshold),
            "y_proba": model.predict_proba(X),
            "threshold": threshold,
        }
    )

    # Add sensitive attributes
    for col in sensitive_attrs.columns:
        predictions[col] = sensitive_attrs[col].values

    # Add analysis-only metadata (carried, never a feature or grouping key)
    if extra_meta is not None:
        for col in extra_meta.columns:
            predictions[col] = extra_meta[col].values

    # Add features for individual fairness calculation
    for col in X.columns:
        predictions[col] = X[col].values

    # Add prediction confidence
    predictions["confidence"] = np.abs(predictions["y_proba"] - 0.5)

    # Flag near-threshold predictions
    predictions["near_threshold"] = np.abs(predictions["y_proba"] - threshold) < 0.1

    return predictions
