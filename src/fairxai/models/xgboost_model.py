"""XGBoost model wrapper (optional dependency)."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .sklearn_wrapper import SklearnClassifierWrapper


class XGBoostModel(SklearnClassifierWrapper):
    """XGBoost baseline for cardiac disease prediction."""

    def __init__(
        self,
        n_estimators: int = 250,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        subsample: float = 0.9,
        colsample_bytree: float = 0.9,
        reg_lambda: float = 1.0,
        min_child_weight: int = 3,
        gamma: float = 0.1,
        random_state: int = 42,
        n_jobs: int = -1,
        eval_metric: str = "logloss",
        tree_method: str = "hist",
        device: str = "cpu",
    ):
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError(
                "xgboost is not installed. Install it before selecting model_type='xgboost'."
            ) from exc

        # XGBoost ≥2.0: tree_method must be 'hist' when device='cuda'
        resolved_tree_method = "hist" if device == "cuda" else tree_method

        estimator = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_lambda=reg_lambda,
            min_child_weight=min_child_weight,
            gamma=gamma,
            random_state=random_state,
            n_jobs=n_jobs,
            eval_metric=eval_metric,
            tree_method=resolved_tree_method,
            device=device,
        )
        self._device = device
        super().__init__(estimator=estimator, model_name="XGBoost")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self._device != "cuda":
            return super().predict_proba(X)

        try:
            from xgboost import DMatrix

            booster = self.model.get_booster()
            feature_names = None
            if hasattr(X, "columns"):
                cols = list(X.columns)
                if all(isinstance(col, str) for col in cols):
                    feature_names = cols
            elif self.feature_names and all(isinstance(col, str) for col in self.feature_names):
                feature_names = list(self.feature_names)

            if feature_names:
                dmatrix = DMatrix(X, feature_names=feature_names)
            else:
                dmatrix = DMatrix(X)

            raw_proba = np.asarray(booster.predict(dmatrix))
            if raw_proba.ndim == 2 and raw_proba.shape[1] > 1:
                return raw_proba[:, 1]
            return raw_proba.reshape(-1)
        except Exception as exc:
            logging.debug(
                "XGBoostModel: DMatrix prediction path unavailable; falling back to sklearn API (%s)",
                exc,
            )
            return super().predict_proba(X)

    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        # Override base implementation to avoid direct estimator.predict(X) call,
        # which triggers CUDA/CPU mismatch warnings in sklearn's XGBoost adapter.
        self.feature_names = list(X_train.columns)

        logging.info(f"Training {self.model_name}...")
        logging.info(f"  Features: {len(self.feature_names)}")
        logging.info(f"  Training samples: {len(X_train)}")
        logging.info(f"  Positive class: {y_train.sum()} ({y_train.mean():.2%})")

        self.model.fit(X_train, y_train)

        y_train_pred = self.predict(X_train)
        y_train_proba = self.predict_proba(X_train)

        self.training_metrics = self._calculate_metrics(y_train, y_train_pred, y_train_proba)

        train_auc = self.training_metrics.get("auc_roc", 0.0)
        train_f1 = self.training_metrics.get("f1_score", 0.0)
        train_acc = self.training_metrics.get("accuracy", 0.0)
        if train_auc >= 0.98 or train_f1 >= 0.98 or train_acc >= 0.99:
            logging.warning(
                f"[OVERFIT-RISK] {self.model_name}: train_auc_roc={train_auc:.4f} "
                f"train_f1={train_f1:.4f} train_accuracy={train_acc:.4f} "
                f"— check test/CV metrics for memorization."
            )

        logging.info(
            "[SUCCESS] XGBoost training complete: "
            f"train_accuracy={self.training_metrics['accuracy']:.4f} "
            f"train_auc_roc={self.training_metrics['auc_roc']:.4f}"
        )

        return self.training_metrics
