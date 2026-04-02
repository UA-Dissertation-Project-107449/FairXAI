"""Models package public API and model registry utilities."""

from __future__ import annotations

from .baseline import BaselineLogisticRegression, generate_predictions_with_metadata
from .cv_trainer import CVTrainer
from .random_forest import RandomForestModel
from .svm import SVMModel
from .xgboost_model import XGBoostModel

MODEL_REGISTRY = {
    "logistic_regression": BaselineLogisticRegression,
    "random_forest": RandomForestModel,
    "svm": SVMModel,
    "xgboost": XGBoostModel,
}


def get_model_class(model_type: str):
    """Resolve model class by registry key."""
    key = (model_type or "").strip().lower()
    if key not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise ValueError(f"Unknown model_type '{model_type}'. Available: {available}")
    return MODEL_REGISTRY[key]


__all__ = [
    "BaselineLogisticRegression",
    "RandomForestModel",
    "SVMModel",
    "XGBoostModel",
    "CVTrainer",
    "MODEL_REGISTRY",
    "get_model_class",
    "generate_predictions_with_metadata",
]
