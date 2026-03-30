"""XGBoost model wrapper (optional dependency)."""

from __future__ import annotations

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
        random_state: int = 42,
        n_jobs: int = -1,
        eval_metric: str = "logloss",
        tree_method: str = "hist",
    ):
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError(
                "xgboost is not installed. Install it before selecting model_type='xgboost'."
            ) from exc

        estimator = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_lambda=reg_lambda,
            random_state=random_state,
            n_jobs=n_jobs,
            eval_metric=eval_metric,
            tree_method=tree_method,
        )
        super().__init__(estimator=estimator, model_name="XGBoost")
