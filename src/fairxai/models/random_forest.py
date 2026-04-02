"""Random Forest model wrapper."""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier

from .sklearn_wrapper import SklearnClassifierWrapper


class RandomForestModel(SklearnClassifierWrapper):
    """Random Forest baseline for cardiac disease prediction."""

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: str = "sqrt",
        class_weight: str | None = "balanced_subsample",
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        estimator = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        super().__init__(estimator=estimator, model_name="RandomForest")
