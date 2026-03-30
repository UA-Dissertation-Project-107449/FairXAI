"""SVM model wrapper."""

from __future__ import annotations

from sklearn.svm import SVC

from .sklearn_wrapper import SklearnClassifierWrapper


class SVMModel(SklearnClassifierWrapper):
    """SVM baseline for cardiac disease prediction."""

    def __init__(
        self,
        C: float = 1.0,
        kernel: str = "rbf",
        gamma: str | float = "scale",
        probability: bool = True,
        class_weight: str | None = "balanced",
        max_iter: int = -1,
        random_state: int = 42,
    ):
        estimator = SVC(
            C=C,
            kernel=kernel,
            gamma=gamma,
            probability=probability,
            class_weight=class_weight,
            max_iter=max_iter,
            random_state=random_state,
        )
        super().__init__(estimator=estimator, model_name="SVM")
