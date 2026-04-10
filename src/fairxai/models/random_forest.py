"""Random Forest model wrapper with optional RAPIDS cuML GPU backend."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .sklearn_wrapper import SklearnClassifierWrapper

logger = logging.getLogger(__name__)

_CUML_SAMPLE_WEIGHT_SUPPORTED: bool | None = None


class RandomForestModel(SklearnClassifierWrapper):
    """Random Forest baseline for cardiac disease prediction.

    When ``use_gpu=True`` and RAPIDS cuML is available, training uses
    ``cuml.ensemble.RandomForestClassifier`` for GPU-accelerated fitting.
    Falls back to ``sklearn.ensemble.RandomForestClassifier`` silently if
    cuML is not installed or if GPU initialisation fails.

    cuML differences handled transparently:
    - No ``class_weight`` parameter → balanced ``sample_weight`` is computed
      at fit time via :func:`sklearn.utils.class_weight.compute_sample_weight`.
    - Output arrays may be ``cupy.ndarray`` → converted to NumPy before return.
    - ``n_jobs`` is ignored when using the cuML backend (GPU handles parallelism).
    """

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
        use_gpu: bool = False,
    ):
        self._use_gpu = False
        self._class_weight = class_weight

        if use_gpu:
            try:
                from cuml.ensemble import RandomForestClassifier as CumlRF

                # cuML RF: max_depth=None not accepted; use a large value instead.
                effective_depth = max_depth if max_depth is not None else 16
                estimator = CumlRF(
                    n_estimators=n_estimators,
                    max_depth=effective_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    max_features=max_features,
                    random_state=random_state,
                    # n_jobs not applicable for cuML (GPU handles it)
                )
                self._use_gpu = True
                logger.info("RandomForestModel: using RAPIDS cuML GPU backend")
            except Exception as exc:
                logger.warning(
                    f"RandomForestModel: cuML unavailable ({exc}); falling back to sklearn."
                )

        if not self._use_gpu:
            from sklearn.ensemble import RandomForestClassifier

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

    # ------------------------------------------------------------------
    # cuML-aware overrides
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_sample_weight_strategy(class_weight: str | dict | list | None):
        """Map RF class_weight to a strategy accepted by compute_sample_weight."""
        if class_weight == "balanced_subsample":
            # sklearn's compute_sample_weight does not accept this RF-only mode.
            return "balanced"
        return class_weight or "balanced"

    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        if self._use_gpu:
            from sklearn.utils.class_weight import compute_sample_weight

            self.feature_names = list(X_train.columns)
            logger.info("Training RandomForest (cuML GPU)...")
            logger.info(f"  Features: {len(self.feature_names)}")
            logger.info(f"  Training samples: {len(X_train)}")
            logger.info(f"  Positive class: {y_train.sum()} ({y_train.mean():.2%})")

            sample_weight_mode = self._resolve_sample_weight_strategy(self._class_weight)
            sample_weight = compute_sample_weight(sample_weight_mode, y_train.values)
            global _CUML_SAMPLE_WEIGHT_SUPPORTED
            if _CUML_SAMPLE_WEIGHT_SUPPORTED is False:
                self.model.fit(X_train.values, y_train.values)
            else:
                try:
                    self.model.fit(X_train.values, y_train.values, sample_weight=sample_weight)
                    _CUML_SAMPLE_WEIGHT_SUPPORTED = True
                except TypeError as exc:
                    logger.debug(
                        "RandomForestModel (cuML): sample_weight unsupported (%s); "
                        "using unweighted fit.",
                        exc,
                    )
                    _CUML_SAMPLE_WEIGHT_SUPPORTED = False
                    self.model.fit(X_train.values, y_train.values)

            y_train_pred = self.predict(X_train)
            y_train_proba = self.predict_proba(X_train)
            self.training_metrics = self._calculate_metrics(y_train, y_train_pred, y_train_proba)

            logger.info("✓ cuML RandomForest training complete")
            logger.info(f"  Train Accuracy: {self.training_metrics['accuracy']:.4f}")
            logger.info(f"  Train AUC-ROC: {self.training_metrics['auc_roc']:.4f}")
            return self.training_metrics

        return super().train(X_train, y_train)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        proba = super().predict_proba(X)
        # cuML may return cupy arrays — coerce to NumPy for downstream compatibility.
        try:
            import cupy as cp  # type: ignore[import-not-found]

            if isinstance(proba, cp.ndarray):
                proba = cp.asnumpy(proba)
        except ImportError:
            pass
        return np.asarray(proba)
