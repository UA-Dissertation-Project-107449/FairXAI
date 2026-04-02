"""
Fairness mitigation techniques for bias reduction in ML models.

Provides pre-processing, in-processing, and post-processing methods to improve
fairness metrics while maintaining model performance.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from fairlearn.postprocessing import ThresholdOptimizer

# Fairlearn (in-processing and post-processing)
from fairlearn.reductions import DemographicParity, EqualizedOdds, ExponentiatedGradient, GridSearch

# Imbalanced-learn (pre-processing)
from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

# Scikit-learn
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_sample_weight

# Local imports
from ..models.baseline import BaselineLogisticRegression

logger = logging.getLogger(__name__)


class PreProcessingMitigation:
    """Pre-processing fairness mitigation techniques.

    These methods modify the training data before model training to reduce bias.
    """

    @staticmethod
    def apply_reweighting(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        sensitive_features: pd.DataFrame,
        sensitive_attr: str = "sex",
    ) -> np.ndarray:
        """
        Apply sample reweighting to balance representation.

        Computes sample weights to equalize the influence of different groups,
        reducing disparate impact from imbalanced sensitive attribute distributions.

        Args:
            X_train: Training features
            y_train: Training labels
            sensitive_features: DataFrame with sensitive attributes
            sensitive_attr: Which sensitive attribute to use for weighting

        Returns:
            Array of sample weights
        """
        logger.info(f"Applying reweighting based on {sensitive_attr}")

        # Create combined label: target + sensitive attribute
        combined = y_train.astype(str) + "_" + sensitive_features[sensitive_attr].astype(str)

        # Compute balanced weights for combined groups
        sample_weights = compute_sample_weight("balanced", combined)

        logger.info(f"  Weight range: [{sample_weights.min():.3f}, {sample_weights.max():.3f}]")
        return sample_weights

    @staticmethod
    def apply_smote(
        X_train: pd.DataFrame, y_train: pd.Series, k_neighbors: int = 5, random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Apply SMOTE (Synthetic Minority Over-sampling Technique).

        Generates synthetic samples for the minority class to balance the dataset.

        Args:
            X_train: Training features
            y_train: Training labels
            k_neighbors: Number of nearest neighbors for synthesis
            random_state: Random seed

        Returns:
            Tuple of (X_resampled, y_resampled)
        """
        logger.info(f"Applying SMOTE (k={k_neighbors})")
        logger.info(
            f"  Before: {len(X_train)} samples, class dist: {y_train.value_counts().to_dict()}"
        )

        counts = y_train.value_counts()
        min_count = int(counts.min()) if not counts.empty else 0

        if min_count < 2:
            logger.warning("SMOTE skipped: minority class has < 2 samples")
            return X_train.copy(), y_train.copy()

        k_neighbors = min(k_neighbors, max(min_count - 1, 1))

        sampler = SMOTE(
            sampling_strategy="auto", k_neighbors=k_neighbors, random_state=random_state
        )

        try:
            X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
        except ValueError as e:
            logger.warning(f"SMOTE failed ({e}); falling back to no resampling")
            return X_train.copy(), y_train.copy()

        logger.info(
            f"  After: {len(X_resampled)} samples, class dist: {pd.Series(y_resampled).value_counts().to_dict()}"
        )

        return pd.DataFrame(X_resampled, columns=X_train.columns), pd.Series(
            y_resampled, name=y_train.name
        )

    @staticmethod
    def apply_random_oversampling(
        X_train: pd.DataFrame, y_train: pd.Series, random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Apply random over-sampling (duplicate minority samples).

        Args:
            X_train: Training features
            y_train: Training labels
            random_state: Random seed

        Returns:
            Tuple of (X_resampled, y_resampled)
        """
        logger.info("Applying Random Over-Sampling")
        logger.info(f"  Before: {len(X_train)} samples")

        sampler = RandomOverSampler(sampling_strategy="auto", random_state=random_state)

        X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)

        logger.info(f"  After: {len(X_resampled)} samples")

        return pd.DataFrame(X_resampled, columns=X_train.columns), pd.Series(
            y_resampled, name=y_train.name
        )

    @staticmethod
    def apply_random_undersampling(
        X_train: pd.DataFrame, y_train: pd.Series, random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Apply random under-sampling (remove majority samples).

        Args:
            X_train: Training features
            y_train: Training labels
            random_state: Random seed

        Returns:
            Tuple of (X_resampled, y_resampled)
        """
        logger.info("Applying Random Under-Sampling")
        logger.info(f"  Before: {len(X_train)} samples")

        sampler = RandomUnderSampler(sampling_strategy="auto", random_state=random_state)

        X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)

        logger.info(f"  After: {len(X_resampled)} samples")

        return pd.DataFrame(X_resampled, columns=X_train.columns), pd.Series(
            y_resampled, name=y_train.name
        )

    @staticmethod
    def apply_adasyn(
        X_train: pd.DataFrame, y_train: pd.Series, n_neighbors: int = 5, random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Apply ADASYN (Adaptive Synthetic Sampling).

        Similar to SMOTE but focuses on samples that are harder to learn.

        Args:
            X_train: Training features
            y_train: Training labels
            n_neighbors: Number of nearest neighbors
            random_state: Random seed

        Returns:
            Tuple of (X_resampled, y_resampled)
        """
        logger.info(f"Applying ADASYN (n_neighbors={n_neighbors})")
        logger.info(f"  Before: {len(X_train)} samples")

        counts = y_train.value_counts()
        min_count = int(counts.min()) if not counts.empty else 0

        if min_count < 2:
            logger.warning("ADASYN skipped: minority class has < 2 samples")
            return X_train.copy(), y_train.copy()

        n_neighbors = min(n_neighbors, max(min_count - 1, 1))

        sampler = ADASYN(
            sampling_strategy="auto", n_neighbors=n_neighbors, random_state=random_state
        )

        try:
            X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
        except ValueError as e:
            logger.warning(f"ADASYN failed ({e}); falling back to no resampling")
            return X_train.copy(), y_train.copy()

        logger.info(f"  After: {len(X_resampled)} samples")

        return pd.DataFrame(X_resampled, columns=X_train.columns), pd.Series(
            y_resampled, name=y_train.name
        )


class InProcessingMitigation:
    """In-processing fairness mitigation techniques.

    These methods incorporate fairness constraints during model training.
    """

    @staticmethod
    def apply_exponentiated_gradient(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        sensitive_features: pd.DataFrame,
        sensitive_attr: str = "sex",
        constraint_type: str = "demographic_parity",
        eps: float = 0.05,
        max_iter: int = 50,
        random_state: int = 42,
        base_model_params: Optional[Dict[str, Any]] = None,
    ):
        """
        Apply Exponentiated Gradient reduction for fairness.

        Optimizes a fairness-constrained objective using game-theoretic approach.

        Args:
            X_train: Training features
            y_train: Training labels
            sensitive_features: DataFrame with sensitive attributes
            sensitive_attr: Which sensitive attribute to use
            constraint_type: 'demographic_parity' or 'equalized_odds'
            eps: Tolerance for constraint violation
            max_iter: Maximum iterations
            random_state: Random seed
            base_model_params: Optional LogisticRegression kwargs override

        Returns:
            Trained fairness-aware model
        """
        logger.info(f"Applying Exponentiated Gradient ({constraint_type})")
        logger.info(f"  Constraint tolerance: {eps}, max_iter: {max_iter}")

        # Select constraint
        if constraint_type == "demographic_parity":
            constraint = DemographicParity()
        elif constraint_type == "equalized_odds":
            constraint = EqualizedOdds()
        else:
            raise ValueError(f"Unknown constraint type: {constraint_type}")

        # Base estimator
        base_params = dict(base_model_params or {})
        base_params.setdefault("max_iter", 1000)
        base_params.setdefault("random_state", random_state)
        base_model = LogisticRegression(**base_params)

        # Fairness-aware model
        mitigator = ExponentiatedGradient(
            base_model, constraints=constraint, eps=eps, max_iter=max_iter
        )

        # Train with sensitive features
        mitigator.fit(X_train, y_train, sensitive_features=sensitive_features[sensitive_attr])

        logger.info(f"  Trained {len(mitigator.predictors_)} predictors")

        return mitigator

    @staticmethod
    def apply_grid_search(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        sensitive_features: pd.DataFrame,
        sensitive_attr: str = "sex",
        constraint_type: str = "equalized_odds",
        grid_size: int = 20,
        random_state: int = 42,
        base_model_params: Optional[Dict[str, Any]] = None,
    ):
        """
        Apply Grid Search reduction for fairness.

        Searches over lambda values to find optimal fairness-accuracy trade-off.

        Args:
            X_train: Training features
            y_train: Training labels
            sensitive_features: DataFrame with sensitive attributes
            sensitive_attr: Which sensitive attribute to use
            constraint_type: 'demographic_parity' or 'equalized_odds'
            grid_size: Number of lambda values to try
            random_state: Random seed

        Returns:
            Trained fairness-aware model
        """
        logger.info(f"Applying Grid Search ({constraint_type})")
        logger.info(f"  Grid size: {grid_size}")

        # Select constraint
        if constraint_type == "demographic_parity":
            constraint = DemographicParity()
        elif constraint_type == "equalized_odds":
            constraint = EqualizedOdds()
        else:
            raise ValueError(f"Unknown constraint type: {constraint_type}")

        # Base estimator
        base_params = dict(base_model_params or {})
        base_params.setdefault("max_iter", 1000)
        base_params.setdefault("random_state", random_state)
        base_model = LogisticRegression(**base_params)

        # Fairness-aware model
        mitigator = GridSearch(base_model, constraints=constraint, grid_size=grid_size)

        # Train with sensitive features
        mitigator.fit(X_train, y_train, sensitive_features=sensitive_features[sensitive_attr])

        logger.info(f"  Trained {len(mitigator.predictors_)} predictors")

        return mitigator


class PostProcessingMitigation:
    """Post-processing fairness mitigation techniques.

    These methods adjust predictions from a trained model to improve fairness.
    """

    @staticmethod
    def apply_threshold_optimizer(
        base_model,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        sensitive_features: pd.DataFrame,
        sensitive_attr: str = "sex",
        constraint_type: str = "equalized_odds",
        objective: str = "balanced_accuracy_score",
    ):
        """
        Apply Threshold Optimizer for post-processing fairness.

        Learns group-specific decision thresholds to satisfy fairness constraints.

        Args:
            base_model: Pre-trained model with predict_proba method
            X_train: Training features
            y_train: Training labels
            sensitive_features: DataFrame with sensitive attributes
            sensitive_attr: Which sensitive attribute to use
            constraint_type: 'equalized_odds', 'demographic_parity', etc.
            objective: Optimization objective

        Returns:
            ThresholdOptimizer instance
        """
        logger.info(f"Applying Threshold Optimizer ({constraint_type})")
        logger.info(f"  Objective: {objective}")

        postprocessor = ThresholdOptimizer(
            estimator=base_model, constraints=constraint_type, objective=objective, prefit=True
        )

        # Fit optimizer on training data
        postprocessor.fit(X_train, y_train, sensitive_features=sensitive_features[sensitive_attr])

        logger.info("  Threshold optimization complete")

        return postprocessor


class MitigationEngine:
    """Unified interface for applying fairness mitigation techniques.

    Routes technique requests to appropriate pre/in/post-processing methods
    and handles model training with mitigation.
    """

    # Valid stages and constraints
    VALID_STAGES = ["pre-processing", "in-processing", "post-processing"]
    VALID_PREPROCESSING = ["reweighting", "smote", "ros", "rus", "adasyn"]
    VALID_INPROCESSING = ["exponentiated_gradient", "grid_search"]
    VALID_POSTPROCESSING = ["threshold_optimizer"]

    def __init__(self, random_state: int = 42):
        """
        Initialize mitigation engine.

        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.preprocessing = PreProcessingMitigation()
        self.inprocessing = InProcessingMitigation()
        self.postprocessing = PostProcessingMitigation()

    def _compute_metrics(self, y_test, y_pred, y_proba=None) -> Dict[str, float]:
        """Compute standard evaluation metrics.

        Args:
            y_test: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (optional)

        Returns:
            Dictionary of metric names to values
        """
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )

        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        }

        # Add AUC-ROC if probabilities available
        if y_proba is not None and len(np.unique(y_proba)) > 1:
            try:
                metrics["auc_roc"] = float(roc_auc_score(y_test, y_proba))
            except ValueError:
                logger.warning("Could not compute AUC-ROC, setting to 0.0")
                metrics["auc_roc"] = 0.0
        else:
            metrics["auc_roc"] = 0.0

        return metrics

    def apply_technique(
        self,
        technique_name: str,
        stage: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        sensitive_train: pd.DataFrame,
        sensitive_test: pd.DataFrame,
        sensitive_attr: str = "sex",
        base_model=None,
        **kwargs,
    ) -> Dict:
        """
        Apply a mitigation technique and return trained model with metrics.

        Args:
            technique_name: Name of technique (e.g., 'smote', 'exponentiated_gradient')
            stage: 'pre-processing', 'in-processing', or 'post-processing'
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            sensitive_train: Sensitive attributes for training
            sensitive_test: Sensitive attributes for test
            sensitive_attr: Which sensitive attribute to use
            base_model: Pre-trained model (required for post-processing)
            **kwargs: Additional parameters for specific techniques

        Returns:
            Dictionary with 'model', 'predictions', and 'metadata'
        """
        # Validate inputs
        if stage not in self.VALID_STAGES:
            raise ValueError(f"stage must be one of {self.VALID_STAGES}, got: {stage}")

        if not isinstance(X_train, pd.DataFrame):
            raise TypeError(f"X_train must be pd.DataFrame, got: {type(X_train)}")

        if not isinstance(y_train, pd.Series):
            raise TypeError(f"y_train must be pd.Series, got: {type(y_train)}")

        logger.info(f"\n{'='*60}")
        logger.info(f"Applying {technique_name} ({stage})")
        logger.info(f"{'='*60}")

        if stage == "pre-processing":
            return self._apply_preprocessing(
                technique_name,
                X_train,
                y_train,
                X_test,
                y_test,
                sensitive_train,
                sensitive_test,
                sensitive_attr,
                **kwargs,
            )
        elif stage == "in-processing":
            return self._apply_inprocessing(
                technique_name,
                X_train,
                y_train,
                X_test,
                y_test,
                sensitive_train,
                sensitive_test,
                sensitive_attr,
                **kwargs,
            )
        elif stage == "post-processing":
            if base_model is None:
                raise ValueError("base_model required for post-processing techniques")
            return self._apply_postprocessing(
                technique_name,
                base_model,
                X_train,
                y_train,
                X_test,
                y_test,
                sensitive_train,
                sensitive_test,
                sensitive_attr,
                **kwargs,
            )
        else:
            raise ValueError(f"Unknown stage: {stage}")

    @staticmethod
    def _validate_combo_chain(techniques: List[str]) -> None:
        """Validate a combo chain satisfies pre* → in? → post? ordering.

        Rules:
        - At least 2 techniques required.
        - All pre-processing must come before in/post-processing.
        - At most one in-processing technique.
        - At most one post-processing technique.
        - in-processing and post-processing cannot both appear unless order is in → post.
        """
        if len(techniques) < 2:
            raise ValueError(f"apply_combo requires at least 2 techniques, got: {techniques}")

        _PRE = set(MitigationEngine.VALID_PREPROCESSING)
        _IN = set(MitigationEngine.VALID_INPROCESSING)
        _POST = set(MitigationEngine.VALID_POSTPROCESSING)

        stages_seen = []
        for t in techniques:
            if t in _PRE:
                stages_seen.append("pre")
            elif t in _IN:
                stages_seen.append("in")
            elif t in _POST:
                stages_seen.append("post")
            else:
                raise ValueError(f"Unknown technique in combo: '{t}'")

        # Check ordering: once we see 'in' or 'post', no more 'pre' is allowed
        seen_non_pre = False
        for stage in stages_seen:
            if stage != "pre":
                seen_non_pre = True
            if seen_non_pre and stage == "pre":
                raise ValueError(
                    f"Pre-processing techniques must come before in/post-processing in combo chain: {techniques}"
                )

        if stages_seen.count("in") > 1:
            raise ValueError(f"At most one in-processing technique allowed per combo: {techniques}")
        if stages_seen.count("post") > 1:
            raise ValueError(
                f"At most one post-processing technique allowed per combo: {techniques}"
            )

    @staticmethod
    def _extend_sensitive_for_resampled(
        sensitive_train: pd.DataFrame,
        n_resampled: int,
        random_state: int = 42,
    ) -> pd.DataFrame:
        """Extend sensitive_train to cover synthetic samples added by SMOTE/ADASYN.

        Synthetic samples are assigned sensitive attribute values by sampling
        with replacement from the original distribution (preserves group proportions).
        """
        n_original = len(sensitive_train)
        if n_resampled <= n_original:
            return sensitive_train.iloc[:n_resampled].reset_index(drop=True)
        n_synthetic = n_resampled - n_original
        extra = sensitive_train.sample(
            n=n_synthetic, replace=True, random_state=random_state
        ).reset_index(drop=True)
        return pd.concat([sensitive_train.reset_index(drop=True), extra], ignore_index=True)

    def apply_combo(
        self,
        techniques: List[str],
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        sensitive_train: pd.DataFrame,
        sensitive_test: pd.DataFrame,
        sensitive_attr: str = "sex",
        base_model_params=None,
    ) -> Dict:
        """Apply a sequential chain of mitigation techniques.

        Chain execution order: pre-processing* → in-processing? → post-processing?

        Pre-processing steps transform the training data in place.  If the
        data is expanded (SMOTE/ADASYN), ``sensitive_train`` is extended to
        match the new size by sampling with replacement from the original
        distribution.

        Parameters
        ----------
        techniques:
            Ordered list of technique names, e.g.
            ``['smote', 'exponentiated_gradient', 'threshold_optimizer']``.
        X_train, y_train:
            Training features and labels.
        X_test, y_test:
            Test features and labels.
        sensitive_train, sensitive_test:
            Sensitive attribute DataFrames.
        sensitive_attr:
            Which sensitive attribute column to use for fairness constraints.
        base_model_params:
            Optional hyperparameters for the base logistic regression model
            used by in/post-processing steps.

        Returns
        -------
        dict
            Same schema as ``apply_technique()``:
            ``{'model', 'test_metrics', 'predictions', 'metadata'}``
        """
        self._validate_combo_chain(techniques)

        logger.info(f"\n{'='*60}")
        logger.info(f"Applying combo chain: {' → '.join(techniques)}")
        logger.info(f"{'='*60}")

        _PRE = set(self.VALID_PREPROCESSING)
        _IN = set(self.VALID_INPROCESSING)
        _POST = set(self.VALID_POSTPROCESSING)

        pre_steps = [t for t in techniques if t in _PRE]
        in_step = next((t for t in techniques if t in _IN), None)
        post_step = next((t for t in techniques if t in _POST), None)

        # --- Stage 1: apply pre-processing steps sequentially ---
        X_curr = X_train.copy()
        y_curr = y_train.copy()
        sensitive_curr = sensitive_train.copy()
        sample_weights = None

        for technique in pre_steps:
            n_before = len(X_curr)
            if technique == "reweighting":
                sample_weights = self.preprocessing.apply_reweighting(
                    X_curr, y_curr, sensitive_curr, sensitive_attr
                )
                logger.info(
                    f"  [{technique}] sample weights computed; data shape unchanged {X_curr.shape}"
                )
            elif technique == "smote":
                X_curr, y_curr = self.preprocessing.apply_smote(
                    X_curr, y_curr, random_state=self.random_state
                )
                sensitive_curr = self._extend_sensitive_for_resampled(
                    sensitive_curr, len(X_curr), self.random_state
                )
                logger.info(f"  [{technique}] {n_before} → {len(X_curr)} training samples")
            elif technique == "adasyn":
                X_curr, y_curr = self.preprocessing.apply_adasyn(
                    X_curr, y_curr, random_state=self.random_state
                )
                sensitive_curr = self._extend_sensitive_for_resampled(
                    sensitive_curr, len(X_curr), self.random_state
                )
                logger.info(f"  [{technique}] {n_before} → {len(X_curr)} training samples")
            else:
                raise ValueError(
                    f"Pre-processing technique '{technique}' not supported in combos. "
                    f"Supported: {sorted(_PRE - {'ros', 'rus'})}"
                )

        # --- Stage 2: in-processing or baseline model training ---
        if in_step is not None:
            logger.info(f"  [{in_step}] applying in-processing on transformed data")
            result = self._apply_inprocessing(
                in_step,
                X_curr,
                y_curr,
                X_test,
                y_test,
                sensitive_curr,
                sensitive_test,
                sensitive_attr,
                base_model_params=base_model_params,
            )
            trained_model = result["model"]
        else:
            # Train a baseline LR model on the pre-processed data (needed for post-processing)
            logger.info("  [baseline] training logistic regression on pre-processed data")
            from fairxai.models.baseline import BaselineLogisticRegression

            _blr_params = {
                k: v for k, v in (base_model_params or {}).items() if k != "random_state"
            }
            trained_model = BaselineLogisticRegression(
                random_state=self.random_state,
                **_blr_params,
            )
            if sample_weights is not None:
                trained_model.model.fit(X_curr, y_curr, sample_weight=sample_weights)
            else:
                trained_model.train(X_curr, y_curr)

            y_pred = trained_model.predict(X_test)
            y_proba_raw = (
                trained_model.predict_proba(X_test)
                if hasattr(trained_model, "predict_proba")
                else None
            )
            if y_proba_raw is not None and hasattr(y_proba_raw, "ndim") and y_proba_raw.ndim > 1:
                y_proba_raw = y_proba_raw[:, 1]
            result = {
                "model": trained_model,
                "test_metrics": self._compute_metrics(y_test, y_pred, y_proba_raw),
                "predictions": {"y_pred": y_pred, "y_proba": y_proba_raw},
            }

        # --- Stage 3: post-processing ---
        if post_step is not None:
            logger.info(f"  [{post_step}] applying post-processing on trained model")
            # Post-processing needs the original (non-augmented) train split for threshold fitting
            result = self._apply_postprocessing(
                post_step,
                trained_model,
                X_train,
                y_train,
                X_test,
                y_test,
                sensitive_train,
                sensitive_test,
                sensitive_attr,
            )

        result["metadata"] = {
            "technique": "+".join(techniques),
            "stage": "combo",
            "combo_chain": techniques,
            "pre_steps": pre_steps,
            "in_step": in_step,
            "post_step": post_step,
        }
        return result

    def _apply_preprocessing(
        self,
        technique_name,
        X_train,
        y_train,
        X_test,
        y_test,
        sensitive_train,
        sensitive_test,
        sensitive_attr,
        **kwargs,
    ) -> Dict:
        """Apply pre-processing technique and train model."""
        # Apply resampling/reweighting
        if technique_name == "reweighting":
            sample_weights = self.preprocessing.apply_reweighting(
                X_train, y_train, sensitive_train, sensitive_attr
            )
            # Train with sample weights
            model = BaselineLogisticRegression(random_state=self.random_state)
            model.model.fit(X_train, y_train, sample_weight=sample_weights)
            X_train_processed, y_train_processed = X_train, y_train

        elif technique_name == "smote":
            X_train_processed, y_train_processed = self.preprocessing.apply_smote(
                X_train, y_train, random_state=self.random_state
            )
            model = BaselineLogisticRegression(random_state=self.random_state)
            model.train(X_train_processed, y_train_processed)

        elif technique_name == "ros":
            X_train_processed, y_train_processed = self.preprocessing.apply_random_oversampling(
                X_train, y_train, random_state=self.random_state
            )
            model = BaselineLogisticRegression(random_state=self.random_state)
            model.train(X_train_processed, y_train_processed)

        elif technique_name == "rus":
            X_train_processed, y_train_processed = self.preprocessing.apply_random_undersampling(
                X_train, y_train, random_state=self.random_state
            )
            model = BaselineLogisticRegression(random_state=self.random_state)
            model.train(X_train_processed, y_train_processed)

        elif technique_name == "adasyn":
            X_train_processed, y_train_processed = self.preprocessing.apply_adasyn(
                X_train, y_train, random_state=self.random_state
            )
            model = BaselineLogisticRegression(random_state=self.random_state)
            model.train(X_train_processed, y_train_processed)
        else:
            raise ValueError(f"Unknown pre-processing technique: {technique_name}")

        # Evaluate and get predictions
        test_metrics = model.evaluate(X_test, y_test)
        y_pred = model.predict(X_test)
        # BaselineLogisticRegression.predict_proba already returns 1D probs for the positive class
        y_proba_raw = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
        if y_proba_raw is None:
            y_proba = None
        elif hasattr(y_proba_raw, "ndim") and y_proba_raw.ndim > 1:
            y_proba = y_proba_raw[:, 1]
        else:
            y_proba = y_proba_raw

        return {
            "model": model,
            "test_metrics": test_metrics,
            "predictions": {"y_pred": y_pred, "y_proba": y_proba},
            "metadata": {
                "technique": technique_name,
                "stage": "pre-processing",
                "samples_before": len(X_train),
                "samples_after": len(X_train_processed),
            },
        }

    def _apply_inprocessing(
        self,
        technique_name,
        X_train,
        y_train,
        X_test,
        y_test,
        sensitive_train,
        sensitive_test,
        sensitive_attr,
        **kwargs,
    ) -> Dict:
        """Apply in-processing technique."""
        base_model_params = kwargs.pop("base_model_params", None)
        if technique_name == "exponentiated_gradient":
            model = self.inprocessing.apply_exponentiated_gradient(
                X_train,
                y_train,
                sensitive_train,
                sensitive_attr,
                random_state=self.random_state,
                base_model_params=base_model_params,
                **kwargs,
            )
        elif technique_name == "grid_search":
            model = self.inprocessing.apply_grid_search(
                X_train,
                y_train,
                sensitive_train,
                sensitive_attr,
                random_state=self.random_state,
                base_model_params=base_model_params,
                **kwargs,
            )
        else:
            raise ValueError(f"Unknown in-processing technique: {technique_name}")

        # Predict on test set
        y_pred = model.predict(X_test)

        # Get probabilities (Fairlearn models may have multiple predictors)
        y_proba = None
        try:
            if hasattr(model, "predictors_") and len(model.predictors_) > 0:
                predictor = model.predictors_[0]
                if hasattr(predictor, "predict_proba"):
                    y_proba = predictor.predict_proba(X_test)[:, 1]
            elif hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_test)[:, 1]
        except Exception as e:
            logger.warning(f"Could not extract probabilities: {e}")
            y_proba = None

        # Calculate metrics using helper method
        test_metrics = self._compute_metrics(y_test, y_pred, y_proba)

        return {
            "model": model,
            "test_metrics": test_metrics,
            "predictions": {"y_pred": y_pred, "y_proba": y_proba},
            "metadata": {
                "technique": technique_name,
                "stage": "in-processing",
                "n_predictors": len(model.predictors_) if hasattr(model, "predictors_") else 1,
            },
        }

    def _apply_postprocessing(
        self,
        technique_name,
        base_model,
        X_train,
        y_train,
        X_test,
        y_test,
        sensitive_train,
        sensitive_test,
        sensitive_attr,
        **kwargs,
    ) -> Dict:
        """Apply post-processing technique."""
        _ = kwargs.pop("base_model_params", None)
        if technique_name == "threshold_optimizer":
            if sensitive_attr not in sensitive_train.columns:
                raise ValueError(
                    f"Sensitive attribute '{sensitive_attr}' not found in training data"
                )
            group_counts = y_train.groupby(sensitive_train[sensitive_attr]).nunique(dropna=False)
            if (group_counts < 2).any():
                logger.warning(
                    "Degenerate labels for at least one sensitive group; "
                    "skipping threshold optimizer and returning baseline predictions."
                )
                y_pred = base_model.predict(X_test)
                y_proba = None
                if hasattr(base_model, "model") and hasattr(base_model.model, "predict_proba"):
                    y_proba = base_model.model.predict_proba(X_test)[:, 1]
                elif hasattr(base_model, "predict_proba"):
                    y_proba = base_model.predict_proba(X_test)[:, 1]
                test_metrics = self._compute_metrics(y_test, y_pred, y_proba)
                return {
                    "model": base_model,
                    "test_metrics": test_metrics,
                    "predictions": {"y_pred": y_pred, "y_proba": y_proba},
                    "metadata": {
                        "technique": technique_name,
                        "stage": "post-processing",
                        "base_model": type(base_model).__name__,
                        "skipped": True,
                        "skip_reason": "degenerate_labels",
                    },
                }
            # Resolve the underlying sklearn estimator for ThresholdOptimizer.
            # BaselineLogisticRegression wraps sklearn in .model; fairlearn models
            # (ExponentiatedGradient, GridSearch) use predictors_; fall back to base_model.
            if hasattr(base_model, "model"):
                estimator_for_post = base_model.model
            elif hasattr(base_model, "predictors_") and len(base_model.predictors_) > 0:
                estimator_for_post = next(
                    (p for p in base_model.predictors_ if hasattr(p, "predict_proba")),
                    base_model.predictors_[0],
                )
            else:
                estimator_for_post = base_model
            postprocessor = self.postprocessing.apply_threshold_optimizer(
                estimator_for_post, X_train, y_train, sensitive_train, sensitive_attr, **kwargs
            )
        else:
            raise ValueError(f"Unknown post-processing technique: {technique_name}")

        # Predict on test set
        y_pred = postprocessor.predict(X_test, sensitive_features=sensitive_test[sensitive_attr])

        # Get probabilities from base model
        y_proba = None
        if hasattr(base_model, "model") and hasattr(base_model.model, "predict_proba"):
            y_proba = base_model.model.predict_proba(X_test)[:, 1]
        elif hasattr(base_model, "predict_proba"):
            y_proba = base_model.predict_proba(X_test)[:, 1]

        # Calculate metrics using helper method
        test_metrics = self._compute_metrics(y_test, y_pred, y_proba)

        return {
            "model": postprocessor,
            "test_metrics": test_metrics,
            "predictions": {"y_pred": y_pred, "y_proba": y_proba},
            "metadata": {
                "technique": technique_name,
                "stage": "post-processing",
                "base_model": type(base_model).__name__,
            },
        }
