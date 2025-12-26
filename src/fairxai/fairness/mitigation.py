"""
Fairness mitigation techniques for bias reduction in ML models.

Provides pre-processing, in-processing, and post-processing methods to improve
fairness metrics while maintaining model performance.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Union

# Scikit-learn
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_sample_weight

# Imbalanced-learn (pre-processing)
from imblearn.over_sampling import SMOTE, RandomOverSampler, ADASYN
from imblearn.under_sampling import RandomUnderSampler

# Fairlearn (in-processing and post-processing)
from fairlearn.reductions import ExponentiatedGradient, GridSearch, DemographicParity, EqualizedOdds
from fairlearn.postprocessing import ThresholdOptimizer

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
        sensitive_attr: str = 'sex'
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
        sample_weights = compute_sample_weight('balanced', combined)
        
        logger.info(f"  Weight range: [{sample_weights.min():.3f}, {sample_weights.max():.3f}]")
        return sample_weights
    
    @staticmethod
    def apply_smote(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        k_neighbors: int = 5,
        random_state: int = 42
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
        logger.info(f"  Before: {len(X_train)} samples, class dist: {y_train.value_counts().to_dict()}")
        
        sampler = SMOTE(
            sampling_strategy='auto',
            k_neighbors=k_neighbors,
            random_state=random_state
        )
        
        X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
        
        logger.info(f"  After: {len(X_resampled)} samples, class dist: {pd.Series(y_resampled).value_counts().to_dict()}")
        
        return pd.DataFrame(X_resampled, columns=X_train.columns), pd.Series(y_resampled, name=y_train.name)
    
    @staticmethod
    def apply_random_oversampling(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        random_state: int = 42
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
        
        sampler = RandomOverSampler(
            sampling_strategy='auto',
            random_state=random_state
        )
        
        X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
        
        logger.info(f"  After: {len(X_resampled)} samples")
        
        return pd.DataFrame(X_resampled, columns=X_train.columns), pd.Series(y_resampled, name=y_train.name)
    
    @staticmethod
    def apply_random_undersampling(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        random_state: int = 42
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
        
        sampler = RandomUnderSampler(
            sampling_strategy='auto',
            random_state=random_state
        )
        
        X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
        
        logger.info(f"  After: {len(X_resampled)} samples")
        
        return pd.DataFrame(X_resampled, columns=X_train.columns), pd.Series(y_resampled, name=y_train.name)
    
    @staticmethod
    def apply_adasyn(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        n_neighbors: int = 5,
        random_state: int = 42
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
        
        sampler = ADASYN(
            sampling_strategy='auto',
            n_neighbors=n_neighbors,
            random_state=random_state
        )
        
        X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
        
        logger.info(f"  After: {len(X_resampled)} samples")
        
        return pd.DataFrame(X_resampled, columns=X_train.columns), pd.Series(y_resampled, name=y_train.name)


class InProcessingMitigation:
    """In-processing fairness mitigation techniques.
    
    These methods incorporate fairness constraints during model training.
    """
    
    @staticmethod
    def apply_exponentiated_gradient(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        sensitive_features: pd.DataFrame,
        sensitive_attr: str = 'sex',
        constraint_type: str = 'demographic_parity',
        eps: float = 0.05,
        max_iter: int = 50,
        random_state: int = 42
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
            
        Returns:
            Trained fairness-aware model
        """
        logger.info(f"Applying Exponentiated Gradient ({constraint_type})")
        logger.info(f"  Constraint tolerance: {eps}, max_iter: {max_iter}")
        
        # Select constraint
        if constraint_type == 'demographic_parity':
            constraint = DemographicParity()
        elif constraint_type == 'equalized_odds':
            constraint = EqualizedOdds()
        else:
            raise ValueError(f"Unknown constraint type: {constraint_type}")
        
        # Base estimator
        base_model = LogisticRegression(max_iter=1000, random_state=random_state)
        
        # Fairness-aware model
        mitigator = ExponentiatedGradient(
            base_model,
            constraints=constraint,
            eps=eps,
            max_iter=max_iter
        )
        
        # Train with sensitive features
        mitigator.fit(
            X_train,
            y_train,
            sensitive_features=sensitive_features[sensitive_attr]
        )
        
        logger.info(f"  Trained {len(mitigator.predictors_)} predictors")
        
        return mitigator
    
    @staticmethod
    def apply_grid_search(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        sensitive_features: pd.DataFrame,
        sensitive_attr: str = 'sex',
        constraint_type: str = 'equalized_odds',
        grid_size: int = 20,
        random_state: int = 42
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
        if constraint_type == 'demographic_parity':
            constraint = DemographicParity()
        elif constraint_type == 'equalized_odds':
            constraint = EqualizedOdds()
        else:
            raise ValueError(f"Unknown constraint type: {constraint_type}")
        
        # Base estimator
        base_model = LogisticRegression(max_iter=1000, random_state=random_state)
        
        # Fairness-aware model
        mitigator = GridSearch(
            base_model,
            constraints=constraint,
            grid_size=grid_size
        )
        
        # Train with sensitive features
        mitigator.fit(
            X_train,
            y_train,
            sensitive_features=sensitive_features[sensitive_attr]
        )
        
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
        sensitive_attr: str = 'sex',
        constraint_type: str = 'equalized_odds',
        objective: str = 'balanced_accuracy_score'
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
            estimator=base_model,
            constraints=constraint_type,
            objective=objective,
            prefit=True
        )
        
        # Fit optimizer on training data
        postprocessor.fit(
            X_train,
            y_train,
            sensitive_features=sensitive_features[sensitive_attr]
        )
        
        logger.info("  Threshold optimization complete")
        
        return postprocessor


class MitigationEngine:
    """Unified interface for applying fairness mitigation techniques.
    
    Routes technique requests to appropriate pre/in/post-processing methods
    and handles model training with mitigation.
    """
    
    # Valid stages and constraints
    VALID_STAGES = ['pre-processing', 'in-processing', 'post-processing']
    VALID_PREPROCESSING = ['reweighting', 'smote', 'ros', 'rus', 'adasyn']
    VALID_INPROCESSING = ['exponentiated_gradient', 'grid_search']
    VALID_POSTPROCESSING = ['threshold_optimizer']
    
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
            accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        )
        
        metrics = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred, zero_division=0)),
            'recall': float(recall_score(y_test, y_pred, zero_division=0)),
            'f1_score': float(f1_score(y_test, y_pred, zero_division=0)),
        }
        
        # Add AUC-ROC if probabilities available
        if y_proba is not None and len(np.unique(y_proba)) > 1:
            try:
                metrics['auc_roc'] = float(roc_auc_score(y_test, y_proba))
            except ValueError:
                logger.warning("Could not compute AUC-ROC, setting to 0.0")
                metrics['auc_roc'] = 0.0
        else:
            metrics['auc_roc'] = 0.0
        
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
        sensitive_attr: str = 'sex',
        base_model = None,
        **kwargs
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
        
        if stage == 'pre-processing':
            return self._apply_preprocessing(
                technique_name, X_train, y_train, X_test, y_test,
                sensitive_train, sensitive_test, sensitive_attr, **kwargs
            )
        elif stage == 'in-processing':
            return self._apply_inprocessing(
                technique_name, X_train, y_train, X_test, y_test,
                sensitive_train, sensitive_test, sensitive_attr, **kwargs
            )
        elif stage == 'post-processing':
            if base_model is None:
                raise ValueError("base_model required for post-processing techniques")
            return self._apply_postprocessing(
                technique_name, base_model, X_train, y_train, X_test, y_test,
                sensitive_train, sensitive_test, sensitive_attr, **kwargs
            )
        else:
            raise ValueError(f"Unknown stage: {stage}")
    
    def _apply_preprocessing(
        self, technique_name, X_train, y_train, X_test, y_test,
        sensitive_train, sensitive_test, sensitive_attr, **kwargs
    ) -> Dict:
        """Apply pre-processing technique and train model."""
        # Apply resampling/reweighting
        if technique_name == 'reweighting':
            sample_weights = self.preprocessing.apply_reweighting(
                X_train, y_train, sensitive_train, sensitive_attr
            )
            # Train with sample weights
            model = BaselineLogisticRegression(random_state=self.random_state)
            model.model.fit(X_train, y_train, sample_weight=sample_weights)
            X_train_processed, y_train_processed = X_train, y_train
            
        elif technique_name == 'smote':
            X_train_processed, y_train_processed = self.preprocessing.apply_smote(
                X_train, y_train, random_state=self.random_state
            )
            model = BaselineLogisticRegression(random_state=self.random_state)
            model.train(X_train_processed, y_train_processed)
            
        elif technique_name == 'ros':
            X_train_processed, y_train_processed = self.preprocessing.apply_random_oversampling(
                X_train, y_train, random_state=self.random_state
            )
            model = BaselineLogisticRegression(random_state=self.random_state)
            model.train(X_train_processed, y_train_processed)
            
        elif technique_name == 'rus':
            X_train_processed, y_train_processed = self.preprocessing.apply_random_undersampling(
                X_train, y_train, random_state=self.random_state
            )
            model = BaselineLogisticRegression(random_state=self.random_state)
            model.train(X_train_processed, y_train_processed)
            
        elif technique_name == 'adasyn':
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
        y_proba_raw = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
        if y_proba_raw is None:
            y_proba = None
        elif hasattr(y_proba_raw, 'ndim') and y_proba_raw.ndim > 1:
            y_proba = y_proba_raw[:, 1]
        else:
            y_proba = y_proba_raw
        
        return {
            'model': model,
            'test_metrics': test_metrics,
            'predictions': {'y_pred': y_pred, 'y_proba': y_proba},
            'metadata': {
                'technique': technique_name,
                'stage': 'pre-processing',
                'samples_before': len(X_train),
                'samples_after': len(X_train_processed)
            }
        }
    
    def _apply_inprocessing(
        self, technique_name, X_train, y_train, X_test, y_test,
        sensitive_train, sensitive_test, sensitive_attr, **kwargs
    ) -> Dict:
        """Apply in-processing technique."""
        if technique_name == 'exponentiated_gradient':
            model = self.inprocessing.apply_exponentiated_gradient(
                X_train, y_train, sensitive_train, sensitive_attr,
                random_state=self.random_state, **kwargs
            )
        elif technique_name == 'grid_search':
            model = self.inprocessing.apply_grid_search(
                X_train, y_train, sensitive_train, sensitive_attr,
                random_state=self.random_state, **kwargs
            )
        else:
            raise ValueError(f"Unknown in-processing technique: {technique_name}")
        
        # Predict on test set
        y_pred = model.predict(X_test)
        
        # Get probabilities (Fairlearn models may have multiple predictors)
        y_proba = None
        try:
            if hasattr(model, 'predictors_') and len(model.predictors_) > 0:
                predictor = model.predictors_[0]
                if hasattr(predictor, 'predict_proba'):
                    y_proba = predictor.predict_proba(X_test)[:, 1]
            elif hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)[:, 1]
        except Exception as e:
            logger.warning(f"Could not extract probabilities: {e}")
            y_proba = None
        
        # Calculate metrics using helper method
        test_metrics = self._compute_metrics(y_test, y_pred, y_proba)
        
        return {
            'model': model,
            'test_metrics': test_metrics,
            'predictions': {'y_pred': y_pred, 'y_proba': y_proba},
            'metadata': {
                'technique': technique_name,
                'stage': 'in-processing',
                'n_predictors': len(model.predictors_) if hasattr(model, 'predictors_') else 1
            }
        }
    
    def _apply_postprocessing(
        self, technique_name, base_model, X_train, y_train, X_test, y_test,
        sensitive_train, sensitive_test, sensitive_attr, **kwargs
    ) -> Dict:
        """Apply post-processing technique."""
        if technique_name == 'threshold_optimizer':
            postprocessor = self.postprocessing.apply_threshold_optimizer(
                base_model.model, X_train, y_train, sensitive_train, sensitive_attr, **kwargs
            )
        else:
            raise ValueError(f"Unknown post-processing technique: {technique_name}")
        
        # Predict on test set
        y_pred = postprocessor.predict(X_test, sensitive_features=sensitive_test[sensitive_attr])
        
        # Get probabilities from base model
        y_proba = None
        if hasattr(base_model, 'model') and hasattr(base_model.model, 'predict_proba'):
            y_proba = base_model.model.predict_proba(X_test)[:, 1]
        elif hasattr(base_model, 'predict_proba'):
            y_proba = base_model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics using helper method
        test_metrics = self._compute_metrics(y_test, y_pred, y_proba)
        
        return {
            'model': postprocessor,
            'test_metrics': test_metrics,
            'predictions': {'y_pred': y_pred, 'y_proba': y_proba},
            'metadata': {
                'technique': technique_name,
                'stage': 'post-processing',
                'base_model': type(base_model).__name__
            }
        }
