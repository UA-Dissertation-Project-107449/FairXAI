"""Cross-validation trainer with stratification by sensitive attributes."""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from typing import Dict, List, Tuple, Optional
import logging


class CVTrainer:
    """
    Cross-validation trainer that maintains stratification by target and sensitive attributes.
    """
    
    def __init__(self, n_folds: int = 5, random_state: int = 42):
        """
        Initialize CV trainer.
        
        Args:
            n_folds: Number of cross-validation folds
            random_state: Random seed for reproducibility
        """
        self.n_folds = n_folds
        self.random_state = random_state
        self.logger = logging.getLogger(__name__)
    
    def create_stratified_folds(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sensitive_attrs: pd.DataFrame
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create stratified folds based on target and sensitive attributes.
        
        Args:
            X: Features
            y: Target variable
            sensitive_attrs: Sensitive attributes (age_group, sex)
            
        Returns:
            List of (train_idx, val_idx) tuples for each fold
        """
        # Create stratification key combining target and sensitive attributes
        strat_cols = ['target']
        for col in ['age_group', 'sex', 'ethnicity', 'group_cluster']:
            if col in sensitive_attrs.columns:
                strat_cols.append(col)

        strat_df = pd.DataFrame({'target': y.values})
        for col in strat_cols:
            if col == 'target':
                continue
            strat_df[col] = sensitive_attrs[col].values

        # Fallback to target-only strat if no sensitive columns are present
        strat_key = strat_df.astype(str).agg('_'.join, axis=1)
        
        # Create folds
        skf = StratifiedKFold(
            n_splits=self.n_folds,
            shuffle=True,
            random_state=self.random_state
        )
        
        folds = list(skf.split(X, strat_key))
        
        self.logger.info(f"Created {self.n_folds} stratified folds")
        self.logger.info(f"  Total samples: {len(X)}")
        for i, (train_idx, val_idx) in enumerate(folds):
            self.logger.info(f"  Fold {i}: train={len(train_idx)}, val={len(val_idx)}")
        
        return folds
    
    def train_fold(
        self,
        model,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series
    ) -> Dict:
        """
        Train model on one fold.
        
        Args:
            model: Model instance with train() and evaluate() methods
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            
        Returns:
            Dictionary with train and validation metrics
        """
        # Train model
        train_metrics = model.train(X_train, y_train)
        
        # Evaluate on validation set
        val_metrics = model.evaluate(X_val, y_val)
        
        return {
            'train_metrics': train_metrics,
            'val_metrics': val_metrics
        }
    
    def run_cv_experiment(
        self,
        model_class,
        X: pd.DataFrame,
        y: pd.Series,
        sensitive_attrs: pd.DataFrame,
        model_params: Optional[Dict] = None
    ) -> Dict:
        """
        Run full cross-validation experiment.
        
        Args:
            model_class: Model class to instantiate
            X: Full feature set
            y: Full target variable
            sensitive_attrs: Sensitive attributes for stratification
            model_params: Parameters to pass to model constructor
            
        Returns:
            Dictionary with aggregated CV results
        """
        if model_params is None:
            model_params = {}
        
        # Create folds
        folds = self.create_stratified_folds(X, y, sensitive_attrs)
        
        # Store results for each fold
        fold_results = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            self.logger.info(f"\nTraining fold {fold_idx + 1}/{self.n_folds}...")
            
            # Split data
            X_train = X.iloc[train_idx]
            y_train = y.iloc[train_idx]
            X_val = X.iloc[val_idx]
            y_val = y.iloc[val_idx]
            
            # Create fresh model instance for this fold
            model = model_class(**model_params)
            
            # Train and evaluate
            fold_result = self.train_fold(model, X_train, y_train, X_val, y_val)
            fold_result['fold_idx'] = fold_idx
            fold_result['train_indices'] = train_idx
            fold_result['val_indices'] = val_idx
            
            fold_results.append(fold_result)
            
            # Log fold performance
            self.logger.info(f"  Fold {fold_idx} val metrics:")
            self.logger.info(f"    Accuracy: {fold_result['val_metrics']['accuracy']:.3f}")
            self.logger.info(f"    Recall: {fold_result['val_metrics']['recall']:.3f}")
            self.logger.info(f"    F1: {fold_result['val_metrics']['f1_score']:.3f}")
        
        # Aggregate results
        aggregated = self.aggregate_fold_results(fold_results)
        
        return {
            'fold_results': fold_results,
            'aggregated_metrics': aggregated,
            'n_folds': self.n_folds,
            'random_state': self.random_state
        }
    
    def aggregate_fold_results(self, fold_results: List[Dict]) -> Dict:
        """
        Aggregate metrics across folds.
        
        Args:
            fold_results: List of fold result dictionaries
            
        Returns:
            Dictionary with mean, std, and per-fold metrics
        """
        # Extract validation metrics from each fold
        metrics_names = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']
        
        aggregated = {}
        for metric_name in metrics_names:
            values = [fold['val_metrics'][metric_name] for fold in fold_results]
            aggregated[metric_name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'folds': values
            }
        
        # Log aggregated results
        self.logger.info(f"\n{'='*60}")
        self.logger.info("Cross-Validation Results (mean ± std):")
        self.logger.info(f"{'='*60}")
        for metric_name, stats in aggregated.items():
            self.logger.info(
                f"  {metric_name:12s}: {stats['mean']:.3f} ± {stats['std']:.3f} "
                f"(min: {stats['min']:.3f}, max: {stats['max']:.3f})"
            )
        
        return aggregated
    
    def get_fold_predictions(
        self,
        model,
        X: pd.DataFrame,
        y: pd.Series,
        sensitive_attrs: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Generate predictions for all samples using cross-validation.
        
        Each sample is predicted by a model that was NOT trained on it.
        
        Args:
            model_class: Model class to instantiate
            X: Full feature set
            y: Full target variable
            sensitive_attrs: Sensitive attributes
            
        Returns:
            DataFrame with predictions and metadata for all samples
        """
        folds = self.create_stratified_folds(X, y, sensitive_attrs)
        
        all_predictions = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            # Split data
            X_train = X.iloc[train_idx]
            y_train = y.iloc[train_idx]
            X_val = X.iloc[val_idx]
            y_val = y.iloc[val_idx]
            
            # Train model on fold
            model.train(X_train, y_train)
            
            # Predict on validation set
            y_pred = model.predict(X_val)
            y_proba = model.predict_proba(X_val)
            
            # Create predictions dataframe
            fold_preds = pd.DataFrame({
                'fold': fold_idx,
                'sample_idx': val_idx,
                'y_true': y_val.values,
                'y_pred': y_pred,
                'y_proba': y_proba,
            })

            for col in ['age_group', 'sex', 'ethnicity', 'group_cluster']:
                if col in sensitive_attrs.columns:
                    fold_preds[col] = sensitive_attrs.iloc[val_idx][col].values
            
            all_predictions.append(fold_preds)
        
        # Concatenate all fold predictions
        full_predictions = pd.concat(all_predictions, ignore_index=True)
        full_predictions = full_predictions.sort_values('sample_idx').reset_index(drop=True)
        
        return full_predictions
