"""Cross-validation trainer with stratification by sensitive attributes."""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from typing import Any, Dict, List, Tuple, Optional
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
        self._last_effective_folds = n_folds

    @staticmethod
    def _coerce_probability_vector(values: Any) -> np.ndarray:
        """Convert probability outputs into a single positive-class score vector."""
        arr = np.asarray(values)
        if arr.ndim == 0:
            return np.asarray([float(arr)])
        if arr.ndim == 1:
            return arr.reshape(-1)
        if arr.ndim == 2:
            if arr.shape[1] == 0:
                return np.zeros(arr.shape[0], dtype=float)
            if arr.shape[1] == 1:
                return arr[:, 0]
            return arr[:, 1]
        return arr.reshape(arr.shape[0], -1)[:, 0]

    @staticmethod
    def _coerce_label_vector(values: Any) -> np.ndarray:
        """Convert prediction outputs into a 1D label vector."""
        arr = np.asarray(values)
        if arr.ndim <= 1:
            return arr.reshape(-1)
        return np.argmax(arr, axis=1)
    
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

        min_group_size = int(strat_key.value_counts().min())
        if min_group_size < 2:
            # Intersectional strata can become too sparse on fine-grained bins;
            # fall back to target-only stratification for CV feasibility.
            self.logger.warning(
                "Combined sensitive stratification too sparse (min_group_size=%s). "
                "Falling back to target-only stratification.",
                min_group_size,
            )
            strat_key = y.astype(str)
            min_group_size = int(strat_key.value_counts().min())
        effective_folds = min(self.n_folds, min_group_size)
        if effective_folds < 2:
            raise ValueError(
                "Not enough samples per stratification group for CV: "
                f"min_group_size={min_group_size}, requested_folds={self.n_folds}."
            )
        if effective_folds != self.n_folds:
            self.logger.warning(
                "Reducing CV folds from %s to %s due to small stratified groups "
                "(min_group_size=%s).",
                self.n_folds,
                effective_folds,
                min_group_size,
            )
        self._last_effective_folds = effective_folds
        
        # Create folds
        skf = StratifiedKFold(
            n_splits=effective_folds,
            shuffle=True,
            random_state=self.random_state
        )
        
        folds = list(skf.split(X, strat_key))
        
        self.logger.info(f"Created {effective_folds} stratified folds")
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
        model_params: Optional[Dict] = None,
        xai_enabled: bool = False,
        shap_enabled: bool = True,
        allow_svm_shap: bool = False,
        tracked_indices: Optional[List[int]] = None,
        feature_names: Optional[List[str]] = None,
        shap_max_samples: int = 1000,
    ) -> Dict:
        """
        Run full cross-validation experiment.
        
        Args:
            model_class: Model class to instantiate
            X: Full feature set
            y: Full target variable
            sensitive_attrs: Sensitive attributes for stratification
            model_params: Parameters to pass to model constructor
            xai_enabled: If True, run SHAP/LIME per fold and store results.
            shap_enabled: If False, only LIME is computed in XAI stage.
            allow_svm_shap: If True, allows SHAP calls for SVM models.
            tracked_indices: Row indices (into *X*) for LIME stability tracking.
                Each tracked instance is LIME-explained in the fold where it
                lands in the validation set.
            feature_names: Feature names for XAI output (defaults to X.columns).
            shap_max_samples: Cap for SHAP background / computation samples.
            
        Returns:
            Dictionary with aggregated CV results (and XAI data when enabled).

        Note:
            Trained models are used for XAI computation inside the fold loop
            and then discarded — they are not persisted. This is deliberate:
            storing *n_folds* model objects is unnecessary since the
            SHAP / LIME values are the desired output, not the models
            themselves.
        """
        if model_params is None:
            model_params = {}
        if feature_names is None:
            feature_names = list(X.columns)
        
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

            # --- XAI (while model is still alive) ---
            if xai_enabled:
                fold_result['xai'] = self._run_xai_for_fold(
                    model=model,
                    X_train=X_train,
                    X_val=X_val,
                    feature_names=feature_names,
                    tracked_indices=tracked_indices,
                    val_indices=val_idx,
                    fold_idx=fold_idx,
                    max_samples=shap_max_samples,
                    shap_enabled=shap_enabled,
                    allow_svm_shap=allow_svm_shap,
                )
            
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
            'n_folds': self._last_effective_folds,
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

    # ------------------------------------------------------------------
    # XAI helpers
    # ------------------------------------------------------------------

    def _run_xai_for_fold(
        self,
        model: Any,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        feature_names: List[str],
        tracked_indices: Optional[List[int]],
        val_indices: np.ndarray,
        fold_idx: int,
        max_samples: int = 1000,
        shap_enabled: bool = True,
        allow_svm_shap: bool = False,
    ) -> Dict:
        """Run SHAP (on train) and LIME (on tracked val instances) for one fold.

        Args:
            model: Trained wrapper whose ``.model`` attribute is the sklearn
                estimator (e.g. ``BaselineLogisticRegression``).
            X_train: Training features for this fold.
            X_val: Validation features for this fold.
            feature_names: Column names for XAI output.
            tracked_indices: Global row indices to LIME-explain when they
                appear in *val_indices*.
            val_indices: Array of global row indices in this fold's val set.
            fold_idx: Current fold number.
            max_samples: SHAP subsample cap.

        Returns:
            ``{'shap_values': np.ndarray, 'feature_names': list,
            'lime_results': list[dict]}``
        """
        from fairxai.explainability.tabular import (
            shap_explain_tabular,
            lime_explain_instance,
        )

        xai_result: Dict = {
            'shap_values': None,
            'shap_values_local': None,
            'feature_names': feature_names,
            'lime_results': [],
        }

        inner_model = getattr(model, 'model', model)

        if shap_enabled:
            # --- SHAP on training data (global) ---
            try:
                shap_exp = shap_explain_tabular(
                    inner_model, X_train,
                    feature_names=feature_names,
                    max_samples=max_samples,
                    allow_svm=allow_svm_shap,
                )
                xai_result['shap_values'] = shap_exp.shap_values
                self.logger.info(
                    f"  Fold {fold_idx}: SHAP global computed "
                    f"({shap_exp.shap_values.shape[0]} samples)"
                )
            except Exception as exc:
                self.logger.warning(f"  Fold {fold_idx}: SHAP global failed — {exc}")

            # --- SHAP on validation data (local) ---
            try:
                shap_local = shap_explain_tabular(
                    inner_model, X_val,
                    feature_names=feature_names,
                    max_samples=max_samples,
                    allow_svm=allow_svm_shap,
                )
                xai_result['shap_values_local'] = shap_local.shap_values
                self.logger.info(
                    f"  Fold {fold_idx}: SHAP local computed "
                    f"({shap_local.shap_values.shape[0]} samples)"
                )
            except Exception as exc:
                self.logger.warning(f"  Fold {fold_idx}: SHAP local failed — {exc}")
        else:
            self.logger.info(f"  Fold {fold_idx}: SHAP skipped by configuration")

        # --- LIME on tracked instances that landed in this val set ---
        if tracked_indices is not None:
            val_set = set(val_indices.tolist())
            targets_in_fold = [idx for idx in tracked_indices if idx in val_set]

            inner_model = getattr(model, 'model', model)
            for idx in targets_in_fold:
                try:
                    # Locate the row in X_val by its global index
                    row = X_val.loc[idx] if idx in X_val.index else X_val.iloc[
                        list(val_indices).index(idx)
                    ]
                    exp = lime_explain_instance(
                        model=inner_model,
                        data_row=row,
                        training_data=X_train,
                        feature_names=feature_names,
                        class_names=['no_disease', 'disease'],
                        num_features=10,
                    )
                    for feat, weight in exp.weights:
                        xai_result['lime_results'].append({
                            'instance_id': int(idx),
                            'fold_idx': fold_idx,
                            'feature': feat,
                            'weight': weight,
                            'intercept': exp.intercept,
                            'score': exp.score,
                            'local_pred': exp.local_pred,
                        })
                    self.logger.info(
                        f"  Fold {fold_idx}: LIME explained instance {idx}"
                    )
                except Exception as exc:
                    self.logger.warning(
                        f"  Fold {fold_idx}: LIME failed for instance {idx} — {exc}"
                    )

        return xai_result

    # ------------------------------------------------------------------

    @staticmethod
    def aggregate_cv_shap(
        fold_results: List[Dict],
        scope: str = 'global',
    ) -> Optional[pd.DataFrame]:
        """Aggregate per-fold SHAP values into a summary DataFrame.

        Stacks the absolute SHAP value matrices from every fold and computes
        per-feature statistics: ``mean_abs_shap``, ``std_abs_shap``,
        ``p25``, ``p50``, ``p75``.

        Args:
            fold_results: List of fold dicts, each containing an ``xai`` key
                with ``shap_values`` / ``shap_values_local`` and
                ``feature_names``.
            scope: ``'global'`` aggregates ``shap_values`` (training data),
                ``'local'`` aggregates ``shap_values_local`` (validation data).

        Returns:
            DataFrame ``[feature, mean_abs_shap, std_abs_shap, p25, p50, p75]``
            sorted by ``mean_abs_shap`` descending, or *None* if no SHAP data.
        """
        key = 'shap_values' if scope == 'global' else 'shap_values_local'
        arrays = []
        feature_names = None
        for fr in fold_results:
            xai = fr.get('xai')
            if xai is None:
                continue
            sv = xai.get(key)
            if sv is not None:
                arrays.append(np.abs(sv))
                if feature_names is None:
                    feature_names = xai.get('feature_names')

        if not arrays or feature_names is None:
            return None

        stacked = np.vstack(arrays)
        summary = pd.DataFrame({
            'feature': feature_names,
            'mean_abs_shap': np.mean(stacked, axis=0),
            'std_abs_shap': np.std(stacked, axis=0),
            'p25': np.percentile(stacked, 25, axis=0),
            'p50': np.percentile(stacked, 50, axis=0),
            'p75': np.percentile(stacked, 75, axis=0),
        })
        return summary.sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)

    @staticmethod
    def aggregate_cv_lime(fold_results: List[Dict]) -> Optional[pd.DataFrame]:
        """Collect per-fold LIME explanations for tracked instances.

        Each tracked instance appears in exactly one validation fold and thus
        receives one LIME explanation.  The returned DataFrame is a raw record
        table — no cross-instance aggregation — so downstream code can compare
        feature contributions for near-threshold patients.

        Args:
            fold_results: List of fold dicts, each containing an ``xai`` key
                with ``lime_results`` (list of dicts).

        Returns:
            DataFrame ``[instance_id, fold_idx, feature, weight, intercept,
            score, local_pred]`` or *None* if no LIME data.
        """
        rows: List[Dict] = []
        for fr in fold_results:
            xai = fr.get('xai')
            if xai is None:
                continue
            rows.extend(xai.get('lime_results', []))

        if not rows:
            return None
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    
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
            y_pred = self._coerce_label_vector(model.predict(X_val))
            y_proba = self._coerce_probability_vector(model.predict_proba(X_val))
            
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
