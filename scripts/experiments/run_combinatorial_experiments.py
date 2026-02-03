"""Combinatorial experiment orchestration for fairness mitigation analysis."""

import argparse
import logging
import sys
import os
import json
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from joblib import Parallel, delayed

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from fairxai.models.baseline import BaselineLogisticRegression
from fairxai.models.cv_trainer import CVTrainer
from fairxai.fairness.metrics import FairnessMetrics
from fairxai.fairness.mitigation import MitigationEngine
from fairxai.experiments.versioning import ExperimentVersioning
from fairxai.data.schemas import available_sensitive, preferred_sensitive
from fairxai.utils.config import load_yaml_config
from fairxai.experiments.data_io import load_schema_config as load_schema_config_shared, build_schema_excludes, default_exclude_columns
from fairxai.explainability.tabular import shap_explain_tabular, lime_explain_instance
from fairxai.cli.runner_base import get_project_root, setup_phase_logging
from fairxai.cli.runner_utils import (
    append_run_history,
    get_run_root,
    resolve_run_id,
    update_latest_pointer,
)


STAGE_MAP = {
    'reweighting': 'pre-processing',
    'smote': 'pre-processing',
    'ros': 'pre-processing',
    'rus': 'pre-processing',
    'adasyn': 'pre-processing',
    'exponentiated_gradient': 'in-processing',
    'grid_search': 'in-processing',
    'threshold_optimizer': 'post-processing',
}


def load_processed_data(
    dataset_name: str,
    binning_strategy: str,
    processed_dir: Path
) -> Dict[str, pd.DataFrame]:
    """
    Load preprocessed data for specific dataset and binning strategy.
    
    Args:
        dataset_name: Dataset name (cleveland, kaggle_heart)
        binning_strategy: Binning strategy name
        processed_dir: Path to processed data directory
        
    Returns:
        Dictionary with train/test data
    """
    data_dir = processed_dir / f"{dataset_name}_{binning_strategy}"
    
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Processed data not found: {data_dir}\n"
            f"Run preprocessing with --binning-strategy {binning_strategy}"
        )
    
    # Load scaled data with sensitive attributes
    train_df = pd.read_csv(data_dir / f"{dataset_name}_train_scaled.csv")
    test_df = pd.read_csv(data_dir / f"{dataset_name}_test_scaled.csv")
    
    return {
        'train_df': train_df,
        'test_df': test_df
    }


def load_schema_config(project_root: Path, pipeline: str) -> Dict[str, Any]:
    return load_schema_config_shared(project_root, pipeline)


def schema_excludes(schema_cfg: Dict[str, Any], dataset_name: str) -> List[str]:
    return build_schema_excludes(schema_cfg, dataset_name)


def prepare_data_splits(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    exclude_cols: List[str],
    sensitive_attrs: List[str],
    target_col: str
) -> Dict[str, Any]:
    """
    Prepare train/test splits with feature/target separation.
    
    Args:
        train_df: Training dataframe
        test_df: Test dataframe
        exclude_cols: Columns to exclude from features
        
    Returns:
        Dictionary with X, y, sensitive attributes for train and test
    """
    # Filter exclude columns that actually exist
    exclude_cols = [col for col in exclude_cols if col in train_df.columns]
    
    # Separate features, target, and sensitive/group attributes
    X_train = train_df.drop(columns=exclude_cols)
    y_train = train_df[target_col]
    X_test = test_df.drop(columns=exclude_cols)
    y_test = test_df[target_col]

    sens_cols_train = available_sensitive(train_df, sensitive_attrs)
    sens_cols_test = available_sensitive(test_df, sensitive_attrs)

    sensitive_train = train_df[sens_cols_train].copy() if sens_cols_train else pd.DataFrame(index=train_df.index)
    sensitive_test = test_df[sens_cols_test].copy() if sens_cols_test else pd.DataFrame(index=test_df.index)
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'sensitive_train': sensitive_train,
        'X_test': X_test,
        'y_test': y_test,
        'sensitive_test': sensitive_test,
        'sensitive_cols': list(dict.fromkeys(sens_cols_train + sens_cols_test))
    }


def save_experiment_xai(
    exp_id: str,
    model: Any,
    X_ref: pd.DataFrame,
    X_lime: pd.DataFrame,
    versioning: ExperimentVersioning,
    dataset_name: str,
    base_model: Optional[Any] = None
) -> None:
    xai_enabled = os.getenv('XAI_ENABLED', 'true').lower() == 'true'
    if not xai_enabled:
        return

    dataset_xai_dir = versioning.latest_dir / 'xai' / dataset_name
    shap_dir = dataset_xai_dir / 'shap'
    lime_dir = dataset_xai_dir / 'lime'
    shap_dir.mkdir(parents=True, exist_ok=True)
    lime_dir.mkdir(parents=True, exist_ok=True)
    max_samples = int(os.getenv('XAI_MAX_SAMPLES', '200'))
    lime_instances = int(os.getenv('XAI_LIME_INSTANCES', '2'))

    def _mean_abs_shap_values(shap_values: Any) -> np.ndarray:
        def _reduce(arr: np.ndarray) -> np.ndarray:
            if arr.ndim == 3:
                return np.mean(np.abs(arr), axis=(0, 2))
            if arr.ndim == 2:
                return np.mean(np.abs(arr), axis=0)
            if arr.ndim == 1:
                return np.abs(arr)
            return np.mean(np.abs(arr), axis=0)

        if isinstance(shap_values, list):
            values = [_reduce(np.asarray(v)) for v in shap_values]
            return np.mean(values, axis=0)

        return _reduce(np.asarray(shap_values))

    def _resolve_shap_model(raw_model: Any) -> Any:
        candidate = raw_model
        if hasattr(raw_model, 'predictors_'):
            predictor = next((p for p in raw_model.predictors_ if hasattr(p, 'predict_proba') or hasattr(p, 'predict')), None)
            if predictor is not None:
                candidate = predictor
        if hasattr(raw_model, 'model'):
            candidate = raw_model.model

        if hasattr(candidate, 'predict_proba'):
            return lambda X: candidate.predict_proba(X)
        if hasattr(candidate, 'predict'):
            return lambda X: candidate.predict(X)
        return candidate

    def _wrap_decision_function(df_model: Any):
        class _DecisionFunctionWrapper:
            def __init__(self, base_model: Any):
                self.base_model = base_model

            def predict_proba(self, X):
                scores = self.base_model.decision_function(X)
                scores = np.asarray(scores)
                if scores.ndim == 1:
                    prob_pos = 1.0 / (1.0 + np.exp(-scores))
                    return np.vstack([1 - prob_pos, prob_pos]).T
                exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
                return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        return _DecisionFunctionWrapper(df_model)

    def _resolve_lime_model(raw_model: Any) -> Optional[Any]:
        if hasattr(raw_model, 'predict_proba'):
            return raw_model
        if hasattr(raw_model, 'predictors_'):
            predictor = next((p for p in raw_model.predictors_ if hasattr(p, 'predict_proba') or hasattr(p, 'decision_function')), None)
            if predictor is not None:
                return predictor
        if hasattr(raw_model, 'model') and hasattr(raw_model.model, 'predict_proba'):
            return raw_model.model
        if hasattr(raw_model, 'decision_function'):
            return _wrap_decision_function(raw_model)
        if hasattr(raw_model, 'model') and hasattr(raw_model.model, 'decision_function'):
            return _wrap_decision_function(raw_model.model)
        return None

    try:
        xai_model = base_model if base_model is not None else model
        shap_model = _resolve_shap_model(xai_model)
        # Global SHAP (train reference)
        shap_global = shap_explain_tabular(shap_model, X_ref, max_samples=max_samples)
        mean_abs_global = _mean_abs_shap_values(shap_global.shap_values)
        shap_global_df = pd.DataFrame({
            'feature': shap_global.feature_names,
            'mean_abs_shap': mean_abs_global
        }).sort_values('mean_abs_shap', ascending=False)
        shap_global_file = shap_dir / f"{exp_id}_shap_global.csv"
        shap_global_df.to_csv(shap_global_file, index=False)

        # Local SHAP (test/reference for local context)
        shap_local = shap_explain_tabular(shap_model, X_lime, max_samples=max_samples)
        mean_abs_local = _mean_abs_shap_values(shap_local.shap_values)
        shap_local_df = pd.DataFrame({
            'feature': shap_local.feature_names,
            'mean_abs_shap': mean_abs_local
        }).sort_values('mean_abs_shap', ascending=False)
        shap_local_file = shap_dir / f"{exp_id}_shap_local.csv"
        shap_local_df.to_csv(shap_local_file, index=False)
    except Exception as exc:
        logging.getLogger(__name__).warning(f"SHAP failed for {exp_id}: {exc}")

    # LIME examples (requires predict_proba)
    try:
        lime_model = _resolve_lime_model(xai_model)
        if lime_instances > 0 and lime_model is not None:
            lime_rows = X_lime.sample(n=min(lime_instances, len(X_lime)), random_state=42)
            lime_results = []
            for idx, row in lime_rows.iterrows():
                exp = lime_explain_instance(
                    model=lime_model,
                    data_row=row,
                    training_data=X_ref,
                    feature_names=list(X_ref.columns),
                    class_names=["no_disease", "disease"],
                    num_features=10
                )
                for feat, weight in exp.weights:
                    lime_results.append({
                        'instance_id': int(idx),
                        'feature': feat,
                        'weight': weight,
                        'intercept': exp.intercept,
                        'score': exp.score,
                        'local_pred': exp.local_pred
                    })
            lime_df = pd.DataFrame(lime_results)
            lime_file = lime_dir / f"{exp_id}_lime_examples.csv"
            lime_df.to_csv(lime_file, index=False)
        elif lime_instances > 0:
            logging.getLogger(__name__).warning(f"LIME skipped for {exp_id}: no predict_proba/decision_function")
    except Exception as exc:
        logging.getLogger(__name__).warning(f"LIME failed for {exp_id}: {exc}")


def aggregate_dataset_shap(xai_dir: Path, suffix: str) -> None:
    files = [
        p for p in xai_dir.glob(f"*_shap_{suffix}.csv")
        if not p.name.endswith(f"shap_{suffix}_summary.csv")
    ]
    if not files:
        return

    rows = []
    for file_path in files:
        df = pd.read_csv(file_path)
        if 'feature' not in df.columns or 'mean_abs_shap' not in df.columns:
            continue
        df = df[['feature', 'mean_abs_shap']].copy()
        df['source_file'] = file_path.name
        rows.append(df)

    if not rows:
        return

    combined = pd.concat(rows, ignore_index=True)
    grouped = combined.groupby('feature')['mean_abs_shap']
    summary = grouped.agg(
        count='count',
        mean='mean',
        std='std',
        min='min',
        max='max',
    ).reset_index()
    summary['p25'] = grouped.quantile(0.25).values
    summary['p50'] = grouped.quantile(0.50).values
    summary['p75'] = grouped.quantile(0.75).values
    summary = summary.sort_values('mean', ascending=False)

    out_file = xai_dir / f"shap_{suffix}_summary.csv"
    summary.to_csv(out_file, index=False)


def run_single_experiment(
    exp_id: str,
    config: Dict[str, Any],
    versioning: ExperimentVersioning,
    processed_dir: Path,
    schema_cfg: Dict[str, Any],
    logger: logging.Logger,
    target_col: str
) -> Dict[str, Any]:
    """
    Run a single experiment with given configuration.
    
    Args:
        exp_id: Experiment ID
        config: Experiment configuration
        versioning: Versioning system instance
        processed_dir: Path to processed data
        logger: Logger instance
        
    Returns:
        Dictionary with experiment results
    """
    start_time = datetime.now()
    
    try:
        logger.info(f"\n{'='*80}")
        logger.info(f"Experiment {exp_id}")
        logger.info(f"{'='*80}")
        logger.info(f"Dataset: {config['dataset']}")
        logger.info(f"Binning: {config['binning_strategy']}")
        logger.info(f"Mitigation: {config['mitigation_technique']}")
        logger.info(f"Training: {config['training_method']}")
        
        # Load data
        data = load_processed_data(
            config['dataset'],
            config['binning_strategy'],
            processed_dir
        )
        
        # Prepare splits
        exclude_cols = default_exclude_columns(
            schema_cfg,
            config['dataset'],
            target=target_col,
            sensitive_attrs=config.get('sensitive_attributes', [])
        )
        splits = prepare_data_splits(
            data['train_df'],
            data['test_df'],
            exclude_cols,
            config['sensitive_attributes'],
            target_col
        )

        if config['mitigation_technique'] != 'baseline' and not splits['sensitive_cols']:
            raise ValueError("No sensitive/group columns available for mitigation; check preprocessing and config")
        
        logger.info(f"Data loaded: train={len(splits['X_train'])}, test={len(splits['X_test'])}")
        logger.info(f"Features: {splits['X_train'].shape[1]}")
        
        # Train model based on training method
        if config['training_method'] == 'kfold_cv':
            results = run_cv_experiment(exp_id, config, splits, versioning, logger)
        else:
            results = run_single_split_experiment(exp_id, config, splits, versioning, logger)
        
        # Add execution metadata
        duration = (datetime.now() - start_time).total_seconds()
        results['execution'] = {
            'duration_seconds': duration,
            'timestamp': datetime.now().isoformat(),
            'status': 'success'
        }
        
        logger.info(f"[SUCCESS] Experiment {exp_id} completed in {duration:.1f}s")
        return results
        
    except Exception as e:
        logger.error(f"[ERROR] Experiment {exp_id} failed: {str(e)}")
        duration = (datetime.now() - start_time).total_seconds()
        
        return {
            'experiment_id': exp_id,
            'configuration': config,
            'execution': {
                'duration_seconds': duration,
                'timestamp': datetime.now().isoformat(),
                'status': 'failed',
                'error': str(e)
            },
            'results': None
        }


def run_single_split_experiment(
    exp_id: str,
    config: Dict[str, Any],
    splits: Dict[str, Any],
    versioning: ExperimentVersioning,
    logger: logging.Logger
) -> Dict[str, Any]:
    """
    Run experiment with single train/test split.
    """
    logger.info("\nTraining with single train/test split...")
    
    # Initialize mitigation engine
    engine = MitigationEngine()
    
    # Apply mitigation technique
    mitigation = config['mitigation_technique']
    stage = STAGE_MAP.get(mitigation)
    if mitigation == 'baseline':
        stage = 'baseline'
    base_model = None
    sensitive_attr = next(
        (c for c in config['sensitive_attributes'] if c in splits['sensitive_train'].columns and c != 'age_group'),
        next((c for c in splits['sensitive_train'].columns), None)
    )

    if mitigation != 'baseline' and sensitive_attr is None:
        raise ValueError("Mitigation requires at least one sensitive/group column; none found in splits")

    if mitigation == 'baseline':
        # Train baseline model
        model = BaselineLogisticRegression(**config.get('model_params', {}))
        train_metrics = model.train(splits['X_train'], splits['y_train'])
        test_metrics = model.evaluate(splits['X_test'], splits['y_test'])
        
        # Get predictions
        y_pred = model.predict(splits['X_test'])
        y_proba = model.predict_proba(splits['X_test'])
        
        result = {
            'test_metrics': test_metrics,
            'predictions': {'y_pred': y_pred, 'y_proba': y_proba}
        }
    else:
        if stage is None:
            raise ValueError(f"Unknown mitigation technique: {mitigation}")

        if stage == 'post-processing':
            base_model = BaselineLogisticRegression(**config.get('model_params', {}))
            base_model.train(splits['X_train'], splits['y_train'])

        # Apply mitigation technique
        fairness_base_params = config.get('fairness_base_model_params')
        result = engine.apply_technique(
            technique_name=mitigation,
            stage=stage,
            X_train=splits['X_train'],
            y_train=splits['y_train'],
            X_test=splits['X_test'],
            y_test=splits['y_test'],
            sensitive_train=splits['sensitive_train'],
            sensitive_test=splits['sensitive_test'],
            sensitive_attr=sensitive_attr,
            base_model=base_model,
            base_model_params=fairness_base_params
        )
    
    # Calculate fairness metrics
    predictions_df = pd.DataFrame({
        'y_true': splits['y_test'].values,
        'y_pred': result['predictions']['y_pred'],
        'y_proba': result['predictions']['y_proba'],
    })

    for col in splits['sensitive_cols']:
        predictions_df[col] = splits['sensitive_test'][col].values
    
    # Add features for individual fairness
    for col in splits['X_test'].columns:
        predictions_df[col] = splits['X_test'][col].values
    
    fairness_calc = FairnessMetrics(
        available_sensitive(predictions_df, config['sensitive_attributes'])
    )
    fairness_results = fairness_calc.calculate_all_metrics(
        predictions_df,
        feature_cols=list(splits['X_test'].columns)
    )
    
    # Save predictions
    versioning.save_predictions(exp_id, predictions_df, dataset=config['dataset'])

    # XAI outputs (single-split only)
    model_for_xai = result.get('model') if isinstance(result, dict) else None
    if mitigation == 'baseline':
        model_for_xai = model
    if stage == 'post-processing' and base_model is not None:
        model_for_xai = base_model
    if model_for_xai is not None:
        save_experiment_xai(
            exp_id,
            model_for_xai,
            splits['X_train'],
            splits['X_test'],
            versioning,
            config['dataset'],
            base_model=base_model
        )
    
    logger.info(f"  Accuracy: {result['test_metrics']['accuracy']:.3f}")
    logger.info(f"  Recall: {result['test_metrics']['recall']:.3f}")
    logger.info(f"  F1: {result['test_metrics']['f1_score']:.3f}")
    
    return {
        'experiment_id': exp_id,
        'configuration': config,
        'test_metrics': result['test_metrics'],
        'fairness_metrics': fairness_results,
        'training_method': 'single_split',
        'n_folds': 1
    }


def run_cv_experiment(
    exp_id: str,
    config: Dict[str, Any],
    splits: Dict[str, Any],
    versioning: ExperimentVersioning,
    logger: logging.Logger
) -> Dict[str, Any]:
    """
    Run experiment with k-fold cross-validation.
    """
    n_folds = config.get('cv_folds', 5)
    logger.info(f"\nTraining with {n_folds}-fold cross-validation...")
    
    # Combine train and test for CV (we'll use full dataset for CV)
    X_full = pd.concat([splits['X_train'], splits['X_test']], ignore_index=True)
    y_full = pd.concat([splits['y_train'], splits['y_test']], ignore_index=True)
    sensitive_full = pd.concat([splits['sensitive_train'], splits['sensitive_test']], ignore_index=True)
    
    # Initialize CV trainer
    cv_trainer = CVTrainer(n_folds=n_folds, random_state=config.get('random_seed', 42))

    mitigation = config.get('mitigation_technique', 'baseline')
    stage = STAGE_MAP.get(mitigation)

    def _aggregate_fold_metrics(fold_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        metrics_names = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']
        aggregated = {}
        for metric_name in metrics_names:
            values = [fold['val_metrics'][metric_name] for fold in fold_results]
            aggregated[metric_name] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'folds': values
            }
        return aggregated

    if mitigation == 'baseline':
        # Run CV experiment (baseline only)
        cv_results = cv_trainer.run_cv_experiment(
            model_class=BaselineLogisticRegression,
            X=X_full,
            y=y_full,
            sensitive_attrs=sensitive_full,
            model_params=config.get('model_params', {})
        )

        # Get fold predictions for fairness calculation
        model = BaselineLogisticRegression(**config.get('model_params', {}))
        fold_predictions = cv_trainer.get_fold_predictions(model, X_full, y_full, sensitive_full)
    else:
        if stage is None:
            raise ValueError(f"Unknown mitigation technique for CV: {mitigation}")

        engine = MitigationEngine()
        folds = cv_trainer.create_stratified_folds(X_full, y_full, sensitive_full)
        fold_results = []
        all_predictions = []

        sensitive_attr = next(
            (c for c in config['sensitive_attributes'] if c in sensitive_full.columns and c != 'age_group'),
            next((c for c in sensitive_full.columns), None)
        )

        if sensitive_attr is None:
            raise ValueError("Mitigation requires at least one sensitive/group column; none found in CV splits")

        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            logger.info(f"\nMitigation CV fold {fold_idx + 1}/{n_folds}...")

            X_train = X_full.iloc[train_idx]
            y_train = y_full.iloc[train_idx]
            X_val = X_full.iloc[val_idx]
            y_val = y_full.iloc[val_idx]
            sensitive_train = sensitive_full.iloc[train_idx]
            sensitive_val = sensitive_full.iloc[val_idx]

            base_model = None
            if stage == 'post-processing':
                base_model = BaselineLogisticRegression(**config.get('model_params', {}))
                base_model.train(X_train, y_train)

            result = engine.apply_technique(
                technique_name=mitigation,
                stage=stage,
                X_train=X_train,
                y_train=y_train,
                X_test=X_val,
                y_test=y_val,
                sensitive_train=sensitive_train,
                sensitive_test=sensitive_val,
                sensitive_attr=sensitive_attr,
                base_model=base_model
            )

            fold_results.append({
                'fold_idx': fold_idx,
                'train_indices': train_idx,
                'val_indices': val_idx,
                'train_metrics': result.get('train_metrics', None),
                'val_metrics': result['test_metrics']
            })

            y_pred = result['predictions']['y_pred']
            y_proba = result['predictions']['y_proba']
            if y_proba is None:
                y_proba = y_pred

            fold_preds = pd.DataFrame({
                'fold': fold_idx,
                'sample_idx': val_idx,
                'y_true': y_val.values,
                'y_pred': y_pred,
                'y_proba': y_proba
            })

            for col in ['age_group', 'sex', 'ethnicity', 'group_cluster']:
                if col in sensitive_full.columns:
                    fold_preds[col] = sensitive_full.iloc[val_idx][col].values

            all_predictions.append(fold_preds)

        fold_predictions = pd.concat(all_predictions, ignore_index=True)
        fold_predictions = fold_predictions.sort_values('sample_idx').reset_index(drop=True)

        cv_results = {
            'fold_results': fold_results,
            'aggregated_metrics': _aggregate_fold_metrics(fold_results),
            'n_folds': n_folds,
            'random_state': config.get('random_seed', 42)
        }
    
    # Save fold predictions
        versioning.save_predictions(
            exp_id,
            fold_predictions,
            f"cv_predictions_{exp_id}.csv",
            dataset=config['dataset']
        )
    
    # Calculate fairness metrics on full CV predictions
    predictions_df = fold_predictions.copy()
    for col in X_full.columns:
        predictions_df[col] = X_full[col].values
    
    fairness_calc = FairnessMetrics(
        available_sensitive(predictions_df, config['sensitive_attributes'])
    )
    fairness_results = fairness_calc.calculate_all_metrics(
        predictions_df,
        feature_cols=list(X_full.columns)
    )
    
    agg = cv_results['aggregated_metrics']
    logger.info(f"  Accuracy: {agg['accuracy']['mean']:.3f} ± {agg['accuracy']['std']:.3f}")
    logger.info(f"  Recall: {agg['recall']['mean']:.3f} ± {agg['recall']['std']:.3f}")
    logger.info(f"  F1: {agg['f1_score']['mean']:.3f} ± {agg['f1_score']['std']:.3f}")
    
    return {
        'experiment_id': exp_id,
        'configuration': config,
        'cv_results': cv_results['aggregated_metrics'],
        'fold_results': cv_results['fold_results'],
        'fairness_metrics': fairness_results,
        'training_method': 'kfold_cv',
        'n_folds': n_folds
    }


def run_combinatorial_analysis(
    config_path: str,
    pipeline: str = 'cardiac',
    n_jobs: int = 1,
    verbose: bool = False,
    archive_previous: bool = True,
    run_id: Optional[str] = None,
    results_root: Optional[str] = None
):
    """Main orchestration for combinatorial experiments."""
    project_root = get_project_root(Path(__file__))
    use_run_id = bool(run_id or os.getenv('RUN_ID') or os.getenv('PREFECT__RUNTIME__FLOW_RUN_ID'))
    run_id = resolve_run_id(run_id) if use_run_id else None
    log_subdir = f"experiments/{run_id}" if run_id else 'experiments/latest_run'
    setup_phase_logging(project_root, 'combinatorial_experiments.log', verbose=verbose, log_subdir=log_subdir)
    logger = logging.getLogger(__name__)
    
    logger.info("="*80)
    logger.info("COMBINATORIAL FAIRNESS EXPERIMENTS")
    logger.info("="*80)
    logger.info("[PHASE] Combinatorial experiments started")
    
    # Load configuration
    config_path = Path(config_path)
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    if n_jobs is None:
        n_jobs = int(config.get('n_jobs', 1))
    
    logger.info(f"Loaded configuration from: {config_path}")

    sensitive_attrs = preferred_sensitive(config.get('sensitive_attributes'))
    
    # Load pipeline config
    pipeline_cfg = load_yaml_config(str(project_root / f"configs/pipelines/{pipeline}.yaml"))
    target_col = pipeline_cfg.get('training', {}).get('target', 'heart_disease')

    # Initialize versioning
    base_results_dir = Path(results_root) if results_root else Path(config['paths']['results_dir'])
    if run_id:
        run_dir = get_run_root(base_results_dir, run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        versioning = ExperimentVersioning(base_results_dir, run_dir=run_dir)
    else:
        versioning = ExperimentVersioning(base_results_dir)
        # Archive previous run if requested
        if archive_previous:
            versioning.archive_previous_run()

    if run_id:
        append_run_history(base_results_dir, {
            'run_id': run_id,
            'pipeline': pipeline,
            'mode': 'full',
            'phase': 'combinatorial',
            'datasets': config.get('datasets', []),
            'output_dir': str(versioning.latest_dir),
            'status': 'started'
        })
    
    # Generate all experiment combinations
    experiments = []
    fairness_base_params_cfg = config.get('fairness_base_model_params')
    for dataset in config['datasets']:
        if isinstance(fairness_base_params_cfg, dict) and dataset in fairness_base_params_cfg:
            fairness_base_params = fairness_base_params_cfg.get(dataset)
        else:
            fairness_base_params = fairness_base_params_cfg
        if not fairness_base_params:
            fairness_base_params = config.get('model_params', {})
        for binning in config['binning_strategies']:
            for mitigation in config['mitigation_techniques']:
                for training_method in config['training_methods']:
                    exp_id = versioning.generate_experiment_id()
                    exp_config = {
                        'dataset': dataset,
                        'binning_strategy': binning,
                        'mitigation_technique': mitigation,
                        'training_method': training_method,
                        'cv_folds': config.get('cv_folds', 5),
                        'random_seed': config.get('random_seed', 42),
                        'model_params': config.get('model_params', {}),
                        'fairness_base_model_params': fairness_base_params or None,
                        'sensitive_attributes': sensitive_attrs,
                    }
                    
                    experiments.append((exp_id, exp_config))
    
    total_experiments = len(experiments)
    logger.info(f"\nTotal experiments to run: {total_experiments}")
    logger.info(f"  Datasets: {len(config['datasets'])}")
    logger.info(f"  Binning strategies: {len(config['binning_strategies'])}")
    logger.info(f"  Mitigation techniques: {len(config['mitigation_techniques'])}")
    logger.info(f"  Training methods: {len(config['training_methods'])}")
    logger.info(f"  Parallel jobs: {n_jobs}")
    
    # Save manifests first
    logger.info("\nSaving experiment manifests...")
    for exp_id, exp_config in experiments:
        versioning.save_manifest(exp_id, exp_config)
    
    # Run experiments
    logger.info("\nStarting experiments...")
    processed_dir = Path(config['paths']['processed_dir'])
    schema_cfg = load_schema_config(Path(__file__).parent.parent.parent, pipeline)
    
    if n_jobs == 1:
        # Sequential execution
        results = []
        for i, (exp_id, exp_config) in enumerate(experiments, 1):
            logger.info(f"\n[{i}/{total_experiments}] Running experiment {exp_id}...")
            result = run_single_experiment(exp_id, exp_config, versioning, processed_dir, schema_cfg, logger, target_col)
            results.append(result)
            versioning.save_results(exp_id, result)
    else:
        # Parallel execution
        logger.info(f"Running experiments in parallel with {n_jobs} jobs...")
        parallel_verbose = int(config.get('parallel_verbose', 0))
        results = Parallel(n_jobs=n_jobs, verbose=parallel_verbose)(
            delayed(run_single_experiment)(exp_id, exp_config, versioning, processed_dir, schema_cfg, logger, target_col)
            for exp_id, exp_config in experiments
        )
        
        # Save results
        for result in results:
            versioning.save_results(result['experiment_id'], result)
    
    # Create summary
    logger.info("\n" + "="*80)
    logger.info("EXPERIMENTS COMPLETE")
    logger.info("="*80)
    
    summary = versioning.create_summary()
    
    # Count successes and failures
    n_success = sum(1 for r in results if r['execution']['status'] == 'success')
    n_failed = total_experiments - n_success
    
    logger.info(f"\nResults summary:")
    logger.info(f"  Total experiments: {total_experiments}")
    logger.info(f"  Successful: {n_success}")
    logger.info(f"  Failed: {n_failed}")
    logger.info(f"\nResults saved to: {versioning.latest_dir}")
    logger.info("[PHASE] Combinatorial experiments complete")

    if run_id:
        update_latest_pointer(base_results_dir, versioning.latest_dir, logger)
        append_run_history(base_results_dir, {
            'run_id': run_id,
            'pipeline': pipeline,
            'mode': 'full',
            'phase': 'combinatorial',
            'datasets': config.get('datasets', []),
            'output_dir': str(versioning.latest_dir),
            'status': 'completed'
        })

    # Aggregate dataset-level SHAP summaries (global/local)
    xai_root = versioning.latest_dir / 'xai'
    for dataset in config['datasets']:
        shap_dir = xai_root / dataset / 'shap'
        if shap_dir.exists():
            aggregate_dataset_shap(shap_dir, 'global')
            aggregate_dataset_shap(shap_dir, 'local')


def main():
    parser = argparse.ArgumentParser(
        description='Run combinatorial fairness mitigation experiments'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/experiments/combinatorial.yaml',
        help='Path to experiment configuration file'
    )
    parser.add_argument(
        '--pipeline',
        type=str,
        default='cardiac',
        help='Pipeline name (e.g., cardiac, dermatology)'
    )
    parser.add_argument(
        '--n-jobs',
        type=int,
        default=None,
        help='Number of parallel jobs (-1 for all cores)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    parser.add_argument(
        '--archive-previous',
        action='store_true',
        default=os.getenv('ARCHIVE_PREVIOUS', 'true').lower() == 'true',
        help='Archive previous run before starting'
    )
    parser.add_argument(
        '--run-id',
        type=str,
        default=os.getenv('RUN_ID'),
        help='Run identifier (optional, enables run-scoped outputs)'
    )
    parser.add_argument(
        '--results-root',
        type=str,
        default=None,
        help='Base results directory for run outputs'
    )
    
    args = parser.parse_args()
    
    run_combinatorial_analysis(
        config_path=args.config,
        pipeline=args.pipeline,
        n_jobs=args.n_jobs,
        verbose=args.verbose,
        archive_previous=args.archive_previous,
        run_id=args.run_id,
        results_root=args.results_root
    )


if __name__ == '__main__':
    main()