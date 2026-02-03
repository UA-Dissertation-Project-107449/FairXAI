#!/usr/bin/env python3
"""
Train baseline logistic regression models for cardiac disease prediction.

This script:
1. Loads preprocessed train/test datasets
2. Trains logistic regression for each dataset
3. Evaluates on test set with multiple thresholds
4. Generates predictions with probabilities
5. Saves trained models and predictions
6. Logs performance metrics

Usage:
    python scripts/models/train_baseline.py
"""

import sys
import logging
import os
from pathlib import Path
import json
from datetime import datetime
from typing import Optional
import pandas as pd
import numpy as np
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from fairxai.models.baseline import BaselineLogisticRegression, generate_predictions_with_metadata
from fairxai.explainability.tabular import shap_explain_tabular, lime_explain_instance
from fairxai.cli.runner_base import get_project_root, load_pipeline_config, setup_phase_logging
from fairxai.experiments.data_io import build_schema_excludes, resolve_base_dataset
from fairxai.cli.runner_utils import archive_latest_run


def save_xai_outputs(
    model: BaselineLogisticRegression,
    X_ref: pd.DataFrame,
    X_lime: pd.DataFrame,
    output_dir: Path,
    dataset_name: str,
    X_global: Optional[pd.DataFrame] = None
) -> None:
    xai_enabled = os.getenv('XAI_ENABLED', 'true').lower() == 'true'
    if not xai_enabled:
        logging.info("XAI disabled via XAI_ENABLED=false")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    lime_instances = int(os.getenv('XAI_LIME_INSTANCES', '3'))
    global_max = int(os.getenv('XAI_GLOBAL_MAX_SAMPLES', '1000'))

    # Global SHAP summary (dataset-level)
    if X_global is not None:
        try:
            df_global = X_global.copy()
            if len(df_global) > global_max:
                df_global = df_global.sample(n=global_max, random_state=42)
            shap_global = shap_explain_tabular(model.model, df_global, max_samples=global_max)
            shap_vals_global = np.abs(shap_global.shap_values)
            mean_abs_global = np.mean(shap_vals_global, axis=0)
            shap_global_summary = pd.DataFrame({
                'feature': shap_global.feature_names,
                'mean_abs_shap': mean_abs_global
            }).sort_values('mean_abs_shap', ascending=False)
            shap_global_file = output_dir / f"{dataset_name}_shap_global.csv"
            shap_global_summary.to_csv(shap_global_file, index=False)
            logging.info(f"[SUCCESS] SHAP global summary saved: {shap_global_file}")
        except Exception as exc:
            logging.warning(f"Global SHAP failed for {dataset_name}: {exc}")

    # LIME examples
    try:
        def _wrap_decision_function(df_model):
            class _DecisionFunctionWrapper:
                def __init__(self, base_model):
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

        def _resolve_lime_model(raw_model):
            if hasattr(raw_model, 'predict_proba'):
                return raw_model
            if hasattr(raw_model, 'model') and hasattr(raw_model.model, 'predict_proba'):
                return raw_model.model
            if hasattr(raw_model, 'decision_function'):
                return _wrap_decision_function(raw_model)
            if hasattr(raw_model, 'model') and hasattr(raw_model.model, 'decision_function'):
                return _wrap_decision_function(raw_model.model)
            return None

        lime_model = _resolve_lime_model(model)
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
            lime_file = output_dir / f"{dataset_name}_lime_examples.csv"
            lime_df.to_csv(lime_file, index=False)
            logging.info(f"[SUCCESS] LIME examples saved: {lime_file}")
        elif lime_instances > 0:
            logging.warning(f"LIME skipped for {dataset_name}: no predict_proba/decision_function")
    except Exception as exc:
        logging.warning(f"LIME failed for {dataset_name}: {exc}")




def main():
    parser = argparse.ArgumentParser(description='Train baseline models')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose console output')
    args = parser.parse_args()

    # Paths
    project_root = get_project_root(Path(__file__))
    pipeline_cfg = load_pipeline_config(project_root, "cardiac")
    schema_path = project_root / pipeline_cfg['runtime']['schema_mapping_json']
    with open(schema_path, 'r') as f:
        schema_cfg = json.load(f)
    data_processed = project_root / pipeline_cfg['paths']['processed_dir']
    run_id = os.getenv('RUN_ID')
    if run_id:
        baseline_root = project_root / f"results/cardiac/runs/{run_id}/baseline"
        experiments_dir = baseline_root / "results"
        models_dir = baseline_root / "models"
    else:
        experiments_dir = project_root / pipeline_cfg['paths']['experiments_dir']
        models_dir = project_root / pipeline_cfg['paths']['models_dir']
    log_dir = setup_phase_logging(project_root, 'training_baseline.log', verbose=args.verbose)
    if not run_id:
        baseline_root = project_root / 'results/cardiac/baseline'
    
    # Setup
    logging.info("[PHASE] Baseline training started")
    if not run_id:
        archive_latest_run(
            baseline_root,
            enabled=(os.getenv('ARCHIVE_BASELINE', 'true').lower() == 'true'),
            logger=logging.getLogger(__name__)
        )

    models_dir.mkdir(parents=True, exist_ok=True)
    experiments_dir.mkdir(parents=True, exist_ok=True)
    
    # Configuration
    random_state = 42
    thresholds_to_test = [0.3, 0.4, 0.5, 0.6, 0.7]
    
    logging.info(f"Configuration:")
    logging.info(f"  Model: Logistic Regression")
    logging.info(f"  Random state: {random_state}")
    logging.info(f"  Decision thresholds: {thresholds_to_test}")
    
    # Find processed datasets
    train_files = list(data_processed.glob('*_train_scaled.csv'))
    
    if not train_files:
        logging.error(f"No training datasets found in {data_processed}")
        logging.error("Please run scripts/data/preprocess_cardiac.py first.")
        return
    
    logging.info(f"\nFound {len(train_files)} datasets to train on")
    
    # Train models for each dataset
    results_summary = {}
    
    for train_file in train_files:
        dataset_name = train_file.stem.replace('_train_scaled', '')
        test_file = train_file.parent / f'{dataset_name}_test_scaled.csv'
        
        if not test_file.exists():
            logging.warning(f"Test file not found for {dataset_name}, skipping")
            continue
        
        logging.info(f"\n{'='*60}")
        logging.info(f"Training: {dataset_name}")
        logging.info(f"{'='*60}")
        
        # Load data
        train_df = pd.read_csv(train_file)
        test_df = pd.read_csv(test_file)
        
        logging.info(f"Loaded data:")
        logging.info(f"  Train: {len(train_df)} samples")
        logging.info(f"  Test: {len(test_df)} samples")
        
        # Separate features, target, and sensitive attributes
        # Note: scaled files have both encoded and categorical versions
        # We keep the encoded numerical versions for modeling
        sensitive_cols = ['age_group', 'sex']
        target_col = 'heart_disease'

        base_dataset = resolve_base_dataset(schema_cfg, dataset_name)
        schema_exclude = build_schema_excludes(schema_cfg, base_dataset)

        extra_excludes = [
            col for col in train_df.columns
            if col.startswith('sex_bin') or col.startswith('sex_extended')
        ]
        exclude_cols = [target_col] + sensitive_cols + schema_exclude + extra_excludes
        
        # Get feature columns (all except target and categorical sensitive attrs)
        feature_cols = [col for col in train_df.columns if col not in exclude_cols]
        
        X_train = train_df[feature_cols].select_dtypes(include=[np.number])
        y_train = train_df[target_col]
        X_test = test_df[feature_cols].select_dtypes(include=[np.number])
        y_test = test_df[target_col]
        
        # Keep categorical versions for fairness analysis
        sensitive_train = train_df[sensitive_cols]
        sensitive_test = test_df[sensitive_cols]
        
        logging.info(f"Features: {len(feature_cols)}")
        
        # Train model
        logging.info(f"\n--- Training ---")
        model = BaselineLogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=random_state,
            class_weight='balanced'  # Handle class imbalance
        )
        
        train_metrics = model.train(X_train, y_train)
        
        # Evaluate on test set
        logging.info(f"\n--- Evaluation ---")
        test_metrics_default = model.evaluate(X_test, y_test, threshold=0.5)
        
        logging.info(f"Test Set Performance (threshold=0.5):")
        logging.info(f"  Accuracy:  {test_metrics_default['accuracy']:.4f}")
        logging.info(f"  Precision: {test_metrics_default['precision']:.4f}")
        logging.info(f"  Recall:    {test_metrics_default['recall']:.4f}")
        logging.info(f"  F1 Score:  {test_metrics_default['f1_score']:.4f}")
        logging.info(f"  AUC-ROC:   {test_metrics_default['auc_roc']:.4f}")
        
        cm = test_metrics_default['confusion_matrix']
        logging.info(f"  Confusion Matrix:")
        logging.info(f"    TN: {cm['tn']:3d}  FP: {cm['fp']:3d}")
        logging.info(f"    FN: {cm['fn']:3d}  TP: {cm['tp']:3d}")
        
        # Test multiple thresholds
        logging.info(f"\n--- Threshold Analysis ---")
        threshold_results = []
        
        for threshold in thresholds_to_test:
            metrics = model.evaluate(X_test, y_test, threshold=threshold)
            threshold_results.append(metrics)
            logging.info(f"  Threshold {threshold:.1f}: "
                        f"Acc={metrics['accuracy']:.3f} "
                        f"Prec={metrics['precision']:.3f} "
                        f"Rec={metrics['recall']:.3f} "
                        f"F1={metrics['f1_score']:.3f}")
        
        # Feature importance
        logging.info(f"\n--- Feature Importance (Top 10) ---")
        feature_importance = model.get_feature_importance()
        for idx, row in feature_importance.head(10).iterrows():
            logging.info(f"  {row['feature']:20s}: {row['coefficient']:+.4f}")
        
        # Generate predictions with metadata
        logging.info(f"\n--- Generating Predictions ---")
        
        train_predictions = generate_predictions_with_metadata(
            model, X_train, y_train, sensitive_train, threshold=0.5
        )
        test_predictions = generate_predictions_with_metadata(
            model, X_test, y_test, sensitive_test, threshold=0.5
        )
        
        # Analyze near-threshold predictions
        n_near_threshold_train = train_predictions['near_threshold'].sum()
        n_near_threshold_test = test_predictions['near_threshold'].sum()
        
        logging.info(f"  Train predictions: {len(train_predictions)}")
        logging.info(f"    Near threshold (±0.1): {n_near_threshold_train} "
                    f"({n_near_threshold_train/len(train_predictions):.1%})")
        logging.info(f"  Test predictions: {len(test_predictions)}")
        logging.info(f"    Near threshold (±0.1): {n_near_threshold_test} "
                    f"({n_near_threshold_test/len(test_predictions):.1%})")
        
        # Save model
        model_file = models_dir / f'{dataset_name}_logistic_regression.pkl'
        model.save(str(model_file))
        
        # Save predictions
        train_pred_file = experiments_dir / f'{dataset_name}_train_predictions.csv'
        test_pred_file = experiments_dir / f'{dataset_name}_test_predictions.csv'
        
        train_predictions.to_csv(train_pred_file, index=False)
        test_predictions.to_csv(test_pred_file, index=False)
        
        logging.info(f"\n[SUCCESS] Predictions saved:")
        logging.info(f"  Train: {train_pred_file}")
        logging.info(f"  Test: {test_pred_file}")
        
        # Save feature importance
        importance_file = experiments_dir / f'{dataset_name}_feature_importance.csv'
        feature_importance.to_csv(importance_file, index=False)
        logging.info(f"  Feature importance: {importance_file}")

        # XAI outputs
        xai_dir = experiments_dir / 'xai'
        save_xai_outputs(model, X_train, X_test, xai_dir, dataset_name, X_global=X_train)
        
        # Save metrics
        results_summary[dataset_name] = {
            'train_metrics': train_metrics,
            'test_metrics': test_metrics_default,
            'threshold_analysis': threshold_results,
            'n_features': len(feature_cols),
            'n_train': len(train_df),
            'n_test': len(test_df),
            'near_threshold_pct_test': float(n_near_threshold_test / len(test_predictions))
        }
    
    # Save overall results
    results_file = experiments_dir / 'training_results.json'
    with open(results_file, 'w') as f:
        json.dump(results_summary, f, indent=2, default=str)
    
    logging.info(f"\n{'='*60}")
    logging.info("[PHASE] Baseline training complete")
    logging.info(f"{'='*60}")
    logging.info(f"Models saved to: {models_dir}")
    logging.info(f"Results saved to: {experiments_dir}")
    logging.info(f"\nNext step: Run fairness assessment on predictions (Phase 4)")
    

if __name__ == "__main__":
    main()
