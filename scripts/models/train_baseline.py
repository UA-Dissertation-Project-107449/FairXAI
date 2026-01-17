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
import shutil
from datetime import datetime
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from fairxai.models.baseline import BaselineLogisticRegression, generate_predictions_with_metadata


def setup_logging(log_dir: Path):
    """Configure logging to file and console."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / 'training_baseline.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler()
        ]
    )
    
    logging.info("="*60)
    logging.info("BASELINE MODEL TRAINING - LOGISTIC REGRESSION")
    logging.info("="*60)


def archive_latest_run(base_dir: Path, enabled: bool) -> None:
    if not enabled:
        return

    latest_dir = base_dir / 'latest_run'
    archives_dir = base_dir / 'archived_runs'
    archives_dir.mkdir(parents=True, exist_ok=True)

    has_files = latest_dir.exists() and any(latest_dir.rglob('*'))
    if not has_files:
        return

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    archive_path = archives_dir / f'run_{timestamp}'
    shutil.copytree(latest_dir, archive_path)

    # Clean latest_run
    for item in latest_dir.glob('*'):
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()


def main():
    # Paths
    project_root = Path(__file__).parent.parent.parent
    from fairxai.utils.config import load_yaml_config
    
    pipeline_cfg = load_yaml_config(str(project_root / 'configs/pipelines/cardiac.yaml'))
    schema_path = project_root / pipeline_cfg['runtime']['schema_mapping_json']
    with open(schema_path, 'r') as f:
        schema_cfg = json.load(f)
    data_processed = project_root / pipeline_cfg['paths']['processed_dir']
    experiments_dir = project_root / pipeline_cfg['paths']['experiments_dir']
    models_dir = project_root / pipeline_cfg['paths']['models_dir']
    log_dir = project_root / 'logs/cardiac'
    baseline_root = project_root / 'results/cardiac/baseline'
    
    # Setup
    setup_logging(log_dir)
    archive_latest_run(baseline_root, enabled=(os.getenv('ARCHIVE_BASELINE', 'true').lower() == 'true'))

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

        base_dataset = next(
            (ds for ds in schema_cfg.get('cardiac_relevant_datasets', []) if dataset_name.startswith(ds)),
            dataset_name
        )
        dataset_cfg = schema_cfg.get('datasets', {}).get(base_dataset, {})
        unified_cfg = schema_cfg.get('unified_schema', {})
        schema_exclude = list(dataset_cfg.get('exclude_features') or [])
        schema_exclude += list(unified_cfg.get('exclude_features') or [])
        label_col = dataset_cfg.get('label') or dataset_cfg.get('target')
        if label_col:
            schema_exclude.append(label_col)

        exclude_cols = [target_col] + sensitive_cols + schema_exclude
        
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
        
        logging.info(f"\n✓ Predictions saved:")
        logging.info(f"  Train: {train_pred_file}")
        logging.info(f"  Test: {test_pred_file}")
        
        # Save feature importance
        importance_file = experiments_dir / f'{dataset_name}_feature_importance.csv'
        feature_importance.to_csv(importance_file, index=False)
        logging.info(f"  Feature importance: {importance_file}")
        
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
    logging.info("TRAINING COMPLETE")
    logging.info(f"{'='*60}")
    logging.info(f"Models saved to: {models_dir}")
    logging.info(f"Results saved to: {experiments_dir}")
    logging.info(f"\nNext step: Run fairness assessment on predictions (Phase 4)")
    

if __name__ == "__main__":
    main()
