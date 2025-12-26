#!/usr/bin/env python3
"""
Run fairness mitigation comparison experiment.

This script:
1. Loads preprocessed train/test datasets
2. Trains baseline models (no mitigation)
3. Applies pre-processing mitigation techniques (SMOTE, ROS, RUS, ADASYN, reweighting)
4. Applies in-processing techniques (ExponentiatedGradient, GridSearch)
5. Applies post-processing techniques (ThresholdOptimizer)
6. Computes fairness metrics for each technique
7. Saves comprehensive comparison results

Usage:
    python scripts/experiments/run_mitigation_comparison.py
    python scripts/experiments/run_mitigation_comparison.py --datasets cleveland
    python scripts/experiments/run_mitigation_comparison.py --config configs/experiments/mitigation.yaml
"""

import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime
import json
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from fairxai.models.baseline import BaselineLogisticRegression, generate_predictions_with_metadata
from fairxai.fairness.metrics import FairnessMetrics
from fairxai.fairness.mitigation import MitigationEngine
from fairxai.utils.config import load_yaml_config


def setup_logging(log_dir: Path, timestamp: str):
    """Configure logging to file and console."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f'mitigation_comparison_{timestamp}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler()
        ]
    )
    
    logging.info("="*80)
    logging.info("FAIRNESS MITIGATION COMPARISON EXPERIMENT")
    logging.info("="*80)


def load_dataset(dataset_name: str, data_dir: Path):
    """
    Load train/test splits for a dataset.
    
    Args:
        dataset_name: Name of dataset ('cleveland' or 'kaggle_heart')
        data_dir: Path to processed data directory
        
    Returns:
        Tuple of (X_train, y_train, sensitive_train, X_test, y_test, sensitive_test)
    """
    logging.info(f"\nLoading dataset: {dataset_name}")
    
    train_file = data_dir / f'{dataset_name}_train.csv'
    test_file = data_dir / f'{dataset_name}_test.csv'
    
    if not train_file.exists() or not test_file.exists():
        raise FileNotFoundError(f"Dataset files not found: {train_file}, {test_file}")
    
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    
    logging.info(f"  Train: {len(train_df)} samples")
    logging.info(f"  Test: {len(test_df)} samples")
    
    # Encode sex if it's categorical
    if train_df['sex'].dtype == 'object':
        logging.info("  Encoding categorical 'sex' variable...")
        sex_map = {'Male': 1, 'Female': 0, 'M': 1, 'F': 0}
        train_df['sex'] = train_df['sex'].map(sex_map)
        test_df['sex'] = test_df['sex'].map(sex_map)
    
    # Separate features, target, and sensitive attributes
    # Exclude target, sensitive attrs, metadata, and original categorical columns
    exclude = [
        'heart_disease', 'age_group', 'sex', 'sex_extended', 'sex_bin',
        'Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope',
        '_dataset_source', '_dataset_file', 'age_raw', 'HeartDisease'
    ]
    # Only exclude columns that actually exist
    exclude = [col for col in exclude if col in train_df.columns]
    
    X_train = train_df.drop(columns=exclude)
    y_train = train_df['heart_disease']
    sensitive_train = train_df[['age_group', 'sex']]
    
    X_test = test_df.drop(columns=exclude)
    y_test = test_df['heart_disease']
    sensitive_test = test_df[['age_group', 'sex']]
    
    logging.info(f"  Features: {X_train.shape[1]}")
    logging.info(f"  Sensitive attributes: {list(sensitive_train.columns)}")
    
    return X_train, y_train, sensitive_train, X_test, y_test, sensitive_test


def train_baseline(X_train, y_train, X_test, y_test, sensitive_test, dataset_name):
    """Train baseline model without mitigation."""
    logging.info(f"\n{'='*60}")
    logging.info(f"Training Baseline: {dataset_name}")
    logging.info(f"{'='*60}")
    
    model = BaselineLogisticRegression(class_weight='balanced', random_state=42)
    train_metrics = model.train(X_train, y_train)
    test_metrics = model.evaluate(X_test, y_test)
    
    logging.info(f"Baseline metrics:")
    logging.info(f"  Accuracy: {test_metrics['accuracy']:.3f}")
    logging.info(f"  Precision: {test_metrics['precision']:.3f}")
    logging.info(f"  Recall: {test_metrics['recall']:.3f}")
    logging.info(f"  F1: {test_metrics['f1_score']:.3f}")
    logging.info(f"  AUC-ROC: {test_metrics['auc_roc']:.3f}")
    
    # Generate predictions with metadata for fairness assessment
    predictions = generate_predictions_with_metadata(
        model, X_test, y_test, sensitive_test
    )
    
    # Calculate fairness metrics
    # Note: feature_cols for individual fairness should be numeric features only
    # X_test columns are the actual model features (all numeric after exclusions)
    fairness_calc = FairnessMetrics()
    fairness_results = fairness_calc.calculate_all_metrics(
        predictions, 
        feature_cols=list(X_test.columns)  # Use actual model features, not sensitive attrs
    )
    
    return {
        'model': model,
        'test_metrics': test_metrics,
        'fairness': fairness_results,
        'predictions': predictions
    }


def apply_mitigation_techniques(
    X_train, y_train, X_test, y_test, sensitive_train, sensitive_test,
    dataset_name, baseline_model, techniques_config
):
    """
    Apply all mitigation techniques and collect results.
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        sensitive_train, sensitive_test: Sensitive attributes
        dataset_name: Dataset identifier
        baseline_model: Trained baseline model for post-processing
        techniques_config: Dict of techniques to test
        
    Returns:
        List of result dictionaries
    """
    engine = MitigationEngine(random_state=42)
    results = []
    
    for technique_name, config in techniques_config.items():
        stage = config['stage']
        
        logging.info(f"\n{'='*60}")
        logging.info(f"Testing: {technique_name} ({stage})")
        logging.info(f"{'='*60}")
        
        try:
            # Apply technique
            if stage == 'post-processing':
                result = engine.apply_technique(
                    technique_name=technique_name,
                    stage=stage,
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    sensitive_train=sensitive_train,
                    sensitive_test=sensitive_test,
                    sensitive_attr='sex',  # Primary sensitive attribute
                    base_model=baseline_model
                )
            else:
                result = engine.apply_technique(
                    technique_name=technique_name,
                    stage=stage,
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    sensitive_train=sensitive_train,
                    sensitive_test=sensitive_test,
                    sensitive_attr='sex'
                )
            
            # Generate predictions DataFrame for fairness assessment
            y_proba = result['predictions']['y_proba']
            if y_proba is None:
                # Use predictions as fallback if no probabilities
                y_proba = result['predictions']['y_pred']
                logging.warning(f"No probabilities available for {technique_name}, using predictions")
            
            predictions_df = pd.DataFrame({
                'y_true': y_test.values,
                'y_pred': result['predictions']['y_pred'],
                'y_proba': y_proba,
                'age_group': sensitive_test['age_group'].values,
                'sex': sensitive_test['sex'].values
            })
            
            # Add model features for individual fairness calculation
            for col in X_test.columns:
                predictions_df[col] = X_test[col].values
            
            # Calculate fairness metrics
            # Use actual model features (numeric) for individual fairness, not sensitive attrs
            fairness_calc = FairnessMetrics()
            fairness_results = fairness_calc.calculate_all_metrics(
                predictions_df,
                feature_cols=list(X_test.columns)
            )
            
            # Compile result
            results.append({
                'dataset': dataset_name,
                'technique': technique_name,
                'stage': stage,
                'test_metrics': result['test_metrics'],
                'fairness': fairness_results,
                'metadata': result['metadata']
            })
            
            logging.info(f"✓ {technique_name} complete")
            logging.info(f"  Accuracy: {result['test_metrics']['accuracy']:.3f}")
            logging.info(f"  Recall: {result['test_metrics']['recall']:.3f}")
            
        except Exception as e:
            logging.error(f"✗ Failed to apply {technique_name}: {e}")
            logging.exception(e)
    
    return results


def create_comparison_table(all_results):
    """Create comparison DataFrame from results."""
    comparison_data = []
    
    for result in all_results:
        metrics = result['test_metrics']
        
        # Extract key fairness metrics
        fairness = result.get('fairness', {})
        demographic_parity = fairness.get('demographic_parity', {})
        
        row = {
            'dataset': result['dataset'],
            'technique': result['technique'],
            'stage': result['stage'],
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1_score': metrics['f1_score'],
            'auc_roc': metrics['auc_roc'],
        }
        
        # Add fairness metrics if available
        if demographic_parity:
            for attr, dp_metrics in demographic_parity.items():
                if isinstance(dp_metrics, dict) and 'max_difference' in dp_metrics:
                    row[f'dp_{attr}'] = dp_metrics['max_difference']
        
        comparison_data.append(row)
    
    return pd.DataFrame(comparison_data)


def main():
    parser = argparse.ArgumentParser(description='Run mitigation comparison experiment')
    parser.add_argument('--config', type=str, 
                       default='configs/experiments/mitigation.yaml',
                       help='Path to experiment config file')
    parser.add_argument('--datasets', type=str, nargs='+',
                       help='Datasets to process (default: from config)')
    parser.add_argument('--output-dir', type=str,
                       help='Output directory (default: results/experiments/mitigation)')
    args = parser.parse_args()
    
    # Paths
    project_root = Path(__file__).parent.parent.parent
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Load config
    config_path = project_root / args.config
    if not config_path.exists():
        logging.error(f"Config file not found: {config_path}")
        return
    
    experiment_cfg = load_yaml_config(str(config_path))
    
    # Validate config
    REQUIRED_KEYS = ['data', 'mitigation_strategies']
    missing = [k for k in REQUIRED_KEYS if k not in experiment_cfg]
    if missing:
        logging.error(f"Config missing required keys: {missing}")
        sys.exit(1)
    
    # Determine datasets to process
    datasets = args.datasets if args.datasets else experiment_cfg['data']['datasets']
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = project_root / 'results/experiments/mitigation'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    log_dir = project_root / 'logs/experiments'
    setup_logging(log_dir, timestamp)
    
    logging.info(f"Configuration:")
    logging.info(f"  Datasets: {datasets}")
    logging.info(f"  Output: {output_dir}")
    logging.info(f"  Timestamp: {timestamp}")
    
    # Data directory
    data_dir = project_root / 'data/processed/cardiac'
    
    # Techniques to test (from config)
    techniques = experiment_cfg['mitigation_strategies']
    
    # Filter to implemented techniques
    implemented = {
        'smote': techniques['smote'],
        'ros': techniques['ros'],
        'rus': techniques['rus'],
        'adasyn': techniques['adasyn'],
        'reweighting': techniques['reweighting'],
        'exponentiated_gradient': techniques['exponentiated_gradient'],
        'grid_search': techniques['grid_search'],
        'threshold_optimizer': techniques['threshold_optimizer']
    }
    
    logging.info(f"\nTechniques to test: {list(implemented.keys())}")
    
    # Process each dataset
    all_results = []
    baseline_results = []
    
    for dataset_name in datasets:
        logging.info(f"\n{'='*80}")
        logging.info(f"PROCESSING: {dataset_name.upper()}")
        logging.info(f"{'='*80}")
        
        try:
            # Load data
            X_train, y_train, sensitive_train, X_test, y_test, sensitive_test = \
                load_dataset(dataset_name, data_dir)
            
            # Train baseline
            baseline = train_baseline(
                X_train, y_train, X_test, y_test, sensitive_test, dataset_name
            )
            
            baseline_results.append({
                'dataset': dataset_name,
                'technique': 'baseline',
                'stage': 'none',
                'test_metrics': baseline['test_metrics'],
                'fairness': baseline['fairness'],
                'metadata': {}
            })
            
            # Apply mitigation techniques
            mitigation_results = apply_mitigation_techniques(
                X_train, y_train, X_test, y_test,
                sensitive_train, sensitive_test,
                dataset_name, baseline['model'], implemented
            )
            
            all_results.extend(mitigation_results)
            
        except Exception as e:
            logging.error(f"Failed to process {dataset_name}: {e}")
            logging.exception(e)
    
    # Combine baseline and mitigation results
    all_results = baseline_results + all_results
    
    # Summary statistics
    logging.info(f"\n{'='*80}")
    logging.info("PROCESSING SUMMARY")
    logging.info(f"{'='*80}")
    logging.info(f"Datasets processed: {len(set(r['dataset'] for r in all_results))}")
    logging.info(f"Techniques tested: {len(set(r['technique'] for r in all_results))}")
    logging.info(f"Total results: {len(all_results)}")
    
    # Create comparison table
    logging.info(f"\n{'='*80}")
    logging.info("GENERATING COMPARISON REPORT")
    logging.info(f"{'='*80}")
    
    comparison_df = create_comparison_table(all_results)
    
    # Save results
    csv_file = output_dir / f'mitigation_comparison_{timestamp}.csv'
    json_file = output_dir / f'mitigation_comparison_{timestamp}.json'
    
    comparison_df.to_csv(csv_file, index=False)
    logging.info(f"\n✓ Saved CSV: {csv_file}")
    
    with open(json_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    logging.info(f"✓ Saved JSON: {json_file}")
    
    # Print summary
    logging.info(f"\n{'='*80}")
    logging.info("RESULTS SUMMARY")
    logging.info(f"{'='*80}")
    logging.info("\n" + comparison_df.round(3).to_string(index=False))
    
    # Clinical constraint check (recall >= 0.70)
    meets_clinical = comparison_df[comparison_df['recall'] >= 0.70]
    logging.info(f"\n\nTechniques meeting clinical constraint (recall ≥ 0.70): {len(meets_clinical)}")
    if len(meets_clinical) > 0:
        logging.info("\n" + meets_clinical[['dataset', 'technique', 'recall', 'accuracy']].to_string(index=False))
    
    logging.info(f"\n{'='*80}")
    logging.info("EXPERIMENT COMPLETE")
    logging.info(f"{'='*80}")


if __name__ == '__main__':
    main()
