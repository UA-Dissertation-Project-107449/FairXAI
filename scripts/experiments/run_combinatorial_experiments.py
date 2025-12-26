"""Combinatorial experiment orchestration for fairness mitigation analysis."""

import argparse
import logging
import sys
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import pandas as pd
import numpy as np
from joblib import Parallel, delayed

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fairxai.models.baseline import BaselineLogisticRegression
from fairxai.models.cv_trainer import CVTrainer
from fairxai.fairness.metrics import FairnessMetrics
from fairxai.fairness.mitigation import MitigationEngine
from fairxai.experiments.versioning import ExperimentVersioning
from fairxai.data.schemas import available_sensitive, preferred_sensitive


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


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


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


def prepare_data_splits(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    exclude_cols: List[str],
    sensitive_attrs: List[str]
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
    y_train = train_df['heart_disease']
    X_test = test_df.drop(columns=exclude_cols)
    y_test = test_df['heart_disease']

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


def run_single_experiment(
    exp_id: str,
    config: Dict[str, Any],
    versioning: ExperimentVersioning,
    processed_dir: Path,
    logger: logging.Logger
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
        exclude_cols = [
            'heart_disease', 'age_group', 'sex', 'sex_extended', 'sex_bin',
            'Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope',
            '_dataset_source', '_dataset_file', 'age_raw', 'HeartDisease'
        ]
        exclude_cols.extend(config.get('sensitive_attributes', []))
        splits = prepare_data_splits(
            data['train_df'],
            data['test_df'],
            exclude_cols,
            config['sensitive_attributes']
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
        
        logger.info(f"✓ Experiment {exp_id} completed in {duration:.1f}s")
        return results
        
    except Exception as e:
        logger.error(f"✗ Experiment {exp_id} failed: {str(e)}")
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
        stage = STAGE_MAP.get(mitigation)
        if stage is None:
            raise ValueError(f"Unknown mitigation technique: {mitigation}")

        base_model = None
        if stage == 'post-processing':
            base_model = BaselineLogisticRegression(**config.get('model_params', {}))
            base_model.train(splits['X_train'], splits['y_train'])

        # Apply mitigation technique
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
            base_model=base_model
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
    versioning.save_predictions(exp_id, predictions_df)
    
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
    
    # Run CV experiment
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
    
    # Save fold predictions
    versioning.save_predictions(exp_id, fold_predictions, f"cv_predictions_{exp_id}.csv")
    
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


def main():
    """Main orchestration for combinatorial experiments."""
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
        '--n-jobs',
        type=int,
        default=1,
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
        help='Archive previous run before starting'
    )
    
    args = parser.parse_args()
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    logger.info("="*80)
    logger.info("COMBINATORIAL FAIRNESS EXPERIMENTS")
    logger.info("="*80)
    
    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Loaded configuration from: {config_path}")

    sensitive_attrs = preferred_sensitive(config.get('sensitive_attributes'))
    
    # Initialize versioning
    base_results_dir = Path(config['paths']['results_dir'])
    versioning = ExperimentVersioning(base_results_dir)
    
    # Archive previous run if requested
    if args.archive_previous:
        versioning.archive_previous_run()
    
    # Generate all experiment combinations
    experiments = []
    for dataset in config['datasets']:
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
                        'sensitive_attributes': sensitive_attrs,
                    }
                    
                    experiments.append((exp_id, exp_config))
    
    total_experiments = len(experiments)
    logger.info(f"\nTotal experiments to run: {total_experiments}")
    logger.info(f"  Datasets: {len(config['datasets'])}")
    logger.info(f"  Binning strategies: {len(config['binning_strategies'])}")
    logger.info(f"  Mitigation techniques: {len(config['mitigation_techniques'])}")
    logger.info(f"  Training methods: {len(config['training_methods'])}")
    logger.info(f"  Parallel jobs: {args.n_jobs}")
    
    # Save manifests first
    logger.info("\nSaving experiment manifests...")
    for exp_id, exp_config in experiments:
        versioning.save_manifest(exp_id, exp_config)
    
    # Run experiments
    logger.info("\nStarting experiments...")
    processed_dir = Path(config['paths']['processed_dir'])
    
    if args.n_jobs == 1:
        # Sequential execution
        results = []
        for i, (exp_id, exp_config) in enumerate(experiments, 1):
            logger.info(f"\n[{i}/{total_experiments}] Running experiment {exp_id}...")
            result = run_single_experiment(exp_id, exp_config, versioning, processed_dir, logger)
            results.append(result)
            versioning.save_results(exp_id, result)
    else:
        # Parallel execution
        logger.info(f"Running experiments in parallel with {args.n_jobs} jobs...")
        results = Parallel(n_jobs=args.n_jobs, verbose=10)(
            delayed(run_single_experiment)(exp_id, exp_config, versioning, processed_dir, logger)
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
    logger.info(f"\nNext steps:")
    logger.info(f"  1. Compare experiments: python scripts/analysis/compare_experiments.py")
    logger.info(f"  2. Visualize results: python scripts/analysis/visualize_results.py")


if __name__ == '__main__':
    main()
