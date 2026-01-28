#!/usr/bin/env python3
"""
Run age binning strategies analysis experiment.

This script:
1. Loads standardized raw datasets (with age_raw column)
2. Tests multiple age binning strategies
3. Computes fairness metrics for each strategy
4. Analyzes sensitive attribute distribution within bins
5. Scores strategies based on sample size, balance, and fairness
6. Generates comprehensive comparison report

Usage:
    python scripts/experiments/run_age_binning_analysis.py
    python scripts/experiments/run_age_binning_analysis.py --strategies fixed_10yr clinical
    python scripts/experiments/run_age_binning_analysis.py --config configs/experiments/age_binning.yaml
"""

import sys
import logging
import argparse
import os
import shutil
from pathlib import Path
from datetime import datetime
import json
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from fairxai.experiments.age_binning import (
    create_binning_strategy,
    analyze_strategy_comprehensive,
    compare_strategies,
    generate_summary_report
)
from fairxai.utils.config import load_yaml_config
from fairxai.utils.logging_utils import setup_logging


def archive_latest_run(base_dir: Path, enabled: bool, logger: logging.Logger):
    if not enabled:
        return
    latest_dir = base_dir / 'latest_run'
    archives_dir = base_dir / 'archived_runs'
    archives_dir.mkdir(parents=True, exist_ok=True)

    has_files = latest_dir.exists() and any(p.is_file() for p in latest_dir.rglob('*'))
    if not has_files:
        return

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    archive_path = archives_dir / f'run_{timestamp}'
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if archive_path.exists():
        shutil.rmtree(archive_path)
    shutil.move(str(latest_dir), str(archive_path))
    latest_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Archived previous latest_run to: {archive_path}")


def load_dataset_for_binning(dataset_name: str, data_dir: Path):
    """
    Load standardized raw dataset with age_raw column.
    
    Args:
        dataset_name: Name of dataset ('cleveland' or 'kaggle_heart')
        data_dir: Path to raw data directory
        
    Returns:
        DataFrame with age_raw, sex, and heart_disease columns
    """
    logging.info(f"\nLoading dataset: {dataset_name}")
    
    file_path = data_dir / f'{dataset_name}_standardized.csv'
    
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    
    df = pd.read_csv(file_path)
    
    # Verify required columns
    REQUIRED_COLUMNS = ['age_raw', 'sex', 'heart_disease']
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    logging.info(f"  Loaded: {len(df)} samples")
    logging.info(f"  Age range: [{df['age_raw'].min()}, {df['age_raw'].max()}]")
    logging.info(f"  Target prevalence: {df['heart_disease'].mean():.2%}")
    
    return df


def run_strategy_analysis(df, strategy_name, dataset_name):
    """
    Analyze a single binning strategy on a dataset.
    
    Args:
        df: DataFrame with age_raw, sex, heart_disease
        strategy_name: Strategy to test
        dataset_name: Dataset identifier
        
    Returns:
        Analysis result dictionary
    """
    logging.info(f"\nAnalyzing strategy: {strategy_name}")
    
    try:
        # Create binning strategy
        bins, labels = create_binning_strategy(df, strategy_name, age_col='age_raw')
        
        # Comprehensive analysis
        result = analyze_strategy_comprehensive(
            df=df,
            strategy_name=strategy_name,
            bins=bins,
            labels=labels,
            dataset_name=dataset_name,
            age_col='age_raw',
            sensitive_col='sex',
            target_col='heart_disease'
        )
        
        return result
        
    except Exception as e:
        logging.error(f"Failed to analyze {strategy_name}: {e}")
        logging.exception(e)
        return None


def main():
    parser = argparse.ArgumentParser(description='Run age binning analysis experiment')
    parser.add_argument('--config', type=str,
                       default='configs/experiments/age_binning.yaml',
                       help='Path to experiment config file')
    parser.add_argument('--datasets', type=str, nargs='+',
                       help='Datasets to process (default: from config)')
    parser.add_argument('--strategies', type=str, nargs='+',
                       help='Strategies to test (default: from config)')
    parser.add_argument('--output-dir', type=str,
                       help='Output directory (default: results/cardiac/experiments/latest_run/age_binning)')
    parser.add_argument('--run-mode', type=str, choices=['full', 'partial'],
                       default=os.getenv('EXPERIMENT_RUN_MODE', 'partial'),
                       help='Run mode (full or partial)')
    parser.add_argument('--archive-previous', action='store_true',
                       default=os.getenv('ARCHIVE_PREVIOUS', 'true').lower() == 'true',
                       help='Archive previous latest_run (full runs only)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose console output')
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
    required_keys = ['data', 'binning_strategies']
    missing = [k for k in required_keys if k not in experiment_cfg]
    if missing:
        logging.error(f"Config missing required keys: {missing}")
        sys.exit(1)
    
    # Determine datasets and strategies to process
    datasets = args.datasets if args.datasets else experiment_cfg['data']['datasets']
    
    if args.strategies:
        strategies = args.strategies
    else:
        strategies = list(experiment_cfg['binning_strategies'].keys())
    
    # Setup logging
    log_dir = project_root / 'logs/experiments/latest_run'
    setup_logging(log_dir / 'age_binning_analysis.log', verbose=args.verbose)
    logger = logging.getLogger(__name__)
    logging.info("[PHASE] Age binning analysis started")

    # Determine output directory
    if args.run_mode == 'partial':
        base_results = project_root / 'results/cardiac/experiments/partial'
    else:
        base_results = project_root / 'results/cardiac/experiments/full'
    latest_dir = base_results / 'latest_run'

    if args.run_mode == 'partial':
        archive_latest_run(base_results, enabled=True, logger=logger)
    else:
        archive_latest_run(base_results, enabled=args.archive_previous, logger=logger)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = latest_dir / 'age_binning'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"Configuration:")
    logging.info(f"  Datasets: {datasets}")
    logging.info(f"  Strategies: {strategies}")
    logging.info(f"  Output: {output_dir}")
    logging.info(f"  Run mode: {args.run_mode}")
    logging.info(f"  Timestamp: {timestamp}")
    
    # Data directory (use raw standardized data)
    data_dir = project_root / 'data/raw/cardiac'
    
    # Get scoring weights from config
    scoring_cfg = experiment_cfg.get('scoring', {})
    scoring_weights = {
        'sample_size': scoring_cfg.get('sample_size_weight', 0.40),
        'balance': scoring_cfg.get('group_balance_weight', 0.30),
        'fairness': scoring_cfg.get('fairness_sensitivity_weight', 0.30)
    }
    
    logging.info(f"\nScoring weights:")
    logging.info(f"  Sample size: {scoring_weights['sample_size']:.0%}")
    logging.info(f"  Group balance: {scoring_weights['balance']:.0%}")
    logging.info(f"  Fairness sensitivity: {scoring_weights['fairness']:.0%}")
    
    # Process each dataset and strategy
    all_results = []
    
    for dataset_name in datasets:
        logging.info(f"\n{'='*80}")
        logging.info(f"PROCESSING: {dataset_name.upper()}")
        logging.info(f"{'='*80}")
        
        try:
            # Load dataset
            df = load_dataset_for_binning(dataset_name, data_dir)
        except Exception as e:
            logging.error(f"Failed to load {dataset_name}: {e}")
            logging.exception(e)
            continue  # Skip this dataset but continue with others
        
        # Test each strategy (errors handled within run_strategy_analysis)
        for strategy_name in strategies:
            result = run_strategy_analysis(df, strategy_name, dataset_name)
            if result:
                all_results.append(result)
    
    if not all_results:
        logging.error("No results generated. Exiting.")
        return
    
    # Compute scores for all results (done once, used by report and recommendations)
    from fairxai.experiments.age_binning import compute_strategy_score
    for result in all_results:
        result['score'] = compute_strategy_score(
            result,
            scoring_weights['sample_size'],
            scoring_weights['balance'],
            scoring_weights['fairness']
        )
    
    # Generate comparison and reports
    logging.info(f"\n{'='*80}")
    logging.info("GENERATING COMPARISON REPORT")
    logging.info(f"{'='*80}")
    
    # Comparison table
    comparison_df = compare_strategies(all_results, by_dataset=True)
    
    # Save results
    csv_file = output_dir / f'age_binning_comparison_{timestamp}.csv'
    json_file = output_dir / f'age_binning_analysis_{timestamp}.json'
    report_file = output_dir / f'age_binning_report_{timestamp}.md'
    
    comparison_df.to_csv(csv_file, index=False)
    logging.info(f"\n[SUCCESS] Saved CSV: {csv_file}")
    
    with open(json_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    logging.info(f"[SUCCESS] Saved JSON: {json_file}")
    
    # Generate markdown report
    generate_summary_report(all_results, report_file, scoring_weights)
    logging.info(f"[SUCCESS] Saved Report: {report_file}")
    
    # Print summary
    logging.info(f"\n{'='*80}")
    logging.info("RESULTS SUMMARY")
    logging.info(f"{'='*80}")
    logging.info("\n" + comparison_df.to_string(index=False))
    
    # Recommendations
    logging.info(f"\n{'='*80}")
    logging.info("RECOMMENDATIONS")
    logging.info(f"{'='*80}")
    
    # Find top strategies per dataset (using pre-computed scores)
    for dataset in datasets:
        dataset_results = [r for r in all_results if r['dataset'] == dataset]
        if not dataset_results:
            continue
        
        # Sort by pre-computed score
        dataset_results.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        logging.info(f"\nTop 3 strategies for {dataset}:")
        for i, result in enumerate(dataset_results[:3], 1):
            metrics = result['fairness_metrics']
            logging.info(f"  {i}. {result['strategy']} (score: {result['score']:.3f})")
            logging.info(f"     - Groups: {metrics['n_groups']}, "
                        f"Min size: {metrics['min_group_size']}, "
                        f"Balance CV: {metrics['group_balance_cv']:.3f}, "
                        f"SP diff: {metrics['max_sp_difference']:.3f}")
    
    logging.info(f"\n{'='*80}")
    logging.info("EXPERIMENT COMPLETE")
    logging.info(f"{'='*80}")
    logging.info(f"\nResults saved to: {output_dir}")
    logging.info(f"  - Comparison CSV: {csv_file.name}")
    logging.info(f"  - Detailed JSON: {json_file.name}")
    logging.info(f"  - Summary Report: {report_file.name}")
    logging.info("[PHASE] Age binning analysis complete")


if __name__ == '__main__':
    main()
