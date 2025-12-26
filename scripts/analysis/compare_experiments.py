"""Compare results across combinatorial experiments."""

import argparse
import json
import yaml
from pathlib import Path
import pandas as pd
import numpy as np
import sys
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fairxai.experiments.versioning import ExperimentVersioning


def setup_logging():
    """Configure logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s - %(message)s'
    )


def load_all_results(versioning: ExperimentVersioning) -> pd.DataFrame:
    """
    Load all experiment results and combine into DataFrame.
    
    Args:
        versioning: Versioning system instance
        
    Returns:
        DataFrame with all experiment results
    """
    experiments = versioning.list_experiments()
    
    if not experiments:
        logging.warning("No experiments found in latest_run")
        return pd.DataFrame()
    
    all_results = []
    
    for exp_summary in experiments:
        exp_id = exp_summary['experiment_id']
        
        try:
            # Load full experiment data
            exp_data = versioning.load_experiment(exp_id)
            
            if exp_data['results'] is None:
                logging.warning(f"No results for experiment {exp_id}, skipping")
                continue
            
            config = exp_data['manifest']['configuration']
            results = exp_data['results']
            
            # Extract metrics
            row = {
                'experiment_id': exp_id,
                'dataset': config['dataset'],
                'binning_strategy': config['binning_strategy'],
                'mitigation_technique': config['mitigation_technique'],
                'training_method': config['training_method'],
                'status': results['execution']['status']
            }
            
            # Add performance metrics
            if results['execution']['status'] == 'success':
                if config['training_method'] == 'kfold_cv':
                    # CV results
                    cv_results = results.get('cv_results', {})
                    for metric_name in ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']:
                        if metric_name in cv_results:
                            row[f'{metric_name}_mean'] = cv_results[metric_name]['mean']
                            row[f'{metric_name}_std'] = cv_results[metric_name]['std']
                else:
                    # Single split results
                    test_metrics = results.get('test_metrics', {})
                    for metric_name, value in test_metrics.items():
                        row[metric_name] = value
                
                # Add fairness metrics
                fairness = results.get('fairness_metrics', {})
                if fairness:
                    # Demographic parity
                    if 'demographic_parity' in fairness:
                        for attr, metrics in fairness['demographic_parity'].items():
                            row[f'dem_parity_{attr}_max_diff'] = metrics.get('max_difference', np.nan)
                    
                    # Equalized odds
                    if 'equalized_odds' in fairness:
                        for attr, metrics in fairness['equalized_odds'].items():
                            row[f'eq_odds_{attr}_tpr_diff'] = metrics.get('tpr_difference', np.nan)
                            row[f'eq_odds_{attr}_fpr_diff'] = metrics.get('fpr_difference', np.nan)
            
            all_results.append(row)
            
        except Exception as e:
            logging.error(f"Failed to load experiment {exp_id}: {e}")
            continue
    
    return pd.DataFrame(all_results)


def create_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Create summary statistics grouped by key factors."""
    if df.empty:
        return pd.DataFrame()
    
    # Group by mitigation technique
    summary = df.groupby('mitigation_technique').agg({
        'accuracy_mean' if 'accuracy_mean' in df.columns else 'accuracy': ['mean', 'std', 'min', 'max'],
        'recall_mean' if 'recall_mean' in df.columns else 'recall': ['mean', 'std', 'min', 'max'],
        'f1_score_mean' if 'f1_score_mean' in df.columns else 'f1_score': ['mean', 'std', 'min', 'max']
    })
    
    return summary


def compare_binning_strategies(df: pd.DataFrame, output_dir: Path):
    """
    Create binning strategy comparison table.
    
    Pivot table: binning_strategy × dataset with performance metrics
    """
    if df.empty:
        logging.warning("No data for binning comparison")
        return
    
    # Use appropriate metric columns based on training method
    metric_col = 'f1_score_mean' if 'f1_score_mean' in df.columns else 'f1_score'
    
    # Pivot: rows=binning, cols=dataset, values=avg F1
    pivot = df.pivot_table(
        index='binning_strategy',
        columns='dataset',
        values=metric_col,
        aggfunc='mean'
    )
    
    # Save
    output_file = output_dir / 'binning_comparison.csv'
    pivot.to_csv(output_file)
    logging.info(f"✓ Saved binning comparison: {output_file}")
    
    return pivot


def compare_mitigation_techniques(df: pd.DataFrame, output_dir: Path):
    """
    Create mitigation technique comparison table.
    
    Pivot table: mitigation_technique × dataset with performance metrics
    """
    if df.empty:
        logging.warning("No data for mitigation comparison")
        return
    
    metric_col = 'f1_score_mean' if 'f1_score_mean' in df.columns else 'f1_score'
    
    # Pivot: rows=mitigation, cols=dataset, values=avg F1
    pivot = df.pivot_table(
        index='mitigation_technique',
        columns='dataset',
        values=metric_col,
        aggfunc='mean'
    )
    
    # Save
    output_file = output_dir / 'mitigation_comparison.csv'
    pivot.to_csv(output_file)
    logging.info(f"✓ Saved mitigation comparison: {output_file}")
    
    return pivot


def filter_best_configurations(
    df: pd.DataFrame,
    output_dir: Path,
    fairness_threshold: float = 0.10,
    performance_threshold: float = 0.15
):
    """
    Filter configurations meeting fairness/performance criteria.
    
    Args:
        df: Results dataframe
        fairness_threshold: Min fairness improvement (fraction)
        performance_threshold: Max performance drop (fraction)
    """
    if df.empty:
        logging.warning("No data for filtering")
        return
    
    # Calculate baseline metrics for each dataset
    baseline_df = df[df['mitigation_technique'] == 'baseline']
    
    best_configs = []
    
    for dataset in df['dataset'].unique():
        dataset_df = df[df['dataset'] == dataset].copy()
        baseline_row = baseline_df[baseline_df['dataset'] == dataset]
        
        if baseline_row.empty:
            logging.warning(f"No baseline for {dataset}, skipping")
            continue
        
        # Get baseline metrics
        metric_col = 'f1_score_mean' if 'f1_score_mean' in df.columns else 'f1_score'
        baseline_f1 = baseline_row[metric_col].values[0]
        
        # Filter non-baseline techniques
        mitigated_df = dataset_df[dataset_df['mitigation_technique'] != 'baseline']
        
        for _, row in mitigated_df.iterrows():
            f1 = row[metric_col]
            
            # Check performance threshold
            performance_drop = (baseline_f1 - f1) / baseline_f1
            
            if performance_drop <= performance_threshold:
                # Add to best configs
                best_configs.append({
                    'dataset': dataset,
                    'binning': row['binning_strategy'],
                    'mitigation': row['mitigation_technique'],
                    'training_method': row['training_method'],
                    'f1_score': f1,
                    'baseline_f1': baseline_f1,
                    'f1_drop_pct': performance_drop * 100,
                    'experiment_id': row['experiment_id']
                })
    
    best_df = pd.DataFrame(best_configs)
    
    if not best_df.empty:
        # Sort by F1 score
        best_df = best_df.sort_values(['dataset', 'f1_score'], ascending=[True, False])
        
        output_file = output_dir / 'best_configurations.csv'
        best_df.to_csv(output_file, index=False)
        logging.info(f"✓ Saved best configurations: {output_file}")
        logging.info(f"  Found {len(best_df)} configurations meeting criteria")
    else:
        logging.warning("No configurations met the criteria")
    
    return best_df


def main():
    """Main comparison script."""
    parser = argparse.ArgumentParser(
        description='Compare combinatorial experiment results'
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default='results/experiments',
        help='Base results directory'
    )
    parser.add_argument(
        '--fairness-threshold',
        type=float,
        default=0.10,
        help='Minimum fairness improvement (10%% default)'
    )
    parser.add_argument(
        '--performance-threshold',
        type=float,
        default=0.15,
        help='Maximum performance drop (15%% default)'
    )
    
    args = parser.parse_args()
    setup_logging()
    
    logging.info("="*80)
    logging.info("EXPERIMENT COMPARISON")
    logging.info("="*80)
    
    # Initialize versioning
    versioning = ExperimentVersioning(Path(args.results_dir))
    
    # Check if latest_run exists
    if not versioning.latest_dir.exists():
        logging.error(f"No latest_run found in {args.results_dir}")
        logging.error("Run combinatorial experiments first")
        return
    
    # Load all results
    logging.info("\nLoading experiment results...")
    df = load_all_results(versioning)
    
    if df.empty:
        logging.error("No results loaded")
        return
    
    logging.info(f"Loaded {len(df)} experiments")
    logging.info(f"  Successful: {(df['status'] == 'success').sum()}")
    logging.info(f"  Failed: {(df['status'] == 'failed').sum()}")
    
    # Filter successful experiments
    df_success = df[df['status'] == 'success'].copy()
    
    if df_success.empty:
        logging.error("No successful experiments to analyze")
        return
    
    # Create output directory
    output_dir = versioning.latest_dir / 'comparisons'
    output_dir.mkdir(exist_ok=True)
    
    # Save full results table
    full_results_file = output_dir / 'full_comparison.csv'
    df_success.to_csv(full_results_file, index=False)
    logging.info(f"\n✓ Saved full results: {full_results_file}")
    
    # Create comparison tables
    logging.info("\nGenerating comparison tables...")
    
    binning_pivot = compare_binning_strategies(df_success, output_dir)
    mitigation_pivot = compare_mitigation_techniques(df_success, output_dir)
    
    # Filter best configurations
    logging.info("\nFiltering best configurations...")
    best_configs = filter_best_configurations(
        df_success,
        output_dir,
        args.fairness_threshold,
        args.performance_threshold
    )
    
    # Print summary
    logging.info("\n" + "="*80)
    logging.info("COMPARISON COMPLETE")
    logging.info("="*80)
    logging.info(f"Results saved to: {output_dir}")
    logging.info(f"\nFiles created:")
    logging.info(f"  - full_comparison.csv ({len(df_success)} experiments)")
    logging.info(f"  - binning_comparison.csv")
    logging.info(f"  - mitigation_comparison.csv")
    if best_configs is not None and not best_configs.empty:
        logging.info(f"  - best_configurations.csv ({len(best_configs)} configs)")


if __name__ == '__main__':
    main()
