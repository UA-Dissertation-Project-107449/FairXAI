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
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from fairxai.experiments.versioning import ExperimentVersioning
from fairxai.cli.runner_base import get_project_root, setup_phase_logging
from fairxai.visualization.plots import (
    save_comparison_heatmap,
    save_tradeoff_scatter,
    save_pareto_frontier
)

# Composite score weights (must sum to 1.0)
SCORE_WEIGHTS = {
    'f1_value': 0.40,
    'recall_value': 0.30,
    'accuracy_value': 0.20,
    'auc_value': 0.10
}


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
                'status': results['execution']['status'],
                'error': results['execution'].get('error')
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
                    # Unified metric columns for comparisons
                    if 'f1_score' in cv_results:
                        row['f1_value'] = cv_results['f1_score']['mean']
                    if 'accuracy' in cv_results:
                        row['accuracy_value'] = cv_results['accuracy']['mean']
                    if 'recall' in cv_results:
                        row['recall_value'] = cv_results['recall']['mean']
                    if 'precision' in cv_results:
                        row['precision_value'] = cv_results['precision']['mean']
                    if 'auc_roc' in cv_results:
                        row['auc_value'] = cv_results['auc_roc']['mean']
                else:
                    # Single split results
                    test_metrics = results.get('test_metrics', {})
                    for metric_name, value in test_metrics.items():
                        row[metric_name] = value
                    # Unified metric columns for comparisons
                    row['f1_value'] = test_metrics.get('f1_score')
                    row['accuracy_value'] = test_metrics.get('accuracy')
                    row['recall_value'] = test_metrics.get('recall')
                    row['precision_value'] = test_metrics.get('precision')
                    row['auc_value'] = test_metrics.get('auc_roc')

                # Standardize metric columns to reduce missing values
                row['accuracy'] = row.get('accuracy_value', row.get('accuracy'))
                row['precision'] = row.get('precision_value', row.get('precision'))
                row['recall'] = row.get('recall_value', row.get('recall'))
                row['f1_score'] = row.get('f1_value', row.get('f1_score'))
                row['auc_roc'] = row.get('auc_value', row.get('auc_roc'))
                
                # Add fairness metrics
                fairness = results.get('fairness_metrics', {})
                if fairness:
                    dp_diffs = []
                    eq_diffs = []

                    # New structure: group_fairness -> {attr} -> {demographic_parity, equalized_odds}
                    group_fairness = fairness.get('group_fairness', {})
                    if group_fairness:
                        for attr, metrics in group_fairness.items():
                            dp = metrics.get('demographic_parity') if isinstance(metrics, dict) else None
                            if isinstance(dp, dict):
                                max_diff = dp.get('max_difference', np.nan)
                                row[f'dem_parity_{attr}_max_diff'] = max_diff
                                if pd.notna(max_diff):
                                    dp_diffs.append(max_diff)

                            eq = metrics.get('equalized_odds') if isinstance(metrics, dict) else None
                            if isinstance(eq, dict):
                                tpr_diff = eq.get('tpr_max_difference', eq.get('tpr_difference', np.nan))
                                fpr_diff = eq.get('fpr_max_difference', eq.get('fpr_difference', np.nan))
                                row[f'eq_odds_{attr}_tpr_diff'] = tpr_diff
                                row[f'eq_odds_{attr}_fpr_diff'] = fpr_diff
                                if pd.notna(tpr_diff):
                                    eq_diffs.append(tpr_diff)
                                if pd.notna(fpr_diff):
                                    eq_diffs.append(fpr_diff)

                    # Legacy structure: fairness_metrics -> demographic_parity / equalized_odds
                    if 'demographic_parity' in fairness:
                        for attr, metrics in fairness['demographic_parity'].items():
                            max_diff = metrics.get('max_difference', np.nan)
                            row[f'dem_parity_{attr}_max_diff'] = max_diff
                            if pd.notna(max_diff):
                                dp_diffs.append(max_diff)

                    if 'equalized_odds' in fairness:
                        for attr, metrics in fairness['equalized_odds'].items():
                            tpr_diff = metrics.get('tpr_difference', np.nan)
                            fpr_diff = metrics.get('fpr_difference', np.nan)
                            row[f'eq_odds_{attr}_tpr_diff'] = tpr_diff
                            row[f'eq_odds_{attr}_fpr_diff'] = fpr_diff
                            if pd.notna(tpr_diff):
                                eq_diffs.append(tpr_diff)
                            if pd.notna(fpr_diff):
                                eq_diffs.append(fpr_diff)

                    if dp_diffs:
                        row['dp_max_diff'] = float(np.nanmax(dp_diffs))
                    if eq_diffs:
                        row['eq_odds_max_diff'] = float(np.nanmax(eq_diffs))
            
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
        'accuracy_value': ['mean', 'std', 'min', 'max'],
        'recall_value': ['mean', 'std', 'min', 'max'],
        'f1_value': ['mean', 'std', 'min', 'max'],
        'auc_value': ['mean', 'std', 'min', 'max'],
        'score_value': ['mean', 'std', 'min', 'max']
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
    metric_col = 'score_value'
    
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
    logging.info(f"[SUCCESS] Saved binning comparison: {output_file}")

    # Heatmap
    heatmap_file = output_dir / 'binning_comparison_heatmap.png'
    save_comparison_heatmap(
        pivot,
        title='Binning Strategy Comparison (Composite Score)',
        output_file=heatmap_file
    )
    logging.info(f"[SUCCESS] Saved binning heatmap: {heatmap_file}")
    
    return pivot


def compare_mitigation_techniques(df: pd.DataFrame, output_dir: Path):
    """
    Create mitigation technique comparison table.
    
    Pivot table: mitigation_technique × dataset with performance metrics
    """
    if df.empty:
        logging.warning("No data for mitigation comparison")
        return
    
    metric_col = 'score_value'
    
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
    logging.info(f"[SUCCESS] Saved mitigation comparison: {output_file}")

    # Heatmap
    heatmap_file = output_dir / 'mitigation_comparison_heatmap.png'
    save_comparison_heatmap(
        pivot,
        title='Mitigation Technique Comparison (Composite Score)',
        output_file=heatmap_file
    )
    logging.info(f"[SUCCESS] Saved mitigation heatmap: {heatmap_file}")
    
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
        metric_col = 'score_value'
        baseline_score = baseline_row[metric_col].values[0]
        
        # Filter non-baseline techniques
        mitigated_df = dataset_df[dataset_df['mitigation_technique'] != 'baseline']
        
        for _, row in mitigated_df.iterrows():
            score = row[metric_col]
            fairness_gap = row.get('fairness_gap')
            
            # Check performance threshold
            if baseline_score == 0 or pd.isna(baseline_score) or pd.isna(score):
                continue
            performance_drop = (baseline_score - score) / baseline_score
            
            if fairness_gap is None or pd.isna(fairness_gap):
                continue
            if fairness_gap > fairness_threshold:
                continue

            if performance_drop <= performance_threshold:
                # Add to best configs
                best_configs.append({
                    'dataset': dataset,
                    'binning': row['binning_strategy'],
                    'mitigation': row['mitigation_technique'],
                    'training_method': row['training_method'],
                    'score': score,
                    'baseline_score': baseline_score,
                    'score_drop_pct': performance_drop * 100,
                    'fairness_gap': fairness_gap,
                    'baseline_fairness_gap': row.get('baseline_fairness_gap'),
                    'fairness_gain_score': row.get('fairness_gain_score'),
                    'fairness_gain_pct': row.get('fairness_gain_pct'),
                    'f1_score': row.get('f1_value'),
                    'recall': row.get('recall_value'),
                    'accuracy': row.get('accuracy_value'),
                    'auc_roc': row.get('auc_value'),
                    'experiment_id': row['experiment_id']
                })
    
    best_df = pd.DataFrame(best_configs)
    
    if not best_df.empty:
        # Sort by F1 score
        best_df = best_df.sort_values(['dataset', 'score'], ascending=[True, False])
        
        output_file = output_dir / 'best_configurations.csv'
        best_df.to_csv(output_file, index=False)
        logging.info(f"[SUCCESS] Saved best configurations: {output_file}")
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
        default=None,
        help='Base results directory'
    )
    parser.add_argument(
        '--pipeline',
        type=str,
        default='cardiac',
        help='Pipeline name (e.g., cardiac, dermatology)'
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
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Disable plot generation'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Verbose console output'
    )
    
    args = parser.parse_args()
    project_root = get_project_root(Path(__file__))
    setup_phase_logging(project_root, 'experiment_comparison.log', verbose=args.verbose, log_subdir='experiments/latest_run')
    
    logging.info("="*80)
    logging.info("EXPERIMENT COMPARISON")
    logging.info("="*80)
    logging.info("[PHASE] Comparison started")
    
    results_dir = Path(args.results_dir) if args.results_dir else (project_root / f"results/{args.pipeline}/experiments/full")
    # Initialize versioning
    versioning = ExperimentVersioning(results_dir)
    
    # Check if latest_run exists
    if not versioning.latest_dir.exists():
        logging.error(f"No latest_run found in {results_dir}")
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

    # Compute composite score for ranking
    for metric, weight in SCORE_WEIGHTS.items():
        if metric not in df_success.columns:
            df_success[metric] = np.nan

    df_success['score_value'] = (
        df_success['f1_value'] * SCORE_WEIGHTS['f1_value'] +
        df_success['recall_value'] * SCORE_WEIGHTS['recall_value'] +
        df_success['accuracy_value'] * SCORE_WEIGHTS['accuracy_value'] +
        df_success['auc_value'] * SCORE_WEIGHTS['auc_value']
    )

    # Compute fairness gap for trade-off analysis
    fairness_cols = ['dp_max_diff', 'eq_odds_max_diff']
    for col in fairness_cols:
        if col not in df_success.columns:
            df_success[col] = np.nan
    df_success['fairness_gap'] = df_success[fairness_cols].max(axis=1, skipna=True)

    # Compute fairness gains per metric vs baseline (same dataset/binning/training_method)
    fairness_metric_cols = [
        c for c in df_success.columns
        if c.startswith('dem_parity_') or c.startswith('eq_odds_')
    ]

    baseline_lookup = {}
    for _, row in df_success[df_success['mitigation_technique'] == 'baseline'].iterrows():
        key = (row['dataset'], row['binning_strategy'], row['training_method'])
        baseline_lookup[key] = row

    for col in fairness_metric_cols:
        gain_col = f'gain_{col}'
        df_success[gain_col] = np.nan

    df_success['baseline_fairness_gap'] = np.nan
    df_success['fairness_gain_score'] = np.nan
    df_success['fairness_gain_pct'] = np.nan

    for idx, row in df_success.iterrows():
        key = (row['dataset'], row['binning_strategy'], row['training_method'])
        baseline = baseline_lookup.get(key)
        if baseline is None:
            continue

        gains = []
        for col in fairness_metric_cols:
            base_val = baseline.get(col)
            curr_val = row.get(col)
            if pd.isna(base_val) or pd.isna(curr_val):
                continue
            gain = base_val - curr_val
            df_success.at[idx, f'gain_{col}'] = gain
            gains.append(gain)

        base_gap = baseline.get('fairness_gap')
        df_success.at[idx, 'baseline_fairness_gap'] = base_gap

        if gains:
            gain_score = float(np.mean(gains))
            df_success.at[idx, 'fairness_gain_score'] = gain_score
            if base_gap and base_gap > 0:
                df_success.at[idx, 'fairness_gain_pct'] = gain_score / base_gap
    
    # Save full results table
    full_results_file = output_dir / 'full_comparison.csv'
    df_success.to_csv(full_results_file, index=False)
    logging.info(f"\n[SUCCESS] Saved full results: {full_results_file}")
    
    # Create comparison tables
    logging.info("\nGenerating comparison tables...")
    
    compare_binning_strategies(df_success, output_dir)
    compare_mitigation_techniques(df_success, output_dir)
    
    # Filter best configurations
    logging.info("\nFiltering best configurations...")
    best_configs = filter_best_configurations(
        df_success,
        output_dir,
        args.fairness_threshold,
        args.performance_threshold
    )

    # Diagnostics: repeated metrics
    dup_cols = [
        'dataset', 'training_method', 'binning_strategy',
        'accuracy_value', 'recall_value', 'f1_value', 'auc_value'
    ]
    dup_groups = (
        df_success.groupby(dup_cols, dropna=False)
        .agg(
            count=('experiment_id', 'count'),
            techniques=('mitigation_technique', lambda x: ','.join(sorted(set(x))))
        )
        .reset_index()
    )
    dup_groups = dup_groups[dup_groups['count'] > 1]
    dup_file = output_dir / 'diagnostics_duplicate_metrics.csv'
    dup_groups.to_csv(dup_file, index=False)
    logging.info(f"[SUCCESS] Saved duplicate metrics diagnostics: {dup_file}")

    # Diagnostics: failures (e.g., non-convergence)
    failures = df[df['status'] == 'failed'].copy()
    if not failures.empty:
        failure_file = output_dir / 'diagnostics_failures.csv'
        failures.to_csv(failure_file, index=False)
        logging.info(f"[SUCCESS] Saved failure diagnostics: {failure_file}")

    # Diagnostics: experiment duplication counts
    combo_counts = (
        df.groupby(['dataset', 'binning_strategy', 'mitigation_technique', 'training_method'])
        .agg(count=('experiment_id', 'count'))
        .reset_index()
    )
    combo_file = output_dir / 'diagnostics_experiment_counts.csv'
    combo_counts.to_csv(combo_file, index=False)
    logging.info(f"[SUCCESS] Saved experiment counts: {combo_file}")

    # Trade-off visuals (per dataset)
    if not args.no_plots:
        for dataset in sorted(df_success['dataset'].unique()):
            subset = df_success[df_success['dataset'] == dataset].copy()
            if subset.empty:
                continue

            tradeoff_base = output_dir / f"tradeoff_{dataset}"
            tradeoff_csv = tradeoff_base.with_suffix('.csv')
            subset.to_csv(tradeoff_csv, index=False)

            tradeoff_png = tradeoff_base.with_suffix('.png')
            tradeoff_result = save_tradeoff_scatter(
                subset,
                x_col='score_value',
                y_col='fairness_gap',
                hue_col='mitigation_technique',
                style_col='training_method',
                title=f"{dataset} - Performance vs Fairness",
                output_file=tradeoff_png
            )
            if tradeoff_result:
                logging.info(f"[SUCCESS] Saved tradeoff plot: {tradeoff_png}")
            else:
                logging.warning(f"Tradeoff plot skipped (no data): {tradeoff_png}")

            pareto_base = output_dir / f"pareto_{dataset}"
            pareto_csv = pareto_base.with_suffix('.csv')
            subset.to_csv(pareto_csv, index=False)

            pareto_png = pareto_base.with_suffix('.png')
            pareto_result = save_pareto_frontier(
                subset,
                x_col='score_value',
                y_col='fairness_gap',
                title=f"{dataset} - Pareto Frontier",
                output_file=pareto_png
            )
            if pareto_result:
                logging.info(f"[SUCCESS] Saved pareto plot: {pareto_png}")
            else:
                logging.warning(f"Pareto plot skipped (no data): {pareto_png}")

    # Summary outputs
    summary_rows = []
    for dataset in sorted(df_success['dataset'].unique()):
        subset = df_success[df_success['dataset'] == dataset].copy()
        if subset.empty:
            continue
        best_score = subset.sort_values('score_value', ascending=False).head(1)
        best_gain = subset.sort_values('fairness_gain_score', ascending=False).head(1)
        summary_rows.append({
            'dataset': dataset,
            'best_score_experiment': best_score['experiment_id'].values[0],
            'best_score': best_score['score_value'].values[0],
            'best_score_fairness_gap': best_score['fairness_gap'].values[0],
            'best_gain_experiment': best_gain['experiment_id'].values[0],
            'best_fairness_gain': best_gain['fairness_gain_score'].values[0],
            'best_gain_score': best_gain['score_value'].values[0]
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = output_dir / 'summary_top_configs.csv'
    summary_df.to_csv(summary_csv, index=False)
    logging.info(f"[SUCCESS] Saved summary: {summary_csv}")
    
    # Print summary
    logging.info("\n" + "="*80)
    logging.info("COMPARISON COMPLETE")
    logging.info("="*80)
    logging.info("[PHASE] Comparison complete")
    logging.info(f"Results saved to: {output_dir}")
    logging.info(f"\nFiles created:")
    logging.info(f"  - full_comparison.csv ({len(df_success)} experiments)")
    logging.info(f"  - binning_comparison.csv")
    logging.info(f"  - mitigation_comparison.csv")
    if best_configs is not None and not best_configs.empty:
        logging.info(f"  - best_configurations.csv ({len(best_configs)} configs)")


if __name__ == '__main__':
    main()