#!/usr/bin/env python3
"""
Age binning strategies analysis - Reusable functions for notebooks and scripts.

This module provides utilities for:
- Creating various age binning strategies
- Analyzing sensitive attribute distribution within bins
- Computing fairness metrics per binning strategy
- Generating comparison reports

Can be imported by notebooks for detailed analysis or run as standalone script
for quick summary statistics.
"""

import sys
import logging
from pathlib import Path
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from fairxai.data.loaders import load_standardized_raw


def setup_logging(log_dir: Path):
    """Setup logging configuration."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / 'age_binning_analysis.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler()
        ]
    )
    logging.info("Age binning analysis")


def create_binning_strategy(
    df: pd.DataFrame, 
    strategy_name: str,
    age_col: str = 'age_raw',
    **kwargs
) -> Tuple[List[float], Optional[List[str]]]:
    """
    Create bins and labels for a given strategy.
    
    Args:
        df: DataFrame with age data
        strategy_name: Name of strategy ('fixed_10yr', 'quantile_5', etc.)
        age_col: Column name containing numeric age
        **kwargs: Additional parameters (n_bins for quantile, etc.)
    
    Returns:
        Tuple of (bins, labels)
    """
    if strategy_name == 'fixed_10yr':
        bins = [0, 40, 50, 60, 70, 100]
        labels = ["<40", "40-49", "50-59", "60-69", "70+"]
    
    elif strategy_name == 'fixed_5yr':
        bins = [0, 35, 40, 45, 50, 55, 60, 65, 70, 75, 100]
        labels = ["<35", "35-39", "40-44", "45-49", "50-54", 
                  "55-59", "60-64", "65-69", "70-74", "75+"]
    
    elif strategy_name == 'clinical':
        bins = [0, 45, 55, 65, 100]
        labels = ["<45", "45-54", "55-64", "65+"]
    
    elif strategy_name.startswith('quantile'):
        n_bins = kwargs.get('n_bins', 5)
        bins = pd.qcut(df[age_col], q=n_bins, retbins=True, duplicates='drop')[1]
        labels = None  # Auto-generated
    
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")
    
    return bins, labels


def apply_binning(
    df: pd.DataFrame,
    bins: List[float],
    labels: Optional[List[str]],
    age_col: str = 'age_raw',
    output_col: str = 'age_group_exp'
) -> pd.DataFrame:
    """
    Apply binning to a DataFrame.
    
    Args:
        df: Input DataFrame
        bins: Bin edges
        labels: Bin labels (or None for auto-generation)
        age_col: Column with numeric age
        output_col: Name for binned column
    
    Returns:
        DataFrame with new binned column
    """
    df = df.copy()
    df[output_col] = pd.cut(
        df[age_col], 
        bins=bins, 
        labels=labels,
        include_lowest=True
    )
    return df


def sensitive_attribute_distribution(
    df: pd.DataFrame, 
    bin_col: str,
    sensitive_col: str = 'sex'
) -> pd.DataFrame:
    """
    Calculate percentage distribution of sensitive attribute within bins.
    
    Args:
        df: DataFrame with binned data
        bin_col: Column with age bins
        sensitive_col: Sensitive attribute column (e.g., 'sex')
    
    Returns:
        DataFrame with percentages per bin
    """
    grouped = df.groupby(bin_col, observed=True)[sensitive_col].value_counts(
        normalize=True
    ).rename('pct').reset_index()
    
    pivot = grouped.pivot(index=bin_col, columns=sensitive_col, values='pct').fillna(0)
    pivot = (pivot * 100).round(2)
    
    # Rename columns to be more descriptive
    new_cols = {col: f"{col.lower()}_pct" for col in pivot.columns}
    pivot = pivot.rename(columns=new_cols)
    
    return pivot.reset_index()


def compute_fairness_metrics(
    df: pd.DataFrame,
    bin_col: str,
    target_col: str = 'heart_disease'
) -> Dict:
    """
    Compute fairness metrics for a binning strategy.
    
    Args:
        df: DataFrame with binned data and target
        bin_col: Column with age bins
        target_col: Target variable column
    
    Returns:
        Dictionary with fairness metrics
    """
    # Group size statistics
    group_counts = df[bin_col].value_counts()
    
    # Positive rate per group (statistical parity)
    positive_rates = df.groupby(bin_col, observed=True)[target_col].mean()
    
    metrics = {
        'n_groups': len(group_counts),
        'min_group_size': int(group_counts.min()),
        'max_group_size': int(group_counts.max()),
        'mean_group_size': float(group_counts.mean()),
        'group_balance_cv': float(group_counts.std() / group_counts.mean()),
        'max_sp_difference': float(positive_rates.max() - positive_rates.min()),
        'overall_positive_rate': float(df[target_col].mean()),
        'group_sizes': group_counts.to_dict(),
        'positive_rates_by_group': positive_rates.to_dict()
    }
    
    return metrics


def analyze_strategy_comprehensive(
    df: pd.DataFrame,
    strategy_name: str,
    bins: List[float],
    labels: Optional[List[str]],
    dataset_name: str,
    age_col: str = 'age_raw',
    sensitive_col: str = 'sex',
    target_col: str = 'heart_disease'
) -> Dict:
    """
    Comprehensive analysis of a binning strategy.
    
    Combines fairness metrics and sensitive attribute distribution.
    
    Args:
        df: Input DataFrame
        strategy_name: Name of strategy
        bins: Bin edges
        labels: Bin labels
        dataset_name: Dataset identifier
        age_col: Age column name
        sensitive_col: Sensitive attribute column
        target_col: Target variable column
    
    Returns:
        Dictionary with complete analysis
    """
    # Apply binning
    df_binned = apply_binning(df, bins, labels, age_col, 'age_group_exp')
    
    # Fairness metrics
    fairness = compute_fairness_metrics(df_binned, 'age_group_exp', target_col)
    
    # Sensitive attribute distribution
    sensitive_dist = sensitive_attribute_distribution(
        df_binned, 'age_group_exp', sensitive_col
    )
    
    # Overall sensitive attribute distribution
    overall_sensitive = df[sensitive_col].value_counts(normalize=True)
    overall_sensitive_pct = {k: round(v*100, 2) for k, v in overall_sensitive.items()}
    
    return {
        'dataset': dataset_name,
        'strategy': strategy_name,
        'fairness_metrics': fairness,
        'sensitive_distribution': sensitive_dist.to_dict(orient='records'),
        'overall_sensitive_distribution': overall_sensitive_pct,
        'bins': [float(b) for b in bins],
        'labels': labels if labels else "auto-generated"
    }


def compare_strategies(
    results: List[Dict],
    by_dataset: bool = True
) -> pd.DataFrame:
    """
    Create comparison table across strategies.
    
    Args:
        results: List of analysis results
        by_dataset: Whether to separate by dataset
    
    Returns:
        Comparison DataFrame
    """
    comparison_data = []
    
    for result in results:
        metrics = result['fairness_metrics']
        comparison_data.append({
            'dataset': result['dataset'],
            'strategy': result['strategy'],
            'n_groups': metrics['n_groups'],
            'min_group_size': metrics['min_group_size'],
            'max_group_size': metrics['max_group_size'],
            'group_balance_cv': round(metrics['group_balance_cv'], 3),
            'max_sp_difference': round(metrics['max_sp_difference'], 3),
        })
    
    df = pd.DataFrame(comparison_data)
    
    if by_dataset:
        df = df.sort_values(['dataset', 'strategy'])
    else:
        df = df.sort_values(['strategy', 'dataset'])
    
    return df


def generate_summary_report(
    results: List[Dict],
    output_file: Path
):
    """
    Generate markdown summary report.
    
    Args:
        results: List of analysis results
        output_file: Path to save report
    """
    comparison = compare_strategies(results)
    
    report = ["# Age Binning Strategy Analysis Report\n"]
    report.append("## Overview\n")
    report.append(f"- Datasets analyzed: {len(set(r['dataset'] for r in results))}")
    report.append(f"- Strategies tested: {len(set(r['strategy'] for r in results))}")
    report.append(f"- Total configurations: {len(results)}\n")
    
    report.append("## Comparison Table\n")
    report.append("```")
    report.append(comparison.to_string(index=False))
    report.append("```\n")
    
    report.append("## Interpretation Guidelines\n")
    report.append("### Statistical Parity Difference:")
    report.append("- < 0.10: Acceptable fairness")
    report.append("- 0.10-0.20: Moderate concern")
    report.append("- > 0.20: Significant fairness violation\n")
    
    report.append("### Group Balance (Coefficient of Variation):")
    report.append("- < 0.30: Well-balanced groups")
    report.append("- 0.30-0.60: Moderate imbalance")
    report.append("- > 0.60: Severe imbalance (risk of small sample issues)\n")
    
    with open(output_file, 'w') as f:
        f.write('\n'.join(report))


def main():
    """Main execution for standalone script."""
    root = Path(__file__).parent.parent.parent
    results_dir = root / 'results/experiments/binning'
    logs_dir = root / 'logs/cardiac'

    setup_logging(logs_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    datasets = ['cleveland', 'kaggle_heart']
    strategies = ['fixed_10yr', 'quantile_5']  # Quick summary
    
    all_results = []

    for ds_name in datasets:
        logging.info(f"\nAnalyzing dataset: {ds_name}")
        df = load_standardized_raw(ds_name, str(root))
        
        if 'sex' not in df.columns or 'age_raw' not in df.columns:
            logging.warning(f"Missing columns in {ds_name}; skipping")
            continue

        for strategy_name in strategies:
            logging.info(f"  Testing strategy: {strategy_name}")
            
            # Create binning
            if strategy_name.startswith('quantile'):
                n_bins = int(strategy_name.split('_')[1])
                bins, labels = create_binning_strategy(
                    df, strategy_name, n_bins=n_bins
                )
            else:
                bins, labels = create_binning_strategy(df, strategy_name)
            
            # Comprehensive analysis
            result = analyze_strategy_comprehensive(
                df, strategy_name, bins, labels, ds_name
            )
            all_results.append(result)
            
            # Log sensitive attribute distribution
            logging.info(f"    Sensitive attribute distribution:")
            for record in result['sensitive_distribution']:
                logging.info(f"      {record}")

    # Save results
    comparison_df = compare_strategies(all_results)
    comparison_df.to_csv(results_dir / 'quick_comparison.csv', index=False)
    
    summary_file = results_dir / 'age_binning_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    report_file = results_dir / 'age_binning_report.md'
    generate_summary_report(all_results, report_file)
    
    logging.info(f"\n✓ Analysis complete. Results saved to: {results_dir}")
    logging.info(f"  - Comparison: quick_comparison.csv")
    logging.info(f"  - Full results: age_binning_summary.json")
    logging.info(f"  - Report: age_binning_report.md")


if __name__ == '__main__':
    main()
