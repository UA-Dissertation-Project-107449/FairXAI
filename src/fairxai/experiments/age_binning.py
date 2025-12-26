"""
Age binning strategies analysis - Reusable functions for experiments.

This module provides utilities for:
- Creating various age binning strategies
- Analyzing sensitive attribute distribution within bins
- Computing fairness metrics per binning strategy
- Generating comparison reports

Designed to be imported by notebooks and experiment scripts.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path


logger = logging.getLogger(__name__)

# Constants for scoring and interpretation
MAX_EXPECTED_SP_DIFF = 0.5  # Typical max statistical parity difference for age groups
MAX_EXPECTED_CV = 1.0  # Typical max coefficient of variation for reasonable binnings
MIN_SAMPLE_SIZE = 30  # Minimum recommended group size for statistical validity


def create_binning_strategy(
    df: pd.DataFrame, 
    strategy_name: str,
    age_col: str = 'age_raw',
    **kwargs
) -> Tuple[List[float], Optional[List[str]]]:
    """
    Create bins and labels for a given age binning strategy.
    
    Supported strategies:
    - 'fixed_10yr': Fixed 10-year intervals [<40, 40-49, 50-59, 60-69, 70+]
    - 'fixed_5yr': Fixed 5-year intervals [<35, 35-39, ..., 75+]
    - 'clinical': Clinical age groups [<45, 45-54, 55-64, 65+]
    - 'quantile_N': N quantile-based bins (data-driven)
    
    Args:
        df: DataFrame with age data
        strategy_name: Name of strategy
        age_col: Column name containing numeric age
        **kwargs: Additional parameters (n_bins for quantile strategies)
    
    Returns:
        Tuple of (bins, labels). Labels may be None for auto-generation.
    
    Example:
        >>> bins, labels = create_binning_strategy(df, 'fixed_10yr')
        >>> df['age_group'] = pd.cut(df['age_raw'], bins=bins, labels=labels)
    """
    logger.info(f"Creating binning strategy: {strategy_name}")
    
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
        # Extract number of bins from strategy name (e.g., 'quantile_5' -> 5)
        if '_' in strategy_name:
            n_bins = int(strategy_name.split('_')[1])
        else:
            n_bins = kwargs.get('n_bins', 5)
        
        logger.info(f"  Computing {n_bins} quantile bins from data")
        
        try:
            bins = pd.qcut(df[age_col], q=n_bins, retbins=True, duplicates='drop')[1]
            actual_bins = len(bins) - 1
            if actual_bins < n_bins:
                logger.warning(f"  Only created {actual_bins} bins due to duplicate edges")
        except ValueError as e:
            logger.error(f"  Failed to create {n_bins} quantile bins: {e}")
            logger.info(f"  Falling back to {n_bins-1} bins")
            try:
                bins = pd.qcut(df[age_col], q=n_bins-1, retbins=True, duplicates='drop')[1]
            except ValueError:
                raise ValueError(f"Cannot create quantile bins for {strategy_name}. "
                               f"Dataset may be too small or have insufficient unique values.")
        
        labels = None  # Auto-generated labels like "(29.0, 45.0]"
    
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}. "
                        f"Supported: fixed_10yr, fixed_5yr, clinical, quantile_N")
    
    logger.info(f"  Created {len(bins)-1} bins")
    return bins, labels


def apply_binning(
    df: pd.DataFrame,
    bins: List[float],
    labels: Optional[List[str]],
    age_col: str = 'age_raw',
    output_col: str = 'age_group_exp'
) -> pd.DataFrame:
    """
    Apply age binning to a DataFrame.
    
    Args:
        df: Input DataFrame
        bins: Bin edges (from create_binning_strategy)
        labels: Bin labels (or None for auto-generation)
        age_col: Column containing numeric age values
        output_col: Name for the new binned age column
    
    Returns:
        DataFrame with new binned column added
    
    Example:
        >>> bins, labels = create_binning_strategy(df, 'clinical')
        >>> df_binned = apply_binning(df, bins, labels)
        >>> print(df_binned['age_group_exp'].value_counts())
    """
    df = df.copy()
    df[output_col] = pd.cut(
        df[age_col], 
        bins=bins, 
        labels=labels,
        include_lowest=True
    )
    logger.info(f"Applied binning: {df[output_col].nunique()} unique groups created")
    return df


def sensitive_attribute_distribution(
    df: pd.DataFrame, 
    bin_col: str,
    sensitive_col: str = 'sex'
) -> pd.DataFrame:
    """
    Calculate percentage distribution of sensitive attribute within each bin.
    
    Useful for understanding if certain sensitive groups are concentrated
    in specific age bins (representation bias).
    
    Args:
        df: DataFrame with binned age data
        bin_col: Column with age bins
        sensitive_col: Sensitive attribute column (e.g., 'sex')
    
    Returns:
        DataFrame with percentage distribution per bin
        Columns: [bin_col, <sensitive_value_1>_pct, <sensitive_value_2>_pct, ...]
    
    Example:
        >>> dist = sensitive_attribute_distribution(df, 'age_group_exp', 'sex')
        >>> print(dist)
        age_group_exp  female_pct  male_pct
        <40                  45.2      54.8
        40-49                38.1      61.9
    """
    # Validation
    if len(df) == 0:
        raise ValueError("DataFrame is empty")
    
    if df[bin_col].isna().any():
        n_missing = df[bin_col].isna().sum()
        logger.warning(f"Found {n_missing} NaN values in {bin_col}, they will be excluded")
        df = df[df[bin_col].notna()]
    
    grouped = df.groupby(bin_col, observed=True)[sensitive_col].value_counts(
        normalize=True
    ).rename('pct').reset_index()
    
    pivot = grouped.pivot(index=bin_col, columns=sensitive_col, values='pct').fillna(0)
    pivot = (pivot * 100).round(2)
    
    # Rename columns to be more descriptive
    new_cols = {col: f"{str(col).lower()}_pct" for col in pivot.columns}
    pivot = pivot.rename(columns=new_cols)
    
    logger.info(f"Calculated sensitive attribute distribution for {len(pivot)} bins")
    return pivot.reset_index()


def compute_fairness_metrics(
    df: pd.DataFrame,
    bin_col: str,
    target_col: str = 'heart_disease'
) -> Dict:
    """
    Compute fairness-relevant metrics for an age binning strategy.
    
    Metrics include:
    - Group size statistics (min, max, mean, balance coefficient of variation)
    - Statistical parity: positive rate per group and max difference
    - Overall positive rate for context
    
    Args:
        df: DataFrame with binned age data and target
        bin_col: Column with age bins
        target_col: Binary target variable column
    
    Returns:
        Dictionary with fairness metrics
        Keys: n_groups, min_group_size, max_group_size, mean_group_size,
              group_balance_cv, max_sp_difference, overall_positive_rate,
              group_sizes (dict), positive_rates_by_group (dict)
    
    Example:
        >>> metrics = compute_fairness_metrics(df, 'age_group_exp')
        >>> print(f"Max statistical parity diff: {metrics['max_sp_difference']:.3f}")
    """
    # Group size statistics
    group_counts = df[bin_col].value_counts()
    
    # Positive rate per group (statistical parity indicator)
    positive_rates = df.groupby(bin_col, observed=True)[target_col].mean()
    
    # Coefficient of variation for group balance (lower = more balanced)
    cv = float(group_counts.std() / group_counts.mean())
    
    # Statistical parity difference (max - min positive rate)
    sp_diff = float(positive_rates.max() - positive_rates.min())
    
    metrics = {
        'n_groups': len(group_counts),
        'min_group_size': int(group_counts.min()),
        'max_group_size': int(group_counts.max()),
        'mean_group_size': float(group_counts.mean()),
        'group_balance_cv': cv,
        'max_sp_difference': sp_diff,
        'overall_positive_rate': float(df[target_col].mean()),
        'group_sizes': {str(k): int(v) for k, v in group_counts.to_dict().items()},
        'positive_rates_by_group': {str(k): float(v) for k, v in positive_rates.to_dict().items()}
    }
    
    logger.info(f"Computed fairness metrics: {metrics['n_groups']} groups, "
                f"SP diff={sp_diff:.3f}, balance CV={cv:.3f}")
    
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
    Comprehensive analysis of a single age binning strategy.
    
    Combines:
    - Fairness metrics (group balance, statistical parity)
    - Sensitive attribute distribution within bins
    - Overall population statistics
    
    Args:
        df: Input DataFrame
        strategy_name: Name of binning strategy
        bins: Bin edges
        labels: Bin labels (or None)
        dataset_name: Dataset identifier for tracking
        age_col: Age column name
        sensitive_col: Sensitive attribute column
        target_col: Target variable column
    
    Returns:
        Dictionary with complete analysis results
        Keys: dataset, strategy, fairness_metrics, sensitive_distribution,
              overall_sensitive_distribution, bins, labels
    
    Example:
        >>> bins, labels = create_binning_strategy(df, 'clinical')
        >>> analysis = analyze_strategy_comprehensive(
        ...     df, 'clinical', bins, labels, 'cleveland'
        ... )
        >>> print(analysis['fairness_metrics']['max_sp_difference'])
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Analyzing strategy: {strategy_name} on {dataset_name}")
    logger.info(f"{'='*60}")
    
    # Apply binning
    df_binned = apply_binning(df, bins, labels, age_col, 'age_group_exp')
    
    # Fairness metrics
    fairness = compute_fairness_metrics(df_binned, 'age_group_exp', target_col)
    
    # Sensitive attribute distribution within bins
    sensitive_dist = sensitive_attribute_distribution(
        df_binned, 'age_group_exp', sensitive_col
    )
    
    # Overall sensitive attribute distribution (for context)
    overall_sensitive = df[sensitive_col].value_counts(normalize=True)
    overall_sensitive_pct = {str(k): round(v*100, 2) for k, v in overall_sensitive.items()}
    
    result = {
        'dataset': dataset_name,
        'strategy': strategy_name,
        'fairness_metrics': fairness,
        'sensitive_distribution': sensitive_dist.to_dict(orient='records'),
        'overall_sensitive_distribution': overall_sensitive_pct,
        'bins': [float(b) for b in bins],
        'labels': labels if labels else "auto-generated"
    }
    
    logger.info(f"Analysis complete for {strategy_name}")
    
    return result


def compare_strategies(
    results: List[Dict],
    by_dataset: bool = True
) -> pd.DataFrame:
    """
    Create comparison table across multiple binning strategies.
    
    Extracts key metrics from analysis results for easy comparison.
    
    Args:
        results: List of analysis results from analyze_strategy_comprehensive
        by_dataset: Whether to group by dataset first
    
    Returns:
        Comparison DataFrame with columns:
        [dataset, strategy, n_groups, min_group_size, max_group_size,
         group_balance_cv, max_sp_difference]
    
    Example:
        >>> results = []
        >>> for strategy in ['fixed_10yr', 'clinical', 'quantile_5']:
        ...     bins, labels = create_binning_strategy(df, strategy)
        ...     result = analyze_strategy_comprehensive(df, strategy, bins, labels, 'cleveland')
        ...     results.append(result)
        >>> comparison = compare_strategies(results)
        >>> print(comparison.to_string(index=False))
    """
    logger.info(f"Creating comparison table for {len(results)} strategy configurations")
    
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
    
    logger.info(f"Comparison table created with {len(df)} rows")
    
    return df


def compute_strategy_score(
    result: Dict,
    sample_size_weight: float = 0.40,
    balance_weight: float = 0.30,
    fairness_weight: float = 0.30
) -> float:
    """
    Compute overall score for a binning strategy.
    
    Score components:
    1. Sample size: Penalize small min group size (need statistical power)
    2. Balance: Reward low CV (even group sizes)
    3. Fairness: Reward low statistical parity difference
    
    Args:
        result: Analysis result from analyze_strategy_comprehensive
        sample_size_weight: Weight for sample size component [0-1]
        balance_weight: Weight for balance component [0-1]
        fairness_weight: Weight for fairness component [0-1]
    
    Returns:
        Overall score (higher is better)
    
    Note:
        Weights should sum to 1.0 for interpretable scores
    
    Example:
        >>> score = compute_strategy_score(result, 0.4, 0.3, 0.3)
        >>> print(f"Strategy score: {score:.3f}")
    """
    metrics = result['fairness_metrics']
    
    # Sample size score: min_group_size / mean_group_size (normalize to [0,1])
    sample_score = min(metrics['min_group_size'] / metrics['mean_group_size'], 1.0)
    
    # Balance score: 1 - (CV / MAX_EXPECTED_CV)
    balance_score = max(1.0 - (metrics['group_balance_cv'] / MAX_EXPECTED_CV), 0.0)
    
    # Fairness score: 1 - (sp_diff / MAX_EXPECTED_SP_DIFF)
    fairness_score = max(1.0 - (metrics['max_sp_difference'] / MAX_EXPECTED_SP_DIFF), 0.0)
    
    # Weighted combination
    overall_score = (
        sample_size_weight * sample_score +
        balance_weight * balance_score +
        fairness_weight * fairness_score
    )
    
    logger.info(f"Computed score for {result['strategy']}: {overall_score:.3f} "
                f"(sample={sample_score:.3f}, balance={balance_score:.3f}, "
                f"fairness={fairness_score:.3f})")
    
    return overall_score


def generate_summary_report(
    results: List[Dict],
    output_file: Path,
    scoring_weights: Dict[str, float] = None
):
    """
    Generate markdown summary report for age binning analysis.
    
    Includes:
    - Overview statistics
    - Comparison table
    - Strategy scores
    - Interpretation guidelines
    
    Args:
        results: List of analysis results
        output_file: Path to save markdown report
        scoring_weights: Optional dict with 'sample_size', 'balance', 'fairness' weights
    
    Example:
        >>> generate_summary_report(results, Path('results/age_binning_report.md'))
    """
    logger.info(f"Generating summary report: {output_file}")
    
    if scoring_weights is None:
        scoring_weights = {'sample_size': 0.40, 'balance': 0.30, 'fairness': 0.30}
    
    # Compute scores for ranking
    for result in results:
        result['score'] = compute_strategy_score(
            result,
            scoring_weights['sample_size'],
            scoring_weights['balance'],
            scoring_weights['fairness']
        )
    
    comparison = compare_strategies(results)
    
    # Add scores to comparison
    scores = {(r['dataset'], r['strategy']): r['score'] for r in results}
    comparison['score'] = comparison.apply(
        lambda row: scores.get((row['dataset'], row['strategy']), 0.0),
        axis=1
    )
    comparison['score'] = comparison['score'].round(3)
    
    # Generate report content
    report = ["# Age Binning Strategy Analysis Report\n"]
    
    report.append("## Overview\n")
    report.append(f"- **Datasets analyzed**: {len(set(r['dataset'] for r in results))}")
    report.append(f"- **Strategies tested**: {len(set(r['strategy'] for r in results))}")
    report.append(f"- **Total configurations**: {len(results)}\n")
    
    report.append("## Scoring Weights\n")
    report.append(f"- Sample Size: {scoring_weights['sample_size']:.0%}")
    report.append(f"- Group Balance: {scoring_weights['balance']:.0%}")
    report.append(f"- Fairness Sensitivity: {scoring_weights['fairness']:.0%}\n")
    
    report.append("## Comparison Table\n")
    report.append("```")
    report.append(comparison.to_string(index=False))
    report.append("```\n")
    
    report.append("## Top Strategies by Score\n")
    top_strategies = comparison.nlargest(5, 'score')
    report.append("```")
    report.append(top_strategies[['dataset', 'strategy', 'score']].to_string(index=False))
    report.append("```\n")
    
    report.append("## Interpretation Guidelines\n")
    report.append("### Statistical Parity Difference:")
    report.append("- **< 0.10**: Low bias - groups have similar positive rates")
    report.append("- **0.10-0.20**: Moderate bias - may require mitigation")
    report.append("- **> 0.20**: High bias - strong fairness concerns\n")
    
    report.append("### Group Balance CV (Coefficient of Variation):")
    report.append("- **< 0.30**: Well-balanced groups")
    report.append("- **0.30-0.50**: Moderate imbalance")
    report.append("- **> 0.50**: Highly imbalanced - consider different strategy\n")
    
    report.append("### Sample Size:")
    report.append("- **min_group_size ≥ 30**: Adequate for statistical tests")
    report.append("- **min_group_size < 30**: Risk of unstable estimates\n")
    
    report.append("## Recommendations\n")
    report.append("1. **Prioritize strategies with score ≥ 0.70**")
    report.append("2. **Ensure min_group_size ≥ 30** for statistical validity")
    report.append("3. **Test top strategies** across both datasets for consistency")
    report.append("4. **Consider clinical interpretability** in final selection\n")
    
    # Write report
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        f.write('\n'.join(report))
    
    logger.info(f"✓ Summary report saved: {output_file}")
