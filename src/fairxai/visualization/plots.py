"""Reusable plotting functions for fairness analysis and model evaluation."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any


def plot_dataset_characteristics(datasets: Dict[str, pd.DataFrame], 
                                 target_col: str = 'heart_disease',
                                 age_col: str = 'age_group',
                                 sex_col: str = 'sex'):
    """
    Plot dataset characteristics including target, age, and sex distributions.
    
    Args:
        datasets: Dictionary mapping dataset names to DataFrames
        target_col: Name of target column
        age_col: Name of age group column
        sex_col: Name of sex column
    """
    n_datasets = len(datasets)
    fig, axes = plt.subplots(n_datasets, 3, figsize=(15, 4 * n_datasets))
    
    if n_datasets == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle('Dataset Characteristics', fontsize=16, fontweight='bold')
    
    for row, (dataset_name, df) in enumerate(datasets.items()):
        # Target distribution
        df[target_col].value_counts().plot(
            kind='bar', ax=axes[row, 0], color=['#2ecc71', '#e74c3c']
        )
        axes[row, 0].set_title(f'{dataset_name}: Target Distribution')
        axes[row, 0].set_xlabel('Heart Disease')
        axes[row, 0].set_ylabel('Count')
        axes[row, 0].set_xticklabels(['No Disease', 'Disease'], rotation=0)
        
        # Age groups
        df[age_col].value_counts().sort_index().plot(
            kind='bar', ax=axes[row, 1], color='steelblue'
        )
        axes[row, 1].set_title(f'{dataset_name}: Age Distribution')
        axes[row, 1].set_xlabel('Age Group')
        axes[row, 1].set_ylabel('Count')
        axes[row, 1].tick_params(axis='x', rotation=45)
        
        # Sex distribution
        df[sex_col].value_counts().plot(
            kind='bar', ax=axes[row, 2], color=['#ff6b6b', '#4ecdc4']
        )
        axes[row, 2].set_title(f'{dataset_name}: Sex Distribution')
        axes[row, 2].set_xlabel('Sex')
        axes[row, 2].set_ylabel('Count')
        axes[row, 2].set_xticklabels(axes[row, 2].get_xticklabels(), rotation=0)
    
    plt.tight_layout()
    return fig


def plot_preprocessing_fairness(profiles: Dict[str, Dict[str, Any]], 
                                dataset_names: list = None):
    """
    Plot pre-processing statistical parity violations.
    
    Args:
        profiles: Dictionary mapping dataset names to fairness profile dictionaries
        dataset_names: Optional list of dataset names to display (defaults to profile keys)
    """
    if dataset_names is None:
        dataset_names = list(profiles.keys())
    
    n_datasets = len(profiles)
    fig, axes = plt.subplots(1, n_datasets, figsize=(7 * n_datasets, 5))
    
    if n_datasets == 1:
        axes = [axes]
    
    fig.suptitle('Pre-Processing Statistical Parity Violations', 
                 fontsize=14, fontweight='bold')
    
    colors = ['steelblue', 'coral', 'lightgreen', 'orange']
    
    for idx, (dataset_key, profile) in enumerate(profiles.items()):
        age_stats = profile['label_imbalance_by_group']['age_group']
        groups = list(age_stats['positive_rates'].keys())
        rates = [age_stats['positive_rates'][g] for g in groups]
        overall = profile['basic_stats']['target_prevalence']
        
        x = np.arange(len(groups))
        axes[idx].bar(x, rates, color=colors[idx % len(colors)], alpha=0.7, 
                     label='Group Rate')
        axes[idx].axhline(overall, color='red', linestyle='--', linewidth=2, 
                         label=f'Overall Rate ({overall:.2%})')
        axes[idx].set_xlabel('Age Group')
        axes[idx].set_ylabel('Positive Rate (Heart Disease)')
        
        max_diff = age_stats['statistical_parity_difference']['max_difference']
        display_name = dataset_names[idx] if dataset_names else dataset_key
        axes[idx].set_title(f'{display_name} - Max Diff: {max_diff:.2%}')
        
        axes[idx].set_xticks(x)
        axes[idx].set_xticklabels(groups, rotation=45, ha='right')
        axes[idx].legend()
        axes[idx].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_feature_importance(importance_dfs: Dict[str, pd.DataFrame],
                           top_n: int = 10,
                           coef_col: str = 'coefficient',
                           abs_coef_col: str = 'abs_coefficient',
                           feature_col: str = 'feature'):
    """
    Plot feature importance for multiple datasets.
    
    Args:
        importance_dfs: Dictionary mapping dataset names to feature importance DataFrames
        top_n: Number of top features to display
        coef_col: Name of coefficient column
        abs_coef_col: Name of absolute coefficient column
        feature_col: Name of feature column
    """
    n_datasets = len(importance_dfs)
    fig, axes = plt.subplots(1, n_datasets, figsize=(7.5 * n_datasets, 6))
    
    if n_datasets == 1:
        axes = [axes]
    
    fig.suptitle('Feature Importance (Logistic Regression Coefficients)', 
                 fontsize=14, fontweight='bold')
    
    for idx, (dataset_name, importance_df) in enumerate(importance_dfs.items()):
        top_features = importance_df.nlargest(top_n, abs_coef_col)
        colors = ['green' if x > 0 else 'red' for x in top_features[coef_col]]
        
        axes[idx].barh(range(len(top_features)), top_features[coef_col], 
                      color=colors, alpha=0.7)
        axes[idx].set_yticks(range(len(top_features)))
        axes[idx].set_yticklabels(top_features[feature_col])
        axes[idx].set_xlabel('Coefficient Value')
        axes[idx].set_title(f'{dataset_name} - Top {top_n} Features')
        axes[idx].axvline(0, color='black', linewidth=0.8)
        axes[idx].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_confusion_matrices(training_results: Dict[str, Dict[str, Any]],
                            dataset_names: list = None):
    """
    Plot confusion matrices for multiple datasets.
    
    Args:
        training_results: Dictionary containing training results with confusion matrices
        dataset_names: Optional list of display names for datasets
    """
    n_datasets = len(training_results)
    fig, axes = plt.subplots(1, n_datasets, figsize=(6 * n_datasets, 5))
    
    if n_datasets == 1:
        axes = [axes]
    
    fig.suptitle('Confusion Matrices (Test Set)', fontsize=14, fontweight='bold')
    
    cmaps = ['Blues', 'Oranges', 'Greens', 'Purples']
    
    for idx, (dataset_key, results) in enumerate(training_results.items()):
        cm_dict = results['test_metrics']['confusion_matrix']
        cm = np.array([
            [cm_dict['tn'], cm_dict['fp']],
            [cm_dict['fn'], cm_dict['tp']]
        ])
        
        display_name = dataset_names[idx] if dataset_names else dataset_key
        
        sns.heatmap(cm, annot=True, fmt='d', cmap=cmaps[idx % len(cmaps)], 
                   ax=axes[idx],
                   xticklabels=['Predicted No', 'Predicted Yes'],
                   yticklabels=['Actual No', 'Actual Yes'])
        axes[idx].set_title(display_name)
    
    plt.tight_layout()
    return fig


def plot_fairness_heatmap(summary_df: pd.DataFrame, dataset_name: str):
    """
    Create heatmap of fairness metrics.
    
    Args:
        summary_df: DataFrame containing fairness metrics summary
        dataset_name: Name of dataset for title
    """
    # Prepare data for heatmap
    test_metrics = summary_df[summary_df['split'] == 'test'].copy()
    
    # Create pivot table
    if 'max_difference' in test_metrics.columns:
        pivot = test_metrics.pivot_table(
            values='max_difference',
            index='metric',
            columns='sensitive_attribute',
            aggfunc='first'
        )
    else:
        # Handle equalized odds case
        pivot_data = []
        for _, row in test_metrics.iterrows():
            if row['metric'] == 'equalized_odds':
                if 'tpr_max_difference' in test_metrics.columns:
                    pivot_data.append({
                        'metric': 'equalized_odds_tpr',
                        'sensitive_attribute': row['sensitive_attribute'],
                        'value': row['tpr_max_difference']
                    })
                    pivot_data.append({
                        'metric': 'equalized_odds_fpr',
                        'sensitive_attribute': row['sensitive_attribute'],
                        'value': row['fpr_max_difference']
                    })
            else:
                pivot_data.append({
                    'metric': row['metric'],
                    'sensitive_attribute': row['sensitive_attribute'],
                    'value': row.get('max_difference', 0)
                })
        
        pivot_df = pd.DataFrame(pivot_data)
        pivot = pivot_df.pivot_table(
            values='value',
            index='metric',
            columns='sensitive_attribute',
            aggfunc='first'
        )
    
    # Plot
    fig = plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn_r', center=0.1,
                vmin=0, vmax=0.6, cbar_kws={'label': 'Max Difference'})
    plt.title(f'{dataset_name} - Fairness Metrics (Test Set)', 
             fontsize=14, fontweight='bold')
    plt.xlabel('Sensitive Attribute')
    plt.ylabel('Fairness Metric')
    plt.tight_layout()
    return fig


def plot_train_test_comparison(summary_df: pd.DataFrame, 
                               dataset_name: str, 
                               metric_name: str = 'demographic_parity'):
    """
    Compare fairness metric between train and test sets.
    
    Args:
        summary_df: DataFrame containing fairness metrics
        dataset_name: Name of dataset for title
        metric_name: Name of metric to compare
    """
    metric_data = summary_df[summary_df['metric'] == metric_name].copy()
    
    if len(metric_data) == 0:
        print(f"No data for {metric_name}")
        return None
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'{dataset_name} - {metric_name.replace("_", " ").title()} Comparison', 
                 fontsize=14, fontweight='bold')
    
    for idx, attr in enumerate(metric_data['sensitive_attribute'].unique()):
        attr_data = metric_data[metric_data['sensitive_attribute'] == attr]
        
        splits = attr_data['split'].values
        values = attr_data['max_difference'].values
        
        x = np.arange(len(splits))
        colors = ['steelblue' if s == 'train' else 'coral' for s in splits]
        
        axes[idx].bar(x, values, color=colors, alpha=0.7)
        axes[idx].axhline(0.1, color='red', linestyle='--', linewidth=2, 
                         label='Fairness Threshold (10%)')
        axes[idx].set_xticks(x)
        axes[idx].set_xticklabels([s.capitalize() for s in splits])
        axes[idx].set_ylabel('Max Difference')
        axes[idx].set_title(f'{attr.replace("_", " ").title()}')
        axes[idx].legend()
        axes[idx].grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, v in enumerate(values):
            axes[idx].text(i, v + 0.02, f'{v:.3f}', ha='center', 
                          va='bottom', fontweight='bold')
    
    plt.tight_layout()
    return fig


def plot_equalized_odds_details(fairness_results: Dict[str, Any], 
                                dataset_name: str):
    """
    Visualize TPR and FPR by group for equalized odds.
    
    Args:
        fairness_results: Dictionary containing fairness assessment results
        dataset_name: Name of dataset for title
    """
    test_metrics = fairness_results['test_metrics']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{dataset_name} - Equalized Odds (Test Set)', 
                 fontsize=14, fontweight='bold')
    
    row = 0
    for attr, attr_metrics in test_metrics['group_fairness'].items():
        eq_odds = attr_metrics['equalized_odds']
        
        groups = list(eq_odds['group_metrics'].keys())
        tprs = [eq_odds['group_metrics'][g]['tpr'] for g in groups]
        fprs = [eq_odds['group_metrics'][g]['fpr'] for g in groups]
        
        x = np.arange(len(groups))
        width = 0.35
        
        # TPR
        axes[row, 0].bar(x, tprs, width, label='TPR (Recall)', 
                        color='green', alpha=0.7)
        axes[row, 0].set_ylabel('True Positive Rate')
        axes[row, 0].set_title(f'{attr.replace("_", " ").title()} - TPR by Group')
        axes[row, 0].set_xticks(x)
        axes[row, 0].set_xticklabels(groups, rotation=45, ha='right')
        axes[row, 0].set_ylim(0, 1.1)
        axes[row, 0].axhline(np.mean(tprs), color='red', linestyle='--', 
                            linewidth=2, label=f'Mean: {np.mean(tprs):.3f}')
        axes[row, 0].legend()
        axes[row, 0].grid(axis='y', alpha=0.3)
        
        # FPR
        axes[row, 1].bar(x, fprs, width, label='FPR', color='red', alpha=0.7)
        axes[row, 1].set_ylabel('False Positive Rate')
        axes[row, 1].set_title(f'{attr.replace("_", " ").title()} - FPR by Group')
        axes[row, 1].set_xticks(x)
        axes[row, 1].set_xticklabels(groups, rotation=45, ha='right')
        axes[row, 1].set_ylim(0, 1.1)
        axes[row, 1].axhline(np.mean(fprs), color='blue', linestyle='--', 
                            linewidth=2, label=f'Mean: {np.mean(fprs):.3f}')
        axes[row, 1].legend()
        axes[row, 1].grid(axis='y', alpha=0.3)
        
        row += 1
    
    plt.tight_layout()
    return fig


def plot_calibration_by_group(fairness_results: Dict[str, Any], 
                              dataset_name: str):
    """
    Visualize calibration curves by sensitive attribute group.
    
    Args:
        fairness_results: Dictionary containing fairness assessment results
        dataset_name: Name of dataset for title
    """
    test_metrics = fairness_results['test_metrics']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'{dataset_name} - Calibration by Group (Test Set)', 
                 fontsize=14, fontweight='bold')
    
    col = 0
    for attr, calib_data in test_metrics['calibration'].items():
        ax = axes[col]
        
        for group, group_calib in calib_data['group_calibration'].items():
            bins_data = group_calib['bins']
            
            if len(bins_data) > 0:
                predicted = [b['mean_predicted'] for b in bins_data]
                actual = [b['mean_true'] for b in bins_data]
                
                ax.plot(predicted, actual, marker='o', 
                       label=f'{group} (ECE={group_calib["ece"]:.3f})', 
                       linewidth=2)
        
        # Perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect Calibration')
        ax.set_xlabel('Predicted Probability')
        ax.set_ylabel('Actual Probability')
        ax.set_title(f'{attr.replace("_", " ").title()}')
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        col += 1
    
    plt.tight_layout()
    return fig


def plot_fairness_evolution(pre_profiles: Dict[str, Dict[str, Any]],
                           post_fairness: Dict[str, Dict[str, Any]],
                           dataset_names: list = None):
    """
    Visualize fairness evolution from pre-processing to post-processing.
    
    Args:
        pre_profiles: Dictionary mapping dataset keys to pre-processing profiles
        post_fairness: Dictionary mapping dataset keys to post-processing results
        dataset_names: Optional list of display names for datasets
    """
    n_datasets = len(pre_profiles)
    fig, axes = plt.subplots(1, n_datasets, figsize=(7 * n_datasets, 5))
    
    if n_datasets == 1:
        axes = [axes]
    
    fig.suptitle('Fairness Evolution Through Pipeline', 
                 fontsize=14, fontweight='bold')
    
    stages = ['Pre-Processing', 'Post-Processing']
    x = np.arange(len(stages))
    
    for idx, dataset_key in enumerate(pre_profiles.keys()):
        pre_profile = pre_profiles[dataset_key]
        post_result = post_fairness[dataset_key]
        
        # Extract age group values
        age_values = [
            pre_profile['label_imbalance_by_group']['age_group']['statistical_parity_difference']['max_difference'],
            post_result['test_metrics']['group_fairness']['age_group_cat']['demographic_parity']['max_difference']
        ]
        
        # Extract sex values
        sex_values = [
            pre_profile['label_imbalance_by_group']['sex']['statistical_parity_difference']['max_difference'],
            post_result['test_metrics']['group_fairness']['sex_cat']['demographic_parity']['max_difference']
        ]
        
        display_name = dataset_names[idx] if dataset_names else dataset_key
        
        axes[idx].plot(x, age_values, marker='o', linewidth=2, markersize=10, 
                      label='Age Group', color='steelblue')
        axes[idx].plot(x, sex_values, marker='s', linewidth=2, markersize=10, 
                      label='Sex', color='coral')
        axes[idx].axhline(0.1, color='red', linestyle='--', linewidth=2, 
                         alpha=0.5, label='Fairness Threshold')
        axes[idx].set_xticks(x)
        axes[idx].set_xticklabels(stages)
        axes[idx].set_ylabel('Statistical Parity Difference')
        axes[idx].set_title(display_name)
        axes[idx].legend()
        axes[idx].grid(axis='y', alpha=0.3)
        axes[idx].set_ylim(0, 0.7)
    
    plt.tight_layout()
    return fig


def plot_distribution(data: pd.Series, title: str = "Distribution", bins: int = 30, **kwargs):
    """
    Plot distribution of a single variable.
    
    Args:
        data: Series to plot
        title: Plot title
        bins: Number of bins for histogram
        **kwargs: Additional arguments for matplotlib
    
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=kwargs.get('figsize', (10, 6)))
    
    ax.hist(data.dropna(), bins=bins, color='steelblue', alpha=0.7, edgecolor='black')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel(data.name or 'Value', fontsize=10)
    ax.set_ylabel('Frequency', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_correlation_matrix(df: pd.DataFrame, title: str = "Correlation Matrix", **kwargs):
    """
    Plot correlation matrix heatmap.
    
    Args:
        df: DataFrame with numeric columns
        title: Plot title
        **kwargs: Additional arguments for seaborn heatmap
    
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=kwargs.get('figsize', (12, 10)))
    
    # Compute correlation matrix
    corr = df.corr()
    
    # Create heatmap
    sns.heatmap(
        corr,
        annot=kwargs.get('annot', True),
        fmt=kwargs.get('fmt', '.2f'),
        cmap=kwargs.get('cmap', 'coolwarm'),
        center=0,
        square=True,
        ax=ax,
        cbar_kws={'shrink': 0.8}
    )
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    return fig


def plot_model_performance(y_true, y_pred, y_proba=None, model_name: str = "Model"):
    """
    Plot model performance metrics including confusion matrix and ROC curve.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (optional, for ROC curve)
        model_name: Name of the model for title
    
    Returns:
        matplotlib Figure object
    """
    from sklearn.metrics import confusion_matrix, roc_curve, auc
    
    # Create figure with subplots
    if y_proba is not None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    else:
        fig, axes = plt.subplots(1, 1, figsize=(7, 5))
        axes = [axes]
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        ax=axes[0],
        cbar_kws={'label': 'Count'}
    )
    axes[0].set_title(f'{model_name}: Confusion Matrix', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Predicted Label', fontsize=10)
    axes[0].set_ylabel('True Label', fontsize=10)
    axes[0].set_xticklabels(['No Disease', 'Disease'])
    axes[0].set_yticklabels(['No Disease', 'Disease'])
    
    # ROC Curve (if probabilities provided)
    if y_proba is not None:
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        
        axes[1].plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC curve (AUC = {roc_auc:.3f})')
        axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                    label='Random Classifier')
        axes[1].set_xlim([0.0, 1.0])
        axes[1].set_ylim([0.0, 1.05])
        axes[1].set_xlabel('False Positive Rate', fontsize=10)
        axes[1].set_ylabel('True Positive Rate', fontsize=10)
        axes[1].set_title(f'{model_name}: ROC Curve', fontsize=12, fontweight='bold')
        axes[1].legend(loc='lower right')
        axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    return fig


def save_comparison_heatmap(pivot: pd.DataFrame, title: str, output_file, fmt: str = '.3f', cmap: str = 'viridis'):
    """Save a heatmap for a comparison pivot table."""
    if pivot is None or pivot.empty:
        return None

    fig = plt.figure(figsize=(9, 6))
    sns.heatmap(pivot, annot=True, fmt=fmt, cmap=cmap)
    plt.title(title)
    plt.tight_layout()
    fig.savefig(output_file, dpi=200)
    plt.close(fig)
    return output_file


def save_tradeoff_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    hue_col: str,
    style_col: str,
    title: str,
    output_file
):
    """Save a scatter plot showing performance vs fairness trade-offs."""
    if df is None or df.empty:
        return None

    plot_df = df[[x_col, y_col, hue_col, style_col]].dropna(subset=[x_col, y_col]).copy()
    if plot_df.empty:
        return None

    use_hue = hue_col in plot_df.columns and plot_df[hue_col].notna().any()
    use_style = style_col in plot_df.columns and plot_df[style_col].notna().any()

    fig = plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=plot_df,
        x=x_col,
        y=y_col,
        hue=hue_col if use_hue else None,
        style=style_col if use_style else None,
        alpha=0.8
    )
    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    if use_hue or use_style:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    fig.savefig(output_file, dpi=200)
    plt.close(fig)
    return output_file


def save_pareto_frontier(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    output_file
):
    """Save a Pareto frontier plot (maximize x, minimize y)."""
    if df is None or df.empty:
        return None

    data = df[[x_col, y_col]].dropna().copy()
    if data.empty:
        return None

    # Compute Pareto frontier: maximize x, minimize y
    data = data.sort_values([x_col, y_col], ascending=[False, True])
    pareto = []
    best_y = None
    for _, row in data.iterrows():
        y_val = row[y_col]
        if best_y is None or y_val <= best_y:
            pareto.append(row)
            best_y = y_val

    pareto_df = pd.DataFrame(pareto)

    fig = plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x=x_col, y=y_col, alpha=0.6)
    sns.lineplot(data=pareto_df, x=x_col, y=y_col, color='red', marker='o')
    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.tight_layout()
    fig.savefig(output_file, dpi=200)
    plt.close(fig)
    return output_file

