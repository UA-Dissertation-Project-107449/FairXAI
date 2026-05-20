# src/fairxai/viz/__init__.py
"""
Educational visualization toolkit for fairness-aware ML.

Design Principles:
- Every plot includes interpretive context
- Fairness implications highlighted automatically
- Consistent styling across all functions
- Reusable across notebooks/reports
"""

from .clustering import (
    save_cluster_fairness_heatmap,
    save_cluster_profile_bars,
)
from .comparisons import (
    plot_correlation_heatmap_grid,
    plot_drift_heatmap,
    plot_pca_kmeans_scatter_grid,
    plot_two_dataset_feature_distributions,
    summarize_ks_test_between_datasets,
)
from .constants import (
    CARDIAC_CATEGORY_DISPLAY_ORDER,
    CARDIAC_CATEGORY_VALUE_LABEL_MAPPING,
    normalize_cardiac_category_series,
)
from .distributions import (
    plot_bmi_and_bp_relationship,
    plot_categorical_distribution_grid,
    plot_missing_data_patterns,
    plot_mixed_feature_batches,
    plot_numeric_distribution_comparison,
    plot_outlier_analysis,
    plot_stacked_group_distribution_grid,
    plot_target_distribution_by_group,
)
from .fairness import (
    plot_bias_amplification_waterfall,
    plot_fairness_metric_heatmap,
    plot_group_performance_gaps,
)
from .fairness_comparison import (
    PALETTE_MODEL,
    save_before_after_metric_radar,
    save_binning_strategy_delta_matrix,
    save_cross_model_baseline_radar,
    save_cross_model_best_available_radar,
    save_group_before_after_bars,
    save_group_delta_bars,
    save_group_error_consequence_bars,
    save_group_performance_gap_bars,
    save_intersectional_heatmap,
    save_mitigation_delta_matrix,
    save_model_overfit_gap_bars,
    save_top_n_binning_strategy_age_group_small_multiples,
    save_top_n_binning_strategy_summary,
    select_primary_fairness_row,
)
from .style import (
    PALETTE_DATASET,
    PALETTE_SEX,
    PALETTE_TARGET,
    UNITS,
)
from .transformations import (
    plot_before_after_distributions,
    plot_scaling_effects,
    plot_transformation_impact,
)

__all__ = [
    "plot_categorical_distribution_grid",
    "plot_numeric_distribution_comparison",
    "plot_target_distribution_by_group",
    "plot_stacked_group_distribution_grid",
    "plot_missing_data_patterns",
    "plot_outlier_analysis",
    "plot_mixed_feature_batches",
    "plot_bmi_and_bp_relationship",
    "plot_correlation_heatmap_grid",
    "plot_pca_kmeans_scatter_grid",
    "plot_two_dataset_feature_distributions",
    "summarize_ks_test_between_datasets",
    "plot_drift_heatmap",
    "plot_transformation_impact",
    "plot_before_after_distributions",
    "plot_scaling_effects",
    "plot_fairness_metric_heatmap",
    "plot_group_performance_gaps",
    "plot_bias_amplification_waterfall",
    "PALETTE_DATASET",
    "PALETTE_SEX",
    "PALETTE_TARGET",
    "UNITS",
    "CARDIAC_CATEGORY_VALUE_LABEL_MAPPING",
    "CARDIAC_CATEGORY_DISPLAY_ORDER",
    "normalize_cardiac_category_series",
    "save_intersectional_heatmap",
    "save_before_after_metric_radar",
    "save_mitigation_delta_matrix",
    "save_group_before_after_bars",
    "save_group_delta_bars",
    "save_group_performance_gap_bars",
    "save_cross_model_baseline_radar",
    "save_cross_model_best_available_radar",
    "save_group_error_consequence_bars",
    "save_binning_strategy_delta_matrix",
    "save_model_overfit_gap_bars",
    "save_top_n_binning_strategy_summary",
    "save_top_n_binning_strategy_age_group_small_multiples",
    "select_primary_fairness_row",
    "PALETTE_MODEL",
    "save_cluster_profile_bars",
    "save_cluster_fairness_heatmap",
]
