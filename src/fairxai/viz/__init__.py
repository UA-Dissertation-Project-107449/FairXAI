# src/fairxai/viz/__init__.py
"""
Educational visualization toolkit for fairness-aware ML.

Design Principles:
- Every plot includes interpretive context
- Fairness implications highlighted automatically
- Consistent styling across all functions
- Reusable across notebooks/reports
"""

from .distributions import (
    plot_categorical_distribution_grid,
    plot_numeric_distribution_comparison,
    plot_target_distribution_by_group,
    plot_stacked_group_distribution_grid,
    plot_mixed_feature_batches,
    plot_bmi_and_bp_relationship,
)

from .comparisons import (
    plot_correlation_heatmap_grid,
    plot_pca_kmeans_scatter_grid,
    plot_two_dataset_feature_distributions,
    summarize_ks_test_between_datasets,
    plot_feature_drift_matrix,
    plot_dataset_similarity_radar,
    plot_group_representation_bars,
)

from .transformations import (
    plot_transformation_impact,
    plot_before_after_distributions,
    plot_scaling_effects,
)

from .fairness import (
    plot_fairness_metric_heatmap,
    plot_group_performance_gaps,
    plot_bias_amplification_waterfall,
)

from .style import (
    PALETTE_DATASET,
    PALETTE_SEX,
    PALETTE_TARGET,
    UNITS,
)

from .constants import (
    CARDIAC_CATEGORY_VALUE_LABEL_MAPPING,
    CARDIAC_CATEGORY_DISPLAY_ORDER,
    normalize_cardiac_category_series,
)

__all__ = [
    "plot_categorical_distribution_grid",
    "plot_numeric_distribution_comparison",
    "plot_target_distribution_by_group",
    "plot_stacked_group_distribution_grid",
    "plot_mixed_feature_batches",
    "plot_bmi_and_bp_relationship",
    "plot_correlation_heatmap_grid",
    "plot_pca_kmeans_scatter_grid",
    "plot_two_dataset_feature_distributions",
    "summarize_ks_test_between_datasets",
    "plot_feature_drift_matrix",
    "plot_dataset_similarity_radar",
    "plot_group_representation_bars",
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
]