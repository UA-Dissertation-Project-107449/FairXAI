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
)

from .comparisons import (
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

__all__ = [
    "plot_categorical_distribution_grid",
    "plot_numeric_distribution_comparison",
    "plot_target_distribution_by_group",
    "plot_feature_drift_matrix",
    "plot_dataset_similarity_radar",
    "plot_group_representation_bars",
    "plot_transformation_impact",
    "plot_before_after_distributions",
    "plot_scaling_effects",
    "plot_fairness_metric_heatmap",
    "plot_group_performance_gaps",
    "plot_bias_amplification_waterfall"
]