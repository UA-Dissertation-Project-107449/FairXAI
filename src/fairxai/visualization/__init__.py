"""Visualization utilities for FairXAI."""

from .plots import (
    plot_dataset_characteristics,
    plot_preprocessing_fairness,
    plot_feature_importance,
    plot_confusion_matrices,
    plot_fairness_heatmap,
    plot_train_test_comparison,
    plot_equalized_odds_details,
    plot_calibration_by_group,
    plot_fairness_evolution,
    plot_distribution,
    plot_correlation_matrix,
    plot_model_performance
)

__all__ = [
    'plot_dataset_characteristics',
    'plot_preprocessing_fairness',
    'plot_feature_importance',
    'plot_confusion_matrices',
    'plot_fairness_heatmap',
    'plot_train_test_comparison',
    'plot_equalized_odds_details',
    'plot_calibration_by_group',
    'plot_fairness_evolution',
    'plot_distribution',
    'plot_correlation_matrix',
    'plot_model_performance'
]
