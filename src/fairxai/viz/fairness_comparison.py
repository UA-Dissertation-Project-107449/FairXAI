"""Fairness comparison dissertation plots.

Compatibility note: public functions are imported from the legacy experiment
plot module while callers migrate to this focused namespace.
"""

from .experiment_plots import (
    build_fairness_evidence_summary,
    save_before_after_metric_radar,
    save_cross_model_baseline_radar,
    save_cross_model_best_available_radar,
    save_fairness_evidence_summary,
    save_group_before_after_bars,
    save_group_delta_bars,
    save_mitigation_delta_matrix,
    select_primary_fairness_row,
)

__all__ = [
    "build_fairness_evidence_summary",
    "save_before_after_metric_radar",
    "save_cross_model_baseline_radar",
    "save_cross_model_best_available_radar",
    "save_fairness_evidence_summary",
    "save_group_before_after_bars",
    "save_group_delta_bars",
    "save_mitigation_delta_matrix",
    "select_primary_fairness_row",
]

