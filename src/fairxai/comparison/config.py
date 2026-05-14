"""Comparison-stage YAML configuration."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from fairxai.utils.config import load_yaml_config

DEFAULT_COMPARISON_CONFIG: dict[str, Any] = {
    "canonical_outputs": {
        "enabled": True,
        "compatibility_outputs": True,
    },
    "selection": {
        "primary_model_type": "logistic_regression",
        "primary_model_label": "lr",
        "primary_dataset": None,
        "min_recall_delta": -0.03,
        "top_n": 5,
    },
    "figures": {
        "enabled": True,
        "canonical_naming": True,
        "dpi": 300,
        "include_best_available_appendix": True,
        "dataset_averages": False,
        "sizes": {
            "radar_pair": [14, 6],
            "delta_matrix": [16, 6],
            "group_bars": [14, 5],
        },
    },
    "outputs": {
        "comparison_data_dir": "data",
        "dissertation_plot_dir": "plots",
        "dissertation_data_dir": "data",
    },
    "naming": {
        "figure_templates": {
            "fairness_metric_heatmap": "{dataset}_{sensitive_attr}_fairness_metric_heatmap.png",
            "intersectional_heatmap": "{dataset}_{metric}_intersectional_heatmap.png",
            "primary_mitigation_radar": (
                "{dataset}_{model_label}_primary_mitigation_radar_before_after.png"
            ),
            "mitigation_delta_matrix": "{dataset}_{model_label}_mitigation_metric_delta_matrix.png",
            "group_performance_gaps": (
                "{dataset}_{model_label}_primary_{sensitive_attr}_performance_gaps.png"
            ),
            "group_before_after": "{dataset}_{model_label}_primary_{sensitive_attr}_before_after.png",
            "group_delta": "{dataset}_{model_label}_primary_{sensitive_attr}_delta.png",
            "baseline_cross_model_radar": "{dataset}_baseline_cross_model_radar.png",
            "best_available_cross_model_radar": (
                "{dataset}_unbalanced_best_available_cross_model_radar.png"
            ),
        },
    },
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_comparison_config(project_root: Path, config_path: str | None = None) -> dict[str, Any]:
    """Load comparison config from YAML, merged over code fallback defaults."""
    path = (
        Path(config_path) if config_path else project_root / "configs/experiments/comparison.yaml"
    )
    if not path.is_absolute():
        path = project_root / path
    if path.exists():
        return _deep_merge(DEFAULT_COMPARISON_CONFIG, load_yaml_config(str(path)) or {})
    if config_path:
        raise FileNotFoundError(f"Comparison config YAML not found: {path}")
    return deepcopy(DEFAULT_COMPARISON_CONFIG)
