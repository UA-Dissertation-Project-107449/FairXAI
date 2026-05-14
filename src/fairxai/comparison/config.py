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
    "legacy_score": {
        "plots_enabled": False,
        "dissertation_figures_enabled": False,
        "output_subdir": "legacy_score",
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
    path = Path(config_path) if config_path else project_root / "configs/experiments/comparison.yaml"
    if not path.is_absolute():
        path = project_root / path
    if path.exists():
        return _deep_merge(DEFAULT_COMPARISON_CONFIG, load_yaml_config(str(path)) or {})
    if config_path:
        raise FileNotFoundError(f"Comparison config YAML not found: {path}")
    return deepcopy(DEFAULT_COMPARISON_CONFIG)
