"""Canonical comparison/dissertation filename helpers."""

from __future__ import annotations

import re
from typing import Any

from fairxai.comparison.baseline_matching import normalize_sensitive_attr

DEFAULT_FIGURE_TEMPLATES = {
    "fairness_metric_heatmap": "{dataset}_{sensitive_attr}_fairness_metric_heatmap.png",
    "intersectional_heatmap": "{dataset}_{metric}_intersectional_heatmap.png",
    "primary_mitigation_radar": "{dataset}_{model_label}_primary_mitigation_radar_before_after.png",
    "mitigation_delta_matrix": "{dataset}_{model_label}_mitigation_metric_delta_matrix.png",
    "group_performance_gaps": "{dataset}_{model_label}_primary_{sensitive_attr}_performance_gaps.png",
    "group_before_after": "{dataset}_{model_label}_primary_{sensitive_attr}_before_after.png",
    "group_delta": "{dataset}_{model_label}_primary_{sensitive_attr}_delta.png",
    "group_error_consequences": "{dataset}_{model_label}_primary_{sensitive_attr}_error_consequences.png",
    "baseline_cross_model_radar": "{dataset}_baseline_cross_model_radar.png",
    "best_available_cross_model_radar": (
        "{dataset}_unbalanced_best_available_cross_model_radar.png"
    ),
    "binning_strategy_delta_matrix": (
        "{dataset}_{model_label}_binning_strategy_metric_delta_matrix.png"
    ),
    "binning_strategy_summary": ("{dataset}_{model_label}_top{n}_binning_strategy_summary.png"),
    "binning_strategy_age_group_small_multiples": (
        "{dataset}_{model_label}_top{n}_age_group_delta_small_multiples.png"
    ),
}


def slugify_token(value: object) -> str:
    """Normalize path tokens to lowercase snake-ish text."""
    text = str(value).strip().lower()
    text = text.replace("+", "_plus_").replace("-", "_").replace(" ", "_")
    text = re.sub(r"[^a-z0-9_]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "unknown"


def figure_filename(config: dict[str, Any], key: str, **values: object) -> str:
    """Build a canonical figure filename from config templates."""
    templates = (
        (config or {}).get("naming", {}).get("figure_templates", {})
        if isinstance(config, dict)
        else {}
    )
    template = templates.get(key, DEFAULT_FIGURE_TEMPLATES[key])

    normalized: dict[str, str] = {}
    for name, value in values.items():
        if name == "sensitive_attr":
            value = normalize_sensitive_attr(value)
        normalized[name] = slugify_token(value)
    return template.format(**normalized)
