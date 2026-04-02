"""Notebook context helpers for configuration, paths, and figure output."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

DEFAULT_DATASETS_BY_MEDICAL_AREA: dict[str, list[str]] = {
    "cardiac": ["cleveland", "kaggle_heart", "cardio70k"],
}


def resolve_root_dir(start: Path | None = None) -> Path:
    root = start or Path.cwd().resolve()
    if (root / "configs").exists():
        return root
    for parent in root.parents:
        if (parent / "configs").exists():
            return parent
    return root


def load_domain_config(root_dir: Path, medical_area: str) -> dict:
    config_dir = root_dir / "configs"
    pipeline_cfg_path = config_dir / "pipelines" / f"{medical_area}.yaml"
    feature_map_path = config_dir / "domain" / f"{medical_area}_feature_map.yaml"
    schema_path = config_dir / "schema" / f"{medical_area}.json"

    with open(pipeline_cfg_path, "r") as file:
        pipeline_cfg = yaml.safe_load(file)
    with open(feature_map_path, "r") as file:
        feature_map = yaml.safe_load(file)
    with open(schema_path, "r") as file:
        schema_cfg = json.load(file)

    return {
        "config_dir": config_dir,
        "pipeline_cfg_path": pipeline_cfg_path,
        "feature_map_path": feature_map_path,
        "schema_path": schema_path,
        "pipeline_cfg": pipeline_cfg,
        "feature_map": feature_map,
        "schema_cfg": schema_cfg,
        "external_dir": root_dir / pipeline_cfg["paths"]["external_dir"],
        "raw_dir": root_dir / pipeline_cfg["paths"]["raw_dir"],
        "processed_dir": root_dir / pipeline_cfg["paths"]["processed_dir"],
    }


def get_relevant_datasets(schema_cfg: dict, medical_area: str) -> list[str]:
    key = f"{medical_area}_relevant_datasets"
    fallback = DEFAULT_DATASETS_BY_MEDICAL_AREA.get(medical_area, [])
    return schema_cfg.get(key, fallback)


def make_figure_path_builder(
    root_dir: Path,
    medical_area: str,
    notebook_type: str,
) -> tuple[Path, callable]:
    figures_dir = root_dir / "notebooks" / "figures" / medical_area

    def fig_path(stem: str) -> Path:
        return figures_dir / f"{notebook_type}_{stem}.png"

    return figures_dir, fig_path
