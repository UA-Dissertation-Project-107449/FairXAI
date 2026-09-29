from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml


def load_yaml_config(path: str) -> dict[str, object]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config YAML not found: {p}")
    with open(p, "r") as f:
        return yaml.safe_load(f)


def dataset_excluded_model_types(config: Mapping[str, Any], dataset: str) -> set[str]:
    """Model families a cohort must not run, from ``dataset_overrides`` in config.

    SVM's RBF kernel is O(n^2) in rows and the resampling techniques add rows,
    so cardio70k has always been run with SVM dropped. That was a comment and an
    operator's memory; recorded here it is part of the run instead.
    """
    overrides = config.get("dataset_overrides") or {}
    entry = overrides.get(dataset) or {}
    return {str(m).strip().lower() for m in entry.get("exclude_model_types", []) if str(m).strip()}
