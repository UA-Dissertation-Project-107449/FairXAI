"""Shared data IO helpers for experiments."""

import json
from pathlib import Path
from typing import Dict, List, Optional

from fairxai.utils.config import load_yaml_config


def load_schema_config(project_root: Path, pipeline: str = "cardiac") -> Dict:
    pipeline_cfg = load_yaml_config(str(project_root / f"configs/pipelines/{pipeline}.yaml"))
    schema_path = project_root / pipeline_cfg['runtime']['schema_mapping_json']
    with open(schema_path, 'r') as f:
        return json.load(f)


def build_schema_excludes(schema_cfg: Dict, dataset_name: str) -> List[str]:
    dataset_cfg = schema_cfg.get('datasets', {}).get(dataset_name, {})
    unified_cfg = schema_cfg.get('unified_schema', {})
    schema_exclude = list(dataset_cfg.get('exclude_features') or [])
    schema_exclude += list(unified_cfg.get('exclude_features') or [])
    label_col = dataset_cfg.get('label') or dataset_cfg.get('target')
    if label_col:
        schema_exclude.append(label_col)
    return schema_exclude


def resolve_base_dataset(schema_cfg: Dict, dataset_name: str) -> str:
    return next(
        (ds for ds in schema_cfg.get('cardiac_relevant_datasets', []) if dataset_name.startswith(ds)),
        dataset_name
    )


def merge_excludes(schema_cfg: Dict, dataset_name: str, base_excludes: Optional[List[str]] = None) -> List[str]:
    excludes = list(base_excludes or [])
    excludes.extend(build_schema_excludes(schema_cfg, dataset_name))
    return list(dict.fromkeys(excludes))


def default_exclude_columns(
    schema_cfg: Dict,
    dataset_name: str,
    target: str = "heart_disease",
    sensitive_attrs: Optional[List[str]] = None,
) -> List[str]:
    base = [
        target,
        '_dataset_source',
        '_dataset_file',
        'age_raw',
        'sex_extended',
        'sex_bin',
        'age',
        'Age',
        'Sex',
        'gender',
        'condition',
        'HeartDisease',
        'cardio',
        'id'
    ]
    if sensitive_attrs:
        base.extend(sensitive_attrs)
    return merge_excludes(schema_cfg, dataset_name, base)
