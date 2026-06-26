"""Persistence helpers for generated synthetic datasets.

Each dataset is written as a CSV (NaNs become empty cells, read back as NaN by
pandas) alongside a sidecar ``<id>.meta.json`` capturing every generation
parameter and the ground-truth column descriptions. A single grid-level
``grid_manifest.json`` records the whole sweep for reproducibility.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from .config import GroundTruthColumn, SyntheticConfig


def write_dataset(
    df: pd.DataFrame,
    cfg: SyntheticConfig,
    ground_truth: list[GroundTruthColumn],
    datasets_dir: Path,
) -> tuple[Path, Path]:
    """Write ``df`` and its sidecar metadata; return ``(csv_path, meta_path)``."""
    datasets_dir = Path(datasets_dir)
    datasets_dir.mkdir(parents=True, exist_ok=True)
    dataset_id = cfg.dataset_id()

    csv_path = datasets_dir / f"{dataset_id}.csv"
    df.to_csv(csv_path, index=False)  # NaN -> empty field

    meta_path = datasets_dir / f"{dataset_id}.meta.json"
    meta = {
        "dataset_id": dataset_id,
        "config": asdict(cfg),
        "ground_truth": [asdict(col) for col in ground_truth],
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    return csv_path, meta_path


def write_grid_manifest(
    configs: list[SyntheticConfig],
    records: list[dict],
    out_path: Path,
) -> Path:
    """Write a single manifest of every config + per-dataset record."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "n_datasets": len(records),
        "configs": [asdict(cfg) for cfg in configs],
        "datasets": records,
    }
    out_path.write_text(json.dumps(manifest, indent=2, default=str))
    return out_path
