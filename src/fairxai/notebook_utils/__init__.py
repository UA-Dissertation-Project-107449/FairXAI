"""Shared helpers for FairXAI notebooks."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable
import numpy as np
import pandas as pd
from fairxai.viz.style import PALETTE_DATASET, PALETTE_SEX, PALETTE_TARGET, UNITS
from fairxai.data.schemas import get_sex_mapping
from .context import resolve_root_dir, load_domain_config, get_relevant_datasets, make_figure_path_builder
from .data import (
    load_external_datasets,
    load_raw_datasets,
    load_processed_scaled_datasets,
    summarize_stage,
    canonical_features_for_columns,
)

__all__ = [
    # Palettes & style re-exports
    "PALETTE_DATASET",
    "PALETTE_SEX",
    "PALETTE_TARGET",
    "UNITS",
    # Context helpers
    "resolve_root_dir",
    "load_domain_config",
    "get_relevant_datasets",
    "make_figure_path_builder",
    # Data helpers
    "load_external_datasets",
    "load_raw_datasets",
    "load_processed_scaled_datasets",
    "summarize_stage",
    "canonical_features_for_columns",
    # Local helpers
    "set_schema_cfg",
    "resolve_project_root",
    "detect_csv_sep",
    "dataset_age_unit",
    "age_group_order",
    "apply_age_group_order",
    "age_to_years",
    "resolve_sex_series",
    "add_bar_labels",
    "add_bar_labels_with_counts",
    "add_grouped_bar_labels",
    "add_point_labels",
]

_SCHEMA_CFG = None


def set_schema_cfg(schema_cfg: dict) -> None:
    global _SCHEMA_CFG
    _SCHEMA_CFG = schema_cfg


def resolve_project_root(start: Path | None = None) -> Path:
    root = start or Path.cwd().resolve()
    if (root / "configs").exists():
        return root
    for parent in root.parents:
        if (parent / "configs").exists():
            return parent
    return root


def detect_csv_sep(path: Path) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        header = f.readline()
    return ";" if header.count(";") > header.count(",") else ","


def _get_schema_cfg(schema_cfg: dict | None) -> dict:
    if schema_cfg is not None:
        return schema_cfg
    return _SCHEMA_CFG or {}


def dataset_age_unit(dataset_name: str, schema_cfg: dict | None = None) -> str:
    cfg = _get_schema_cfg(schema_cfg)
    ds_cfg = cfg.get("datasets", {}).get(dataset_name, {})
    sens = ds_cfg.get("sensitive_attributes", {})
    age_key = "age" if "age" in sens else "Age" if "Age" in sens else None
    if not age_key:
        return "years"
    bins = sens.get(age_key, {}).get("bins", [])
    if bins and max(bins) > 130:
        return "days"
    return "years"


def age_group_order(dataset_name: str, schema_cfg: dict | None = None) -> list[str]:
    cfg = _get_schema_cfg(schema_cfg)
    ds_cfg = cfg.get("datasets", {}).get(dataset_name, {})
    sens = ds_cfg.get("sensitive_attributes", {})
    age_key = "age" if "age" in sens else "Age" if "Age" in sens else None
    labels = sens.get(age_key, {}).get("labels") if age_key else None
    if labels:
        return labels
    return ["<40", "40-49", "50-59", "60-69", "70+"]


def apply_age_group_order(series: pd.Series, dataset_name: str, schema_cfg: dict | None = None) -> pd.Series:
    order = age_group_order(dataset_name, schema_cfg)
    return pd.Categorical(series.astype(str), categories=order, ordered=True)


def age_to_years(series: pd.Series, unit: str) -> pd.Series:
    if unit == "days":
        return series / 365.25
    return series


def resolve_sex_series(df: pd.DataFrame) -> pd.Series | None:
    """Resolve a sex column to canonical Female/Male string labels.

    Uses the centralized sex mapping from ``configs/domain/cardiac.yaml``
    via :func:`fairxai.data.schemas.get_sex_mapping`.
    """
    sex_num = get_sex_mapping("numeric")
    sex_str = get_sex_mapping("string")

    if "sex_extended" in df.columns:
        return df["sex_extended"].astype("object")
    if "sex" in df.columns:
        if pd.api.types.is_numeric_dtype(df["sex"]):
            return df["sex"].map(sex_num).astype("object")
        return df["sex"].astype(str).str.strip().map(sex_str)
    if "gender" in df.columns:
        if pd.api.types.is_numeric_dtype(df["gender"]):
            return df["gender"].map({1: "Female", 2: "Male", 0: "Female"}).astype("object")
        return df["gender"].astype(str).str.strip().map({
            "1": "Female",
            "2": "Male",
            "0": "Female",
            "Female": "Female",
            "Male": "Male",
        })
    return None


def add_bar_labels(ax, total: float | None = None, fmt: str = "{count} ({pct:.0%})") -> None:
    heights = [p.get_height() for p in ax.patches]
    total = total if total is not None else sum(heights)
    for patch, height in zip(ax.patches, heights):
        if np.isnan(height):
            continue
        pct = (height / total) if total else 0
        label = fmt.format(count=int(round(height)), pct=pct)
        ax.annotate(
            label,
            (patch.get_x() + patch.get_width() / 2, height),
            ha="center",
            va="bottom",
            fontsize=9,
            xytext=(0, 3),
            textcoords="offset points",
        )


def add_bar_labels_with_counts(ax, counts: pd.Series, fmt: str = "{count} ({pct:.0%})") -> None:
    total = counts.sum()
    for patch, count in zip(ax.patches, counts):
        height = patch.get_height()
        pct = (count / total) if total else 0
        label = fmt.format(count=int(count), pct=pct)
        ax.annotate(
            label,
            (patch.get_x() + patch.get_width() / 2, height),
            ha="center",
            va="bottom",
            fontsize=9,
            xytext=(0, 3),
            textcoords="offset points",
        )


def add_grouped_bar_labels(ax, counts_df: pd.DataFrame, fmt: str = "{count} ({pct:.0%})") -> None:
    for container, col in zip(ax.containers, counts_df.columns):
        total = counts_df[col].sum()
        for patch, count in zip(container.patches, counts_df[col].values):
            if np.isnan(count):
                continue
            pct = (count / total) if total else 0
            label = fmt.format(count=int(count), pct=pct)
            ax.annotate(
                label,
                (patch.get_x() + patch.get_width() / 2, patch.get_height()),
                ha="center",
                va="bottom",
                fontsize=8,
                xytext=(0, 3),
                textcoords="offset points",
            )


def add_point_labels(ax, x_vals: Iterable, y_vals: Iterable, fmt: str = "{pct:.0%}") -> None:
    for x, y in zip(x_vals, y_vals):
        label = fmt.format(pct=y)
        ax.annotate(
            label,
            (x, y),
            ha="center",
            va="bottom",
            fontsize=9,
            xytext=(0, 4),
            textcoords="offset points",
        )
