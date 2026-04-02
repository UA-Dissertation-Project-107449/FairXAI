"""Notebook data loading and lightweight tabular summary helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_external_datasets(
    external_files: dict[str, Path], detect_csv_sep
) -> dict[str, pd.DataFrame]:
    loaded: dict[str, pd.DataFrame] = {}
    for name, path in external_files.items():
        sep = detect_csv_sep(path)
        loaded[name] = pd.read_csv(path, sep=sep)
    return loaded


def load_raw_datasets(raw_dir: Path, datasets: list[str]) -> dict[str, pd.DataFrame]:
    return {name: pd.read_csv(raw_dir / f"{name}_standardized.csv") for name in datasets}


def load_processed_scaled_datasets(
    processed_dir: Path, datasets: list[str]
) -> dict[str, dict[str, pd.DataFrame]]:
    processed: dict[str, dict[str, pd.DataFrame]] = {}
    for name in datasets:
        base = processed_dir / name
        train_path = base / f"{name}_train_scaled.csv"
        test_path = base / f"{name}_test_scaled.csv"
        if train_path.exists() and test_path.exists():
            processed[name] = {
                "train": pd.read_csv(train_path),
                "test": pd.read_csv(test_path),
            }
    return processed


def summarize_stage(dfs: dict[str, pd.DataFrame], stage: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for name, df in dfs.items():
        rows.append(
            {
                "dataset": name,
                "stage": stage,
                "rows": len(df),
                "cols": len(df.columns),
                "missing_cells": int(df.isna().sum().sum()),
                "rows_with_missing": int(df.isna().any(axis=1).sum()),
            }
        )
    return pd.DataFrame(rows)


def canonical_features_for_columns(
    columns: list[str], dataset_name: str, feature_map: dict
) -> set[str]:
    canonical: set[str] = set()

    def add_from_section(section: dict) -> None:
        for _, info in section.items():
            canonical_name = info.get("canonical")
            aliases = info.get("aliases", [])
            candidates = [canonical_name] + aliases if canonical_name else aliases
            if any(col in columns for col in candidates):
                if canonical_name:
                    canonical.add(canonical_name)

    for group in ("sensitive", "common", "target"):
        add_from_section(feature_map.get(group, {}))

    dataset_specific = feature_map.get("dataset_specific", {}).get(dataset_name, {})
    add_from_section(dataset_specific)
    return canonical
