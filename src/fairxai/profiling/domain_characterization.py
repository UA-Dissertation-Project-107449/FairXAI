"""Dataset characterization utilities for WebApp-compatible JSON outputs.

This module provides a focused characterization path (CSV -> metrics JSON)
without invoking the full multi-stage pipeline.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .complexity import compute_complexity_metrics

EBM_FEATURE_ORDER = [
    "F2Imbalance",
    "F3Imbalance",
    "F4Imbalance",
    "L1Imbalance",
    "L2Imbalance",
    "L3Imbalance",
    "N2Imbalance",
    "N3Imbalance",
    "N4Imbalance",
    "T1Imbalance",
    "RaugImbalance",
    "BayesImbalance",
]


def _resolve_input_csv(filename: str, datasets_dir: str | Path | None = None) -> Path:
    input_path = Path(filename)
    if input_path.is_absolute() and input_path.exists():
        return input_path
    if input_path.exists():
        return input_path.resolve()

    candidate_dirs: list[Path] = []
    if datasets_dir:
        candidate_dirs.append(Path(datasets_dir))
    env_dir = os.getenv("FAIRXAI_DATASETS_DIR")
    if env_dir:
        candidate_dirs.append(Path(env_dir))

    for base_dir in candidate_dirs:
        candidate = base_dir / filename
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(
        f"Input dataset not found: {filename}. "
        "Provide an absolute path or set FAIRXAI_DATASETS_DIR/--datasets-dir."
    )


def _resolve_target_column(df: pd.DataFrame, target_column: str | None = None) -> str:
    if target_column and target_column in df.columns:
        return target_column

    if "heart_disease" in df.columns:
        return "heart_disease"

    return str(df.columns[-1])


def _resolve_ebm_model_path(ebm_model_path: str | Path | None = None) -> Path:
    if ebm_model_path:
        model_path = Path(ebm_model_path)
        if model_path.exists():
            return model_path.resolve()

    env_model = os.getenv("FAIRXAI_EBM_MODEL_PATH")
    if env_model:
        model_path = Path(env_model)
        if model_path.exists():
            return model_path.resolve()

    package_model = Path(__file__).resolve().parent / "models" / "ebm_model.joblib"
    if package_model.exists():
        return package_model

    raise FileNotFoundError(
        "EBM model not found. Set FAIRXAI_EBM_MODEL_PATH or provide --ebm-model-path."
    )


def _to_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value_f):
        return None
    return value_f


def _round_metric(value: Any) -> float | None:
    value_f = _to_float_or_none(value)
    if value_f is None:
        return None
    return float(np.round(value_f, 3))


def _predict_ebm_difficulty(metrics: dict[str, Any], ebm_model_path: str | Path | None = None) -> float:
    missing = [name for name in EBM_FEATURE_ORDER if _to_float_or_none(metrics.get(name)) is None]
    if missing:
        raise ValueError(f"Cannot compute ebmDifficulty; missing metrics: {', '.join(missing)}")

    from joblib import load

    try:
        model = load(_resolve_ebm_model_path(ebm_model_path))
    except ModuleNotFoundError as exc:
        if exc.name == "interpret":
            raise RuntimeError(
                "EBM model requires the 'interpret' package at runtime. "
                "Install the FairXAI experiment dependencies in the active environment."
            ) from exc
        raise
    input_features = np.array([float(metrics[name]) for name in EBM_FEATURE_ORDER], dtype=float).reshape(1, -1)
    prediction = model.predict(input_features)[0]
    return float(prediction)


def characterize_dataset(
    filename: str,
    output_dir: str | Path,
    datasets_dir: str | Path | None = None,
    target_column: str | None = None,
    ebm_model_path: str | Path | None = None,
) -> dict[str, Any]:
    """Characterize one dataset and write WebApp-compatible JSON output.

    Parameters
    ----------
    filename : str
        Dataset filename or path.
    output_dir : str | Path
        Directory where ``<jobId>.json`` is written.
    datasets_dir : str | Path | None
        Optional base folder for relative ``filename`` resolution.
    target_column : str | None
        Optional target column override.
    ebm_model_path : str | Path | None
        Optional EBM model path override.
    """
    csv_path = _resolve_input_csv(filename=filename, datasets_dir=datasets_dir)
    file_id = csv_path.stem

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"Dataset is empty: {csv_path}")

    target = _resolve_target_column(df, target_column=target_column)
    X = df.drop(columns=[target], errors="ignore")
    y = df[target]

    complexity = compute_complexity_metrics(df, target=target)
    metrics = {
        "nSamples": int(X.shape[0]),
        "nFeatures": int(X.shape[1]),
        "nClasses": int(pd.Series(y).dropna().nunique()),
    }

    for name in EBM_FEATURE_ORDER:
        metrics[name] = _round_metric(complexity.get(name))

    metrics["ebmDifficulty"] = _predict_ebm_difficulty(metrics=metrics, ebm_model_path=ebm_model_path)

    result = {"jobId": file_id, "metrics": metrics}

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"{file_id}.json"
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(result, fp, indent=4)

    return result
