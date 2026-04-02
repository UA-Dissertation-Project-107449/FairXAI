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
from sklearn.decomposition import PCA as SklearnPCA

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


_TARGET_COLUMN_HINTS = [
    "target",
    "label",
    "class",
    "outcome",
    "result",
    "heart_disease",
    "diagnosis",
    "y",
    "output",
]


def _resolve_target_column(df: pd.DataFrame, target_column: str | None = None) -> str:
    if target_column and target_column in df.columns:
        return target_column

    lower_cols = [c.lower() for c in df.columns]
    for hint in _TARGET_COLUMN_HINTS:
        if hint in lower_cols:
            return df.columns[lower_cols.index(hint)]

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


def _clip_metrics(metrics: dict[str, Any]) -> None:
    """Clip all complexity metric values to [0.0, 1.0] in-place.

    N2 is a distance ratio and can exceed 1; other metrics are theoretically
    bounded but may produce minor floating-point overflows. The frontend and
    AI service assume all values are in [0, 1].
    """
    for name in EBM_FEATURE_ORDER:
        v = metrics.get(name)
        if v is not None:
            metrics[name] = float(np.clip(v, 0.0, 1.0))


def _compute_pca2d(X: pd.DataFrame, y: pd.Series) -> list[list[float | int]]:
    """Reduce dataset to 2D via PCA for the frontend scatter plot.

    Returns a list of [x, y_coord, classLabel] triples.
    """
    X_numeric = X.select_dtypes(include=[np.number]).fillna(0)
    if X_numeric.shape[1] < 2:
        return []
    n_components = min(2, X_numeric.shape[0], X_numeric.shape[1])
    pca = SklearnPCA(n_components=n_components, random_state=42)
    coords = pca.fit_transform(X_numeric.values)
    # Pad to 2 columns if dataset has only 1 numeric feature
    if coords.shape[1] < 2:
        coords = np.hstack([coords, np.zeros((coords.shape[0], 1))])
    return [[float(row[0]), float(row[1]), int(label)] for row, label in zip(coords, y.values)]


def _compute_feature_type_summary(X: pd.DataFrame) -> dict[str, int]:
    summary = {
        "numerical": 0,
        "categorical": 0,
        "binary": 0,
        "datetime": 0,
        "text": 0,
        "unknown": 0,
    }

    for column_name in X.columns:
        series = X[column_name]
        non_null = series.dropna()
        if non_null.empty:
            summary["unknown"] += 1
            continue

        if pd.api.types.is_datetime64_any_dtype(series):
            summary["datetime"] += 1
            continue

        unique_count = int(non_null.nunique())
        if unique_count == 2:
            summary["binary"] += 1
            continue

        if pd.api.types.is_numeric_dtype(series):
            summary["numerical"] += 1
            continue

        if unique_count <= 20:
            summary["categorical"] += 1
            continue

        summary["text"] += 1

    return {key: value for key, value in summary.items() if value > 0}


def _resolve_project_root(project_root: str | Path | None = None) -> Path | None:
    if project_root:
        candidate = Path(project_root)
        if candidate.exists():
            return candidate.resolve()

    env_project_root = os.getenv("FAIRXAI_PROJECT_ROOT")
    if env_project_root:
        candidate = Path(env_project_root)
        if candidate.exists():
            return candidate.resolve()

    current = Path(__file__).resolve()
    for parent in current.parents:
        config_path = parent / "configs" / "recommendations" / "thresholds.yaml"
        if config_path.exists():
            return parent

    return None


def _build_triage_report(
    csv_path: Path,
    dataset_name: str,
    target_column: str,
    index_column: str | None,
    sensitive_columns: list[str] | None,
    project_root: str | Path | None,
) -> tuple[dict[str, Any], dict[str, int]]:
    from fairxai.recommendations.engine import RecommendationEngine

    resolved_project_root = _resolve_project_root(project_root)
    engine = RecommendationEngine(
        project_root=str(resolved_project_root) if resolved_project_root else None,
    )

    ingestion = engine.ingest(
        str(csv_path),
        label_column=target_column,
        sensitive_columns=sensitive_columns or None,
        identifier_columns=[index_column] if index_column else None,
        dataset_name=dataset_name,
    )
    report = engine.generate(ingestion)
    feature_summary = report.feature_type_summary or {}
    return report.to_dict(), feature_summary


def _predict_ebm_difficulty(
    metrics: dict[str, Any], ebm_model_path: str | Path | None = None
) -> float:
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
    input_features = np.array(
        [float(metrics[name]) for name in EBM_FEATURE_ORDER], dtype=float
    ).reshape(1, -1)
    prediction = model.predict(input_features)[0]
    return float(prediction)


def characterize_dataset(
    filename: str,
    output_dir: str | Path,
    datasets_dir: str | Path | None = None,
    target_column: str | None = None,
    index_column: str | None = None,
    ebm_model_path: str | Path | None = None,
    include_triage: bool = False,
    sensitive_columns: list[str] | None = None,
    triage_project_root: str | Path | None = None,
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
    index_column : str | None
        Optional index/identifier column to exclude from feature computation.
    ebm_model_path : str | Path | None
        Optional EBM model path override.
    """
    csv_path = _resolve_input_csv(filename=filename, datasets_dir=datasets_dir)
    file_id = csv_path.stem

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"Dataset is empty: {csv_path}")

    target = _resolve_target_column(df, target_column=target_column)
    drop_cols = [target]
    if index_column and index_column in df.columns and index_column != target:
        drop_cols.append(index_column)

    X = df.drop(columns=drop_cols, errors="ignore")
    y = df[target]

    complexity_df = df.drop(columns=[index_column], errors="ignore") if index_column else df
    complexity = compute_complexity_metrics(complexity_df, target=target)
    metrics = {
        "nSamples": int(X.shape[0]),
        "nFeatures": int(X.shape[1]),
        "nClasses": int(pd.Series(y).dropna().nunique()),
    }

    for name in EBM_FEATURE_ORDER:
        metrics[name] = _round_metric(complexity.get(name))

    metrics["ebmDifficulty"] = float(
        np.clip(
            _predict_ebm_difficulty(metrics=metrics, ebm_model_path=ebm_model_path),
            0.0,
            1.0,
        )
    )

    pca2d = _compute_pca2d(X, y)

    missing_percentages = {
        str(column_name): float(np.round(X[column_name].isna().mean() * 100, 2))
        for column_name in X.columns
    }
    duplicate_count = int(df.drop(columns=[index_column], errors="ignore").duplicated().sum())
    class_distribution = {
        str(class_name): int(count)
        for class_name, count in pd.Series(y).value_counts(dropna=False).to_dict().items()
    }

    feature_type_summary = _compute_feature_type_summary(X)

    result: dict[str, Any] = {
        "jobId": file_id,
        "metrics": metrics,
        "pca2d": pca2d,
        "missing_percentages": missing_percentages,
        "duplicate_count": duplicate_count,
        "class_distribution": class_distribution,
        "feature_type_summary": feature_type_summary,
    }

    if include_triage:
        try:
            triage_report, triage_feature_summary = _build_triage_report(
                csv_path=csv_path,
                dataset_name=file_id,
                target_column=target,
                index_column=index_column,
                sensitive_columns=sensitive_columns,
                project_root=triage_project_root,
            )
            result["triage_report"] = triage_report
            if triage_feature_summary:
                result["feature_type_summary"] = triage_feature_summary
        except Exception as exc:  # pragma: no cover - keep characterize resilient
            result["triage_error"] = str(exc)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"{file_id}.json"
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(result, fp, indent=4)

    return result
