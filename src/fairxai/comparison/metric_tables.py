"""Canonical metric-level comparison tables."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .baseline_matching import (
    build_baseline_lookups,
    find_matching_baseline,
    normalize_sensitive_attr,
)

IDENTITY_COLS = [
    "experiment_id",
    "dataset",
    "model_type",
    "model_variant",
    "binning_strategy",
    "training_method",
    "mitigation_technique",
]

PERFORMANCE_METRICS = {
    "f1": ["f1_value", "f1_score", "f1"],
    "recall": ["recall_value", "recall"],
    "precision": ["precision_value", "precision"],
    "auc_roc": ["auc_value", "auc_roc", "auc"],
    "accuracy": ["accuracy_value", "accuracy"],
}

HIGHER_IS_BETTER = {
    "f1": True,
    "recall": True,
    "precision": True,
    "auc_roc": True,
    "accuracy": True,
    "fairness_gap": False,
    "demographic_parity_gap": False,
    "equalized_odds_gap": False,
    "equalized_odds_tpr_gap": False,
    "equalized_odds_fpr_gap": False,
    "tpr": True,
    "equal_opportunity_tpr": True,
    "predictive_parity_precision": True,
    "fpr": False,
    "fnr": False,
    "demographic_parity_rate": False,
}


def _as_float(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _first_numeric(row: pd.Series, columns: list[str]) -> float | None:
    for col in columns:
        if col in row.index:
            value = _as_float(row.get(col))
            if value is not None:
                return value
    return None


def _max_numeric(row: pd.Series, columns: list[str]) -> float | None:
    values = [_as_float(row.get(col)) for col in columns if col in row.index]
    values = [v for v in values if v is not None]
    if not values:
        return None
    return float(np.nanmax(values))


def _identity(row: pd.Series) -> dict[str, Any]:
    return {col: row.get(col) for col in IDENTITY_COLS}


def _gap_columns(df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    dp_cols = [c for c in df.columns if c.startswith("dem_parity_") and c.endswith("_max_diff")]
    eq_tpr_cols = [c for c in df.columns if c.startswith("eq_odds_") and "_tpr_" in c]
    eq_fpr_cols = [c for c in df.columns if c.startswith("eq_odds_") and "_fpr_" in c]
    return dp_cols, eq_tpr_cols, eq_fpr_cols


def _metric_specs_for_row(row: pd.Series, full_df: pd.DataFrame) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for metric, columns in PERFORMANCE_METRICS.items():
        specs.append(
            {
                "metric_family": "performance",
                "metric": metric,
                "value": _first_numeric(row, columns),
                "higher_is_better": True,
                "unit": "rate",
            }
        )

    dp_cols, eq_tpr_cols, eq_fpr_cols = _gap_columns(full_df)
    eq_cols = eq_tpr_cols + eq_fpr_cols
    dp_gap = _first_numeric(row, ["dp_max_diff"])
    if dp_gap is None:
        dp_gap = _max_numeric(row, dp_cols)
    eq_gap = _first_numeric(row, ["eq_odds_max_diff"])
    if eq_gap is None:
        eq_gap = _max_numeric(row, eq_cols)
    gap_specs = [
        ("fairness_gap", _first_numeric(row, ["fairness_gap"])),
        ("demographic_parity_gap", dp_gap),
        ("equalized_odds_gap", eq_gap),
        ("equalized_odds_tpr_gap", _max_numeric(row, eq_tpr_cols)),
        ("equalized_odds_fpr_gap", _max_numeric(row, eq_fpr_cols)),
    ]
    for metric, value in gap_specs:
        specs.append(
            {
                "metric_family": "fairness_gap",
                "metric": metric,
                "value": value,
                "higher_is_better": False,
                "unit": "rate",
            }
        )
    return specs


def build_experiment_index(full_df: pd.DataFrame) -> pd.DataFrame:
    exact, no_variant = build_baseline_lookups(full_df)
    rows = []
    for _, row in full_df.iterrows():
        baseline, source = find_matching_baseline(row, exact, no_variant)
        is_baseline = str(row.get("mitigation_technique")) == "baseline"
        rows.append(
            {
                **_identity(row),
                "is_baseline": is_baseline,
                "status": row.get("status"),
                "baseline_experiment_id": (
                    baseline.get("experiment_id") if baseline is not None else None
                ),
                "baseline_source": "self" if is_baseline else source,
            }
        )
    return pd.DataFrame(rows)


def build_metric_values(full_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in full_df.iterrows():
        for spec in _metric_specs_for_row(row, full_df):
            if spec["value"] is None:
                continue
            rows.append({**_identity(row), **spec})
    return pd.DataFrame(rows)


def build_metric_deltas(full_df: pd.DataFrame) -> pd.DataFrame:
    exact, no_variant = build_baseline_lookups(full_df)
    rows = []
    for _, row in full_df.iterrows():
        if str(row.get("mitigation_technique")) == "baseline":
            continue
        baseline, source = find_matching_baseline(row, exact, no_variant)
        if baseline is None:
            continue
        baseline_specs = {spec["metric"]: spec for spec in _metric_specs_for_row(baseline, full_df)}
        for spec in _metric_specs_for_row(row, full_df):
            base_spec = baseline_specs.get(spec["metric"])
            if base_spec is None or spec["value"] is None or base_spec["value"] is None:
                continue
            delta = float(spec["value"] - base_spec["value"])
            improvement = delta if spec["higher_is_better"] else -delta
            rows.append(
                {
                    **_identity(row),
                    "baseline_experiment_id": baseline.get("experiment_id"),
                    "baseline_source": source,
                    "metric_family": spec["metric_family"],
                    "metric": spec["metric"],
                    "baseline_value": base_spec["value"],
                    "experiment_value": spec["value"],
                    "delta": delta,
                    "improvement": improvement,
                    "higher_is_better": spec["higher_is_better"],
                    "unit": spec["unit"],
                }
            )
    return pd.DataFrame(rows)


def build_group_metric_values(per_group_df: pd.DataFrame | None) -> pd.DataFrame:
    if per_group_df is None or per_group_df.empty:
        return pd.DataFrame()
    rows = []
    for _, row in per_group_df.iterrows():
        value = _as_float(row.get("experiment_value"))
        if value is None:
            continue
        metric = row.get("metric")
        rows.append(
            {
                **{col: row.get(col) for col in IDENTITY_COLS},
                "sensitive_attr": normalize_sensitive_attr(row.get("sensitive_attr")),
                "group": row.get("group"),
                "metric": metric,
                "value": value,
                "overall_value": _as_float(row.get("experiment_overall_value")),
                "group_count": row.get("group_count"),
                "positive_count": row.get("positive_count"),
                "negative_count": row.get("negative_count"),
                "higher_is_better": HIGHER_IS_BETTER.get(str(metric), True),
            }
        )
    return pd.DataFrame(rows)


def _group_improvement(row: pd.Series) -> tuple[float | None, float | None, float | None]:
    base = _as_float(row.get("baseline_value"))
    exp = _as_float(row.get("experiment_value"))
    if base is None or exp is None:
        return None, None, None

    metric = str(row.get("metric"))
    delta = exp - base
    if metric in {"tpr", "equal_opportunity_tpr", "predictive_parity_precision"}:
        return delta, None, None
    if metric in {"fpr", "fnr"}:
        return -delta, None, None
    if metric == "demographic_parity_rate":
        base_overall = _as_float(row.get("baseline_overall_value"))
        exp_overall = _as_float(row.get("experiment_overall_value"))
        if base_overall is None or exp_overall is None:
            return None, None, None
        base_distance = abs(base - base_overall)
        exp_distance = abs(exp - exp_overall)
        return base_distance - exp_distance, base_distance, exp_distance
    return delta, None, None


def build_group_metric_deltas(per_group_df: pd.DataFrame | None) -> pd.DataFrame:
    if per_group_df is None or per_group_df.empty:
        return pd.DataFrame()
    rows = []
    for _, row in per_group_df.iterrows():
        if str(row.get("mitigation_technique")) == "baseline":
            continue
        base = _as_float(row.get("baseline_value"))
        exp = _as_float(row.get("experiment_value"))
        if base is None or exp is None:
            continue
        improvement, base_distance, exp_distance = _group_improvement(row)
        if improvement is None:
            continue
        metric = row.get("metric")
        rows.append(
            {
                **{col: row.get(col) for col in IDENTITY_COLS},
                "baseline_experiment_id": row.get("baseline_experiment_id"),
                "baseline_source": row.get("baseline_source"),
                "sensitive_attr": normalize_sensitive_attr(row.get("sensitive_attr")),
                "group": row.get("group"),
                "metric": metric,
                "baseline_value": base,
                "experiment_value": exp,
                "delta": exp - base,
                "improvement": improvement,
                "baseline_overall_value": _as_float(row.get("baseline_overall_value")),
                "experiment_overall_value": _as_float(row.get("experiment_overall_value")),
                "baseline_distance_to_overall": base_distance,
                "experiment_distance_to_overall": exp_distance,
                "distance_improvement": (
                    base_distance - exp_distance
                    if base_distance is not None and exp_distance is not None
                    else None
                ),
                "group_count": row.get("group_count"),
                "positive_count": row.get("positive_count"),
                "negative_count": row.get("negative_count"),
                "higher_is_better": HIGHER_IS_BETTER.get(str(metric), True),
            }
        )
    return pd.DataFrame(rows)


def build_fairness_evidence_summary(
    full_df: pd.DataFrame,
    group_metric_deltas: pd.DataFrame | None,
    config: dict[str, Any],
) -> pd.DataFrame:
    selection_cfg = config.get("selection", {})
    primary_model = selection_cfg.get("primary_model_type", "logistic_regression")
    primary_dataset = selection_cfg.get("primary_dataset")
    min_recall_delta = float(selection_cfg.get("min_recall_delta", -0.03))
    top_n = int(selection_cfg.get("top_n", 5))

    if full_df is None or full_df.empty or "mitigation_technique" not in full_df.columns:
        return pd.DataFrame()

    df = full_df[full_df["mitigation_technique"] != "baseline"].copy()
    if "model_type" in df.columns:
        df = df[df["model_type"].astype(str) == str(primary_model)]
    if primary_dataset and "dataset" in df.columns:
        df = df[df["dataset"].astype(str) == str(primary_dataset)]
    if df.empty:
        return pd.DataFrame()

    for col in [
        "delta_fairness_gap",
        "delta_recall",
        "delta_f1",
        "delta_precision",
        "delta_auc",
        "delta_accuracy",
        "delta_dp_gap",
        "delta_eq_tpr_gap",
        "delta_eq_fpr_gap",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = np.nan

    eligible = df[(df["delta_fairness_gap"] > 0) & (df["delta_recall"] >= min_recall_delta)]
    selection_reason = f"delta_fairness_gap > 0 and delta_recall >= {min_recall_delta}"
    if eligible.empty:
        eligible = df[df["delta_fairness_gap"] > 0]
        selection_reason = "delta_fairness_gap > 0"
    if eligible.empty:
        eligible = df
        selection_reason = "fallback: best available mitigation row"

    eligible = eligible.sort_values(
        ["delta_fairness_gap", "delta_recall", "delta_f1"],
        ascending=[False, False, False],
        na_position="last",
    )

    group_counts = {}
    if group_metric_deltas is not None and not group_metric_deltas.empty:
        gd = group_metric_deltas.copy()
        gd["improvement"] = pd.to_numeric(gd["improvement"], errors="coerce")
        gd["group_key"] = gd["sensitive_attr"].astype(str) + ":" + gd["group"].astype(str)
        for exp_id, group in gd.dropna(subset=["improvement"]).groupby("experiment_id"):
            scores = group.groupby("group_key")["improvement"].mean()
            group_counts[str(exp_id)] = (
                int((scores > 1e-12).sum()),
                int((scores < -1e-12).sum()),
            )

    rows = []
    seen_mitigations = set()
    for _, row in eligible.iterrows():
        mitigation = row.get("mitigation_technique")
        if mitigation in seen_mitigations:
            continue
        seen_mitigations.add(mitigation)
        improved, worsened = group_counts.get(str(row.get("experiment_id")), (0, 0))
        rows.append(
            {
                "rank": len(rows) + 1,
                **_identity(row),
                "delta_fairness_gap": row.get("delta_fairness_gap"),
                "delta_dp_gap": row.get("delta_dp_gap"),
                "delta_eq_tpr_gap": row.get("delta_eq_tpr_gap"),
                "delta_eq_fpr_gap": row.get("delta_eq_fpr_gap"),
                "delta_f1": row.get("delta_f1"),
                "delta_recall": row.get("delta_recall"),
                "delta_precision": row.get("delta_precision"),
                "delta_auc": row.get("delta_auc"),
                "delta_accuracy": row.get("delta_accuracy"),
                "groups_improved": improved,
                "groups_worsened": worsened,
                "selection_reason": selection_reason,
            }
        )
        if len(rows) >= top_n:
            break
    return pd.DataFrame(rows)


def write_canonical_comparison_outputs(
    full_df: pd.DataFrame,
    per_group_df: pd.DataFrame | None,
    output_dir: Path,
    config: dict[str, Any],
    run_id: str | None = None,
    input_paths: dict[str, str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Write canonical comparison evidence tables and manifest."""
    output_dir.mkdir(parents=True, exist_ok=True)

    experiment_index = build_experiment_index(full_df)
    metric_values = build_metric_values(full_df)
    metric_deltas = build_metric_deltas(full_df)
    group_metric_values = build_group_metric_values(per_group_df)
    group_metric_deltas = build_group_metric_deltas(per_group_df)
    fairness_summary = build_fairness_evidence_summary(full_df, group_metric_deltas, config)

    tables = {
        "experiment_index": experiment_index,
        "metric_values": metric_values,
        "metric_deltas": metric_deltas,
        "group_metric_values": group_metric_values,
        "group_metric_deltas": group_metric_deltas,
        "fairness_evidence_summary": fairness_summary,
    }
    for name, df in tables.items():
        df.to_csv(output_dir / f"{name}.csv", index=False)

    manifest = {
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_paths": input_paths or {},
        "row_counts": {name: int(len(df)) for name, df in tables.items()},
        "canonical_outputs": [
            "experiment_index.csv",
            "metric_values.csv",
            "metric_deltas.csv",
            "group_metric_values.csv",
            "group_metric_deltas.csv",
            "fairness_evidence_summary.csv",
        ],
        "compatibility_outputs": [
            "full_comparison.csv",
            "per_group.csv",
            "binning_summary.csv",
            "mitigation_summary.csv",
            "dataset_summary.csv",
            "cross_model_summary.csv",
            "tradeoff_<dataset>.csv",
            "pareto_<dataset>.csv",
            "top_configs.csv",
        ],
        "dissertation_evidence_policy": (
            "Use canonical metric-level tables for claims and figures. Compatibility "
            "outputs may contain score_value for transitional ranking/model-promotion "
            "logic, but score_value is not dissertation evidence."
        ),
        "baseline_matching_policy": (
            "dataset + model_type + model_variant + binning_strategy + training_method; "
            "fallback omits model_variant"
        ),
        "metric_direction_policy": {
            "raw_delta": "experiment_value - baseline_value",
            "improvement": "positive always better",
            "higher_is_better": [
                "f1",
                "recall",
                "precision",
                "auc_roc",
                "accuracy",
                "tpr",
                "predictive_parity_precision",
            ],
            "lower_is_better": [
                "fairness_gap",
                "demographic_parity_gap",
                "equalized_odds_gap",
                "equalized_odds_tpr_gap",
                "equalized_odds_fpr_gap",
                "fpr",
            ],
            "demographic_parity_group": "improvement = reduced distance to overall rate",
        },
        "warnings": [],
    }
    with (output_dir / "comparison_manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)

    return tables
