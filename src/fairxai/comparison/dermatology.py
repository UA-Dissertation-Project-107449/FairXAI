"""Dermatology baseline comparison: collate per-model metrics + fairness deltas.

Lean, manifest-free comparison stage. Reads the baseline artifacts a dermatology
run already produces and writes a single canonical table across architectures:

- ``baseline/results/<dataset>_<model>_metrics.json`` -> test performance + run cost
- ``baseline/prediction_fairness/fairness_report.json`` -> per-attribute deltas
  (optional; produced by stage 8 / :mod:`fairxai.fairness.image_assessment`)

Outputs ``baseline/comparison/model_comparison.csv`` (one row per dataset x model)
and ``model_comparison.md``. Optional figures are rendered from those same rows by
:mod:`fairxai.viz.dermatology_comparison` (imported only when requested, so this
module stays matplotlib-free on the CSV/Markdown path). No model load, no
retraining, no experiment manifest.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Fairness deltas pulled per sensitive attribute, as
#   column suffix -> (group_fairness metric key, field within that metric).
_FAIRNESS_DELTAS: dict[str, tuple[str, str]] = {
    "dp_delta": ("demographic_parity", "max_difference"),
    "tpr_delta": ("equalized_odds", "tpr_max_difference"),
    "fpr_delta": ("equalized_odds", "fpr_max_difference"),
    "eo_delta": ("equal_opportunity", "max_difference"),
}

# Fixed performance columns, in display order.
_PERF_COLUMNS = ["accuracy", "precision", "recall", "f1", "auc"]
_FAIRNESS_LABELS = {
    "dp_delta": "DP delta",
    "tpr_delta": "TPR delta",
    "fpr_delta": "FPR delta",
    "eo_delta": "Equal-opp delta",
}


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _discover_metrics(
    run_root: Path,
    datasets: Optional[Iterable[str]],
    model_types: Optional[Iterable[str]],
) -> dict[str, dict[str, Any]]:
    """Return ``{run_key: metrics_json}`` for successful baseline models."""
    results_dir = run_root / "baseline" / "results"
    if not results_dir.is_dir():
        logger.warning("No baseline results dir at %s", results_dir)
        return {}

    dataset_filter = set(datasets) if datasets else None
    model_filter = set(model_types) if model_types else None

    metrics: dict[str, dict[str, Any]] = {}
    for path in sorted(results_dir.glob("*_metrics.json")):
        run_key = path.name[: -len("_metrics.json")]
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Skipping unreadable metrics %s: %s", path.name, exc)
            continue
        if data.get("status") not in (None, "success"):
            logger.info("Skipping non-success model %s (status=%s)", run_key, data.get("status"))
            continue
        model_type = data.get("model_type", "")
        dataset = run_key[: -(len(model_type) + 1)] if model_type else run_key
        if dataset_filter and dataset not in dataset_filter:
            continue
        if model_filter and model_type not in model_filter:
            continue
        data["_dataset"] = dataset
        metrics[run_key] = data
    return metrics


def _load_fairness(run_root: Path) -> dict[str, Any]:
    """Return the stage-8 fairness report keyed by run_key, or ``{}`` if absent."""
    path = run_root / "baseline" / "prediction_fairness" / "fairness_report.json"
    if not path.is_file():
        logger.info("No fairness report at %s (fairness columns will be blank).", path)
        return {}
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Unreadable fairness report %s: %s", path, exc)
        return {}


def _sensitive_attrs(fairness: dict[str, Any]) -> list[str]:
    """Stable union of sensitive attributes that have fairness output."""
    attrs: list[str] = []
    for report in fairness.values():
        for attr in report.get("sensitive_attributes", {}):
            if attr not in attrs:
                attrs.append(attr)
    return attrs


def _fairness_deltas(report: dict[str, Any], attr: str) -> dict[str, Optional[float]]:
    """Extract the configured deltas for one attribute (None if not comparable)."""
    out: dict[str, Optional[float]] = {suffix: None for suffix in _FAIRNESS_DELTAS}
    gf = report.get("sensitive_attributes", {}).get(attr, {}).get("group_fairness", {})
    for suffix, (metric, field) in _FAIRNESS_DELTAS.items():
        out[suffix] = _safe_float(gf.get(metric, {}).get(field))
    return out


def build_rows(
    fairness: dict[str, Any], metrics: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    """One row per dataset x model: identity, performance, cost, fairness deltas."""
    attrs = _sensitive_attrs(fairness)
    rows: list[dict[str, Any]] = []
    for run_key, data in sorted(metrics.items()):
        test = data.get("test_metrics", {})
        row: dict[str, Any] = {
            "dataset": data.get("_dataset", run_key),
            "model": data.get("model_type", run_key),
            "architecture": data.get("architecture"),
            "weights_name": data.get("weights_name"),
            "feature_cache": data.get("feature_cache"),
            "n_train": data.get("n_train"),
            "n_test": data.get("n_test"),
            "train_time_seconds": _safe_float(data.get("train_time_seconds")),
            "epochs_run": data.get("epochs_run"),
            "best_epoch": data.get("best_epoch"),
            "early_stopped": data.get("early_stopped"),
            "accuracy": _safe_float(test.get("accuracy")),
            "precision": _safe_float(test.get("precision")),
            "recall": _safe_float(test.get("recall")),
            "f1": _safe_float(test.get("f1_score")),
            "auc": _safe_float(test.get("auc_roc")),
        }
        report = fairness.get(run_key, {})
        for attr in attrs:
            deltas = _fairness_deltas(report, attr) if report else dict.fromkeys(_FAIRNESS_DELTAS)
            for suffix, value in deltas.items():
                row[f"{attr}_{suffix}"] = value
        rows.append(row)
    return rows


def render_markdown(rows: list[dict[str, Any]], attrs: list[str]) -> str:
    """Performance table + one fairness table per sensitive attribute."""
    lines = ["# Dermatology Model Comparison", ""]
    if not rows:
        lines.append("_No baseline models found for this run._")
        return "\n".join(lines) + "\n"

    lines += [
        "## Test performance",
        "",
        "| model | n_test | accuracy | precision | recall | f1 | auc | train_s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in sorted(rows, key=lambda x: (x["auc"] is None, -(x["auc"] or 0.0))):
        lines.append(
            f"| {r['model']} | {r['n_test'] or 'n/a'} | "
            + " | ".join(_fmt(r[c]) for c in _PERF_COLUMNS)
            + f" | {_fmt(r['train_time_seconds'])} |"
        )

    for attr in attrs:
        lines += [
            "",
            f"## Fairness — {attr} (max group differences, lower = fairer)",
            "",
            "| model | DP Δ | TPR Δ | FPR Δ | equal-opp Δ |",
            "|---|---:|---:|---:|---:|",
        ]
        for r in sorted(rows, key=lambda x: x["model"]):
            lines.append(
                f"| {r['model']} | {_fmt(r.get(f'{attr}_dp_delta'))} | "
                f"{_fmt(r.get(f'{attr}_tpr_delta'))} | {_fmt(r.get(f'{attr}_fpr_delta'))} | "
                f"{_fmt(r.get(f'{attr}_eo_delta'))} |"
            )

    lines += [
        "",
        "_Deltas exclude degenerate groups (no positives/negatives) and groups below "
        "the min-support gate; see the stage-8 fairness report for details._",
    ]
    return "\n".join(lines) + "\n"


def _render_figures(
    rows: list[dict[str, Any]],
    attrs: list[str],
    out_dir: Path,
    metrics: dict[str, dict[str, Any]],
) -> None:
    """Delegate figure rendering to viz (matplotlib imported only here)."""
    try:
        from fairxai.viz.dermatology_comparison import (
            render_comparison_figures,
            render_learning_curves,
        )
    except ImportError as exc:
        logger.warning("Skipping dermatology comparison figures (viz/matplotlib): %s", exc)
        return
    render_comparison_figures(
        rows,
        attrs,
        out_dir,
        perf_columns=_PERF_COLUMNS,
        fairness_labels=_FAIRNESS_LABELS,
    )
    render_learning_curves(metrics, out_dir)


def compare_run(
    run_root: Path,
    *,
    datasets: Optional[Iterable[str]] = None,
    model_types: Optional[Iterable[str]] = None,
    write_figures: bool = False,
) -> list[dict[str, Any]]:
    """Collate baseline metrics + fairness into ``baseline/comparison/`` outputs."""
    metrics = _discover_metrics(run_root, datasets, model_types)
    fairness = _load_fairness(run_root)
    attrs = _sensitive_attrs(fairness)
    rows = build_rows(fairness, metrics)

    out_dir = run_root / "baseline" / "comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(rows).to_csv(out_dir / "model_comparison.csv", index=False)
    md = render_markdown(rows, attrs)
    (out_dir / "model_comparison.md").write_text(md)
    if write_figures:
        _render_figures(rows, attrs, out_dir, metrics)
    logger.info("Wrote comparison for %d model(s) to %s", len(rows), out_dir)
    return rows
