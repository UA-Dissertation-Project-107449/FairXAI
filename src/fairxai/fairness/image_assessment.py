"""Post-prediction fairness assessment for dermatology image baselines.

This is the dermatology counterpart to ``scripts/common/assess_predictions.py``,
deliberately kept lighter: image prediction CSVs already carry ``y_true``,
``y_pred``, ``y_proba`` plus plaintext/encoded sensitive columns, so there is no
scaled-tabular decode layer and no combinatorial-experiment manifest to read.

Pipeline boundary: this reads saved baseline outputs only. No model, no
retraining. The heavy lifting (per-attribute group fairness) is delegated to the
modality-agnostic :class:`fairxai.fairness.metrics.FairnessMetrics`; this module
adds discovery, min-group gating, per-group performance, label decoding, and the
JSON/Markdown report shape.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from fairxai.fairness.metrics import FairnessMetrics

logger = logging.getLogger(__name__)

# Human-readable labels for encoded sensitive columns. The loader stores sex as
# Female=0 / Male=1 / unknown=-1 (src/fairxai/data/loaders.py); decode it so the
# report reads "Female" rather than "0". Both int and str keys cover CSV dtypes.
_VALUE_LABELS: dict[str, dict[Any, str]] = {
    "sex": {
        0: "Female",
        1: "Male",
        -1: "Unknown",
        "0": "Female",
        "1": "Male",
        "-1": "Unknown",
    },
}

DEFAULT_MIN_GROUP_SAMPLES = 50
DEFAULT_INTERSECTION_MIN_GROUP_SAMPLES = 30
DEFAULT_GROUP_VIEWS = [
    "age_coarse",
    "sex",
    "fitzpatrick_group",
    "sex_x_fitzpatrick",
    "age_coarse_x_fitzpatrick",
]


def decode_groups(df: pd.DataFrame, attr: str) -> pd.Series:
    """Return *attr* as readable group labels, decoding known encoded columns.

    Unmapped values (and unknown attributes) pass through as strings, so grouping
    is always well-defined.
    """
    series = df[attr]
    mapping = _VALUE_LABELS.get(attr)
    if mapping is not None:
        series = series.map(lambda v: mapping.get(v, str(v)))
    return series.astype(str)


def derive_age_coarse(df: pd.DataFrame) -> pd.Series:
    """Map PAD age groups to a small post-hoc view: <40, 40-59, 60+, Unknown."""

    def _map(value: Any) -> str:
        if pd.isna(value):
            return "Unknown"
        text = str(value).strip()
        if not text or text.lower() in {"nan", "none", "unknown"}:
            return "Unknown"
        if text in {"<20", "20-39"}:
            return "<40"
        if text == "40-59":
            return "40-59"
        if text in {"60-79", "80+"}:
            return "60+"
        return text

    if "age_group" not in df.columns:
        return pd.Series(["Unknown"] * len(df), index=df.index)
    return df["age_group"].map(_map).astype(str)


def derive_group_view_columns(
    df: pd.DataFrame,
    views: list[str],
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    """Add configured post-hoc group-view columns to a copy of *df*.

    Returns ``(frame, metadata)`` where metadata records source attributes and
    whether a view is exploratory/intersectional.
    """
    work = df.copy()
    metadata: dict[str, dict[str, Any]] = {}
    for view in views:
        if view == "age_coarse":
            work[view] = derive_age_coarse(work)
            metadata[view] = {"source_attributes": ["age_group"], "exploratory": False}
        elif view == "sex":
            if "sex" not in work.columns:
                logger.warning("Group view '%s' skipped: missing sex column", view)
                continue
            work[view] = decode_groups(work, "sex")
            metadata[view] = {"source_attributes": ["sex"], "exploratory": False}
        elif view == "fitzpatrick_group":
            if "fitzpatrick_group" not in work.columns:
                logger.warning("Group view '%s' skipped: missing fitzpatrick_group column", view)
                continue
            work[view] = decode_groups(work, "fitzpatrick_group")
            metadata[view] = {"source_attributes": ["fitzpatrick_group"], "exploratory": False}
        elif view == "sex_x_fitzpatrick":
            if "sex" not in work.columns or "fitzpatrick_group" not in work.columns:
                logger.warning("Group view '%s' skipped: missing sex or fitzpatrick_group", view)
                continue
            sex = decode_groups(work, "sex")
            fitz = decode_groups(work, "fitzpatrick_group")
            work[view] = sex + " x " + fitz
            metadata[view] = {
                "source_attributes": ["sex", "fitzpatrick_group"],
                "exploratory": True,
            }
        elif view == "age_coarse_x_fitzpatrick":
            if "age_group" not in work.columns or "fitzpatrick_group" not in work.columns:
                logger.warning(
                    "Group view '%s' skipped: missing age_group or fitzpatrick_group", view
                )
                continue
            age = derive_age_coarse(work)
            fitz = decode_groups(work, "fitzpatrick_group")
            work[view] = age + " x " + fitz
            metadata[view] = {
                "source_attributes": ["age_group", "fitzpatrick_group"],
                "exploratory": True,
            }
        else:
            logger.warning("Unknown group view '%s'; skipping", view)
    return work, metadata


def _binary_performance(
    y_true: pd.Series, y_pred: pd.Series, y_proba: Optional[pd.Series]
) -> dict[str, Optional[float]]:
    """Accuracy/precision/recall/F1/AUC, with ``None`` for undefined quantities.

    Recall is undefined when a group has no actual positives, precision when it
    makes no positive predictions, F1 when either is undefined, AUC when only one
    class is present. Returning ``None`` (rather than 0) keeps degenerate-group
    rows honest instead of reporting a misleading zero.
    """
    has_pos = bool((y_true == 1).any())
    pred_pos = bool((y_pred == 1).any())
    recall = float(recall_score(y_true, y_pred, zero_division=0)) if has_pos else None
    precision = float(precision_score(y_true, y_pred, zero_division=0)) if pred_pos else None
    f1 = (
        float(f1_score(y_true, y_pred, zero_division=0))
        if recall is not None and precision is not None
        else None
    )
    auc = (
        float(roc_auc_score(y_true, y_proba))
        if y_proba is not None and y_true.nunique() == 2
        else None
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
    }


def _degenerate_groups(df: pd.DataFrame, group_col: str, groups: list[str]) -> list[dict[str, Any]]:
    """Groups with no positives or no negatives — TPR/FPR there are undefined.

    Such groups must be excluded from group-difference metrics, otherwise their
    undefined rate (treated as 0) inflates the max-difference deltas.
    """
    degenerate: list[dict[str, Any]] = []
    for group in groups:
        gdf = df[df[group_col] == group]
        pos = int((gdf["y_true"] == 1).sum())
        neg = int((gdf["y_true"] == 0).sum())
        if pos == 0 or neg == 0:
            degenerate.append(
                {
                    "group": group,
                    "count": int(len(gdf)),
                    "reason": "no positives" if pos == 0 else "no negatives",
                }
            )
    return degenerate


def _partition_groups(
    counts: "pd.Series", min_group_samples: int
) -> tuple[list[str], list[dict[str, Any]]]:
    """Split group value-counts into kept (>= min) and skipped (< min)."""
    kept = [str(g) for g, n in counts.items() if n >= min_group_samples]
    skipped = [
        {"group": str(g), "count": int(n)} for g, n in counts.items() if n < min_group_samples
    ]
    return kept, skipped


def _group_performance(
    df: pd.DataFrame, group_col: str, kept: list[str]
) -> dict[str, dict[str, Optional[float]]]:
    """Per-kept-group prevalence + performance from an already-decoded frame."""
    out: dict[str, dict[str, Optional[float]]] = {}
    for group in kept:
        gdf = df[df[group_col] == group]
        proba = gdf["y_proba"] if "y_proba" in gdf.columns else None
        perf = _binary_performance(gdf["y_true"], gdf["y_pred"], proba)
        perf["n"] = int(len(gdf))
        perf["prevalence"] = float(gdf["y_true"].mean())
        out[group] = perf
    return out


def assess_predictions_frame(
    df: pd.DataFrame,
    sensitive_attrs: list[str],
    *,
    min_group_samples: int = DEFAULT_MIN_GROUP_SAMPLES,
) -> dict[str, Any]:
    """Compute the fairness report for a single prediction DataFrame.

    Group fairness for an attribute is computed only over groups meeting
    *min_group_samples*; undersized groups are dropped from the metrics but
    reported under ``skipped_groups`` so nothing is silently ignored.
    """
    proba = df["y_proba"] if "y_proba" in df.columns else None
    report: dict[str, Any] = {
        "n_test": int(len(df)),
        "min_group_samples": int(min_group_samples),
        "overall_performance": _binary_performance(df["y_true"], df["y_pred"], proba),
        "sensitive_attributes": {},
    }

    for attr in sensitive_attrs:
        if attr not in df.columns:
            logger.warning("Sensitive attribute '%s' not in predictions; skipping", attr)
            continue

        decoded_col = f"__group__{attr}"
        work = df.copy()
        work[decoded_col] = decode_groups(work, attr)

        counts = work[decoded_col].value_counts()
        kept, skipped = _partition_groups(counts, min_group_samples)
        degenerate = _degenerate_groups(work, decoded_col, kept)
        degenerate_names = {d["group"] for d in degenerate}
        comparison = [g for g in kept if g not in degenerate_names]

        attr_report: dict[str, Any] = {
            "kept_groups": {g: int(counts[g]) for g in kept},
            "skipped_groups": skipped,
            "degenerate_groups": degenerate,
            "group_performance": _group_performance(work, decoded_col, kept),
        }

        # Deltas are computed only over comparison groups (≥ min support AND both
        # classes present); degenerate groups stay in the performance table above.
        if len(comparison) >= 2:
            comp_df = work[work[decoded_col].isin(comparison)]
            fm = FairnessMetrics(sensitive_attributes=[decoded_col])
            all_metrics = fm.calculate_all_metrics(comp_df)
            attr_report["group_fairness"] = all_metrics["group_fairness"].get(decoded_col, {})
            attr_report["calibration"] = all_metrics["calibration"].get(decoded_col, {})
        else:
            attr_report["group_fairness"] = {}
            attr_report["calibration"] = {}
            attr_report["note"] = (
                "fewer than 2 comparison groups (min support + both classes present); "
                "no group-difference metrics"
            )

        report["sensitive_attributes"][attr] = attr_report

    return report


def _json_default(value: Any) -> Any:
    """Coerce numpy scalars (FairnessMetrics returns them) to native types."""
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _fmt(value: Optional[float]) -> str:
    """Format a metric for Markdown (``n/a`` for None)."""
    return "n/a" if value is None else f"{value:.3f}"


def render_markdown(reports: dict[str, dict[str, Any]]) -> str:
    """Render ``{<dataset>_<model>: report}`` as a Markdown fairness summary."""
    lines = ["# Dermatology Prediction Fairness Report", ""]
    for key in sorted(reports):
        report = reports[key]
        lines.append(f"## {key}")
        op = report["overall_performance"]
        lines.append(
            f"- n_test: {report['n_test']} · acc {_fmt(op['accuracy'])} · "
            f"f1 {_fmt(op['f1'])} · auc {_fmt(op['auc'])} · "
            f"min_group_samples {report['min_group_samples']}"
        )
        lines.append("")
        for attr, ar in sorted(report["sensitive_attributes"].items()):
            lines.append(f"### {attr}")
            if ar["skipped_groups"]:
                skipped = ", ".join(f"{s['group']} (n={s['count']})" for s in ar["skipped_groups"])
                lines.append(f"- skipped (under min): {skipped}")
            if ar.get("degenerate_groups"):
                degen = ", ".join(f"{d['group']} ({d['reason']})" for d in ar["degenerate_groups"])
                lines.append(f"- excluded from deltas (degenerate): {degen}")
            if not ar["group_performance"]:
                lines.append("- no groups met the minimum support threshold")
                lines.append("")
                continue

            lines.append("")
            lines.append("| group | n | prevalence | accuracy | recall | auc |")
            lines.append("|---|---:|---:|---:|---:|---:|")
            for group, perf in sorted(ar["group_performance"].items()):
                lines.append(
                    f"| {group} | {perf['n']} | {_fmt(perf['prevalence'])} | "
                    f"{_fmt(perf['accuracy'])} | {_fmt(perf['recall'])} | {_fmt(perf['auc'])} |"
                )
            lines.append("")

            gf = ar.get("group_fairness", {})
            if gf:
                dp = gf.get("demographic_parity", {}).get("max_difference")
                eo = gf.get("equalized_odds", {})
                eopp = gf.get("equal_opportunity", {}).get("max_difference")
                lines.append(
                    f"- demographic_parity Δ {_fmt(dp)} · "
                    f"TPR Δ {_fmt(eo.get('tpr_max_difference'))} · "
                    f"FPR Δ {_fmt(eo.get('fpr_max_difference'))} · "
                    f"equal_opportunity Δ {_fmt(eopp)}"
                )
                lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _flatten_for_csv(reports: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """One row per (dataset_model, attribute, group) for dissertation import."""
    rows: list[dict[str, Any]] = []
    for key, report in reports.items():
        for attr, ar in report["sensitive_attributes"].items():
            gf = ar.get("group_fairness", {})
            dp = gf.get("demographic_parity", {}).get("max_difference")
            eo = gf.get("equalized_odds", {})
            degenerate_names = {d["group"] for d in ar.get("degenerate_groups", [])}
            for group, perf in ar["group_performance"].items():
                rows.append(
                    {
                        "run_key": key,
                        "sensitive_attribute": attr,
                        "group": group,
                        "n": perf["n"],
                        "prevalence": perf["prevalence"],
                        "accuracy": perf["accuracy"],
                        "recall": perf["recall"],
                        "auc": perf["auc"],
                        "degenerate": group in degenerate_names,
                        "demographic_parity_max_diff": dp,
                        "tpr_max_diff": eo.get("tpr_max_difference"),
                        "fpr_max_diff": eo.get("fpr_max_difference"),
                    }
                )
    return pd.DataFrame(rows)


def assess_group_views_frame(
    df: pd.DataFrame,
    views: list[str],
    *,
    min_group_samples: int = DEFAULT_MIN_GROUP_SAMPLES,
    intersection_min_group_samples: int = DEFAULT_INTERSECTION_MIN_GROUP_SAMPLES,
) -> dict[str, Any]:
    """Compute post-hoc group-view reports for one prediction DataFrame."""
    work, metadata = derive_group_view_columns(df, views)
    proba = work["y_proba"] if "y_proba" in work.columns else None
    report: dict[str, Any] = {
        "n_test": int(len(work)),
        "overall_performance": _binary_performance(work["y_true"], work["y_pred"], proba),
        "group_views": {},
    }
    for view in views:
        if view not in metadata or view not in work.columns:
            continue
        view_min = (
            intersection_min_group_samples if metadata[view]["exploratory"] else min_group_samples
        )
        view_report = assess_predictions_frame(work, [view], min_group_samples=view_min)[
            "sensitive_attributes"
        ][view]
        view_report["min_group_samples"] = int(view_min)
        view_report["source_attributes"] = metadata[view]["source_attributes"]
        view_report["exploratory"] = bool(metadata[view]["exploratory"])
        report["group_views"][view] = view_report
    return report


def render_group_view_markdown(reports: dict[str, dict[str, Any]]) -> str:
    """Render ``{<dataset>_<model>: group_view_report}`` as Markdown."""
    lines = ["# Dermatology Post-Hoc Group View Fairness Report", ""]
    for key in sorted(reports):
        report = reports[key]
        op = report["overall_performance"]
        lines.append(f"## {key}")
        lines.append(
            f"- n_test: {report['n_test']} · acc {_fmt(op['accuracy'])} · "
            f"f1 {_fmt(op['f1'])} · auc {_fmt(op['auc'])}"
        )
        lines.append("")
        for view, vr in sorted(report["group_views"].items()):
            label = "exploratory" if vr.get("exploratory") else "primary"
            lines.append(f"### {view} ({label}, min_group_samples {vr['min_group_samples']})")
            if vr["skipped_groups"]:
                skipped = ", ".join(f"{s['group']} (n={s['count']})" for s in vr["skipped_groups"])
                lines.append(f"- skipped (under min): {skipped}")
            if vr.get("degenerate_groups"):
                degen = ", ".join(f"{d['group']} ({d['reason']})" for d in vr["degenerate_groups"])
                lines.append(f"- excluded from deltas (degenerate): {degen}")
            lines.append("")
            lines.append("| group | n | prevalence | accuracy | recall | auc |")
            lines.append("|---|---:|---:|---:|---:|---:|")
            for group, perf in sorted(vr["group_performance"].items()):
                lines.append(
                    f"| {group} | {perf['n']} | {_fmt(perf['prevalence'])} | "
                    f"{_fmt(perf['accuracy'])} | {_fmt(perf['recall'])} | {_fmt(perf['auc'])} |"
                )
            lines.append("")

            gf = vr.get("group_fairness", {})
            if gf:
                dp = gf.get("demographic_parity", {}).get("max_difference")
                eo = gf.get("equalized_odds", {})
                eopp = gf.get("equal_opportunity", {}).get("max_difference")
                lines.append(
                    f"- demographic_parity Δ {_fmt(dp)} · "
                    f"TPR Δ {_fmt(eo.get('tpr_max_difference'))} · "
                    f"FPR Δ {_fmt(eo.get('fpr_max_difference'))} · "
                    f"equal_opportunity Δ {_fmt(eopp)}"
                )
                lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _flatten_group_views_for_csv(reports: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """One row per (dataset_model, group_view, group) for dissertation import."""
    rows: list[dict[str, Any]] = []
    for key, report in reports.items():
        for view, vr in report.get("group_views", {}).items():
            gf = vr.get("group_fairness", {})
            dp = gf.get("demographic_parity", {}).get("max_difference")
            eo = gf.get("equalized_odds", {})
            degenerate_names = {d["group"] for d in vr.get("degenerate_groups", [])}
            for group, perf in vr["group_performance"].items():
                rows.append(
                    {
                        "run_key": key,
                        "group_view": view,
                        "source_attributes": ",".join(vr.get("source_attributes", [])),
                        "exploratory": bool(vr.get("exploratory")),
                        "min_group_samples": vr.get("min_group_samples"),
                        "group": group,
                        "n": perf["n"],
                        "prevalence": perf["prevalence"],
                        "accuracy": perf["accuracy"],
                        "recall": perf["recall"],
                        "auc": perf["auc"],
                        "degenerate": group in degenerate_names,
                        "demographic_parity_max_diff": dp,
                        "tpr_max_diff": eo.get("tpr_max_difference"),
                        "fpr_max_diff": eo.get("fpr_max_difference"),
                    }
                )
    return pd.DataFrame(rows)


def _discover_predictions(
    results_dir: Path,
    datasets: Optional[list[str]],
    model_types: Optional[list[str]],
) -> list[tuple[str, Path]]:
    """Find ``<dataset>_<model>`` keys and their test-prediction CSVs.

    Drives discovery off the metrics JSONs (each references its CSV), so the run
    layout stays the single source of truth.
    """
    found: list[tuple[str, Path]] = []
    for metrics_path in sorted(results_dir.glob("*_metrics.json")):
        key = metrics_path.name[: -len("_metrics.json")]
        try:
            meta = json.loads(metrics_path.read_text())
        except (OSError, json.JSONDecodeError):
            logger.warning("Could not read %s; skipping", metrics_path)
            continue
        csv_ref = meta.get("test_predictions")
        csv_path = Path(csv_ref) if csv_ref else results_dir / "predictions" / f"{key}_test.csv"
        if not csv_path.exists():
            logger.warning("Test predictions CSV missing for %s (%s)", key, csv_path)
            continue
        if datasets and not any(key.startswith(f"{d}_") or key == d for d in datasets):
            continue
        if model_types and not any(key.endswith(f"_{m}") for m in model_types):
            continue
        found.append((key, csv_path))
    return found


def _render_figures(
    reports: dict[str, dict[str, Any]],
    group_view_reports: dict[str, dict[str, Any]],
    out_dir: Path,
    attrs: list[str],
) -> None:
    """Delegate stage-8 figure rendering to viz (matplotlib imported only here)."""
    try:
        from fairxai.viz.dermatology_fairness import (
            render_group_view_figures,
            render_subgroup_heatmaps,
        )
    except ImportError as exc:
        logger.warning("Skipping dermatology fairness figures (viz/matplotlib): %s", exc)
        return
    render_subgroup_heatmaps(_flatten_for_csv(reports).to_dict("records"), out_dir, attrs=attrs)
    if group_view_reports:
        render_group_view_figures(
            _flatten_group_views_for_csv(group_view_reports).to_dict("records"),
            out_dir / "group_views",
        )


def assess_run(
    run_root: Path,
    sensitive_attrs: list[str],
    *,
    min_group_samples: int = DEFAULT_MIN_GROUP_SAMPLES,
    datasets: Optional[list[str]] = None,
    model_types: Optional[list[str]] = None,
    write_group_views: bool = False,
    group_views: Optional[list[str]] = None,
    group_view_min_group_samples: int = DEFAULT_MIN_GROUP_SAMPLES,
    intersection_min_group_samples: int = DEFAULT_INTERSECTION_MIN_GROUP_SAMPLES,
    write_figures: bool = False,
) -> dict[str, dict[str, Any]]:
    """Assess every baseline prediction CSV in *run_root* and write the report.

    Outputs land in ``<run_root>/baseline/prediction_fairness/``:
    ``fairness_report.json``, ``fairness_report.md``, ``fairness_groups.csv`` (and
    ``figures/`` plus ``group_views/figures/`` when *write_figures*).
    """
    results_dir = run_root / "baseline" / "results"
    discovered = _discover_predictions(results_dir, datasets, model_types)
    if not discovered:
        logger.warning("No prediction CSVs found under %s", results_dir)

    reports: dict[str, dict[str, Any]] = {}
    group_view_reports: dict[str, dict[str, Any]] = {}
    for key, csv_path in discovered:
        df = pd.read_csv(csv_path)
        reports[key] = assess_predictions_frame(
            df, sensitive_attrs, min_group_samples=min_group_samples
        )
        if write_group_views:
            group_view_reports[key] = assess_group_views_frame(
                df,
                group_views or DEFAULT_GROUP_VIEWS,
                min_group_samples=group_view_min_group_samples,
                intersection_min_group_samples=intersection_min_group_samples,
            )
        logger.info("Assessed %s (%d rows)", key, len(df))

    out_dir = run_root / "baseline" / "prediction_fairness"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "fairness_report.json").write_text(
        json.dumps(reports, indent=2, default=_json_default) + "\n"
    )
    (out_dir / "fairness_report.md").write_text(render_markdown(reports))
    _flatten_for_csv(reports).to_csv(out_dir / "fairness_groups.csv", index=False)
    logger.info("Wrote fairness report to %s", out_dir)

    if write_group_views:
        group_dir = out_dir / "group_views"
        group_dir.mkdir(parents=True, exist_ok=True)
        (group_dir / "group_view_report.json").write_text(
            json.dumps(group_view_reports, indent=2, default=_json_default) + "\n"
        )
        (group_dir / "group_view_report.md").write_text(
            render_group_view_markdown(group_view_reports)
        )
        _flatten_group_views_for_csv(group_view_reports).to_csv(
            group_dir / "group_view_groups.csv", index=False
        )
        logger.info("Wrote group-view fairness report to %s", group_dir)

    if write_figures and reports:
        _render_figures(reports, group_view_reports, out_dir, sensitive_attrs)

    return reports
