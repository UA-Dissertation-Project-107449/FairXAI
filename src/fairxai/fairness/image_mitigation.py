"""Post-processing fairness mitigation for dermatology image baselines.

Deliberately post-processing only. CNN pre/in-processing mitigation would require
retraining and is tabular-first in this codebase (see ``mitigation.py``); the
dermatology contribution is *measurement plus a cheap post-hoc correction*, not
full bias removal. This module never loads a model: it reuses the saved train/test
prediction CSVs (``y_true``, ``y_pred``, ``y_proba``, sensitive columns) and learns
group-specific decision thresholds via fairlearn's ``ThresholdOptimizer``.

For each sensitive attribute *in isolation* and each requested fairness constraint,
thresholds are fit on the **train** predictions and applied to the **test**
predictions (so we never fit and evaluate on the same rows), then the stage-8
fairness report is recomputed on the mitigated ``y_pred`` for before/after deltas.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

from .image_assessment import assess_predictions_frame, decode_groups
from .mitigation import PostProcessingMitigation

logger = logging.getLogger(__name__)

# fairlearn ThresholdOptimizer-valid constraints. All reported side-by-side; a
# combo that fairlearn rejects (objective/constraint mismatch) is caught per cell.
DEFAULT_CONSTRAINTS = [
    "demographic_parity",
    "equalized_odds",
    "true_positive_rate_parity",
    "false_positive_rate_parity",
]
DEFAULT_OBJECTIVE = "balanced_accuracy_score"
DEFAULT_MIN_GROUP_SAMPLES = 50


class _PrecomputedScoreEstimator(ClassifierMixin, BaseEstimator):
    """Minimal sklearn-style wrapper exposing stored positive-class scores.

    Lets fairlearn's ``ThresholdOptimizer`` (``prefit=True``) operate model-free on
    saved probabilities. The single input column is the positive-class probability;
    ``predict_proba`` returns the 2-column form fairlearn expects, ``predict``
    thresholds at 0.5.

    Subclasses ``BaseEstimator``/``ClassifierMixin`` so it exposes
    ``__sklearn_tags__`` (required by scikit-learn >= 1.6, which fairlearn calls on
    the prefit estimator); without it CI on newer sklearn raises
    ``'_PrecomputedScoreEstimator' object has no attribute '__sklearn_tags__'``.
    """

    def __init__(self) -> None:
        # Trailing-underscore instance attrs satisfy sklearn's check_is_fitted,
        # which fairlearn runs on the prefit base estimator.
        self.classes_ = np.array([0, 1])
        self.is_fitted_ = True

    def fit(self, X, y=None):  # noqa: D401 - prefit; nothing to learn
        return self

    @staticmethod
    def _scores(X) -> np.ndarray:
        arr = np.asarray(X, dtype=float)
        return arr[:, 0] if arr.ndim == 2 else arr.ravel()

    def predict_proba(self, X) -> np.ndarray:
        p = self._scores(X)
        return np.column_stack([1.0 - p, p])

    def predict(self, X) -> np.ndarray:
        return (self._scores(X) >= 0.5).astype(int)


def _group_fairness_summary(attr_report: dict[str, Any]) -> dict[str, Optional[float]]:
    """Pull the headline max-difference metrics from an assess attr report."""
    gf = attr_report.get("group_fairness", {}) if attr_report else {}
    eo = gf.get("equalized_odds", {})
    return {
        "demographic_parity_max_diff": gf.get("demographic_parity", {}).get("max_difference"),
        "tpr_max_diff": eo.get("tpr_max_difference"),
        "fpr_max_diff": eo.get("fpr_max_difference"),
        "equal_opportunity_max_diff": gf.get("equal_opportunity", {}).get("max_difference"),
    }


def _deltas(
    before: dict[str, Optional[float]], after: dict[str, Optional[float]]
) -> dict[str, Optional[float]]:
    """after - before per metric (None if either side is undefined)."""
    out: dict[str, Optional[float]] = {}
    for k in before:
        b, a = before.get(k), after.get(k)
        out[k] = (a - b) if (b is not None and a is not None) else None
    return out


def _eligible_groups(groups: pd.Series, y_true: pd.Series, min_group_samples: int) -> list[str]:
    """Groups that fairlearn can fit: >= min support AND both classes present.

    ThresholdOptimizer rejects any fit group with degenerate (single-class) labels,
    so tiny/unknown subgroups (e.g. PAD ``<20`` age, ``Unknown`` sex/Fitzpatrick)
    are excluded from the fit. Their test rows keep the baseline prediction.
    """
    eligible: list[str] = []
    for group, n in groups.value_counts().items():
        if n < min_group_samples:
            continue
        if y_true[groups == group].nunique() < 2:
            continue
        eligible.append(str(group))
    return eligible


def mitigate_predictions_frame(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    sensitive_attrs: list[str],
    *,
    constraints: Optional[list[str]] = None,
    objective: str = DEFAULT_OBJECTIVE,
    min_group_samples: int = DEFAULT_MIN_GROUP_SAMPLES,
    random_state: int = 42,
) -> dict[str, Any]:
    """Mitigate one model's test predictions per attribute x constraint.

    Thresholds are fit only on **eligible** train groups (>= min support and both
    classes present); excluded groups keep their baseline ``y_pred``. Returns the
    unmitigated ``baseline`` and, per attribute, an ``after`` fairness report plus
    before/after summary deltas for each constraint. A constraint fairlearn still
    rejects is recorded with an ``error`` so the rest of the matrix completes.
    """
    constraints = constraints or DEFAULT_CONSTRAINTS
    baseline = assess_predictions_frame(
        test_df, sensitive_attrs, min_group_samples=min_group_samples
    )
    result: dict[str, Any] = {
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "objective": objective,
        "constraints": list(constraints),
        "min_group_samples": int(min_group_samples),
        "overall_baseline": baseline["overall_performance"],
        "sensitive_attributes": {},
    }

    x_train = train_df[["y_proba"]].to_numpy(dtype=float)
    y_train = train_df["y_true"].to_numpy()
    x_test = test_df[["y_proba"]].to_numpy(dtype=float)
    base_pred_test = test_df["y_pred"].to_numpy()

    for attr in sensitive_attrs:
        if attr not in train_df.columns or attr not in test_df.columns:
            logger.warning("Sensitive attribute '%s' absent from train/test; skipping", attr)
            continue

        base_attr = baseline["sensitive_attributes"].get(attr, {})
        before_summary = _group_fairness_summary(base_attr)

        groups_train = decode_groups(train_df, attr)
        groups_test = decode_groups(test_df, attr)
        eligible = _eligible_groups(groups_train, train_df["y_true"], min_group_samples)
        attr_out: dict[str, Any] = {
            "baseline": base_attr,
            "eligible_groups": eligible,
            "constraints": {},
        }

        # Fewer than 2 fittable groups -> no group-wise threshold tuning possible.
        if len(eligible) < 2:
            attr_out["note"] = (
                "fewer than 2 eligible groups (min support + both classes in train); "
                "no threshold mitigation"
            )
            result["sensitive_attributes"][attr] = attr_out
            continue

        train_mask = groups_train.isin(eligible).to_numpy()
        test_mask = groups_test.isin(eligible).to_numpy()
        sens_fit = pd.DataFrame({attr: groups_train[train_mask].to_numpy()})
        sens_apply = groups_test[test_mask].to_numpy()

        for constraint in constraints:
            try:
                optimizer = PostProcessingMitigation.apply_threshold_optimizer(
                    base_model=_PrecomputedScoreEstimator(),
                    X_train=x_train[train_mask],
                    y_train=y_train[train_mask],
                    sensitive_features=sens_fit,
                    sensitive_attr=attr,
                    constraint_type=constraint,
                    objective=objective,
                )
                mitigated_sub = optimizer.predict(
                    x_test[test_mask], sensitive_features=sens_apply, random_state=random_state
                )
            except Exception as exc:  # noqa: BLE001 - one bad combo must not kill the matrix
                logger.warning("Mitigation %s/%s failed: %s", attr, constraint, exc)
                attr_out["constraints"][constraint] = {"error": str(exc)}
                continue

            # Eligible test rows get tuned thresholds; excluded groups keep baseline.
            new_pred = base_pred_test.copy()
            new_pred[test_mask] = np.asarray(mitigated_sub).astype(int)
            after_df = test_df.copy()
            after_df["y_pred"] = new_pred
            after = assess_predictions_frame(after_df, [attr], min_group_samples=min_group_samples)
            after_attr = after["sensitive_attributes"].get(attr, {})
            after_summary = _group_fairness_summary(after_attr)
            attr_out["constraints"][constraint] = {
                "overall_after": after["overall_performance"],
                "after": after_attr,
                "summary_before": before_summary,
                "summary_after": after_summary,
                "summary_deltas": _deltas(before_summary, after_summary),
            }

        result["sensitive_attributes"][attr] = attr_out

    return result


def _discover_prediction_pairs(
    results_dir: Path,
    datasets: Optional[list[str]],
    model_types: Optional[list[str]],
) -> list[tuple[str, Path, Path]]:
    """Find ``<dataset>_<model>`` keys with both train and test prediction CSVs."""
    found: list[tuple[str, Path, Path]] = []
    for metrics_path in sorted(results_dir.glob("*_metrics.json")):
        key = metrics_path.name[: -len("_metrics.json")]
        try:
            meta = json.loads(metrics_path.read_text())
        except (OSError, json.JSONDecodeError):
            logger.warning("Could not read %s; skipping", metrics_path)
            continue
        train_ref = meta.get("train_predictions")
        test_ref = meta.get("test_predictions")
        train_path = (
            Path(train_ref) if train_ref else results_dir / "predictions" / f"{key}_train.csv"
        )
        test_path = Path(test_ref) if test_ref else results_dir / "predictions" / f"{key}_test.csv"
        if not train_path.exists() or not test_path.exists():
            logger.warning("Train/test predictions missing for %s; skipping", key)
            continue
        if datasets and not any(key.startswith(f"{d}_") or key == d for d in datasets):
            continue
        if model_types and not any(key.endswith(f"_{m}") for m in model_types):
            continue
        found.append((key, train_path, test_path))
    return found


def _flatten_for_csv(reports: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """One row per (run_key, attr, constraint) for dissertation import."""
    rows: list[dict[str, Any]] = []
    for key, report in reports.items():
        for attr, ar in report.get("sensitive_attributes", {}).items():
            for constraint, cr in ar.get("constraints", {}).items():
                if "error" in cr:
                    rows.append(
                        {
                            "run_key": key,
                            "attr": attr,
                            "constraint": constraint,
                            "error": cr["error"],
                        }
                    )
                    continue
                deltas = cr.get("summary_deltas", {})
                before = cr.get("summary_before", {})
                after = cr.get("summary_after", {})
                rows.append(
                    {
                        "run_key": key,
                        "attr": attr,
                        "constraint": constraint,
                        "objective": report.get("objective"),
                        "overall_acc_before": report.get("overall_baseline", {}).get("accuracy"),
                        "overall_acc_after": cr.get("overall_after", {}).get("accuracy"),
                        "dp_before": before.get("demographic_parity_max_diff"),
                        "dp_after": after.get("demographic_parity_max_diff"),
                        "dp_delta": deltas.get("demographic_parity_max_diff"),
                        "tpr_before": before.get("tpr_max_diff"),
                        "tpr_after": after.get("tpr_max_diff"),
                        "tpr_delta": deltas.get("tpr_max_diff"),
                        "fpr_before": before.get("fpr_max_diff"),
                        "fpr_after": after.get("fpr_max_diff"),
                        "fpr_delta": deltas.get("fpr_max_diff"),
                    }
                )
    return pd.DataFrame(rows)


def _fmt(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def render_markdown(reports: dict[str, dict[str, Any]]) -> str:
    """Render the mitigation reports as a Markdown before/after summary."""
    lines = ["# Dermatology Post-Processing Mitigation Report", ""]
    lines.append(
        "Post-processing only (group-wise thresholds via fairlearn ThresholdOptimizer). "
        "Thresholds fit on train predictions, applied to test. No retraining."
    )
    lines.append("")
    for key in sorted(reports):
        report = reports[key]
        lines.append(f"## {key}")
        lines.append(
            f"- n_train {report['n_train']} · n_test {report['n_test']} · "
            f"objective {report['objective']} · min_group_samples {report['min_group_samples']}"
        )
        lines.append("")
        for attr, ar in sorted(report["sensitive_attributes"].items()):
            lines.append(f"### {attr}")
            for constraint, cr in ar["constraints"].items():
                if "error" in cr:
                    lines.append(f"- {constraint}: error — {cr['error']}")
                    continue
                d = cr.get("summary_deltas", {})
                lines.append(
                    f"- {constraint}: DP Δ {_fmt(d.get('demographic_parity_max_diff'))} · "
                    f"TPR Δ {_fmt(d.get('tpr_max_diff'))} · "
                    f"FPR Δ {_fmt(d.get('fpr_max_diff'))} · "
                    f"acc {_fmt(report['overall_baseline'].get('accuracy'))}"
                    f"→{_fmt(cr.get('overall_after', {}).get('accuracy'))}"
                )
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _render_figures(reports: dict[str, dict[str, Any]], out_dir: Path, attrs: list[str]) -> None:
    """Delegate before/after figure rendering to viz (matplotlib imported only here)."""
    try:
        from fairxai.viz.dermatology_mitigation import render_mitigation_figures
    except ImportError as exc:
        logger.warning("Skipping dermatology mitigation figures (viz/matplotlib): %s", exc)
        return
    rows = _flatten_for_csv(reports).to_dict("records")
    render_mitigation_figures(rows, out_dir, attrs=attrs)


def mitigate_run(
    run_root: Path,
    sensitive_attrs: list[str],
    *,
    constraints: Optional[list[str]] = None,
    objective: str = DEFAULT_OBJECTIVE,
    min_group_samples: int = DEFAULT_MIN_GROUP_SAMPLES,
    datasets: Optional[list[str]] = None,
    model_types: Optional[list[str]] = None,
    random_state: int = 42,
    write_figures: bool = False,
) -> dict[str, dict[str, Any]]:
    """Mitigate every baseline prediction pair in *run_root* and write reports.

    Outputs land in ``<run_root>/baseline/mitigation/``: ``mitigation_report.json``,
    ``mitigation_report.md``, ``mitigation_groups.csv`` (and ``figures/`` when
    *write_figures*).
    """
    results_dir = run_root / "baseline" / "results"
    discovered = _discover_prediction_pairs(results_dir, datasets, model_types)
    if not discovered:
        logger.warning("No train/test prediction pairs found under %s", results_dir)

    reports: dict[str, dict[str, Any]] = {}
    for key, train_path, test_path in discovered:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        reports[key] = mitigate_predictions_frame(
            train_df,
            test_df,
            sensitive_attrs,
            constraints=constraints,
            objective=objective,
            min_group_samples=min_group_samples,
            random_state=random_state,
        )
        logger.info("Mitigated %s (train=%d test=%d)", key, len(train_df), len(test_df))

    out_dir = run_root / "baseline" / "mitigation"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "mitigation_report.json").write_text(
        json.dumps(reports, indent=2, default=_json_default) + "\n"
    )
    (out_dir / "mitigation_report.md").write_text(render_markdown(reports))
    _flatten_for_csv(reports).to_csv(out_dir / "mitigation_groups.csv", index=False)
    if write_figures and reports:
        _render_figures(reports, out_dir, sensitive_attrs)
    logger.info("Wrote mitigation report to %s", out_dir)
    return reports


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
