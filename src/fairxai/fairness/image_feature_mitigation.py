"""Feature-space fairness mitigation for dermatology image baselines (stage 11).

Companion to :mod:`fairxai.fairness.image_mitigation`, which is post-processing
only. This module adds the **pre-processing** and **in-processing** arms so the
dermatology mitigation matrix matches cardiac's technique-for-technique instead
of being a smaller, differently-shaped selection — a cross-domain comparison is
only a comparison when both sides run the same interventions.

How it is possible without retraining a CNN
-------------------------------------------
Dermatology trains with ``freeze_backbone: true``, so the network is a fixed
feature extractor and only the linear head is learned. That makes the learning
problem tabular: a ``n_rows x n_channels`` matrix and a binary label, which is
exactly what :class:`~fairxai.fairness.mitigation.MitigationEngine` already
consumes. The backbone is run once in eval mode to rebuild that matrix from the
saved checkpoint (one forward pass, no gradients), then every technique is the
*same implementation* cardiac uses — not an image-specific reimplementation.

What this is not
----------------
The intervention is on the **head**, not on the representation. Reweighting,
SMOTE, ADASYN and the fairlearn reductions all act on features the frozen
backbone already produced, so any bias baked into those features by ImageNet
pre-training survives. Learning a fair *representation* would need the backbone
unfrozen (adversarial debiasing and friends), which is a different, far more
expensive experiment and stays out of scope. Every report carries this caveat in
its ``scope`` field so a downstream reader cannot lose it.

Features are standardised (train-fit :class:`~sklearn.preprocessing.StandardScaler`)
before any technique runs. SMOTE and ADASYN are distance-based; raw pooled CNN
activations have wildly different per-channel scales, so without scaling the
widest channel would own the neighbourhood. Cardiac mitigates scaled features
too, so this also keeps the two domains on the same footing.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .image_assessment import assess_predictions_frame, decode_groups
from .image_mitigation import DEFAULT_MIN_GROUP_SAMPLES, _deltas, _group_fairness_summary
from .mitigation import MitigationEngine

logger = logging.getLogger(__name__)

# Mirrors the cardiac selection (configs/experiments/combinatorial.yaml).
# ros/rus are excluded there — random resampling has no fairness guarantee and
# discards information — and stay excluded here for the same reason.
DEFAULT_FEATURE_TECHNIQUES = [
    "reweighting",
    "smote",
    "adasyn",
    "exponentiated_gradient",
    "grid_search",
]

_SCOPE_NOTE = (
    "Mitigation is applied to the linear head over frozen-backbone features. "
    "The representation itself is unchanged, so bias encoded by the pre-trained "
    "backbone is not removed - only the decision rule on top of it is corrected."
)


def technique_stage(name: str) -> str:
    """Return the mitigation stage for *name*, as the shared engine defines it.

    Post-processing names are rejected: they belong to
    :mod:`fairxai.fairness.image_mitigation`, which needs no feature matrix.
    """
    key = (name or "").strip().lower()
    if key in MitigationEngine.VALID_PREPROCESSING:
        return "pre-processing"
    if key in MitigationEngine.VALID_INPROCESSING:
        return "in-processing"
    raise ValueError(
        f"Unknown feature-space technique '{name}'. "
        f"Expected one of {MitigationEngine.VALID_PREPROCESSING + MitigationEngine.VALID_INPROCESSING}"
    )


# --------------------------------------------------------------------------- #
# Frozen-backbone feature extraction
# --------------------------------------------------------------------------- #
def extract_frozen_features(
    checkpoint_path: Path,
    csv_path: Path,
    *,
    image_col: str = "image_path",
    target_col: str = "skin_cancer",
    device: str = "cpu",
    batch_size: int = 32,
    num_workers: int = 0,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Rebuild the pooled-feature matrix for one split from a saved checkpoint.

    The training run does not persist its cached features (and the augmentation
    path never caches them at all), so they are recomputed here: rebuild the
    architecture, load the saved weights, swap the classifier head for
    ``Identity`` and run one eval-mode, no-grad forward pass.

    The loader is **unshuffled**, so row *i* of the returned matrix is row *i* of
    the split CSV; the returned frame is that CSV, giving label and sensitive
    columns already aligned. The transform is rebuilt from the checkpoint's own
    metadata, so the pixels match the deterministic eval framing training scored
    with.
    """
    import torch  # type: ignore
    import torch.nn as nn  # type: ignore
    from torch.utils.data import DataLoader  # type: ignore

    from fairxai.explainability.image import _build_transform, _load_model
    from fairxai.training.vision import (
        _MODEL_REGISTRY,
        _CsvImageDataset,
        _detach_head,
        _extract_features,
    )

    torch_device = torch.device(device)
    model, ckpt = _load_model(Path(checkpoint_path), torch_device)
    transform = _build_transform(ckpt)
    dataset = _CsvImageDataset(
        Path(csv_path), transform, ckpt.get("image_col", image_col), target_col
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False
    )

    head_strategy = _MODEL_REGISTRY[ckpt["model_name"]][2]
    _detach_head(model, head_strategy, nn)
    model.eval()
    features, _labels, row_indices = _extract_features(model, loader, torch_device, torch)

    frame = dataset.df.reset_index(drop=True)
    # shuffle=False, but assert rather than assume: a silent reorder would
    # misalign every sensitive attribute against its feature row.
    if row_indices != list(range(len(frame))):
        frame = frame.iloc[row_indices].reset_index(drop=True)
    return features.numpy(), frame


# --------------------------------------------------------------------------- #
# Core: one model's feature matrix -> the technique x attribute matrix
# --------------------------------------------------------------------------- #
def _predictions_frame(
    meta: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray]
) -> pd.DataFrame:
    """Assemble the frame ``assess_predictions_frame`` expects."""
    out = meta.reset_index(drop=True).copy()
    out["y_true"] = np.asarray(y_true).astype(int)
    out["y_pred"] = np.asarray(y_pred).astype(int)
    if y_proba is not None:
        out["y_proba"] = np.asarray(y_proba, dtype=float)
    elif "y_proba" in out.columns:
        out = out.drop(columns="y_proba")
    return out


def _as_frame(features: np.ndarray) -> pd.DataFrame:
    arr = np.asarray(features, dtype=float)
    return pd.DataFrame(arr, columns=[f"f{i}" for i in range(arr.shape[1])])


def _positive_proba(model, X: pd.DataFrame) -> Optional[np.ndarray]:
    if not hasattr(model, "predict_proba"):
        return None
    proba = np.asarray(model.predict_proba(X))
    return proba[:, 1] if proba.ndim > 1 else proba


def mitigate_features_frame(
    train_features: np.ndarray,
    train_meta: pd.DataFrame,
    test_features: np.ndarray,
    test_meta: pd.DataFrame,
    sensitive_attrs: list[str],
    *,
    techniques: Optional[list[str]] = None,
    model_type: str = "logistic_regression",
    model_params: Optional[dict[str, Any]] = None,
    min_group_samples: int = DEFAULT_MIN_GROUP_SAMPLES,
    random_state: int = 42,
) -> dict[str, Any]:
    """Run every technique x sensitive attribute on one model's feature matrix.

    *train_meta* / *test_meta* are the split frames row-aligned with the feature
    matrices; they must carry ``y_true`` and the sensitive columns.

    The delta reference is an **unmitigated head of the same family trained on
    the same standardised features** — never the CNN's own softmax head. Both
    heads and the CNN are different classifiers, so measuring a technique
    against the CNN would fold the head swap into the mitigation effect. The CNN
    numbers travel alongside as context only (see :func:`mitigate_run_features`).

    Unlike the post-processing path, small groups are **not** excluded from the
    fit: these techniques train a classifier that scores any row, rather than a
    per-group threshold that cannot exist for a group never fit. Undersized
    groups are still dropped from the fairness *metrics* by
    ``assess_predictions_frame``; ``group_support_train`` records what each
    technique actually saw.
    """
    techniques = list(techniques or DEFAULT_FEATURE_TECHNIQUES)
    scaler = StandardScaler().fit(np.asarray(train_features, dtype=float))
    X_train = _as_frame(scaler.transform(np.asarray(train_features, dtype=float)))
    X_test = _as_frame(scaler.transform(np.asarray(test_features, dtype=float)))
    y_train = pd.Series(np.asarray(train_meta["y_true"]).astype(int), name="y_true")
    y_test = pd.Series(np.asarray(test_meta["y_true"]).astype(int), name="y_true")

    engine = MitigationEngine(
        random_state=random_state, model_type=model_type, model_params=model_params
    )

    head = engine.build_model()
    head.train(X_train, y_train)
    base_pred = np.asarray(head.predict(X_test)).astype(int)
    base_proba = _positive_proba(head, X_test)
    base_frame = _predictions_frame(test_meta, y_test.to_numpy(), base_pred, base_proba)
    baseline = assess_predictions_frame(
        base_frame, sensitive_attrs, min_group_samples=min_group_samples
    )

    result: dict[str, Any] = {
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_features": int(X_train.shape[1]),
        "model_type": engine.model_type,
        "techniques": techniques,
        "standardized": True,
        "min_group_samples": int(min_group_samples),
        "scope": _SCOPE_NOTE,
        "head_baseline": baseline,
        "overall_baseline": baseline["overall_performance"],
        "sensitive_attributes": {},
    }

    for attr in sensitive_attrs:
        if attr not in train_meta.columns or attr not in test_meta.columns:
            logger.warning("Sensitive attribute '%s' absent from split metadata; skipping", attr)
            continue

        groups_train = decode_groups(train_meta, attr)
        sens_train = pd.DataFrame({attr: groups_train.to_numpy()})
        sens_test = pd.DataFrame({attr: decode_groups(test_meta, attr).to_numpy()})

        base_attr = baseline["sensitive_attributes"].get(attr, {})
        before_summary = _group_fairness_summary(base_attr)
        attr_out: dict[str, Any] = {
            "baseline": base_attr,
            "summary_baseline": before_summary,
            "group_support_train": {str(g): int(n) for g, n in groups_train.value_counts().items()},
            "techniques": {},
        }

        for technique in techniques:
            stage = technique_stage(technique)
            try:
                outcome = engine.apply_technique(
                    technique,
                    stage,
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    sens_train,
                    sens_test,
                    sensitive_attr=attr,
                )
            except Exception as exc:  # noqa: BLE001 - one bad cell must not kill the matrix
                logger.warning("Feature mitigation %s/%s failed: %s", attr, technique, exc)
                attr_out["techniques"][technique] = {"stage": stage, "error": str(exc)}
                continue

            preds = outcome.get("predictions", {})
            after_frame = _predictions_frame(
                test_meta, y_test.to_numpy(), preds.get("y_pred"), preds.get("y_proba")
            )
            after = assess_predictions_frame(
                after_frame, [attr], min_group_samples=min_group_samples
            )
            after_attr = after["sensitive_attributes"].get(attr, {})
            after_summary = _group_fairness_summary(after_attr)
            attr_out["techniques"][technique] = {
                "stage": stage,
                "overall_after": after["overall_performance"],
                "after": after_attr,
                "summary_before": before_summary,
                "summary_after": after_summary,
                "summary_deltas": _deltas(before_summary, after_summary),
                "metadata": outcome.get("metadata", {}),
            }

        result["sensitive_attributes"][attr] = attr_out

    return result


# --------------------------------------------------------------------------- #
# Run-level orchestration
# --------------------------------------------------------------------------- #
def _discover_checkpoints(
    results_dir: Path,
    datasets: Optional[list[str]],
    model_types: Optional[list[str]],
) -> list[tuple[str, str, Path, dict[str, Any]]]:
    """Find ``(run_key, dataset_name, checkpoint_path, metrics)`` for trained models."""
    found: list[tuple[str, str, Path, dict[str, Any]]] = []
    for metrics_path in sorted(results_dir.glob("*_metrics.json")):
        key = metrics_path.name[: -len("_metrics.json")]
        try:
            meta = json.loads(metrics_path.read_text())
        except (OSError, json.JSONDecodeError):
            logger.warning("Could not read %s; skipping", metrics_path)
            continue
        model_type = str(meta.get("model_type") or "")
        dataset_name = (
            key[: -len(f"_{model_type}")] if model_type and key.endswith(model_type) else key
        )
        checkpoint = meta.get("model_file")
        if not checkpoint or not Path(checkpoint).exists():
            logger.warning("No checkpoint for %s; skipping feature-space mitigation", key)
            continue
        if datasets and dataset_name not in datasets:
            continue
        if model_types and model_type not in model_types:
            continue
        found.append((key, dataset_name, Path(checkpoint), meta))
    return found


def _resolve_split_csvs(processed_dir: Path, dataset_name: str) -> tuple[Path, Path]:
    """Locate ``<dataset>_train.csv`` / ``_test.csv``, dataset subdir or flat."""
    for base in (Path(processed_dir) / dataset_name, Path(processed_dir)):
        train = base / f"{dataset_name}_train.csv"
        test = base / f"{dataset_name}_test.csv"
        if train.exists() and test.exists():
            return train, test
    # Nothing on disk: return the canonical location so the caller's extractor
    # (real or injected) still receives a meaningful, inspectable path.
    base = Path(processed_dir) / dataset_name
    return base / f"{dataset_name}_train.csv", base / f"{dataset_name}_test.csv"


def _flatten_for_csv(reports: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """One row per (run_key, attr, technique) for dissertation import."""
    rows: list[dict[str, Any]] = []
    for key, report in reports.items():
        for attr, ar in report.get("sensitive_attributes", {}).items():
            for technique, cell in ar.get("techniques", {}).items():
                if "error" in cell:
                    rows.append(
                        {
                            "run_key": key,
                            "attr": attr,
                            "technique": technique,
                            "stage": cell.get("stage"),
                            "error": cell["error"],
                        }
                    )
                    continue
                before = cell.get("summary_before", {})
                after = cell.get("summary_after", {})
                deltas = cell.get("summary_deltas", {})
                rows.append(
                    {
                        "run_key": key,
                        "attr": attr,
                        "technique": technique,
                        "stage": cell.get("stage"),
                        "model_type": report.get("model_type"),
                        "cnn_acc": (report.get("cnn_test_metrics") or {}).get("accuracy"),
                        "head_acc_before": report.get("overall_baseline", {}).get("accuracy"),
                        "head_acc_after": cell.get("overall_after", {}).get("accuracy"),
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


def _fmt(value: Any) -> str:
    return (
        "n/a" if value is None or (isinstance(value, float) and np.isnan(value)) else f"{value:.4f}"
    )


def render_markdown(reports: dict[str, dict[str, Any]]) -> str:
    """Human-readable before/after table per model x attribute."""
    lines = ["# Dermatology Feature-Space Mitigation", ""]
    if not reports:
        lines.append("_No models mitigated._")
        return "\n".join(lines) + "\n"
    lines += [_SCOPE_NOTE, ""]
    for key in sorted(reports):
        report = reports[key]
        cnn = report.get("cnn_test_metrics") or {}
        lines += [
            f"## {key}",
            "",
            f"- head family: `{report.get('model_type')}` over "
            f"{report.get('n_features')} frozen features "
            f"(train={report.get('n_train')}, test={report.get('n_test')})",
            f"- CNN head accuracy (context only): {_fmt(cnn.get('accuracy'))}",
            f"- linear-head baseline accuracy (delta reference): "
            f"{_fmt(report.get('overall_baseline', {}).get('accuracy'))}",
            "",
        ]
        for attr, ar in sorted(report.get("sensitive_attributes", {}).items()):
            lines += [
                f"### {attr}",
                "",
                "| technique | stage | acc | DP max-diff | TPR max-diff | FPR max-diff |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
            base = ar.get("summary_baseline", {})
            lines.append(
                f"| _baseline (no mitigation)_ | - | "
                f"{_fmt(report.get('overall_baseline', {}).get('accuracy'))} | "
                f"{_fmt(base.get('demographic_parity_max_diff'))} | "
                f"{_fmt(base.get('tpr_max_diff'))} | {_fmt(base.get('fpr_max_diff'))} |"
            )
            for technique, cell in ar.get("techniques", {}).items():
                if "error" in cell:
                    lines.append(
                        f"| {technique} | {cell.get('stage')} | failed | "
                        f"{cell['error'][:60]} | | |"
                    )
                    continue
                after = cell.get("summary_after", {})
                lines.append(
                    f"| {technique} | {cell.get('stage')} | "
                    f"{_fmt(cell.get('overall_after', {}).get('accuracy'))} | "
                    f"{_fmt(after.get('demographic_parity_max_diff'))} | "
                    f"{_fmt(after.get('tpr_max_diff'))} | {_fmt(after.get('fpr_max_diff'))} |"
                )
            lines.append("")
    return "\n".join(lines) + "\n"


def mitigate_run_features(
    run_root: Path,
    sensitive_attrs: list[str],
    *,
    processed_dir: Path,
    image_col: str = "image_path",
    target_col: str = "skin_cancer",
    techniques: Optional[list[str]] = None,
    model_type: str = "logistic_regression",
    model_params: Optional[dict[str, Any]] = None,
    min_group_samples: int = DEFAULT_MIN_GROUP_SAMPLES,
    datasets: Optional[list[str]] = None,
    model_types: Optional[list[str]] = None,
    random_state: int = 42,
    device: str = "cpu",
    batch_size: int = 32,
    num_workers: int = 0,
    extractor: Optional[Callable[..., tuple[np.ndarray, pd.DataFrame]]] = None,
) -> dict[str, dict[str, Any]]:
    """Rebuild features per trained model and run the mitigation matrix.

    *extractor* defaults to :func:`extract_frozen_features`; it is injectable so
    the orchestration can be tested without torch or a real checkpoint.

    Outputs land in ``<run_root>/baseline/mitigation/feature_space/``.
    """
    extract = extractor or extract_frozen_features
    results_dir = Path(run_root) / "baseline" / "results"
    discovered = _discover_checkpoints(results_dir, datasets, model_types)
    if not discovered:
        logger.warning("No checkpoints found under %s", results_dir)

    reports: dict[str, dict[str, Any]] = {}
    for key, dataset_name, checkpoint, meta in discovered:
        train_csv, test_csv = _resolve_split_csvs(Path(processed_dir), dataset_name)
        try:
            train_features, train_meta = extract(
                checkpoint,
                train_csv,
                image_col=image_col,
                target_col=target_col,
                device=device,
                batch_size=batch_size,
                num_workers=num_workers,
            )
            test_features, test_meta = extract(
                checkpoint,
                test_csv,
                image_col=image_col,
                target_col=target_col,
                device=device,
                batch_size=batch_size,
                num_workers=num_workers,
            )
        except Exception as exc:  # noqa: BLE001 - one bad model must not kill the run
            logger.warning("Feature extraction failed for %s: %s", key, exc)
            continue

        if target_col in train_meta.columns and "y_true" not in train_meta.columns:
            train_meta = train_meta.assign(y_true=train_meta[target_col])
        if target_col in test_meta.columns and "y_true" not in test_meta.columns:
            test_meta = test_meta.assign(y_true=test_meta[target_col])

        report = mitigate_features_frame(
            train_features,
            train_meta,
            test_features,
            test_meta,
            sensitive_attrs,
            techniques=techniques,
            model_type=model_type,
            model_params=model_params,
            min_group_samples=min_group_samples,
            random_state=random_state,
        )
        # Context, never the delta reference: the CNN's own softmax head is a
        # different classifier from the linear head mitigation is applied to.
        report["cnn_test_metrics"] = meta.get("test_metrics", {})
        report["cnn_model_type"] = meta.get("model_type")
        report["freeze_backbone"] = meta.get("config", {}).get("freeze_backbone")
        report["checkpoint"] = str(checkpoint)
        reports[key] = report
        logger.info(
            "Feature-space mitigated %s (train=%d test=%d features=%d)",
            key,
            report["n_train"],
            report["n_test"],
            report["n_features"],
        )

    out_dir = Path(run_root) / "baseline" / "mitigation" / "feature_space"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "feature_mitigation_report.json").write_text(
        json.dumps(reports, indent=2, default=_json_default) + "\n"
    )
    (out_dir / "feature_mitigation_report.md").write_text(render_markdown(reports))
    _flatten_for_csv(reports).to_csv(out_dir / "feature_mitigation_summary.csv", index=False)
    logger.info("Wrote feature-space mitigation report to %s", out_dir)
    return reports


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
