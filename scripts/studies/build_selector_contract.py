#!/usr/bin/env python3
"""Build a selector contract from available study artifacts.

The selector contract is a lightweight JSON artifact consumed by downstream
pipeline stages to wire studies output into execution defaults.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from fairxai.cli.runner_base import get_project_root
from fairxai.utils.config import load_yaml_config

logger = logging.getLogger(__name__)


def _configure_logging(verbose: int) -> None:
    level = logging.WARNING
    if verbose >= 2:
        level = logging.DEBUG
    elif verbose >= 1:
        level = logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s")


def _read_json(path: Path) -> Optional[dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            return payload
        return None
    except Exception as exc:
        logger.warning("Could not read JSON at %s: %s", path, exc)
        return None


def _normalise_model_types(raw_values: Optional[list[Any]]) -> list[str]:
    if not raw_values:
        return []
    normalized: list[str] = []
    for raw in raw_values:
        value = str(raw).strip().lower()
        if value and value not in normalized:
            normalized.append(value)
    return normalized


def _is_selected_dataset(dataset_name: str, selected_datasets: list[str]) -> bool:
    if not selected_datasets:
        return True
    return any(dataset_name == ds or dataset_name.startswith(f"{ds}_") for ds in selected_datasets)


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_test_metrics(model_payload: Any) -> Optional[dict[str, float]]:
    if not isinstance(model_payload, dict):
        return None

    test_metrics = model_payload.get("test_metrics")
    if isinstance(test_metrics, dict):
        return test_metrics

    methods = model_payload.get("training_methods")
    if isinstance(methods, dict):
        single_split = methods.get("single_split")
        if isinstance(single_split, dict):
            nested = single_split.get("test_metrics")
            if isinstance(nested, dict):
                return nested

    return None


def _composite_score(metrics: Optional[dict[str, Any]]) -> Optional[float]:
    if not isinstance(metrics, dict):
        return None
    accuracy = _safe_float(metrics.get("accuracy"))
    recall = _safe_float(metrics.get("recall"))
    f1_score = _safe_float(metrics.get("f1_score"))
    auc_roc = _safe_float(metrics.get("auc_roc"))
    if None in (accuracy, recall, f1_score, auc_roc):
        return None
    return (0.40 * f1_score) + (0.30 * recall) + (0.20 * accuracy) + (0.10 * auc_roc)


def _mean(values: list[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / len(values)


def _find_latest_study_dir(studies_root: Path) -> tuple[Optional[str], Optional[Path]]:
    if not studies_root.exists():
        return None, None

    latest_txt = studies_root / "latest.txt"
    if latest_txt.exists():
        study_id = latest_txt.read_text(encoding="utf-8").strip()
        if study_id:
            candidate = studies_root / study_id
            if candidate.exists() and candidate.is_dir():
                return study_id, candidate

    candidates = [path for path in studies_root.iterdir() if path.is_dir()]
    if not candidates:
        return None, None

    latest = max(candidates, key=lambda path: path.stat().st_mtime)
    return latest.name, latest


def _resolve_hpo_root(project_root: Path, pipeline: str) -> Path:
    default_root = project_root / f"output/{pipeline}/studies/hpo"
    cfg_path = project_root / "configs" / "experiments" / "hpo.yaml"
    if not cfg_path.exists():
        return default_root
    try:
        hpo_cfg = load_yaml_config(str(cfg_path))
    except Exception as exc:
        logger.warning("Could not read HPO config at %s: %s", cfg_path, exc)
        return default_root

    output_dir = hpo_cfg.get("output_dir")
    if not output_dir:
        return default_root
    return project_root / str(output_dir)


def _scan_hpo(
    project_root: Path,
    pipeline: str,
    selected_datasets: list[str],
    requested_model_types: list[str],
) -> dict[str, Any]:
    hpo_root = _resolve_hpo_root(project_root, pipeline)
    study_id, study_dir = _find_latest_study_dir(hpo_root)
    legacy_layout = False

    if not study_dir:
        legacy_files = sorted(hpo_root.glob("best_params_*.json")) if hpo_root.exists() else []
        if legacy_files:
            study_dir = hpo_root
            files = legacy_files
            legacy_layout = True
        else:
            return {
                "available": False,
                "dir": None,
                "study_id": None,
                "study_dir": None,
                "layout": "missing",
                "total_best_params_files": 0,
                "coverage": {},
                "available_pairs": [],
                "use_hpo": False,
            }
    else:
        files = sorted(study_dir.glob("best_params_*.json"))

    coverage: dict[str, bool] = {}
    available_pairs: list[dict[str, str]] = []

    for dataset in selected_datasets:
        for model_type in requested_model_types:
            file_path = study_dir / f"best_params_{dataset}_{model_type}.json"
            has_params = file_path.exists()
            coverage[f"{dataset}:{model_type}"] = has_params
            if has_params:
                available_pairs.append(
                    {
                        "dataset": dataset,
                        "model_type": model_type,
                        "path": str(file_path),
                    }
                )

    if selected_datasets and requested_model_types:
        use_hpo = bool(available_pairs)
    else:
        use_hpo = bool(files)

    return {
        "available": bool(files),
        "dir": str(study_dir),
        "study_id": study_id,
        "study_dir": str(study_dir),
        "layout": "legacy_flat" if legacy_layout else "run_scoped",
        "total_best_params_files": len(files),
        "coverage": coverage,
        "available_pairs": available_pairs,
        "use_hpo": use_hpo,
    }


def _scan_feature_selection(
    project_root: Path,
    pipeline: str,
    selected_datasets: list[str],
    requested_model_types: list[str],
) -> dict[str, Any]:
    fs_root = project_root / f"output/{pipeline}/studies/feature_selection"
    study_id, study_dir = _find_latest_study_dir(fs_root)

    if not study_dir:
        return {
            "available": False,
            "study_id": None,
            "study_dir": None,
            "summary_path": None,
            "manifest_path": None,
            "recommended_mode": None,
            "recommended_model_types": [],
            "rfe_top_k": None,
            "mode_scores": {},
            "model_scores": {},
            "used_successful_runs": 0,
        }

    summary_path = study_dir / "study_summary.json"
    manifest_path = study_dir / "study_manifest.json"
    summary = _read_json(summary_path) or {}
    manifest = _read_json(manifest_path) or {}

    rfe_top_k_raw = manifest.get("rfe_top_k", summary.get("rfe_top_k"))
    rfe_top_k = int(rfe_top_k_raw) if isinstance(rfe_top_k_raw, int) else None

    runs = manifest.get("runs") or summary.get("runs") or []
    mode_scores_raw: dict[str, list[float]] = {}
    model_scores_raw: dict[str, list[float]] = {}
    used_successful_runs = 0

    for run in runs:
        if not isinstance(run, dict):
            continue
        if str(run.get("status", "")).strip().lower() != "success":
            continue

        mode = str(run.get("mode", "")).strip()
        model_type = str(run.get("model", "")).strip().lower()
        if not mode or not model_type:
            continue
        if requested_model_types and model_type not in requested_model_types:
            continue

        baseline_root_raw = run.get("baseline_root")
        if baseline_root_raw:
            baseline_root = Path(str(baseline_root_raw))
            if not baseline_root.is_absolute():
                baseline_root = project_root / baseline_root
        else:
            sub_key = str(run.get("sub_key", f"fs_{mode}__{model_type}"))
            baseline_root = study_dir / "runs" / sub_key / "baseline"

        results_path = baseline_root / "results" / "training_results.json"
        training_results = _read_json(results_path)
        if not training_results:
            continue

        run_has_score = False
        for dataset_name, dataset_payload in training_results.items():
            if not isinstance(dataset_payload, dict):
                continue
            if not _is_selected_dataset(str(dataset_name), selected_datasets):
                continue

            model_payload = dataset_payload.get(model_type)
            metrics = _extract_test_metrics(model_payload)
            score = _composite_score(metrics)
            if score is None:
                continue

            mode_scores_raw.setdefault(mode, []).append(score)
            model_scores_raw.setdefault(model_type, []).append(score)
            run_has_score = True

        if run_has_score:
            used_successful_runs += 1

    mode_scores: dict[str, float] = {}
    for mode, values in mode_scores_raw.items():
        mean_score = _mean(values)
        if mean_score is not None:
            mode_scores[mode] = mean_score

    model_scores: dict[str, float] = {}
    for model_type, values in model_scores_raw.items():
        mean_score = _mean(values)
        if mean_score is not None:
            model_scores[model_type] = mean_score

    recommended_mode = None
    if mode_scores:
        recommended_mode = max(sorted(mode_scores.keys()), key=lambda mode: mode_scores[mode])

    recommended_model_types = sorted(
        model_scores.keys(),
        key=lambda model_type: (-model_scores[model_type], model_type),
    )

    return {
        "available": True,
        "study_id": study_id,
        "study_dir": str(study_dir),
        "summary_path": str(summary_path) if summary_path.exists() else None,
        "manifest_path": str(manifest_path) if manifest_path.exists() else None,
        "recommended_mode": recommended_mode,
        "recommended_model_types": recommended_model_types,
        "rfe_top_k": rfe_top_k,
        "mode_scores": mode_scores,
        "model_scores": model_scores,
        "used_successful_runs": used_successful_runs,
    }


def _resolve_output_path(
    project_root: Path,
    pipeline: str,
    run_id: Optional[str],
    explicit_output_file: Optional[str],
) -> Path:
    if explicit_output_file:
        return Path(explicit_output_file)
    if run_id:
        return (
            project_root / f"output/{pipeline}/runs/{run_id}/recommendations/selector_contract.json"
        )
    return project_root / f"output/{pipeline}/recommendations/selector_contract.latest.json"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build selector contract from HPO and feature-selection study outputs."
    )
    parser.add_argument("--pipeline", default="cardiac", help="Pipeline name")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Optional dataset override used when scoring study outputs.",
    )
    parser.add_argument(
        "--model-types",
        nargs="+",
        default=None,
        help="Optional model types override used when scoring study outputs.",
    )
    parser.add_argument(
        "--run-id",
        default=os.getenv("RUN_ID"),
        help="Optional run identifier for run-scoped output path.",
    )
    parser.add_argument(
        "--feature-mode-fallback",
        default=None,
        help="Fallback feature-selection mode when study outputs are unavailable.",
    )
    parser.add_argument(
        "--output-file",
        default=None,
        help="Optional explicit output file path for selector contract.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Verbosity: -v=info, -vv=debug",
    )
    args = parser.parse_args()

    _configure_logging(args.verbose)

    project_root = get_project_root(Path(__file__))
    pipeline_cfg = load_yaml_config(str(project_root / f"configs/pipelines/{args.pipeline}.yaml"))
    training_cfg = pipeline_cfg.get("training", {})

    selected_datasets = [
        str(dataset).strip()
        for dataset in (args.datasets or pipeline_cfg.get("runtime", {}).get("datasets", []))
    ]
    selected_datasets = [dataset for dataset in selected_datasets if dataset]

    requested_model_types = _normalise_model_types(args.model_types)
    if not requested_model_types:
        requested_model_types = _normalise_model_types(training_cfg.get("model_types"))
    if not requested_model_types:
        requested_model_types = _normalise_model_types(
            [training_cfg.get("model", "logistic_regression")]
        )

    fallback_feature_mode = (
        str(args.feature_mode_fallback).strip()
        if args.feature_mode_fallback
        else str(training_cfg.get("feature_selection_mode", "exclude_sensitive")).strip()
    )

    hpo_scan = _scan_hpo(project_root, args.pipeline, selected_datasets, requested_model_types)
    fs_scan = _scan_feature_selection(
        project_root,
        args.pipeline,
        selected_datasets,
        requested_model_types,
    )

    if not hpo_scan.get("use_hpo"):
        logger.warning(
            "No HPO study output found — use_hpo=False, falling back to base model configs. "
            "Run the HPO study (stage 5) to enable tuned hyperparameters."
        )
    if not fs_scan.get("recommended_mode"):
        logger.warning(
            "No feature-selection study output found — using YAML fallback mode '%s'. "
            "Run the feature-selection study (stage 6) to enable data-driven mode selection.",
            fallback_feature_mode,
        )

    recommended_mode = fs_scan.get("recommended_mode") or fallback_feature_mode
    recommended_rfe_top_k = fs_scan.get("rfe_top_k")
    if recommended_rfe_top_k is None:
        recommended_rfe_top_k = int(training_cfg.get("rfe_top_k", 10))

    recommended_model_types = fs_scan.get("recommended_model_types") or requested_model_types

    contract = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "pipeline": args.pipeline,
        "run_id": args.run_id,
        "inputs": {
            "datasets": selected_datasets,
            "requested_model_types": requested_model_types,
            "feature_mode_fallback": fallback_feature_mode,
        },
        "studies": {
            "hpo": hpo_scan,
            "feature_selection": fs_scan,
        },
        "recommendations": {
            "use_hpo": bool(hpo_scan.get("use_hpo")),
            "feature_selection_mode": recommended_mode,
            "feature_selection_mode_source": (
                "feature_selection_study" if fs_scan.get("recommended_mode") else "fallback"
            ),
            "rfe_top_k": int(recommended_rfe_top_k),
            "model_types": recommended_model_types,
            "model_types_source": (
                "feature_selection_study" if fs_scan.get("recommended_model_types") else "requested"
            ),
        },
    }

    output_path = _resolve_output_path(
        project_root,
        args.pipeline,
        args.run_id,
        args.output_file,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(contract, handle, indent=2)

    logger.info("Selector contract written: %s", output_path)
    logger.info(
        "Recommendations: use_hpo=%s feature_selection_mode=%s model_types=%s",
        contract["recommendations"]["use_hpo"],
        contract["recommendations"]["feature_selection_mode"],
        contract["recommendations"]["model_types"],
    )
    print(output_path)


if __name__ == "__main__":
    main()
