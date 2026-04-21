#!/usr/bin/env python3
"""
Train baseline logistic regression models for a pipeline.

This script:
1. Loads preprocessed train/test datasets
2. Trains logistic regression for each dataset
3. Evaluates on test set with multiple thresholds
4. Generates predictions with probabilities
5. Saves trained models and predictions
6. Logs performance metrics

Usage:
    python scripts/common/train_baseline.py --pipeline cardiac
"""

import argparse
import inspect
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from fairxai.cli.runner_base import load_pipeline_config, resolve_project_root, setup_phase_logging
from fairxai.data.feature_selection import build_feature_set
from fairxai.experiments.data_io import build_schema_excludes, resolve_base_dataset
from fairxai.explainability.tabular import (
    build_lime_explainer,
    lime_explain_instance,
    shap_explain_tabular,
)
from fairxai.models import get_model_class
from fairxai.models.baseline import generate_predictions_with_metadata
from fairxai.models.cv_trainer import CVTrainer
from fairxai.utils.config import load_yaml_config

ALLOWED_TRAINING_METHODS = {"single_split", "kfold_cv"}


def save_xai_outputs(
    model: Any,
    model_type: str,
    X_ref: pd.DataFrame,
    X_lime: pd.DataFrame,
    output_dir: Path,
    dataset_name: str,
    X_global: Optional[pd.DataFrame] = None,
    xai_cfg: Optional[dict] = None,
) -> None:
    """Save holdout-based SHAP and LIME outputs.

    Outputs are placed under::

        output_dir/{dataset_name}/holdout/shap/summary.csv
        output_dir/{dataset_name}/holdout/lime/examples.csv

    SHAP global summary includes ``std_abs_shap`` and percentile columns
    (p25, p50, p75) alongside the original ``mean_abs_shap``.
    """
    if xai_cfg is None:
        xai_cfg = {}
    if not xai_cfg.get("enabled", True):
        logging.info("XAI disabled via config xai.enabled=false")
        return

    holdout_shap_dir = output_dir / dataset_name / "holdout" / "shap"
    holdout_lime_dir = output_dir / dataset_name / "holdout" / "lime"
    holdout_shap_dir.mkdir(parents=True, exist_ok=True)
    holdout_lime_dir.mkdir(parents=True, exist_ok=True)
    lime_instances = int(xai_cfg.get("lime_instances", 3))
    global_max = int(xai_cfg.get("global_max_samples", 1000))
    allow_svm_shap = bool(xai_cfg.get("allow_svm_shap", False))
    shap_skip_models = {
        str(m).strip().lower() for m in xai_cfg.get("skip_shap_model_types", ["svm"])
    }
    shap_enabled = model_type not in shap_skip_models

    if not shap_enabled:
        logging.info(f"SHAP skipped for model_type={model_type} via xai.skip_shap_model_types")

    # Global SHAP summary (dataset-level) with percentiles
    if X_global is not None and shap_enabled:
        try:
            df_global = X_global.copy()
            if len(df_global) > global_max:
                df_global = df_global.sample(n=global_max, random_state=42)
            shap_global = shap_explain_tabular(
                model.model,
                df_global,
                max_samples=global_max,
                allow_svm=allow_svm_shap,
            )
            shap_vals_global = np.abs(shap_global.shap_values)
            shap_global_summary = pd.DataFrame(
                {
                    "feature": shap_global.feature_names,
                    "mean_abs_shap": np.mean(shap_vals_global, axis=0),
                    "std_abs_shap": np.std(shap_vals_global, axis=0),
                    "p25": np.percentile(shap_vals_global, 25, axis=0),
                    "p50": np.percentile(shap_vals_global, 50, axis=0),
                    "p75": np.percentile(shap_vals_global, 75, axis=0),
                }
            ).sort_values("mean_abs_shap", ascending=False)
            shap_global_file = holdout_shap_dir / "summary.csv"
            shap_global_summary.to_csv(shap_global_file, index=False)
            logging.info(f"[SUCCESS] Holdout SHAP summary saved: {shap_global_file}")
        except Exception as exc:
            logging.warning(f"Global SHAP failed for {dataset_name}: {exc}")

    # LIME examples
    try:

        def _wrap_decision_function(df_model):
            class _DecisionFunctionWrapper:
                def __init__(self, base_model):
                    self.base_model = base_model

                def predict_proba(self, X):
                    scores = self.base_model.decision_function(X)
                    scores = np.asarray(scores)
                    if scores.ndim == 1:
                        prob_pos = 1.0 / (1.0 + np.exp(-scores))
                        return np.vstack([1 - prob_pos, prob_pos]).T
                    exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
                    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

            return _DecisionFunctionWrapper(df_model)

        def _resolve_lime_model(raw_model):
            if hasattr(raw_model, "predict_proba"):
                return raw_model
            if hasattr(raw_model, "model") and hasattr(raw_model.model, "predict_proba"):
                return raw_model.model
            if hasattr(raw_model, "decision_function"):
                return _wrap_decision_function(raw_model)
            if hasattr(raw_model, "model") and hasattr(raw_model.model, "decision_function"):
                return _wrap_decision_function(raw_model.model)
            return None

        lime_model = _resolve_lime_model(model)
        if lime_instances > 0 and lime_model is not None:
            lime_rows = X_lime.sample(n=min(lime_instances, len(X_lime)), random_state=42)
            lime_results = []
            # Build explainer once and reuse across all LIME instances.
            lime_expl = build_lime_explainer(
                X_ref,
                feature_names=list(X_ref.columns),
                class_names=["no_disease", "disease"],
            )
            for idx, row in lime_rows.iterrows():
                exp = lime_explain_instance(
                    model=lime_model,
                    data_row=row,
                    training_data=X_ref,
                    feature_names=list(X_ref.columns),
                    class_names=["no_disease", "disease"],
                    num_features=10,
                    explainer=lime_expl,
                )
                for feat, weight in exp.weights:
                    lime_results.append(
                        {
                            "instance_id": int(idx),
                            "feature": feat,
                            "weight": weight,
                            "intercept": exp.intercept,
                            "score": exp.score,
                            "local_pred": exp.local_pred,
                        }
                    )
            lime_df = pd.DataFrame(lime_results)
            lime_file = holdout_lime_dir / "examples.csv"
            lime_df.to_csv(lime_file, index=False)
            logging.info(f"[SUCCESS] Holdout LIME examples saved: {lime_file}")
        elif lime_instances > 0:
            logging.warning(f"LIME skipped for {dataset_name}: no predict_proba/decision_function")
    except Exception as exc:
        logging.warning(f"LIME failed for {dataset_name}: {exc}")


def _normalise_model_types(raw_values: Optional[list[Any]]) -> list[str]:
    if not raw_values:
        return []
    normalized: list[str] = []
    for raw in raw_values:
        value = str(raw).strip().lower()
        if value and value not in normalized:
            normalized.append(value)
    return normalized


def _load_selector_recommendations(selector_contract_path: Optional[str]) -> dict[str, Any]:
    if not selector_contract_path:
        return {}

    contract_path = Path(selector_contract_path)
    if not contract_path.exists():
        logging.warning("Selector contract not found at %s; ignoring.", contract_path)
        return {}

    try:
        with open(contract_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:
        logging.warning("Could not read selector contract %s: %s", contract_path, exc)
        return {}

    recommendations = payload.get("recommendations")
    if isinstance(recommendations, dict):
        return recommendations

    logging.warning(
        "Selector contract %s has no recommendations block; ignoring.",
        contract_path,
    )
    return {}


def _resolve_model_types(
    args_model_types: Optional[list[str]],
    training_cfg: dict,
    selector_model_types: Optional[list[Any]] = None,
) -> list[str]:
    """Resolve model types with CLI > selector > config > legacy precedence."""
    cli_types = _normalise_model_types(args_model_types)
    if cli_types:
        return cli_types

    selector_types = _normalise_model_types(selector_model_types)
    if selector_types:
        return selector_types

    cfg_types = training_cfg.get("model_types")
    resolved_cfg = _normalise_model_types(cfg_types)
    if resolved_cfg:
        return resolved_cfg

    legacy = training_cfg.get("model", "logistic_regression")
    return _normalise_model_types([legacy])


def _resolve_training_methods(
    args_training_methods: Optional[list[str]], training_cfg: dict
) -> list[str]:
    """Resolve training methods from CLI args, config, then code defaults.

    Precedence is strictly:
      1) CLI flags
      2) pipeline YAML config
      3) code defaults
    """
    cfg_methods = training_cfg.get("training_methods")
    raw_methods = (
        args_training_methods
        if args_training_methods
        else cfg_methods if cfg_methods else ["single_split", "kfold_cv"]
    )

    resolved: list[str] = []
    for raw in raw_methods:
        method = str(raw).strip().lower()
        if method not in ALLOWED_TRAINING_METHODS:
            allowed = ", ".join(sorted(ALLOWED_TRAINING_METHODS))
            raise ValueError(f"Unsupported training method '{raw}'. Allowed: {allowed}")
        if method not in resolved:
            resolved.append(method)
    return resolved


def _resolve_feature_selection_mode(
    args_mode: Optional[str],
    training_cfg: dict,
    selector_mode: Optional[str] = None,
) -> str:
    """Resolve feature-selection mode with CLI > selector > config > default precedence."""
    if args_mode:
        return str(args_mode).strip()

    if selector_mode:
        return str(selector_mode).strip()

    cfg_mode = training_cfg.get("feature_selection_mode")
    if cfg_mode:
        return str(cfg_mode).strip()

    return "exclude_sensitive"


def _resolve_bool_setting(
    cli_value: Optional[bool],
    cfg: dict,
    key: str,
    default_value: bool,
    selector_value: Optional[Any] = None,
) -> bool:
    """Resolve bool setting with CLI > selector > config > default precedence."""
    if cli_value is not None:
        return bool(cli_value)

    if selector_value is not None:
        return bool(selector_value)

    cfg_value = cfg.get(key)
    if cfg_value is not None:
        return bool(cfg_value)

    return default_value


def _resolve_int_setting(
    cli_value: Optional[int],
    cfg: dict,
    key: str,
    default_value: int,
    selector_value: Optional[Any] = None,
) -> int:
    """Resolve int setting with CLI > selector > config > default precedence."""
    if cli_value is not None:
        return int(cli_value)

    if selector_value is not None:
        try:
            return int(selector_value)
        except (TypeError, ValueError):
            logging.warning(
                "Invalid selector int override for key '%s': %r. Falling back.",
                key,
                selector_value,
            )

    cfg_value = cfg.get(key)
    if cfg_value is not None:
        return int(cfg_value)

    return int(default_value)


def _is_selected_dataset(dataset_name: str, selected_datasets: Optional[set[str]]) -> bool:
    if not selected_datasets:
        return True
    return any(dataset_name == d or dataset_name.startswith(f"{d}_") for d in selected_datasets)


def _apply_model_thread_override(
    model_class: Any, model_params: dict, model_n_jobs: Optional[int]
) -> dict:
    """Apply an explicit n_jobs override only when the model accepts it."""
    if model_n_jobs is None:
        return model_params

    try:
        signature = inspect.signature(model_class.__init__)
    except (TypeError, ValueError):
        return model_params

    if "n_jobs" in signature.parameters:
        updated = dict(model_params)
        updated["n_jobs"] = model_n_jobs
        return updated

    logging.debug(
        "Model class %s does not accept n_jobs; override ignored.",
        getattr(model_class, "__name__", str(model_class)),
    )
    return model_params


def _build_model_params(
    model_type: str,
    training_cfg: dict,
    random_state: int,
    project_root: Path,
    hpo_dir: Optional[Path] = None,
    dataset_name: Optional[str] = None,
) -> dict:
    """Load base hyperparameters from model config file.

    When ``hpo_dir`` and ``dataset_name`` are provided, best params from a
    previous :func:`~fairxai.training.grid_search.run_hpo` run are merged on
    top of the base config, overriding only the searched keys.
    """
    from fairxai.training.grid_search import load_hpo_params

    model_cfg_path = project_root / "configs" / "models" / f"{model_type}.yaml"
    params = dict(load_yaml_config(str(model_cfg_path)).get("hyperparameters", {}))
    params.setdefault("random_state", random_state)

    if hpo_dir is not None and dataset_name is not None:
        hpo_best = load_hpo_params(hpo_dir, dataset_name, model_type)
        if hpo_best:
            logging.info(f"  [HPO] Loaded best params for {model_type}/{dataset_name}: {hpo_best}")
            params.update(hpo_best)
        else:
            logging.debug(
                f"  [HPO] No saved params found for {model_type}/{dataset_name}; " "using defaults."
            )
    return params


def _is_shap_enabled_for_model(model_type: str, xai_cfg: dict) -> bool:
    skip_models = {str(m).strip().lower() for m in xai_cfg.get("skip_shap_model_types", ["svm"])}
    return model_type not in skip_models


def main():
    parser = argparse.ArgumentParser(description="Train baseline models")
    parser.add_argument(
        "--project-root",
        type=str,
        default=None,
        help=(
            "Optional override for repository root used to resolve configs/data/outputs. "
            "If omitted, FAIRXAI_PROJECT_ROOT env var is used when set; "
            "otherwise defaults to script-inferred repo root."
        ),
    )
    parser.add_argument(
        "--pipeline",
        type=str,
        default="cardiac",
        choices=["cardiac", "dermatology"],
        help="Pipeline name (e.g., cardiac, dermatology)",
    )
    parser.add_argument(
        "--model-types",
        nargs="+",
        default=None,
        help="Optional model types override (e.g. logistic_regression random_forest svm xgboost)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Optional dataset names to train (CLI override).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Override baseline output directory (baseline_root). "
            "results/ and models/ are created inside this path. "
            "Used by scripts/studies/run_feature_selection_study.py to "
            "place sub-run artifacts under studies/feature_selection/<study_id>/."
        ),
    )
    parser.add_argument(
        "--selector-contract",
        type=str,
        default=None,
        help=(
            "Optional selector contract JSON generated from studies. "
            "Used as precedence layer between CLI flags and pipeline YAML."
        ),
    )
    parser.add_argument(
        "--use-hpo",
        dest="use_hpo",
        action="store_true",
        default=None,
        help=(
            "Load best hyperparameters from a previous HPO run "
            "(output/cardiac/studies/hpo/best_params_<dataset>_<model>.json). "
            "Run scripts/studies/run_hpo.py first."
        ),
    )
    parser.add_argument(
        "--no-use-hpo",
        dest="use_hpo",
        action="store_false",
        help="Disable HPO parameters even if config enables them.",
    )
    parser.add_argument(
        "--training-methods",
        nargs="+",
        default=None,
        help=("Training methods to run (CLI override). " "Supported: single_split kfold_cv"),
    )
    parser.add_argument(
        "--feature-selection-mode",
        default=None,
        help=(
            "Feature selection strategy: "
            "exclude_sensitive (default), include_all_sensitive, "
            "include_sex_only, include_age_only, include_ethnicity_only, rfe_top_k"
        ),
    )
    parser.add_argument(
        "--rfe-top-k",
        type=int,
        default=None,
        help=(
            "Number of features to keep when --feature-selection-mode=rfe_top_k "
            "(CLI override; default from config/code)."
        ),
    )
    parser.add_argument(
        "--model-n-jobs",
        type=int,
        default=None,
        help="Override model n_jobs when supported (default: use model config)",
    )
    parser.add_argument(
        "--cv-n-jobs",
        type=int,
        default=None,
        help="Parallel workers for CV folds when XAI is disabled (CLI override).",
    )
    parser.add_argument(
        "-v", "--verbose", action="count", default=0, help="Verbosity: -v=info, -vv=debug"
    )
    args = parser.parse_args()

    pipeline = args.pipeline

    # Paths
    project_root = resolve_project_root(Path(__file__), cli_project_root=args.project_root)
    pipeline_cfg = load_pipeline_config(project_root, pipeline)
    pipeline_cfg_path = project_root / f"configs/pipelines/{pipeline}.yaml"
    schema_path = project_root / pipeline_cfg["runtime"]["schema_mapping_json"]
    with open(schema_path, "r") as f:
        schema_cfg = json.load(f)
    data_processed = project_root / pipeline_cfg["paths"]["processed_dir"]
    run_id = os.getenv("RUN_ID")
    if args.output_dir:
        baseline_root = Path(args.output_dir)
    else:
        if not run_id:
            raise RuntimeError(
                "RUN_ID is not set and --output-dir was not provided. "
                "train_baseline.py must be called from the pipeline (RUN_ID exported) "
                "or via a study script that passes --output-dir."
            )
        baseline_root = project_root / f"output/{pipeline}/runs/{run_id}/baseline"
    experiments_dir = baseline_root / "results"
    predictions_dir = experiments_dir / "predictions"
    models_dir = baseline_root / "models"
    setup_phase_logging(
        project_root,
        "training_baseline.log",
        verbose=args.verbose,
        run_id=run_id,
        stage_name="train",
    )

    # Setup
    logging.info("[PHASE] Baseline training started")
    logging.info(f"Effective project root: {project_root}")
    logging.info(f"Resolved pipeline config: {pipeline_cfg_path}")
    logging.info(
        "Run context: pipeline=%s run_id=%s data_processed=%s",
        pipeline,
        run_id or "none",
        data_processed,
    )

    logging.info(f"  Run root: {baseline_root}")
    logging.info(f"  Models will be saved to: {models_dir}")
    logging.info(f"  Results will be saved to: {experiments_dir}")

    models_dir.mkdir(parents=True, exist_ok=True)
    experiments_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    # Configuration
    training_cfg = pipeline_cfg.get("training", {})
    selector_recommendations = _load_selector_recommendations(args.selector_contract)
    selector_model_types = selector_recommendations.get("model_types")
    selector_feature_mode = selector_recommendations.get("feature_selection_mode")
    selector_rfe_top_k = selector_recommendations.get("rfe_top_k")
    selector_use_hpo = selector_recommendations.get("use_hpo")

    random_state = training_cfg.get("random_state", 42)
    thresholds_to_test = training_cfg.get("thresholds", [0.3, 0.4, 0.5, 0.6, 0.7])
    model_types = _resolve_model_types(args.model_types, training_cfg, selector_model_types)
    training_methods = _resolve_training_methods(args.training_methods, training_cfg)
    feature_selection_mode = _resolve_feature_selection_mode(
        args.feature_selection_mode,
        training_cfg,
        selector_feature_mode,
    )
    rfe_top_k = _resolve_int_setting(
        args.rfe_top_k,
        training_cfg,
        "rfe_top_k",
        10,
        selector_rfe_top_k,
    )
    use_hpo = _resolve_bool_setting(
        args.use_hpo,
        training_cfg,
        "use_hpo",
        False,
        selector_use_hpo,
    )
    cv_n_jobs = _resolve_int_setting(args.cv_n_jobs, training_cfg, "cv_n_jobs", 1)

    logging.info("Configuration:")
    logging.info(f"  Models: {model_types}")
    logging.info(f"  Training methods: {training_methods}")
    logging.info(f"  Random state: {random_state}")
    logging.info(f"  Decision thresholds: {thresholds_to_test}")
    logging.info(f"  Feature-selection mode: {feature_selection_mode}")
    logging.info(f"  RFE top-k: {rfe_top_k}")
    logging.info(f"  Use HPO params: {use_hpo}")
    if args.selector_contract:
        logging.info(f"  Selector contract: {args.selector_contract}")
    logging.info(f"  Model n_jobs override: {args.model_n_jobs}")
    logging.info(f"  CV n_jobs: {cv_n_jobs}")

    # Find processed datasets
    train_files = list(data_processed.glob("*_train_scaled.csv"))
    selected_datasets = set(d.strip() for d in args.datasets) if args.datasets else None
    if selected_datasets:
        train_files = [
            p
            for p in train_files
            if _is_selected_dataset(p.stem.replace("_train_scaled", ""), selected_datasets)
        ]

    if not train_files:
        logging.error(f"No training datasets found in {data_processed}")
        logging.error(
            "Please run scripts/common/preprocess_data.py --pipeline %s first." % pipeline
        )
        return

    logging.info(f"Found {len(train_files)} datasets to train on")

    # Train model(s) for each dataset
    results_summary = {}
    split_info: dict = {}

    for train_file in train_files:
        dataset_name = train_file.stem.replace("_train_scaled", "")
        test_file = train_file.parent / f"{dataset_name}_test_scaled.csv"

        if not test_file.exists():
            logging.warning(f"Test file not found for {dataset_name}, skipping")
            continue

        logging.info("[DATASET] Training dataset=%s", dataset_name)

        # Load data
        train_df = pd.read_csv(train_file)
        test_df = pd.read_csv(test_file)

        logging.info("Loaded data:")
        logging.info(f"  Train: {len(train_df)} samples")
        logging.info(f"  Test: {len(test_df)} samples")
        logging.info(f"  Feature-selection mode: {feature_selection_mode}")

        # Separate features, target, and sensitive attributes
        # Note: scaled files have both encoded and categorical versions
        # We keep the encoded numerical versions for modeling
        target_col = training_cfg.get("target", "heart_disease")

        # Capture split metadata for split_info.json
        _split_cfg = pipeline_cfg.get("split", {})
        split_info[dataset_name] = {
            "n_train": len(train_df),
            "n_test": len(test_df),
            "test_size": _split_cfg.get("test_size", 0.3),
            "random_state": _split_cfg.get("random_state", 42),
            "target_column": target_col,
            "train_target_dist": (
                train_df[target_col].value_counts(normalize=True).round(4).to_dict()
                if target_col in train_df.columns
                else {}
            ),
            "test_target_dist": (
                test_df[target_col].value_counts(normalize=True).round(4).to_dict()
                if target_col in test_df.columns
                else {}
            ),
        }
        sensitive_candidates = pipeline_cfg.get("fairness", {}).get(
            "sensitive_attributes", ["age_group", "sex"]
        )
        sensitive_cols = [col for col in sensitive_candidates if col in train_df.columns]

        base_dataset = resolve_base_dataset(schema_cfg, dataset_name)
        schema_exclude = build_schema_excludes(schema_cfg, base_dataset)

        extra_excludes = [
            col
            for col in train_df.columns
            if col.startswith("sex_bin") or col.startswith("sex_extended")
        ]
        # Always exclude target and schema/extra cols; sensitive cols are handled by
        # build_feature_set according to the selected mode.
        hard_exclude = set([target_col] + schema_exclude + extra_excludes)

        # Candidate features: everything not hard-excluded (includes sensitive attrs).
        candidate_cols = [c for c in train_df.columns if c not in hard_exclude]
        X_train_full = train_df[candidate_cols].select_dtypes(include=[np.number])
        X_test_full = test_df[candidate_cols].select_dtypes(include=[np.number])

        fs_mode = feature_selection_mode
        if fs_mode != "rfe_top_k":
            X_train, feature_cols = build_feature_set(
                X_train_full,
                sensitive_attrs=sensitive_cols,
                mode=fs_mode,
                top_k=rfe_top_k,
                trained_model=None,
            )
            X_test = X_test_full[feature_cols]
        else:
            # rfe_top_k needs a fitted model — deferred to inside the model loop (two-pass)
            X_train = X_train_full
            X_test = X_test_full
            feature_cols = list(X_train_full.columns)

        y_train = train_df[target_col]
        y_test = test_df[target_col]

        # Keep categorical versions for fairness analysis
        sensitive_train = train_df[sensitive_cols]
        sensitive_test = test_df[sensitive_cols]

        logging.info(f"Features: {len(feature_cols)} (mode={fs_mode})")

        results_summary.setdefault(dataset_name, {})

        for model_type in model_types:
            logging.info("[MODEL] Training model=%s dataset=%s", model_type, dataset_name)

            try:
                model_class = get_model_class(model_type)
                hpo_dir = project_root / f"output/{pipeline}/studies/hpo" if use_hpo else None
                model_params = _build_model_params(
                    model_type,
                    training_cfg,
                    random_state,
                    project_root,
                    hpo_dir=hpo_dir,
                    dataset_name=dataset_name,
                )
                model_params = _apply_model_thread_override(
                    model_class, model_params, args.model_n_jobs
                )
                model = model_class(**model_params)
            except Exception as exc:
                logging.warning(f"Skipping model_type={model_type} for {dataset_name}: {exc}")
                results_summary[dataset_name][model_type] = {
                    "status": "skipped",
                    "reason": str(exc),
                }
                continue

            if fs_mode == "rfe_top_k":
                # Two-pass: quick first fit on all features to importances to reduce to retrain.
                logging.info(
                    f"  [rfe_top_k] first-pass fit on {len(X_train_full.columns)} features"
                )
                first_pass = model_class(**model_params)
                first_pass.train(X_train_full, y_train)
                X_train, feature_cols = build_feature_set(
                    X_train_full,
                    sensitive_attrs=sensitive_cols,
                    mode="rfe_top_k",
                    top_k=rfe_top_k,
                    trained_model=first_pass,
                )
                X_test = X_test_full[feature_cols]
                feature_preview = feature_cols[:10]
                preview_suffix = "..." if len(feature_cols) > 10 else ""
                logging.info(
                    "  [rfe_top_k] selected %d features: %s%s",
                    len(feature_cols),
                    feature_preview,
                    preview_suffix,
                )

            model_result: dict[str, Any] = {
                "status": "success",
                "model_params": model_params,
                "n_features": len(feature_cols),
                "n_train": len(train_df),
                "n_test": len(test_df),
                "training_methods_requested": list(training_methods),
                "training_methods": {},
            }

            xai_cfg = pipeline_cfg.get("xai", {})
            xai_dir = experiments_dir / "xai"
            xai_dataset_key = f"{dataset_name}__{model_type}"
            X_full = pd.concat([X_train, X_test], ignore_index=True)
            y_full = pd.concat([y_train, y_test], ignore_index=True)
            sensitive_full = pd.concat([sensitive_train, sensitive_test], ignore_index=True)

            if "single_split" in training_methods:
                model = model_class(**model_params)
                train_metrics = model.train(X_train, y_train)

                logging.info("Evaluation:")
                test_metrics_default = model.evaluate(X_test, y_test, threshold=0.5)

                logging.info("Test Set Performance (threshold=0.5):")
                logging.info(f"  Accuracy:  {test_metrics_default['accuracy']:.4f}")
                logging.info(f"  Precision: {test_metrics_default['precision']:.4f}")
                logging.info(f"  Recall:    {test_metrics_default['recall']:.4f}")
                logging.info(f"  F1 Score:  {test_metrics_default['f1_score']:.4f}")
                logging.info(f"  AUC-ROC:   {test_metrics_default['auc_roc']:.4f}")

                cm = test_metrics_default["confusion_matrix"]
                logging.info("  Confusion Matrix:")
                logging.info(f"    TN: {cm['tn']:3d}  FP: {cm['fp']:3d}")
                logging.info(f"    FN: {cm['fn']:3d}  TP: {cm['tp']:3d}")

                logging.info("Threshold analysis:")
                threshold_results = []

                for threshold in thresholds_to_test:
                    metrics = model.evaluate(X_test, y_test, threshold=threshold)
                    threshold_results.append(metrics)
                    logging.info(
                        f"  Threshold {threshold:.1f}: "
                        f"Acc={metrics['accuracy']:.3f} "
                        f"Prec={metrics['precision']:.3f} "
                        f"Rec={metrics['recall']:.3f} "
                        f"F1={metrics['f1_score']:.3f}"
                    )

                logging.info("Feature importance (top 10):")
                feature_importance = model.get_feature_importance()
                display_metric = (
                    "coefficient"
                    if "coefficient" in feature_importance.columns
                    else "importance" if "importance" in feature_importance.columns else None
                )
                if display_metric is not None:
                    for _, row in feature_importance.head(10).iterrows():
                        val = row.get(display_metric)
                        if pd.notna(val):
                            logging.info(f"  {row['feature']:20s}: {float(val):+.4f}")

                logging.info("Generating predictions:")
                train_predictions = generate_predictions_with_metadata(
                    model, X_train, y_train, sensitive_train, threshold=0.5
                )
                test_predictions = generate_predictions_with_metadata(
                    model, X_test, y_test, sensitive_test, threshold=0.5
                )

                n_near_threshold_train = train_predictions["near_threshold"].sum()
                n_near_threshold_test = test_predictions["near_threshold"].sum()

                logging.info(f"  Train predictions: {len(train_predictions)}")
                logging.info(
                    f"    Near threshold (+/-0.1): {n_near_threshold_train} "
                    f"({n_near_threshold_train/len(train_predictions):.1%})"
                )
                logging.info(f"  Test predictions: {len(test_predictions)}")
                logging.info(
                    f"    Near threshold (+/-0.1): {n_near_threshold_test} "
                    f"({n_near_threshold_test/len(test_predictions):.1%})"
                )

                model_file = models_dir / f"{dataset_name}_{model_type}.pkl"
                model.save(str(model_file))

                train_pred_file = predictions_dir / f"{dataset_name}_{model_type}_train.csv"
                test_pred_file = predictions_dir / f"{dataset_name}_{model_type}_test.csv"

                train_predictions.to_csv(train_pred_file, index=False)
                test_predictions.to_csv(test_pred_file, index=False)

                logging.info(
                    "[SUCCESS] Predictions saved for dataset=%s model=%s",
                    dataset_name,
                    model_type,
                )
                logging.info(f"  Train: {train_pred_file}")
                logging.info(f"  Test: {test_pred_file}")

                importance_file = predictions_dir / f"{dataset_name}_{model_type}_importance.csv"
                feature_importance.to_csv(importance_file, index=False)
                logging.info(f"  Feature importance: {importance_file}")

                save_xai_outputs(
                    model,
                    model_type,
                    X_train,
                    X_test,
                    xai_dir,
                    xai_dataset_key,
                    X_global=X_train,
                    xai_cfg=xai_cfg,
                )

                if xai_cfg.get("cv_enabled", True) and xai_cfg.get("enabled", True):
                    logging.info("Cross-validated XAI:")
                    try:
                        cv_lime_n = int(xai_cfg.get("cv_lime_instances", 3))
                        cv_shap_max = int(xai_cfg.get("global_max_samples", 1000))
                        allow_svm_shap = bool(xai_cfg.get("allow_svm_shap", False))

                        full_predictions = generate_predictions_with_metadata(
                            model, X_full, y_full, sensitive_full, threshold=0.5
                        )
                        near_mask = full_predictions["near_threshold"]
                        if near_mask.sum() > 0:
                            tracked = (
                                full_predictions[near_mask]
                                .sample(n=min(cv_lime_n, int(near_mask.sum())), random_state=42)
                                .index.tolist()
                            )
                        else:
                            tracked = full_predictions.sample(
                                n=min(cv_lime_n, len(full_predictions)), random_state=42
                            ).index.tolist()
                        logging.info(f"  Tracked LIME instances: {tracked}")

                        cv_trainer = CVTrainer(
                            n_folds=training_cfg.get("cv_folds", 5),
                            random_state=random_state,
                        )
                        shap_enabled = _is_shap_enabled_for_model(model_type, xai_cfg)
                        if not shap_enabled:
                            logging.info(
                                f"  CV SHAP skipped for model_type={model_type}; LIME remains enabled"
                            )
                        cv_xai_results = cv_trainer.run_cv_experiment(
                            model_class=model_class,
                            X=X_full,
                            y=y_full,
                            sensitive_attrs=sensitive_full,
                            model_params=model_params,
                            xai_enabled=True,
                            shap_enabled=shap_enabled,
                            allow_svm_shap=allow_svm_shap,
                            tracked_indices=tracked,
                            feature_names=list(X_full.columns),
                            shap_max_samples=cv_shap_max,
                            cv_n_jobs=cv_n_jobs,
                        )

                        cv_shap_dir = xai_dir / xai_dataset_key / "cv" / "shap"
                        cv_lime_dir = xai_dir / xai_dataset_key / "cv" / "lime"
                        cv_shap_dir.mkdir(parents=True, exist_ok=True)
                        cv_lime_dir.mkdir(parents=True, exist_ok=True)

                        cv_shap_global = CVTrainer.aggregate_cv_shap(
                            cv_xai_results["fold_results"], scope="global"
                        )
                        if cv_shap_global is not None:
                            cv_shap_file = cv_shap_dir / "global_summary.csv"
                            cv_shap_global.to_csv(cv_shap_file, index=False)
                            logging.info(f"[SUCCESS] CV SHAP global summary saved: {cv_shap_file}")

                        cv_shap_local = CVTrainer.aggregate_cv_shap(
                            cv_xai_results["fold_results"], scope="local"
                        )
                        if cv_shap_local is not None:
                            cv_shap_file = cv_shap_dir / "local_summary.csv"
                            cv_shap_local.to_csv(cv_shap_file, index=False)
                            logging.info(f"[SUCCESS] CV SHAP local summary saved: {cv_shap_file}")

                        cv_lime_df = CVTrainer.aggregate_cv_lime(cv_xai_results["fold_results"])
                        if cv_lime_df is not None:
                            cv_lime_file = cv_lime_dir / "tracked.csv"
                            cv_lime_df.to_csv(cv_lime_file, index=False)
                            logging.info(f"[SUCCESS] CV LIME tracked saved: {cv_lime_file}")

                        agg = cv_xai_results["aggregated_metrics"]
                        logging.info(
                            f"  CV performance: "
                            f"F1={agg['f1_score']['mean']:.3f}±{agg['f1_score']['std']:.3f}, "
                            f"AUC={agg['auc_roc']['mean']:.3f}±{agg['auc_roc']['std']:.3f}"
                        )
                    except Exception as exc:
                        logging.warning(f"CV XAI failed for {dataset_name}/{model_type}: {exc}")
                        logging.debug("CV XAI traceback:", exc_info=True)

                model_result.update(
                    {
                        "train_metrics": train_metrics,
                        "test_metrics": test_metrics_default,
                        "threshold_analysis": threshold_results,
                        "near_threshold_pct_test": float(
                            n_near_threshold_test / len(test_predictions)
                        ),
                    }
                )
                model_result["training_methods"]["single_split"] = {
                    "train_metrics": train_metrics,
                    "test_metrics": test_metrics_default,
                    "threshold_analysis": threshold_results,
                    "near_threshold_pct_test": float(n_near_threshold_test / len(test_predictions)),
                }
            else:
                logging.info("Single-split training skipped by configuration")

            if "kfold_cv" in training_methods:
                logging.info("Cross-validation baseline metrics:")
                cv_trainer = CVTrainer(
                    n_folds=training_cfg.get("cv_folds", 5),
                    random_state=random_state,
                )
                cv_results = cv_trainer.run_cv_experiment(
                    model_class=model_class,
                    X=X_full,
                    y=y_full,
                    sensitive_attrs=sensitive_full,
                    model_params=model_params,
                    xai_enabled=False,
                    cv_n_jobs=cv_n_jobs,
                )

                cv_metrics = {}
                for metric_name, stats in cv_results["aggregated_metrics"].items():
                    cv_metrics[metric_name] = {
                        "mean": float(stats["mean"]),
                        "std": float(stats["std"]),
                        "min": float(stats["min"]),
                        "max": float(stats["max"]),
                        "folds": [float(v) for v in stats.get("folds", [])],
                    }

                cv_rows = []
                for fold in cv_results["fold_results"]:
                    vm = fold["val_metrics"]
                    cv_rows.append(
                        {
                            "fold": int(fold["fold_idx"]),
                            "accuracy": float(vm["accuracy"]),
                            "precision": float(vm["precision"]),
                            "recall": float(vm["recall"]),
                            "f1_score": float(vm["f1_score"]),
                            "auc_roc": float(vm["auc_roc"]),
                        }
                    )

                cv_dir = experiments_dir / "cv"
                cv_dir.mkdir(parents=True, exist_ok=True)
                cv_fold_metrics_file = cv_dir / f"{dataset_name}_{model_type}_fold_metrics.csv"
                pd.DataFrame(cv_rows).to_csv(cv_fold_metrics_file, index=False)

                cv_pred_model = model_class(**model_params)
                cv_predictions = cv_trainer.get_fold_predictions(
                    cv_pred_model,
                    X_full,
                    y_full,
                    sensitive_full,
                )
                features_with_index = X_full.reset_index(drop=True).copy()
                features_with_index["sample_idx"] = features_with_index.index
                cv_predictions = cv_predictions.merge(
                    features_with_index, on="sample_idx", how="left"
                )
                cv_predictions["threshold"] = 0.5
                cv_predictions["confidence"] = np.abs(cv_predictions["y_proba"] - 0.5)
                cv_predictions["near_threshold"] = np.abs(cv_predictions["y_proba"] - 0.5) < 0.1

                cv_pred_file = predictions_dir / f"{dataset_name}_{model_type}_cv.csv"
                cv_predictions.to_csv(cv_pred_file, index=False)

                logging.info(
                    "[SUCCESS] CV outputs saved for dataset=%s model=%s",
                    dataset_name,
                    model_type,
                )
                logging.info(f"  Fold metrics: {cv_fold_metrics_file}")
                logging.info(f"  CV predictions: {cv_pred_file}")

                model_result["cv_results"] = {
                    "metrics": cv_metrics,
                    "n_folds": int(cv_results["n_folds"]),
                    "fold_metrics_file": str(cv_fold_metrics_file),
                    "predictions_file": str(cv_pred_file),
                    "near_threshold_pct": float(cv_predictions["near_threshold"].mean()),
                }
                model_result["training_methods"]["kfold_cv"] = {
                    "metrics": cv_metrics,
                    "n_folds": int(cv_results["n_folds"]),
                    "near_threshold_pct": float(cv_predictions["near_threshold"].mean()),
                }
            else:
                logging.info("K-fold CV training skipped by configuration")

            results_summary[dataset_name][model_type] = model_result

    # Save overall results
    results_file = experiments_dir / "training_results.json"
    with open(results_file, "w") as f:
        json.dump(results_summary, f, indent=2, default=str)

    # Save split metadata
    split_info_file = baseline_root / "split_info.json"
    with open(split_info_file, "w") as f:
        json.dump(split_info, f, indent=2, default=str)

    logging.info("[PHASE] Baseline training complete")
    n_models_trained = sum(
        1
        for ds_results in results_summary.values()
        for r in ds_results.values()
        if r.get("status") == "success"
    )
    logging.info(
        "Training summary: datasets=%d models_trained=%d",
        len(results_summary),
        n_models_trained,
    )
    logging.info(f"Models saved to: {models_dir}")
    logging.info(f"Results saved to: {experiments_dir}")


if __name__ == "__main__":
    main()
