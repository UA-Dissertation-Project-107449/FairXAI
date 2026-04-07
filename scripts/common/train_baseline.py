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
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from fairxai.cli.runner_base import get_project_root, load_pipeline_config, setup_phase_logging
from fairxai.cli.runner_utils import archive_latest_run
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


def _resolve_model_types(args_model_types: Optional[list[str]], training_cfg: dict) -> list[str]:
    """Resolve model types from CLI args or pipeline config."""
    if args_model_types:
        return [m.strip().lower() for m in args_model_types]

    cfg_types = training_cfg.get("model_types")
    if cfg_types:
        return [str(m).strip().lower() for m in cfg_types]

    legacy = training_cfg.get("model", "logistic_regression")
    return [str(legacy).strip().lower()]


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
            logging.info(
                f"  [HPO] Loaded best params for {model_type}/{dataset_name}: {hpo_best}"
            )
            params.update(hpo_best)
        else:
            logging.debug(
                f"  [HPO] No saved params found for {model_type}/{dataset_name}; "
                "using defaults."
            )
    return params


def _is_shap_enabled_for_model(model_type: str, xai_cfg: dict) -> bool:
    skip_models = {str(m).strip().lower() for m in xai_cfg.get("skip_shap_model_types", ["svm"])}
    return model_type not in skip_models


def main():
    parser = argparse.ArgumentParser(description="Train baseline models")
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
        "--use-hpo",
        action="store_true",
        default=False,
        help=(
            "Load best hyperparameters from a previous HPO run "
            "(output/cardiac/hpo/best_params_<dataset>_<model>.json). "
            "Run scripts/experiments/run_hpo.py first."
        ),
    )
    parser.add_argument(
        "--feature-selection-mode",
        default="exclude_sensitive",
        help=(
            "Feature selection strategy: "
            "exclude_sensitive (default), include_all_sensitive, "
            "include_sex_only, include_age_only, include_ethnicity_only, rfe_top_k"
        ),
    )
    parser.add_argument(
        "--rfe-top-k",
        type=int,
        default=10,
        help="Number of features to keep when --feature-selection-mode=rfe_top_k (default: 10)",
    )
    parser.add_argument(
        "-v", "--verbose", action="count", default=0, help="Verbosity: -v=info, -vv=debug"
    )
    args = parser.parse_args()

    pipeline = args.pipeline

    # Paths
    project_root = get_project_root(Path(__file__))
    pipeline_cfg = load_pipeline_config(project_root, pipeline)
    schema_path = project_root / pipeline_cfg["runtime"]["schema_mapping_json"]
    with open(schema_path, "r") as f:
        schema_cfg = json.load(f)
    data_processed = project_root / pipeline_cfg["paths"]["processed_dir"]
    run_id = os.getenv("RUN_ID")
    if run_id:
        baseline_root = project_root / f"output/{pipeline}/runs/{run_id}/baseline"
        experiments_dir = baseline_root / "results"
        models_dir = baseline_root / "models"
    else:
        experiments_dir = project_root / pipeline_cfg["paths"]["experiments_dir"]
        models_dir = project_root / pipeline_cfg["paths"]["models_dir"]
    log_dir = setup_phase_logging(
        project_root,
        "training_baseline.log",
        verbose=args.verbose,
        run_id=run_id,
        stage_name="train",
    )
    if not run_id:
        baseline_root = project_root / f"output/{pipeline}/baseline"

    # Setup
    logging.info("[PHASE] Baseline training started")
    if not run_id:
        archive_latest_run(
            baseline_root,
            enabled=(os.getenv("ARCHIVE_BASELINE", "true").lower() == "true"),
            logger=logging.getLogger(__name__),
        )

    models_dir.mkdir(parents=True, exist_ok=True)
    experiments_dir.mkdir(parents=True, exist_ok=True)

    # Configuration
    training_cfg = pipeline_cfg.get("training", {})
    random_state = training_cfg.get("random_state", 42)
    thresholds_to_test = training_cfg.get("thresholds", [0.3, 0.4, 0.5, 0.6, 0.7])
    model_types = _resolve_model_types(args.model_types, training_cfg)

    logging.info(f"Configuration:")
    logging.info(f"  Models: {model_types}")
    logging.info(f"  Random state: {random_state}")
    logging.info(f"  Decision thresholds: {thresholds_to_test}")

    # Find processed datasets
    train_files = list(data_processed.glob("*_train_scaled.csv"))

    if not train_files:
        logging.error(f"No training datasets found in {data_processed}")
        logging.error(
            "Please run scripts/common/preprocess_data.py --pipeline %s first." % pipeline
        )
        return

    logging.info(f"\nFound {len(train_files)} datasets to train on")

    # Train model(s) for each dataset
    results_summary = {}

    for train_file in train_files:
        dataset_name = train_file.stem.replace("_train_scaled", "")
        test_file = train_file.parent / f"{dataset_name}_test_scaled.csv"

        if not test_file.exists():
            logging.warning(f"Test file not found for {dataset_name}, skipping")
            continue

        logging.info(f"\n{'='*60}")
        logging.info(f"Dataset: {dataset_name}")
        logging.info(f"{'='*60}")

        # Load data
        train_df = pd.read_csv(train_file)
        test_df = pd.read_csv(test_file)

        logging.info(f"Loaded data:")
        logging.info(f"  Train: {len(train_df)} samples")
        logging.info(f"  Test: {len(test_df)} samples")

        # Separate features, target, and sensitive attributes
        # Note: scaled files have both encoded and categorical versions
        # We keep the encoded numerical versions for modeling
        target_col = training_cfg.get("target", "heart_disease")
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

        fs_mode = args.feature_selection_mode
        X_train, feature_cols = build_feature_set(
            X_train_full,
            sensitive_attrs=sensitive_cols,
            mode=fs_mode,
            top_k=args.rfe_top_k,
            trained_model=None,  # static modes only at this stage; rfe_top_k needs a fitted model
        )
        X_test = X_test_full[feature_cols]

        y_train = train_df[target_col]
        y_test = test_df[target_col]

        # Keep categorical versions for fairness analysis
        sensitive_train = train_df[sensitive_cols]
        sensitive_test = test_df[sensitive_cols]

        logging.info(f"Features: {len(feature_cols)} (mode={fs_mode})")

        results_summary.setdefault(dataset_name, {})

        for model_type in model_types:
            logging.info(f"\n--- Training model: {model_type} ---")

            try:
                model_class = get_model_class(model_type)
                hpo_dir = (
                    project_root / f"output/{pipeline}/hpo" if args.use_hpo else None
                )
                model_params = _build_model_params(
                    model_type, training_cfg, random_state, project_root,
                    hpo_dir=hpo_dir, dataset_name=dataset_name,
                )
                model = model_class(**model_params)
            except Exception as exc:
                logging.warning(f"Skipping model_type={model_type} for {dataset_name}: {exc}")
                results_summary[dataset_name][model_type] = {
                    "status": "skipped",
                    "reason": str(exc),
                }
                continue

            train_metrics = model.train(X_train, y_train)

            logging.info("\n--- Evaluation ---")
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

            logging.info("\n--- Threshold Analysis ---")
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

            logging.info("\n--- Feature Importance (Top 10) ---")
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

            logging.info("\n--- Generating Predictions ---")
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
                f"    Near threshold (±0.1): {n_near_threshold_train} "
                f"({n_near_threshold_train/len(train_predictions):.1%})"
            )
            logging.info(f"  Test predictions: {len(test_predictions)}")
            logging.info(
                f"    Near threshold (±0.1): {n_near_threshold_test} "
                f"({n_near_threshold_test/len(test_predictions):.1%})"
            )

            model_file = models_dir / f"{dataset_name}_{model_type}.pkl"
            model.save(str(model_file))

            train_pred_file = experiments_dir / f"{dataset_name}_{model_type}_train_predictions.csv"
            test_pred_file = experiments_dir / f"{dataset_name}_{model_type}_test_predictions.csv"

            train_predictions.to_csv(train_pred_file, index=False)
            test_predictions.to_csv(test_pred_file, index=False)

            logging.info("\n[SUCCESS] Predictions saved:")
            logging.info(f"  Train: {train_pred_file}")
            logging.info(f"  Test: {test_pred_file}")

            importance_file = (
                experiments_dir / f"{dataset_name}_{model_type}_feature_importance.csv"
            )
            feature_importance.to_csv(importance_file, index=False)
            logging.info(f"  Feature importance: {importance_file}")

            xai_cfg = pipeline_cfg.get("xai", {})
            xai_dir = experiments_dir / "xai"
            xai_dataset_key = f"{dataset_name}__{model_type}"
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
                logging.info("\n--- Cross-Validated XAI ---")
                try:
                    cv_lime_n = int(xai_cfg.get("cv_lime_instances", 3))
                    cv_shap_max = int(xai_cfg.get("global_max_samples", 1000))
                    allow_svm_shap = bool(xai_cfg.get("allow_svm_shap", False))

                    X_full = pd.concat([X_train, X_test], ignore_index=True)
                    y_full = pd.concat([y_train, y_test], ignore_index=True)
                    sensitive_full = pd.concat([sensitive_train, sensitive_test], ignore_index=True)

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
                        # Standalone baseline run: no outer parallelism, so
                        # parallelize folds with XAI disabled at fold level
                        # (XAI is run separately after CV here).
                        cv_n_jobs=1,
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

            results_summary[dataset_name][model_type] = {
                "status": "success",
                "model_params": model_params,
                "train_metrics": train_metrics,
                "test_metrics": test_metrics_default,
                "threshold_analysis": threshold_results,
                "n_features": len(feature_cols),
                "n_train": len(train_df),
                "n_test": len(test_df),
                "near_threshold_pct_test": float(n_near_threshold_test / len(test_predictions)),
            }

    # Save overall results
    results_file = experiments_dir / "training_results.json"
    with open(results_file, "w") as f:
        json.dump(results_summary, f, indent=2, default=str)

    logging.info(f"\n{'='*60}")
    logging.info("[PHASE] Baseline training complete")
    logging.info(f"{'='*60}")
    logging.info(f"Models saved to: {models_dir}")
    logging.info(f"Results saved to: {experiments_dir}")
    logging.info(f"\nNext step: Run fairness assessment on predictions (Phase 4)")


if __name__ == "__main__":
    main()
