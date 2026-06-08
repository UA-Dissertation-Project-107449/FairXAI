#!/usr/bin/env python3
"""
Run fairness mitigation comparison experiment.

This script:
1. Loads preprocessed train/test datasets
2. Trains baseline models (no mitigation)
3. Applies pre-processing mitigation techniques (SMOTE, ROS, RUS, ADASYN, reweighting)
4. Applies in-processing techniques (ExponentiatedGradient, GridSearch)
5. Applies post-processing techniques (ThresholdOptimizer)
6. Computes fairness metrics for each technique
7. Saves comprehensive comparison results

Usage:
    python scripts/experiments/run_mitigation_comparison.py
    python scripts/experiments/run_mitigation_comparison.py --datasets cleveland
    python scripts/experiments/run_mitigation_comparison.py --config configs/experiments/mitigation.yaml
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from fairxai.cli.runner_base import get_project_root, load_pipeline_config, setup_phase_logging
from fairxai.cli.runner_utils import (
    append_run_history,
    archive_latest_run,
    get_run_root,
    resolve_run_id,
    update_latest_pointer,
)
from fairxai.experiments.data_io import (
    default_exclude_columns,
    resolve_dataset_dir,
    resolve_default_binning,
)
from fairxai.fairness.metrics import FairnessMetrics
from fairxai.fairness.mitigation import MitigationEngine
from fairxai.models.baseline import BaselineLogisticRegression, generate_predictions_with_metadata
from fairxai.utils.config import load_yaml_config


def load_dataset(
    dataset_name: str, data_dir: Path, schema_cfg: dict, target_col: str, sensitive_attrs: list
):
    """
    Load train/test splits for a dataset.

    Args:
        dataset_name: Name of dataset ('cleveland' or 'kaggle_heart')
        data_dir: Path to processed data directory

    Returns:
        Tuple of (X_train, y_train, sensitive_train, X_test, y_test, sensitive_test)
    """

    def _drop_duplicate_alias_columns(df: pd.DataFrame, label: str) -> pd.DataFrame:
        """Drop duplicate Pandas-suffixed aliases like ``col.1`` when ``col`` exists."""
        duplicate_aliases = []
        for col in df.columns:
            if "." not in col:
                continue
            base, suffix = col.rsplit(".", 1)
            if suffix.isdigit() and base in df.columns:
                duplicate_aliases.append(col)

        if duplicate_aliases:
            logging.info("  %s: dropping duplicate alias columns %s", label, duplicate_aliases)
            df = df.drop(columns=duplicate_aliases)
        return df

    logging.info(f"[DATASET] Loading dataset: {dataset_name}")

    train_scaled = data_dir / f"{dataset_name}_train_scaled.csv"
    test_scaled = data_dir / f"{dataset_name}_test_scaled.csv"
    train_raw = data_dir / f"{dataset_name}_train.csv"
    test_raw = data_dir / f"{dataset_name}_test.csv"

    if train_scaled.exists() and test_scaled.exists():
        train_df = pd.read_csv(train_scaled)
        test_df = pd.read_csv(test_scaled)
        if train_raw.exists() and test_raw.exists():
            train_raw_df = pd.read_csv(train_raw)
            test_raw_df = pd.read_csv(test_raw)
        else:
            train_raw_df = train_df
            test_raw_df = test_df
    else:
        if not train_raw.exists() or not test_raw.exists():
            raise FileNotFoundError(f"Dataset files not found: {train_raw}, {test_raw}")
        train_df = pd.read_csv(train_raw)
        test_df = pd.read_csv(test_raw)
        train_raw_df = train_df
        test_raw_df = test_df

    train_df = _drop_duplicate_alias_columns(train_df, "train")
    test_df = _drop_duplicate_alias_columns(test_df, "test")
    train_raw_df = _drop_duplicate_alias_columns(train_raw_df, "train_raw")
    test_raw_df = _drop_duplicate_alias_columns(test_raw_df, "test_raw")

    logging.info(f"  Train: {len(train_df)} samples")
    logging.info(f"  Test: {len(test_df)} samples")

    # Encode sex if it's categorical
    if "sex" in train_raw_df.columns and train_raw_df["sex"].dtype == "object":
        logging.info("  Encoding categorical 'sex' variable...")
        sex_map = {"Male": 1, "Female": 0, "M": 1, "F": 0}
        train_raw_df["sex"] = train_raw_df["sex"].map(sex_map)
        test_raw_df["sex"] = test_raw_df["sex"].map(sex_map)

    # Separate features, target, and sensitive attributes
    # Exclude target, sensitive attrs, metadata, and original categorical columns
    sensitive_cols = [col for col in sensitive_attrs if col in train_raw_df.columns]
    exclude = default_exclude_columns(
        schema_cfg, dataset_name, target=target_col, sensitive_attrs=sensitive_cols
    )
    # Only exclude columns that actually exist
    exclude = [col for col in exclude if col in train_df.columns]

    X_train = train_df.drop(columns=exclude)
    y_train = train_raw_df[target_col]
    sensitive_train = (
        train_raw_df[sensitive_cols].copy()
        if sensitive_cols
        else pd.DataFrame(index=train_raw_df.index)
    )

    X_test = test_df.drop(columns=exclude)
    y_test = test_raw_df[target_col]
    sensitive_test = (
        test_raw_df[sensitive_cols].copy()
        if sensitive_cols
        else pd.DataFrame(index=test_raw_df.index)
    )

    # Coerce continuous targets to binary if needed
    if pd.api.types.is_numeric_dtype(y_train):
        unique_vals = pd.Series(y_train).dropna().unique()
        if len(unique_vals) > 2 and pd.api.types.is_float_dtype(y_train):
            logging.warning("Target appears continuous; coercing to binary with threshold 0.5")
            y_train = (y_train >= 0.5).astype(int)
            y_test = (y_test >= 0.5).astype(int)

    # Ensure only numeric features remain
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns
    dropped = [c for c in X_train.columns if c not in numeric_cols]
    if dropped:
        logging.warning(f"Dropping non-numeric features: {dropped}")
    X_train = X_train[numeric_cols]
    X_test = X_test.reindex(columns=numeric_cols)

    logging.info(f"  Features: {X_train.shape[1]}")
    logging.info(f"  Sensitive attributes: {list(sensitive_train.columns)}")

    return X_train, y_train, sensitive_train, X_test, y_test, sensitive_test


def train_baseline(
    X_train, y_train, X_test, y_test, sensitive_test, dataset_name, model_params=None
):
    """Train baseline model without mitigation."""
    logging.info(f"[BASELINE] Training baseline model for dataset={dataset_name}")

    model = BaselineLogisticRegression(**(model_params or {}))
    model.train(X_train, y_train)
    test_metrics = model.evaluate(X_test, y_test)

    logging.info("Baseline metrics:")
    logging.info(f"  Accuracy: {test_metrics['accuracy']:.3f}")
    logging.info(f"  Precision: {test_metrics['precision']:.3f}")
    logging.info(f"  Recall: {test_metrics['recall']:.3f}")
    logging.info(f"  F1: {test_metrics['f1_score']:.3f}")
    logging.info(f"  AUC-ROC: {test_metrics['auc_roc']:.3f}")

    # Generate predictions with metadata for fairness assessment
    predictions = generate_predictions_with_metadata(model, X_test, y_test, sensitive_test)

    # Calculate fairness metrics
    # Note: feature_cols for individual fairness should be numeric features only
    # X_test columns are the actual model features (all numeric after exclusions)
    fairness_calc = FairnessMetrics(list(sensitive_test.columns))
    fairness_results = fairness_calc.calculate_all_metrics(
        predictions,
        feature_cols=list(X_test.columns),  # Use actual model features, not sensitive attrs
    )

    return {
        "model": model,
        "test_metrics": test_metrics,
        "fairness": fairness_results,
        "predictions": predictions,
    }


def apply_mitigation_techniques(
    X_train,
    y_train,
    X_test,
    y_test,
    sensitive_train,
    sensitive_test,
    dataset_name,
    baseline_model,
    techniques_config,
    base_model_params=None,
    constraint_attrs="all",
):
    """
    Apply all mitigation techniques and collect results.

    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        sensitive_train, sensitive_test: Sensitive attributes
        dataset_name: Dataset identifier
        baseline_model: Trained baseline model for post-processing
        techniques_config: Dict of techniques to test
        constraint_attrs: Which sensitive attrs to constrain on per technique.
            ``"all"`` (default) runs each technique once per available sensitive
            attr (thorough); a list restricts to that subset; ``"none"`` keeps
            the legacy single-attr behavior (first available attr).

    Returns:
        List of result dictionaries (one per technique × constraint attr).
    """
    engine = MitigationEngine(random_state=42)
    results = []

    available_attrs = list(sensitive_test.columns)
    resolved_attrs = _resolve_constraint_attrs(constraint_attrs, available_attrs)
    if not resolved_attrs:
        logging.warning(
            "No sensitive attributes available for mitigation on %s — skipping techniques.",
            dataset_name,
        )
        return results
    logging.info(
        "[MITIGATION] dataset=%s constraint attrs=%s (mode=%s)",
        dataset_name,
        resolved_attrs,
        constraint_attrs,
    )

    for technique_name, config in techniques_config.items():
        stage = config["stage"]

        for sensitive_attr in resolved_attrs:
            logging.info(
                "[MITIGATION] technique=%s stage=%s constraint=%s dataset=%s",
                technique_name,
                stage,
                sensitive_attr,
                dataset_name,
            )

            try:
                # Apply technique
                if stage == "post-processing":
                    result = engine.apply_technique(
                        technique_name=technique_name,
                        stage=stage,
                        X_train=X_train,
                        y_train=y_train,
                        X_test=X_test,
                        y_test=y_test,
                        sensitive_train=sensitive_train,
                        sensitive_test=sensitive_test,
                        sensitive_attr=sensitive_attr,
                        base_model=baseline_model,
                    )
                else:
                    result = engine.apply_technique(
                        technique_name=technique_name,
                        stage=stage,
                        X_train=X_train,
                        y_train=y_train,
                        X_test=X_test,
                        y_test=y_test,
                        sensitive_train=sensitive_train,
                        sensitive_test=sensitive_test,
                        sensitive_attr=sensitive_attr,
                        base_model_params=base_model_params,
                    )

                # Generate predictions DataFrame for fairness assessment
                y_proba = result["predictions"]["y_proba"]
                if y_proba is None:
                    # Use predictions as fallback if no probabilities
                    y_proba = result["predictions"]["y_pred"]
                    logging.warning(
                        f"No probabilities available for {technique_name}, using predictions"
                    )

                predictions_df = pd.DataFrame(
                    {
                        "y_true": y_test.values,
                        "y_pred": result["predictions"]["y_pred"],
                        "y_proba": y_proba,
                    }
                )
                for col in sensitive_test.columns:
                    predictions_df[col] = sensitive_test[col].values

                # Add model features for individual fairness calculation
                for col in X_test.columns:
                    predictions_df[col] = X_test[col].values

                # Calculate fairness metrics across ALL sensitive attrs (measurement
                # is unchanged; only the imposed constraint varies per loop).
                fairness_calc = FairnessMetrics(list(sensitive_test.columns))
                fairness_results = fairness_calc.calculate_all_metrics(
                    predictions_df, feature_cols=list(X_test.columns)
                )

                # Compile result
                results.append(
                    {
                        "dataset": dataset_name,
                        "technique": technique_name,
                        "constraint_attr": sensitive_attr,
                        "stage": stage,
                        "test_metrics": result["test_metrics"],
                        "fairness": fairness_results,
                        "metadata": result["metadata"],
                    }
                )

                logging.info(
                    "[SUCCESS] Mitigation complete dataset=%s technique=%s constraint=%s stage=%s",
                    dataset_name,
                    technique_name,
                    sensitive_attr,
                    stage,
                )
                logging.info(f"  Accuracy: {result['test_metrics']['accuracy']:.3f}")
                logging.info(f"  Recall: {result['test_metrics']['recall']:.3f}")

            except Exception as e:
                logging.error(
                    f"Failed to apply {technique_name} (constraint={sensitive_attr}): {e}"
                )
                logging.exception(e)

    return results


def _resolve_constraint_attrs(constraint_attrs, available_attrs):
    """Resolve the ``constraint_attrs`` config into a list of attr names.

    ``"all"`` / falsy-but-not-list → every available attr; a list → that subset
    intersected with availability (order preserved); ``"none"`` → first attr
    only (legacy). Returns ``[]`` when nothing is available.
    """
    if not available_attrs:
        return []
    if isinstance(constraint_attrs, (list, tuple)):
        chosen = [a for a in constraint_attrs if a in available_attrs]
        return chosen or available_attrs[:1]
    mode = str(constraint_attrs or "all").lower()
    if mode == "none":
        return available_attrs[:1]
    # "all" (default) or any unrecognized scalar → thorough.
    return list(available_attrs)


def create_comparison_table(all_results):
    """Create comparison DataFrame from results."""
    if not all_results:
        return pd.DataFrame()
    comparison_data = []

    for result in all_results:
        metrics = result["test_metrics"]

        # Extract key fairness metrics
        # FairnessMetrics.calculate_all_metrics() returns:
        #   {"group_fairness": {attr: {"demographic_parity": {...}, "equalized_odds": {...}}},
        #    "individual_fairness": {...}}
        fairness = result.get("fairness", {})
        group_fairness = fairness.get("group_fairness", {})
        # Flatten per-attribute metrics into combined dicts keyed by attr
        demographic_parity = {
            attr: metrics.get("demographic_parity", {})
            for attr, metrics in group_fairness.items()
            if "demographic_parity" in metrics
        }
        equalized_odds = {
            attr: metrics.get("equalized_odds", {})
            for attr, metrics in group_fairness.items()
            if "equalized_odds" in metrics
        }

        row = {
            "dataset": result["dataset"],
            "technique": result["technique"],
            "constraint_attr": result.get("constraint_attr", ""),
            "stage": result["stage"],
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1_score"],
            "auc_roc": metrics["auc_roc"],
        }

        # Add fairness metrics if available
        dp_diffs = []
        if demographic_parity:
            for attr, dp_metrics in demographic_parity.items():
                if isinstance(dp_metrics, dict) and "max_difference" in dp_metrics:
                    diff = dp_metrics["max_difference"]
                    row[f"dp_{attr}_max_diff"] = diff
                    if pd.notna(diff):
                        dp_diffs.append(diff)

        eo_diffs = []
        if equalized_odds:
            for attr, eo_metrics in equalized_odds.items():
                if isinstance(eo_metrics, dict):
                    tpr_diff = eo_metrics.get(
                        "tpr_max_difference", eo_metrics.get("tpr_difference", np.nan)
                    )
                    fpr_diff = eo_metrics.get(
                        "fpr_max_difference", eo_metrics.get("fpr_difference", np.nan)
                    )
                    row[f"eq_odds_{attr}_tpr_diff"] = tpr_diff
                    row[f"eq_odds_{attr}_fpr_diff"] = fpr_diff
                    if pd.notna(tpr_diff):
                        eo_diffs.append(tpr_diff)
                    if pd.notna(fpr_diff):
                        eo_diffs.append(fpr_diff)

        if dp_diffs:
            row["dp_max_diff"] = float(np.nanmax(dp_diffs))
        if eo_diffs:
            row["eq_odds_max_diff"] = float(np.nanmax(eo_diffs))

        comparison_data.append(row)

    df = pd.DataFrame(comparison_data)

    # Compute fairness gain vs baseline per dataset
    for col in ["dp_max_diff", "eq_odds_max_diff"]:
        if col not in df.columns:
            df[col] = np.nan
    df["fairness_gap"] = df[["dp_max_diff", "eq_odds_max_diff"]].max(axis=1, skipna=True)
    baseline_rows = df[df["technique"] == "baseline"].set_index("dataset")
    if baseline_rows.empty:
        df["baseline_fairness_gap"] = np.nan
    else:
        df["baseline_fairness_gap"] = df["dataset"].map(baseline_rows["fairness_gap"])
    df["fairness_gain"] = df["baseline_fairness_gap"] - df["fairness_gap"]
    df["fairness_gain_pct"] = df["fairness_gain"] / df["baseline_fairness_gap"]

    return df


def _write_mitigation_report(df: "pd.DataFrame", report_file: "Path") -> None:
    """Write a human-readable markdown summary of mitigation results."""
    from pathlib import Path

    lines = ["# Mitigation Comparison Report\n"]
    for dataset in sorted(df["dataset"].unique() if "dataset" in df.columns else []):
        lines.append(f"## {dataset}\n")
        sub = df[df["dataset"] == dataset] if "dataset" in df.columns else df
        # Best by accuracy
        if "accuracy" in sub.columns and not sub["accuracy"].isna().all():
            best_acc = sub.loc[sub["accuracy"].idxmax()]
            lines.append(
                f"**Best accuracy:** `{best_acc.get('technique', '?')}` — "
                f"{best_acc['accuracy']:.3f}\n"
            )
        # Best fairness (lowest fairness gap if column exists)
        gap_col = next(
            (c for c in ("fairness_gap", "demographic_parity_gap") if c in sub.columns), None
        )
        if gap_col and not sub[gap_col].isna().all():
            best_idx = sub[gap_col].abs().idxmin()
            if pd.notna(best_idx):
                best_fair = sub.loc[best_idx]
                lines.append(
                    f"**Best fairness:** `{best_fair.get('technique', '?')}` — "
                    f"{gap_col}={best_fair[gap_col]:.3f}\n"
                )
        # Summary table: top 5 by accuracy
        if "accuracy" in sub.columns:
            top5 = sub.nlargest(5, "accuracy")
            show_cols = [
                c for c in ("technique", "accuracy", "recall", gap_col) if c and c in sub.columns
            ]
            lines.append(top5[show_cols].round(3).to_markdown(index=False))
            lines.append("")
    Path(report_file).write_text("\n".join(lines))


def run_analysis(
    config_path: str,
    datasets: list = None,
    output_dir: str = None,
    pipeline: str = "cardiac",
    run_mode: str = "partial",
    archive_previous: bool = True,
    run_id: str = None,
    output_root: str = None,
    verbose: int = 0,
):
    """
    Runs the mitigation comparison experiment.
    """
    # Paths
    project_root = get_project_root(Path(__file__))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    experiment_cfg = load_yaml_config(str(config_path))
    pipeline_cfg = load_pipeline_config(project_root, pipeline)
    schema_path = project_root / pipeline_cfg["runtime"]["schema_mapping_json"]
    with open(schema_path, "r") as f:
        schema_cfg = json.load(f)

    _lr_cfg = load_yaml_config(
        str(project_root / "configs" / "models" / "logistic_regression.yaml")
    )
    model_params = dict(_lr_cfg.get("hyperparameters", {}))

    target_col = experiment_cfg.get("data", {}).get("target", "heart_disease")

    sensitive_attrs = experiment_cfg.get("data", {}).get("sensitive_attributes", ["sex"])

    # Which sensitive attrs to impose a fairness constraint on, per technique:
    #   "all"  -> every available sensitive attr (thorough; default)
    #   [list] -> an explicit subset (1-by-1 or combinations)
    #   "none" -> legacy behavior: constrain on the first available attr only
    constraint_attrs_cfg = experiment_cfg.get("constraint_attrs", "all")
    logging.info("Mitigation constraint_attrs mode: %s", constraint_attrs_cfg)

    use_run_id = bool(run_id or os.getenv("RUN_ID") or os.getenv("PREFECT__RUNTIME__FLOW_RUN_ID"))
    run_id = resolve_run_id(run_id) if use_run_id else None

    default_output_dir = experiment_cfg.get("output", {}).get("results_dir")
    if run_id:
        base_output = Path(output_root) if output_root else (project_root / f"output/{pipeline}")
    elif output_root:
        base_output = Path(output_root)
    elif default_output_dir:
        base_output = Path(default_output_dir)
        if base_output.parts and base_output.name == "mitigation":
            base_output = base_output.parents[1]
        if run_mode == "partial" and "full" in base_output.parts:
            parts = list(base_output.parts)
            idx = len(parts) - 1 - parts[::-1].index("full")
            parts[idx] = "partial"
            base_output = Path(*parts)
    else:
        base_output = project_root / f"output/{pipeline}/experiments/{run_mode}"
    # Setup logging
    setup_phase_logging(
        project_root,
        "mitigation_comparison.log",
        verbose=verbose,
        run_id=run_id,
        stage_name="mitigation",
    )
    logger = logging.getLogger(__name__)
    logging.info("[PHASE] Mitigation comparison started")

    if run_id:
        run_dir = get_run_root(base_output, run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        output_dir = Path(output_dir) if output_dir else run_dir / "experiments" / "mitigation"
    else:
        latest_dir = base_output / "latest_run"
        if run_mode == "partial":
            archive_latest_run(base_output, enabled=True, logger=logger)
        else:
            archive_latest_run(base_output, enabled=archive_previous, logger=logger)
        output_dir = Path(output_dir) if output_dir else latest_dir / "mitigation"

    output_dir.mkdir(parents=True, exist_ok=True)
    datasets = datasets if datasets else experiment_cfg["data"]["datasets"]

    logging.info("Configuration:")
    logging.info(f"  Datasets: {datasets}")
    logging.info(f"  Output: {output_dir}")
    logging.info(f"  Run mode: {run_mode}")
    logging.info(f"  Timestamp: {timestamp}")
    if run_id:
        logging.info(f"  Run ID: {run_id}")
        append_run_history(
            base_output,
            {
                "run_id": run_id,
                "pipeline": pipeline,
                "mode": run_mode,
                "phase": "mitigation",
                "datasets": datasets,
                "output_dir": str(output_dir),
                "status": "started",
            },
        )

    # Data directory. Per-dataset subdir resolved below via the shared resolver.
    data_dir = project_root / pipeline_cfg["paths"]["processed_dir"]
    default_binning = resolve_default_binning(pipeline_cfg)
    logging.info(
        "Run context: pipeline=%s run_id=%s processed_dir=%s default_binning=%s",
        pipeline,
        run_id or "none",
        data_dir,
        default_binning,
    )

    # Techniques to test (from config)
    techniques = experiment_cfg["mitigation_strategies"]

    # Filter to implemented techniques
    implemented = {
        "smote": techniques["smote"],
        "ros": techniques["ros"],
        "rus": techniques["rus"],
        "adasyn": techniques["adasyn"],
        "reweighting": techniques["reweighting"],
        "exponentiated_gradient": techniques["exponentiated_gradient"],
        "grid_search": techniques["grid_search"],
        "threshold_optimizer": techniques["threshold_optimizer"],
    }

    logging.info(f"Techniques to test: {list(implemented.keys())}")

    # Process each dataset
    all_results = []
    baseline_results = []

    for dataset_name in datasets:
        logging.info("[DATASET] Processing dataset=%s", dataset_name)

        try:
            dataset_dir = resolve_dataset_dir(data_dir, dataset_name, default_binning)

            # Load data — processed files carry the canonical target column
            X_train, y_train, sensitive_train, X_test, y_test, sensitive_test = load_dataset(
                dataset_name, dataset_dir, schema_cfg, target_col, sensitive_attrs
            )

            # Train baseline
            baseline = train_baseline(
                X_train, y_train, X_test, y_test, sensitive_test, dataset_name, model_params
            )

            baseline_results.append(
                {
                    "dataset": dataset_name,
                    "technique": "baseline",
                    "stage": "none",
                    "test_metrics": baseline["test_metrics"],
                    "fairness": baseline["fairness"],
                    "metadata": {},
                }
            )

            # Apply mitigation techniques
            mitigation_results = apply_mitigation_techniques(
                X_train,
                y_train,
                X_test,
                y_test,
                sensitive_train,
                sensitive_test,
                dataset_name,
                baseline["model"],
                implemented,
                base_model_params=model_params,
                constraint_attrs=constraint_attrs_cfg,
            )

            all_results.extend(mitigation_results)

        except Exception as e:
            logging.error(f"Failed to process {dataset_name}: {e}")
            logging.exception(e)

    # Combine baseline and mitigation results
    all_results = baseline_results + all_results
    if not all_results:
        logging.error("No results produced; aborting report generation")
        return

    # Summary statistics
    logging.info("Processing summary:")
    logging.info(f"Datasets processed: {len(set(r['dataset'] for r in all_results))}")
    logging.info(f"Techniques tested: {len(set(r['technique'] for r in all_results))}")
    logging.info(f"Total results: {len(all_results)}")

    # Create comparison table
    logging.info("[PHASE] Generating comparison report")

    comparison_df = create_comparison_table(all_results)
    if comparison_df.empty:
        logging.error("No comparison data available")
        return

    # Save results (no timestamp — run is already scoped by run_id)
    csv_file = output_dir / "summary.csv"
    json_file = output_dir / "results.json"

    comparison_df.to_csv(csv_file, index=False)
    logging.info(f"[SUCCESS] Saved CSV: {csv_file}")

    with open(json_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logging.info(f"[SUCCESS] Saved JSON: {json_file}")

    # Generate human-readable markdown report
    report_file = output_dir / "report.md"
    _write_mitigation_report(comparison_df, report_file)
    logging.info(f"[SUCCESS] Saved report: {report_file}")

    # Print summary
    logging.info("Results summary: rows=%d", len(comparison_df))
    logging.debug("Comparison table:\n%s", comparison_df.round(3).to_string(index=False))

    # Clinical constraint check (recall >= 0.70)
    meets_clinical = comparison_df[comparison_df["recall"] >= 0.70]
    logging.info(f"Techniques meeting clinical constraint (recall >= 0.70): {len(meets_clinical)}")
    if len(meets_clinical) > 0:
        logging.debug(
            "Clinical-constraint rows:\n%s",
            meets_clinical[["dataset", "technique", "recall", "accuracy"]].to_string(index=False),
        )

    logging.info("[PHASE] Mitigation comparison complete")
    logging.info("Output directory: %s", output_dir)

    if run_id:
        update_latest_pointer(base_output, run_dir, logger)
        append_run_history(
            base_output,
            {
                "run_id": run_id,
                "pipeline": pipeline,
                "mode": run_mode,
                "phase": "mitigation",
                "datasets": datasets,
                "output_dir": str(output_dir),
                "status": "completed",
            },
        )


def main():
    parser = argparse.ArgumentParser(description="Run mitigation comparison experiment")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiments/mitigation.yaml",
        help="Path to experiment config file",
    )
    parser.add_argument(
        "--datasets", type=str, nargs="+", help="Datasets to process (default: from config)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory (default: from config or output/{pipeline}/experiments/{run_mode}/latest_run/mitigation)",
    )
    parser.add_argument(
        "--pipeline", type=str, default="cardiac", help="Pipeline name (e.g., cardiac, dermatology)"
    )
    parser.add_argument(
        "--run-mode",
        type=str,
        choices=["full", "partial"],
        default=os.getenv("EXPERIMENT_RUN_MODE", "partial"),
        help="Run mode (full or partial)",
    )
    parser.add_argument(
        "--archive-previous",
        action="store_true",
        default=os.getenv("ARCHIVE_PREVIOUS", "true").lower() == "true",
        help="Archive previous latest_run (full runs only)",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=os.getenv("RUN_ID"),
        help="Run identifier (optional, enables run-scoped outputs)",
    )
    parser.add_argument(
        "--output-root", type=str, default=None, help="Base output directory for run outputs"
    )
    parser.add_argument(
        "-v", "--verbose", action="count", default=0, help="Verbosity: -v=info, -vv=debug"
    )
    args = parser.parse_args()

    run_analysis(
        config_path=args.config,
        datasets=args.datasets,
        output_dir=args.output_dir,
        pipeline=args.pipeline,
        run_mode=args.run_mode,
        archive_previous=args.archive_previous,
        run_id=args.run_id,
        output_root=args.output_root,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
