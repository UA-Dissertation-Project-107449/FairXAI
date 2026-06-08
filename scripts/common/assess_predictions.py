"""
Post-prediction fairness assessment.

Evaluates fairness of model predictions across sensitive attributes.
Calculates group fairness metrics, calibration, and individual fairness.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

# Add src to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from fairxai.cli.runner_base import load_pipeline_config, setup_phase_logging
from fairxai.fairness.metrics import FairnessMetrics, summarize_fairness_results

_KNOWN_MODELS = ["logistic_regression", "random_forest", "svm", "xgboost"]

# Map a config sensitive-attribute name to the decoded column produced by
# decode_sensitive_attributes(). age_group / sex are decoded from scaled values;
# ethnicity / group_cluster are categorical and pass through as *_cat strings.
_SENSITIVE_DECODE_MAP = {
    "age_group": "age_group_cat",
    "sex": "sex_cat",
    "ethnicity": "ethnicity_cat",
    "group_cluster": "group_cluster_cat",
}
_DEFAULT_SENSITIVE = ["age_group", "sex"]


def resolve_sensitive_columns(df: pd.DataFrame, configured: list) -> list:
    """Resolve configured sensitive attrs to decoded columns present in *df*.

    Falls back to the raw name when no decoded variant exists. Configured names
    whose source column is absent are silently dropped.
    """
    resolved = []
    for name in configured:
        decoded = _SENSITIVE_DECODE_MAP.get(name, name)
        if decoded in df.columns:
            resolved.append(decoded)
        elif name in df.columns:
            resolved.append(name)
    return resolved


def _split_dataset_model(combined: str, known_models: list = _KNOWN_MODELS) -> tuple:
    """Split 'cleveland_logistic_regression' → ('cleveland', 'logistic_regression')."""
    for model in sorted(known_models, key=len, reverse=True):
        if combined.endswith(f"_{model}"):
            return combined[: -(len(model) + 1)], model
    return combined, "unknown"


def _render_fairness_md(nested_results: dict) -> str:
    """Render nested {dataset: {model: {method: results}}} fairness dict as markdown."""

    def _iter_methods(model_results: dict) -> list[tuple[str, dict]]:
        if not isinstance(model_results, dict):
            return []
        if any(key in model_results for key in ("train_metrics", "test_metrics", "cv_metrics")):
            # Backward compatibility with legacy shape {model: direct_results}
            return [("single_split", model_results)]
        rows: list[tuple[str, dict]] = []
        for method in ("single_split", "kfold_cv"):
            value = model_results.get(method)
            if isinstance(value, dict):
                rows.append((method, value))
        return rows

    lines = ["# Prediction Fairness Report\n"]
    for dataset, models in sorted(nested_results.items()):
        lines.append(f"## {dataset.replace('_', ' ').title()}\n")
        for model, res in sorted(models.items()):
            lines.append(f"### {model.replace('_', ' ').title()}\n")
            method_rows = _iter_methods(res)
            if not method_rows:
                continue

            for method_name, method_results in method_rows:
                method_label = "Single Split" if method_name == "single_split" else "K-Fold CV"
                lines.append(f"#### {method_label}\n")

                if "cv_metrics" in method_results:
                    cv_metrics = method_results.get("cv_metrics", {})
                    gf = cv_metrics.get("group_fairness", {})
                    if gf:
                        lines.append("**CV Fairness - Group Fairness**\n")
                        lines.append("| Attribute | Metric | Max Difference | Fair? |")
                        lines.append("|-----------|--------|---------------|-------|")
                        for attr, attr_metrics in sorted(gf.items()):
                            for metric_name, md in sorted(attr_metrics.items()):
                                fair = md.get("is_fair", False)
                                diff = md.get("max_difference") or md.get("tpr_max_difference", "?")
                                flag = "✓" if fair else "✗"
                                if isinstance(diff, float):
                                    diff = f"{diff:.3f}"
                                lines.append(f"| {attr} | {metric_name} | {diff} | {flag} |")
                        lines.append("")

                    calib = cv_metrics.get("calibration", {})
                    for attr, cd in sorted(calib.items()):
                        fair = cd.get("is_fair", False)
                        flag = "✓" if fair else "✗"
                        ece = cd.get("max_ece_difference", "?")
                        if isinstance(ece, float):
                            ece = f"{ece:.4f}"
                        lines.append(f"**CV Calibration ({attr}):** max ECE diff = {ece} {flag}\n")
                    continue

                for split in ("test_metrics", "train_metrics"):
                    label = "Test" if split == "test_metrics" else "Train"
                    metrics = method_results.get(split, {})
                    gf = metrics.get("group_fairness", {})
                    if not gf:
                        continue
                    lines.append(f"**{label} Set - Group Fairness**\n")
                    lines.append("| Attribute | Metric | Max Difference | Fair? |")
                    lines.append("|-----------|--------|---------------|-------|")
                    for attr, attr_metrics in sorted(gf.items()):
                        for metric_name, md in sorted(attr_metrics.items()):
                            fair = md.get("is_fair", False)
                            diff = md.get("max_difference") or md.get("tpr_max_difference", "?")
                            flag = "✓" if fair else "✗"
                            if isinstance(diff, float):
                                diff = f"{diff:.3f}"
                            lines.append(f"| {attr} | {metric_name} | {diff} | {flag} |")
                    lines.append("")

                    calib = metrics.get("calibration", {})
                    for attr, cd in sorted(calib.items()):
                        fair = cd.get("is_fair", False)
                        flag = "✓" if fair else "✗"
                        ece = cd.get("max_ece_difference", "?")
                        if isinstance(ece, float):
                            ece = f"{ece:.4f}"
                        lines.append(
                            f"**{label} Calibration ({attr}):** max ECE diff = {ece} {flag}\n"
                        )
    return "\n".join(lines)


def decode_sensitive_attributes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Decode sensitive attributes from scaled numerical values back to categories.

    Args:
        df: DataFrame with scaled sensitive attributes

    Returns:
        DataFrame with decoded categories
    """
    df = df.copy()

    # For scaled values, we need to find the nearest encoding
    if "age_group" in df.columns:
        # Round to nearest 0.5
        scaled_ages = df["age_group"].values
        unique_scaled = np.unique(scaled_ages)

        # Try to infer categories from unique values
        sorted_unique = sorted(unique_scaled)
        if len(sorted_unique) <= 5:
            # Map sorted values to categories
            categories = ["<40", "40-49", "50-59", "60-69", "70+"]
            age_decode = {val: categories[i] for i, val in enumerate(sorted_unique)}
            df["age_group_cat"] = df["age_group"].map(age_decode)
        else:
            # Use numerical binning if too many unique values
            df["age_group_cat"] = pd.cut(
                df["age_group"], bins=5, labels=["<40", "40-49", "50-59", "60-69", "70+"]
            )

    # Decode sex (0=Female, 1=Male typically, or could be scaled)
    if "sex" in df.columns:
        unique_sex = df["sex"].unique()
        if len(unique_sex) == 2:
            # Binary encoding
            sorted_sex = sorted(unique_sex)
            sex_decode = {sorted_sex[0]: "Female", sorted_sex[1]: "Male"}
            df["sex_cat"] = df["sex"].map(sex_decode)
        else:
            # Fallback
            df["sex_cat"] = df["sex"].apply(lambda x: "Male" if x > 0 else "Female")

    # Categorical sensitive attrs — no scaled decode, just a string passthrough.
    if "group_cluster" in df.columns:
        df["group_cluster_cat"] = df["group_cluster"].astype("Int64").astype(str)
    if "ethnicity" in df.columns:
        df["ethnicity_cat"] = df["ethnicity"].astype(str)

    return df


def assess_dataset_fairness(
    dataset_name: str,
    train_file: Path,
    test_file: Path,
    output_dir: Path,
    metrics_calculator: FairnessMetrics,
    configured_sensitive: list = None,
) -> Dict:
    """
    Assess fairness for a single dataset's predictions.

    Args:
        dataset_name: Name of the dataset
        train_file: Path to training predictions CSV
        test_file: Path to test predictions CSV
        output_dir: Directory to save results
        metrics_calculator: FairnessMetrics instance

    Returns:
        Dictionary with assessment results
    """
    logging.info("[DATASET] Assessing fairness dataset_model=%s", dataset_name)

    # Load predictions
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    logging.info("Loaded predictions:")
    logging.info(f"  Train: {len(train_df)} samples")
    logging.info(f"  Test: {len(test_df)} samples")

    # Decode sensitive attributes
    train_df = decode_sensitive_attributes(train_df)
    test_df = decode_sensitive_attributes(test_df)

    # Resolve sensitive attributes from the pipeline config (honors
    # group_cluster / ethnicity when present), falling back to age_group + sex.
    configured = configured_sensitive or _DEFAULT_SENSITIVE
    resolved = resolve_sensitive_columns(train_df, configured)
    if resolved:
        metrics_calculator.sensitive_attributes = resolved
        logging.info("Sensitive attributes for fairness: %s", resolved)
        for col in resolved:
            if col in train_df.columns:
                logging.info("  %s: %s", col, train_df[col].value_counts().to_dict())
    else:
        logging.warning(
            "Could not resolve any configured sensitive attributes; using scaled values"
        )

    results = {"dataset": dataset_name, "train_metrics": {}, "test_metrics": {}, "comparison": {}}

    # Get numerical feature columns (exclude metadata)
    exclude_cols = [
        "y_true",
        "y_pred",
        "y_proba",
        "threshold",
        "age_group",
        "sex",
        "ethnicity",
        "group_cluster",
        "confidence",
        "near_threshold",
        "age_group_cat",
        "sex_cat",
        "ethnicity_cat",
        "group_cluster_cat",
    ]
    feature_cols = [col for col in train_df.columns if col not in exclude_cols]

    if not feature_cols:
        logging.warning("No feature columns found for individual fairness")
        feature_cols = None

    # Calculate metrics for train set
    logging.info("Train split fairness metrics:")
    train_metrics = metrics_calculator.calculate_all_metrics(train_df, feature_cols)
    results["train_metrics"] = train_metrics

    # Log train results
    log_fairness_metrics(train_metrics, "Train")

    # Calculate metrics for test set
    logging.info("Test split fairness metrics:")
    test_metrics = metrics_calculator.calculate_all_metrics(test_df, feature_cols)
    results["test_metrics"] = test_metrics

    # Log test results
    log_fairness_metrics(test_metrics, "Test")

    # Compare train vs test
    results["comparison"] = compare_fairness(train_metrics, test_metrics)

    # Create summary table (renamed: _fairness_summary.csv → _summary.csv)
    summary_train = summarize_fairness_results(train_metrics)
    summary_test = summarize_fairness_results(test_metrics)

    summary_train["split"] = "train"
    summary_test["split"] = "test"
    summary = pd.concat([summary_train, summary_test], ignore_index=True)

    summary_file = output_dir / f"{dataset_name}_summary.csv"
    summary.to_csv(summary_file, index=False)

    logging.info(f"[SUCCESS] Summary table saved to: {summary_file}")

    return results


def assess_cv_fairness(
    dataset_name: str,
    cv_file: Path,
    output_dir: Path,
    metrics_calculator: FairnessMetrics,
    configured_sensitive: list = None,
) -> Dict:
    """Assess fairness for CV out-of-fold predictions."""
    logging.info("[DATASET] Assessing CV fairness dataset_model=%s", dataset_name)

    cv_df = pd.read_csv(cv_file)
    logging.info("Loaded CV predictions: %d samples", len(cv_df))

    cv_df = decode_sensitive_attributes(cv_df)
    configured = configured_sensitive or _DEFAULT_SENSITIVE
    resolved = resolve_sensitive_columns(cv_df, configured)
    if resolved:
        metrics_calculator.sensitive_attributes = resolved
        logging.info("CV sensitive attributes for fairness: %s", resolved)

    exclude_cols = [
        "fold",
        "sample_idx",
        "y_true",
        "y_pred",
        "y_proba",
        "threshold",
        "age_group",
        "sex",
        "ethnicity",
        "group_cluster",
        "confidence",
        "near_threshold",
        "age_group_cat",
        "sex_cat",
        "ethnicity_cat",
        "group_cluster_cat",
    ]
    feature_cols = [col for col in cv_df.columns if col not in exclude_cols]
    if not feature_cols:
        feature_cols = None

    cv_metrics = metrics_calculator.calculate_all_metrics(cv_df, feature_cols)
    log_fairness_metrics(cv_metrics, "CV")

    summary = summarize_fairness_results(cv_metrics)
    summary["split"] = "cv"
    summary_file = output_dir / f"{dataset_name}_cv_summary.csv"
    summary.to_csv(summary_file, index=False)
    logging.info(f"[SUCCESS] CV summary table saved to: {summary_file}")

    n_folds = int(cv_df["fold"].nunique()) if "fold" in cv_df.columns else None
    return {
        "dataset": dataset_name,
        "cv_metrics": cv_metrics,
        "n_samples": len(cv_df),
        "n_folds": n_folds,
    }


def log_fairness_metrics(metrics: Dict, split_name: str):
    """Log fairness metrics in a readable format."""

    # Group fairness
    for attr, attr_metrics in metrics.get("group_fairness", {}).items():
        logging.info(f"  {split_name} {attr.upper()} group fairness:")

        for metric_name, metric_data in attr_metrics.items():
            fair = metric_data.get("is_fair", False)
            tag = "[SUCCESS]" if fair else "[FAIL]"
            log = logging.info if fair else logging.warning

            if metric_name == "demographic_parity":
                diff = metric_data["max_difference"]
                log(f"    {tag} Demographic Parity: {diff:.3f} max difference")

            elif metric_name == "equalized_odds":
                tpr_diff = metric_data["tpr_max_difference"]
                fpr_diff = metric_data["fpr_max_difference"]
                log(f"    {tag} Equalized Odds: TPR diff={tpr_diff:.3f}, FPR diff={fpr_diff:.3f}")

            elif metric_name == "equal_opportunity":
                diff = metric_data["max_difference"]
                log(f"    {tag} Equal Opportunity: {diff:.3f} TPR difference")

            elif metric_name == "predictive_parity":
                diff = metric_data["max_difference"]
                log(f"    {tag} Predictive Parity: {diff:.3f} precision difference")

    # Calibration
    for attr, calib_data in metrics.get("calibration", {}).items():
        logging.info(f"  {split_name} {attr.upper()} calibration:")
        fair = calib_data.get("is_fair", False)
        tag = "[SUCCESS]" if fair else "[FAIL]"
        log = logging.info if fair else logging.warning
        max_ece_diff = calib_data["max_ece_difference"]
        log(f"    {tag} Max ECE difference: {max_ece_diff:.4f}")

        for group, group_calib in calib_data["group_calibration"].items():
            logging.info(f"      {group}: ECE={group_calib['ece']:.4f}")

    # Individual fairness
    if "individual_fairness" in metrics and metrics["individual_fairness"]:
        ind_fair = metrics["individual_fairness"]
        logging.info(f"  {split_name} individual fairness (k-NN consistency):")
        logging.info(f"    Mean consistency: {ind_fair['mean_consistency']:.3f}")
        logging.info(f"    Median consistency: {ind_fair['median_consistency']:.3f}")


def compare_fairness(train_metrics: Dict, test_metrics: Dict) -> Dict:
    """
    Compare train vs test fairness metrics.

    Args:
        train_metrics: Metrics from training set
        test_metrics: Metrics from test set

    Returns:
        Dictionary with comparison results
    """
    comparison = {}

    # Compare group fairness
    for attr in train_metrics.get("group_fairness", {}).keys():
        train_attr = train_metrics["group_fairness"][attr]
        test_attr = test_metrics["group_fairness"][attr]

        comparison[attr] = {}
        for metric_name in train_attr.keys():
            train_metric = train_attr[metric_name]
            test_metric = test_attr[metric_name]

            # Compare key difference metrics
            if "max_difference" in train_metric:
                train_diff = train_metric["max_difference"]
                test_diff = test_metric["max_difference"]
                comparison[attr][metric_name] = {
                    "train_difference": train_diff,
                    "test_difference": test_diff,
                    "delta": test_diff - train_diff,
                }

    return comparison


def write_overfit_gap_table(training_results_path: Path, output_dir: Path) -> None:
    """Read training_results.json and write overfit_gap_table.csv.

    Emits one row per dataset × model with train/test gaps for F1, recall, and AUC.
    Also logs a compact summary so the gap is visible in pipeline logs.
    """
    if not training_results_path.exists():
        logging.warning(
            "training_results.json not found at %s — skipping overfit gap table.",
            training_results_path,
        )
        return

    with open(training_results_path) as f:
        training_results = json.load(f)

    rows = []
    for dataset, models in training_results.items():
        if not isinstance(models, dict):
            continue
        for model, model_data in models.items():
            if not isinstance(model_data, dict):
                continue
            train_m = model_data.get("train_metrics") or {}
            test_m = model_data.get("test_metrics") or {}

            train_f1 = train_m.get("f1_score")
            test_f1 = test_m.get("f1_score")
            train_recall = train_m.get("recall")
            test_recall = test_m.get("recall")
            train_auc = train_m.get("auc_roc")
            test_auc = test_m.get("auc_roc")
            train_acc = train_m.get("accuracy")
            test_acc = test_m.get("accuracy")

            def _gap(a, b):
                if a is None or b is None:
                    return None
                return round(float(a) - float(b), 4)

            f1_gap = _gap(train_f1, test_f1)
            recall_gap = _gap(train_recall, test_recall)
            auc_gap = _gap(train_auc, test_auc)
            acc_gap = _gap(train_acc, test_acc)

            high = (
                (train_f1 is not None and train_f1 >= 0.98)
                or (f1_gap is not None and f1_gap >= 0.15)
                or (train_acc is not None and train_acc >= 0.99)
            )
            medium = (f1_gap is not None and f1_gap >= 0.08) or (
                acc_gap is not None and acc_gap >= 0.08
            )
            overfit_risk = "high" if high else "medium" if medium else "low"

            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "train_f1": train_f1,
                    "test_f1": test_f1,
                    "f1_gap": f1_gap,
                    "train_recall": train_recall,
                    "test_recall": test_recall,
                    "recall_gap": recall_gap,
                    "train_auc": train_auc,
                    "test_auc": test_auc,
                    "auc_gap": auc_gap,
                    "train_accuracy": train_acc,
                    "test_accuracy": test_acc,
                    "accuracy_gap": acc_gap,
                    "overfit_risk": overfit_risk,
                }
            )

    if not rows:
        logging.warning("No model entries found in training_results.json — gap table empty.")
        return

    df = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "overfit_gap_table.csv"
    df.to_csv(out_path, index=False)
    logging.info("[SUCCESS] Overfit gap table saved to: %s", out_path)

    logging.info("[OVERFIT-GAP] train_f1 / test_f1 / f1_gap / risk:")
    for _, row in df.iterrows():
        tf = f"{row['train_f1']:.3f}" if row["train_f1"] is not None else "N/A"
        ef = f"{row['test_f1']:.3f}" if row["test_f1"] is not None else "N/A"
        gap = f"{row['f1_gap']:+.3f}" if row["f1_gap"] is not None else "N/A"
        logging.info(
            "  %-12s %-22s train=%s test=%s gap=%s risk=%s",
            row["dataset"],
            row["model"],
            tf,
            ef,
            gap,
            row["overfit_risk"].upper(),
        )


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Assess post-prediction fairness")
    parser.add_argument(
        "--pipeline",
        type=str,
        default="cardiac",
        choices=["cardiac", "dermatology"],
        help="Pipeline name (e.g., cardiac, dermatology)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Optional dataset names to assess (CLI override).",
    )
    parser.add_argument(
        "--model-types",
        nargs="+",
        default=None,
        help="Optional model types to assess (CLI override).",
    )
    parser.add_argument(
        "-v", "--verbose", action="count", default=0, help="Verbosity: -v=info, -vv=debug"
    )
    args = parser.parse_args()

    pipeline = args.pipeline

    project_root = Path(__file__).parent.parent.parent
    run_id = os.getenv("RUN_ID")
    if not run_id:
        raise RuntimeError(
            "RUN_ID is not set. assess_predictions.py must be called from the pipeline "
            "with RUN_ID exported."
        )
    baseline_root = project_root / f"output/{pipeline}/runs/{run_id}/baseline"
    predictions_dir = baseline_root / "results" / "predictions"
    results_dir = baseline_root / "prediction_fairness"
    # Setup
    setup_phase_logging(
        project_root,
        "fairness_assessment.log",
        verbose=args.verbose,
        run_id=run_id,
        stage_name="assess",
    )
    logging.info("[PHASE] Fairness assessment started")
    logging.info(
        "Run context: pipeline=%s run_id=%s predictions_dir=%s output_dir=%s",
        pipeline,
        run_id,
        predictions_dir,
        results_dir,
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    # Resolve configured sensitive attributes from the pipeline config so
    # group_cluster / ethnicity are honored when present (per-cluster fairness).
    try:
        pipeline_cfg = load_pipeline_config(project_root, pipeline)
        configured_sensitive = (pipeline_cfg.get("fairness", {}) or {}).get(
            "sensitive_attributes", _DEFAULT_SENSITIVE
        )
    except Exception as exc:  # noqa: BLE001 — config is optional, fall back safely
        logging.warning("Could not load pipeline config (%s); using default sensitive attrs", exc)
        configured_sensitive = _DEFAULT_SENSITIVE
    logging.info("Configured sensitive attributes: %s", configured_sensitive)

    # Initialize fairness calculator (per-dataset override happens downstream).
    metrics_calculator = FairnessMetrics(sensitive_attributes=["age_group_cat", "sex_cat"])

    # Find all prediction files (new path: results/predictions/{ds}_{m}_train.csv)
    train_files = sorted(predictions_dir.glob("*_train.csv"))
    if args.datasets:
        selected = [d.strip() for d in args.datasets]
        train_files = [
            p for p in train_files if any(p.name.startswith(f"{dataset}_") for dataset in selected)
        ]
    if args.model_types:
        selected_models = [m.strip().lower() for m in args.model_types]
        train_files = [
            p
            for p in train_files
            if any(f"_{model}_train.csv" in p.name for model in selected_models)
        ]

    if not train_files:
        logging.error(f"No prediction files found in {predictions_dir}")
        logging.error("Please run baseline training first (Phase 5)")
        return

    logging.info(f"Found {len(train_files)} prediction pairs to assess")

    # Process each dataset — build nested {dataset: {model: {method: results}}} structure
    nested_results: dict = {}

    for train_file in train_files:
        combined_name = train_file.stem.replace("_train", "")
        test_file = predictions_dir / f"{combined_name}_test.csv"

        if not test_file.exists():
            logging.warning(f"Test file not found for {combined_name}, skipping")
            continue

        dataset, model = _split_dataset_model(combined_name)

        try:
            results = assess_dataset_fairness(
                combined_name,
                train_file,
                test_file,
                results_dir,
                metrics_calculator,
                configured_sensitive=configured_sensitive,
            )
            nested_results.setdefault(dataset, {}).setdefault(model, {})["single_split"] = results
        except Exception as e:
            logging.exception(f"Failed to assess {combined_name}: {e}")
            continue

    # Process CV prediction files: results/predictions/{ds}_{model}_cv.csv
    cv_files = sorted(predictions_dir.glob("*_cv.csv"))
    if args.datasets:
        selected = [d.strip() for d in args.datasets]
        cv_files = [
            p for p in cv_files if any(p.name.startswith(f"{dataset}_") for dataset in selected)
        ]
    if args.model_types:
        selected_models = [m.strip().lower() for m in args.model_types]
        cv_files = [
            p for p in cv_files if any(f"_{model}_cv.csv" in p.name for model in selected_models)
        ]

    for cv_file in cv_files:
        combined_name = cv_file.stem.replace("_cv", "")
        dataset, model = _split_dataset_model(combined_name)
        try:
            cv_results = assess_cv_fairness(
                combined_name,
                cv_file,
                results_dir,
                metrics_calculator,
                configured_sensitive=configured_sensitive,
            )
            nested_results.setdefault(dataset, {}).setdefault(model, {})["kfold_cv"] = cv_results
        except Exception as e:
            logging.exception(f"Failed to assess CV file {combined_name}: {e}")
            continue

    # Save combined report
    combined_file = results_dir / "fairness_report.json"
    with open(combined_file, "w") as f:
        json.dump(
            nested_results,
            f,
            indent=2,
            default=lambda o: float(o) if isinstance(o, (np.integer, np.floating)) else str(o),
        )

    # Save human-readable markdown interpretation
    md_file = results_dir / "fairness_report.md"
    md_file.write_text(_render_fairness_md(nested_results))

    # Write overfit gap table from training_results.json
    training_results_path = baseline_root / "results" / "training_results.json"
    write_overfit_gap_table(training_results_path, results_dir)

    logging.info("[PHASE] Fairness assessment complete")
    logging.info(
        "Fairness assessment summary: datasets=%d models=%d",
        len(nested_results),
        sum(len(v) for v in nested_results.values()),
    )
    logging.info(f"Results saved to: {results_dir}")
    logging.info(f"Combined report: {combined_file}")


if __name__ == "__main__":
    main()
