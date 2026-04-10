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
from typing import Dict, List

import numpy as np
import pandas as pd

# Add src to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from fairxai.cli.runner_base import get_project_root, load_pipeline_config, setup_phase_logging
from fairxai.fairness.metrics import FairnessMetrics, summarize_fairness_results


def decode_sensitive_attributes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Decode sensitive attributes from scaled numerical values back to categories.

    Args:
        df: DataFrame with scaled sensitive attributes

    Returns:
        DataFrame with decoded categories
    """
    df = df.copy()

    # Decode age_group (was label-encoded)
    # Typical encoding: 0:<40, 1:40-49, 2:50-59, 3:60-69, 4:70+
    age_mapping = {
        -2.0: "<40",
        -1.5: "<40",  # Handles some variance from scaling
        -1.0: "40-49",
        -0.5: "40-49",
        0.0: "50-59",
        0.5: "50-59",
        1.0: "60-69",
        1.5: "60-69",
        2.0: "70+",
        2.5: "70+",
    }

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

    return df


def assess_dataset_fairness(
    dataset_name: str,
    train_file: Path,
    test_file: Path,
    output_dir: Path,
    metrics_calculator: FairnessMetrics,
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
    logging.info(f"\n{'='*60}")
    logging.info(f"Assessing: {dataset_name}")
    logging.info(f"{'='*60}")

    # Load predictions
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    logging.info(f"Loaded predictions:")
    logging.info(f"  Train: {len(train_df)} samples")
    logging.info(f"  Test: {len(test_df)} samples")

    # Decode sensitive attributes
    train_df = decode_sensitive_attributes(train_df)
    test_df = decode_sensitive_attributes(test_df)

    # Check if decoding worked
    if "age_group_cat" in train_df.columns and "sex_cat" in train_df.columns:
        logging.info(f"\n--- Decoded Sensitive Attributes ---")
        logging.info(f"Age groups: {train_df['age_group_cat'].value_counts().to_dict()}")
        logging.info(f"Sex: {train_df['sex_cat'].value_counts().to_dict()}")

        # Use decoded columns
        metrics_calculator.sensitive_attributes = ["age_group_cat", "sex_cat"]
    else:
        logging.warning("Could not decode sensitive attributes, using scaled values")

    results = {"dataset": dataset_name, "train_metrics": {}, "test_metrics": {}, "comparison": {}}

    # Get numerical feature columns (exclude metadata)
    exclude_cols = [
        "y_true",
        "y_pred",
        "y_proba",
        "threshold",
        "age_group",
        "sex",
        "confidence",
        "near_threshold",
        "age_group_cat",
        "sex_cat",
    ]
    feature_cols = [col for col in train_df.columns if col not in exclude_cols]

    if not feature_cols:
        logging.warning("No feature columns found for individual fairness")
        feature_cols = None

    # Calculate metrics for train set
    logging.info(f"\n--- Train Set Fairness ---")
    train_metrics = metrics_calculator.calculate_all_metrics(train_df, feature_cols)
    results["train_metrics"] = train_metrics

    # Log train results
    log_fairness_metrics(train_metrics, "Train")

    # Calculate metrics for test set
    logging.info(f"\n--- Test Set Fairness ---")
    test_metrics = metrics_calculator.calculate_all_metrics(test_df, feature_cols)
    results["test_metrics"] = test_metrics

    # Log test results
    log_fairness_metrics(test_metrics, "Test")

    # Compare train vs test
    results["comparison"] = compare_fairness(train_metrics, test_metrics)

    # Save results
    output_file = output_dir / f"{dataset_name}_fairness_assessment.json"
    with open(output_file, "w") as f:
        # Convert numpy types for JSON serialization
        json.dump(
            results,
            f,
            indent=2,
            default=lambda o: float(o) if isinstance(o, (np.integer, np.floating)) else str(o),
        )

    logging.info(f"\n[SUCCESS] Fairness assessment saved to: {output_file}")

    # Create summary table
    summary_train = summarize_fairness_results(train_metrics)
    summary_test = summarize_fairness_results(test_metrics)

    summary_train["split"] = "train"
    summary_test["split"] = "test"
    summary = pd.concat([summary_train, summary_test], ignore_index=True)

    summary_file = output_dir / f"{dataset_name}_fairness_summary.csv"
    summary.to_csv(summary_file, index=False)

    logging.info(f"[SUCCESS] Summary table saved to: {summary_file}")

    return results


def log_fairness_metrics(metrics: Dict, split_name: str):
    """Log fairness metrics in a readable format."""

    # Group fairness
    for attr, attr_metrics in metrics.get("group_fairness", {}).items():
        logging.info(f"\n  {attr.upper()} - Group Fairness:")

        for metric_name, metric_data in attr_metrics.items():
            is_fair = "[SUCCESS]" if metric_data.get("is_fair", False) else "[ERROR]"

            if metric_name == "demographic_parity":
                diff = metric_data["max_difference"]
                logging.info(f"    {is_fair} Demographic Parity: {diff:.3f} max difference")

            elif metric_name == "equalized_odds":
                tpr_diff = metric_data["tpr_max_difference"]
                fpr_diff = metric_data["fpr_max_difference"]
                logging.info(
                    f"    {is_fair} Equalized Odds: TPR diff={tpr_diff:.3f}, FPR diff={fpr_diff:.3f}"
                )

            elif metric_name == "equal_opportunity":
                diff = metric_data["max_difference"]
                logging.info(f"    {is_fair} Equal Opportunity: {diff:.3f} TPR difference")

            elif metric_name == "predictive_parity":
                diff = metric_data["max_difference"]
                logging.info(f"    {is_fair} Predictive Parity: {diff:.3f} precision difference")

    # Calibration
    for attr, calib_data in metrics.get("calibration", {}).items():
        logging.info(f"\n  {attr.upper()} - Calibration:")
        is_fair = "[SUCCESS]" if calib_data.get("is_fair", False) else "[ERROR]"
        max_ece_diff = calib_data["max_ece_difference"]
        logging.info(f"    {is_fair} Max ECE difference: {max_ece_diff:.4f}")

        for group, group_calib in calib_data["group_calibration"].items():
            logging.info(f"      {group}: ECE={group_calib['ece']:.4f}")

    # Individual fairness
    if "individual_fairness" in metrics and metrics["individual_fairness"]:
        ind_fair = metrics["individual_fairness"]
        logging.info(f"\n  Individual Fairness (k-NN consistency):")
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
    pipeline_cfg = load_pipeline_config(project_root, pipeline)
    run_id = os.getenv("RUN_ID")
    if run_id:
        baseline_root = project_root / f"output/{pipeline}/runs/{run_id}/baseline"
        experiments_dir = baseline_root / "results"
        results_dir = baseline_root / "fairness"
    else:
        experiments_dir = project_root / pipeline_cfg["paths"]["experiments_dir"]
        results_dir = project_root / pipeline_cfg["paths"]["results_fairness_dir"]
    # Setup
    log_dir = setup_phase_logging(
        project_root,
        "fairness_assessment.log",
        verbose=args.verbose,
        run_id=run_id,
        stage_name="assess",
    )
    logging.info("[PHASE] Fairness assessment started")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Initialize fairness calculator
    metrics_calculator = FairnessMetrics(sensitive_attributes=["age_group_cat", "sex_cat"])

    # Find all prediction files
    train_files = sorted(experiments_dir.glob("*_train_predictions.csv"))
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
            if any(f"_{model}_train_predictions.csv" in p.name for model in selected_models)
        ]

    if not train_files:
        logging.error(f"No prediction files found in {experiments_dir}")
        logging.error("Please run baseline training first (Phase 3)")
        return

    logging.info(f"Found {len(train_files)} datasets to assess")

    # Process each dataset
    all_results = {}

    for train_file in train_files:
        dataset_name = train_file.stem.replace("_train_predictions", "")
        test_file = experiments_dir / f"{dataset_name}_test_predictions.csv"

        if not test_file.exists():
            logging.warning(f"Test file not found for {dataset_name}, skipping")
            continue

        try:
            results = assess_dataset_fairness(
                dataset_name, train_file, test_file, results_dir, metrics_calculator
            )
            all_results[dataset_name] = results
        except Exception as e:
            logging.exception(f"[ERROR] Failed to assess {dataset_name}: {e}")
            continue

    # Save combined results
    combined_file = results_dir / "post_prediction_fairness_report.json"
    with open(combined_file, "w") as f:
        json.dump(
            all_results,
            f,
            indent=2,
            default=lambda o: float(o) if isinstance(o, (np.integer, np.floating)) else str(o),
        )

    logging.info(f"\n{'='*60}")
    logging.info("[PHASE] Fairness assessment complete")
    logging.info(f"{'='*60}")
    logging.info(f"Results saved to: {results_dir}")
    logging.info(f"Combined report: {combined_file}")
    logging.info(f"\nNext step: Create visualization notebook (Phase 5)")


if __name__ == "__main__":
    main()
