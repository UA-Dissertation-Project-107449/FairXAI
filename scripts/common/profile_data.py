#!/usr/bin/env python3
"""
Profile pipeline datasets for fairness assessment (pre-model).

This script:
1. Loads standardized datasets
2. Profiles each dataset for basic statistics and fairness metrics
3. Analyzes representation balance across sensitive groups
4. Identifies label imbalances by demographic groups
5. Generates comprehensive fairness report (PRE-PROCESSING)

Usage:
    python scripts/common/profile_data.py --pipeline cardiac
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from fairxai.cli.runner_base import get_project_root, load_pipeline_config, setup_phase_logging
from fairxai.cli.runner_utils import get_run_root, resolve_run_id
from fairxai.data.profilers import DataProfiler, compare_datasets


def main():
    parser = argparse.ArgumentParser(description="Profile pipeline datasets")
    parser.add_argument(
        "--pipeline",
        type=str,
        default="cardiac",
        choices=["cardiac", "dermatology"],
        help="Pipeline name (e.g., cardiac, dermatology)",
    )
    parser.add_argument(
        "-v", "--verbose", action="count", default=0, help="Verbosity: -v=info, -vv=debug"
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=os.getenv("RUN_ID"),
        help="Run identifier (optional, enables run-scoped outputs)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Optional dataset names to profile (CLI override).",
    )
    args = parser.parse_args()

    pipeline = args.pipeline

    # Paths
    project_root = get_project_root(Path(__file__))
    pipeline_cfg = load_pipeline_config(project_root, pipeline)
    data_raw = project_root / pipeline_cfg["paths"]["raw_dir"]
    run_id = resolve_run_id(args.run_id) if args.run_id else None
    setup_phase_logging(
        project_root,
        "data_profiling.log",
        verbose=args.verbose,
        run_id=run_id,
        stage_name="profile",
    )
    if run_id:
        results_profiling = get_run_root(project_root / f"output/{pipeline}", run_id) / "profiling"
    else:
        results_profiling = project_root / f"output/{pipeline}/profiling"

    # Setup
    logging.info("[PHASE] Data profiling started")
    results_profiling.mkdir(parents=True, exist_ok=True)

    # Initialize profiler
    sensitive_attrs = pipeline_cfg.get("fairness", {}).get(
        "sensitive_attributes", ["age_group", "sex"]
    )
    profiler = DataProfiler(sensitive_attrs=sensitive_attrs)

    # Find all standardized datasets
    dataset_files = list(data_raw.glob("*_standardized.csv"))
    if args.datasets:
        selected = set(d.strip() for d in args.datasets)
        dataset_files = [
            p for p in dataset_files if p.stem.replace("_standardized", "") in selected
        ]

    if not dataset_files:
        logging.error(f"No standardized datasets found in {data_raw}")
        logging.error("Please run scripts/common/load_data.py --pipeline %s first." % pipeline)
        return

    logging.info(f"Found {len(dataset_files)} standardized datasets")
    target_col = pipeline_cfg.get("training", {}).get("target", "heart_disease")

    # Profile each dataset
    all_profiles = []

    for filepath in dataset_files:
        dataset_name = filepath.stem.replace("_standardized", "")
        logging.info(f"\n{'='*60}")
        logging.info(f"Profiling: {dataset_name}")
        logging.info(f"{'='*60}")

        # Load dataset
        df = pd.read_csv(filepath)
        logging.info(f"Loaded: {len(df)} samples, {len(df.columns)} features")

        # Generate profile
        profile = profiler.profile_dataset(df, target=target_col, dataset_name=dataset_name)
        all_profiles.append(profile)

        # Log key findings
        logging.info("\n--- Basic Statistics ---")
        logging.info(f"  Samples: {profile['basic_stats']['n_samples']}")
        logging.info(f"  Features: {profile['basic_stats']['n_features']}")
        logging.info(f"  Disease prevalence: {profile['basic_stats']['target_prevalence']:.2%}")

        logging.info("\n--- Sensitive Attribute Distribution ---")
        for attr, dist in profile["sensitive_attr_distribution"].items():
            logging.info(f"  {attr}:")
            for value, prop in dist["proportions"].items():
                logging.info(f"    {value}: {prop:.2%} (n={dist['counts'][value]})")

        logging.info("\n--- Representation Balance ---")
        for attr, balance in profile["representation_balance"].items():
            cv = balance["coefficient_of_variation"]
            ratio = balance["size_ratio"]
            logging.info(f"  {attr}:")
            logging.info(f"    CV: {cv:.3f} (lower is more balanced)")
            logging.info(f"    Size ratio (max/min): {ratio:.2f}x")

        logging.info("\n--- Label Imbalance by Group (Statistical Parity) ---")
        for attr, imbalance in profile["label_imbalance_by_group"].items():
            logging.info(f"  {attr}:")
            for group, rate in imbalance["positive_rates"].items():
                logging.info(f"    {group}: {rate:.2%} disease prevalence")

            spd = imbalance["statistical_parity_difference"]
            logging.info(f"Max difference: {spd['max_difference']:.2%}")
            if spd["max_ratio"]:
                logging.info(f"Max ratio: {spd['max_ratio']:.2f}x")

        logging.info("\n--- Missing Values ---")
        if profile["missing_value_analysis"]["total_missing"] > 0:
            logging.warning(f"Total missing: {profile['missing_value_analysis']['total_missing']}")
            for col, count in profile["missing_value_analysis"]["columns_with_missing"].items():
                logging.warning(f"    {col}: {count} missing")
        else:
            logging.info("[SUCCESS] No missing values")

        # Save individual profile
        profile_file = results_profiling / f"{dataset_name}_data_profile.json"
        with open(profile_file, "w") as f:
            # Convert numpy types to native Python for JSON serialization
            json.dump(profile, f, indent=2, default=str)
        logging.info(f"\n[SUCCESS] Profile saved to: {profile_file}")

        complexity = profile.get("complexity_metrics")
        if complexity:
            complexity_file = results_profiling / f"{dataset_name}_complexity.json"
            with open(complexity_file, "w") as f:
                json.dump(complexity, f, indent=2, default=str)
            logging.info(f"[SUCCESS] Complexity metrics saved to: {complexity_file}")

    # Compare datasets
    logging.info(f"\n{'='*60}")
    logging.info("CROSS-DATASET COMPARISON")
    logging.info(f"{'='*60}")

    comparison = compare_datasets(all_profiles)
    logging.info(f"Total datasets: {comparison['n_datasets']}")
    logging.info(f"Total samples: {comparison['total_samples']}")
    logging.info("\nSample sizes:")
    for name, size in comparison["sample_sizes"].items():
        logging.info(f"  {name}: {size}")
    logging.info("\nDisease prevalence:")
    for name, prev in comparison["target_prevalence"].items():
        logging.info(f"  {name}: {prev:.2%}")

    # Save comparison
    comparison_file = results_profiling / "dataset_comparison.json"
    with open(comparison_file, "w") as f:
        json.dump(comparison, f, indent=2)
    logging.info(f"\n[SUCCESS] Comparison saved to: {comparison_file}")

    # Generate summary report
    logging.info(f"\n{'='*60}")
    logging.info("PRE-PROCESSING FAIRNESS ASSESSMENT SUMMARY")
    logging.info(f"{'='*60}")

    for profile in all_profiles:
        dataset_name = profile["dataset_name"]
        logging.info(f"\n{dataset_name.upper()}:")

        # Check for fairness issues
        issues = []

        # Check representation balance
        for attr, balance in profile["representation_balance"].items():
            if balance["size_ratio"] and balance["size_ratio"] > 3.0:
                issues.append(
                    f"High representation imbalance in {attr} (ratio: {balance['size_ratio']:.1f}x)"
                )

        # Check label imbalance
        for attr, imbalance in profile["label_imbalance_by_group"].items():
            spd = imbalance["statistical_parity_difference"]
            if spd["max_difference"] > 0.15:  # >15% difference
                issues.append(
                    f"Significant statistical parity violation in {attr} (diff: {spd['max_difference']:.1%})"
                )

        if issues:
            logging.warning(
                "  Potential fairness issues detected for dataset=%s (%d)",
                dataset_name,
                len(issues),
            )
            for issue in issues:
                logging.info("    - %s", issue)
        else:
            logging.info("[SUCCESS] No major fairness issues detected in raw data")

    logging.info(f"\n{'='*60}")
    logging.info("[PHASE] Data profiling complete")
    logging.info(f"{'='*60}")
    logging.info(f"Profiles saved to: {results_profiling}")


if __name__ == "__main__":
    main()
