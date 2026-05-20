#!/usr/bin/env python3
"""
Preprocess pipeline datasets for modeling.

This script:
1. Loads standardized datasets from data/raw/{pipeline}/
2. Analyzes and handles missing values
3. Performs stratified train/test split
4. Scales numerical features
5. Saves processed datasets to data/processed/{pipeline}/
6. Generates post-preprocessing fairness assessment

Usage:
    python scripts/common/preprocess_data.py --pipeline cardiac
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import pandas as pd
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from fairxai.cli.runner_base import get_project_root, load_pipeline_config, setup_phase_logging
from fairxai.data.preprocessors import CardiacPreprocessor
from fairxai.data.profilers import DataProfiler
from fairxai.data.schemas import available_sensitive, get_age_unit, preferred_sensitive
from fairxai.experiments.attribute_binning import apply_binning, create_binning_strategy


def _stringify(obj):
    """Recursively stringify dict keys to make JSON-serializable."""
    if isinstance(obj, dict):
        return {str(k): _stringify(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_stringify(v) for v in obj]
    return obj


def _load_schema_config(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def _apply_schema_rules(df: pd.DataFrame, schema_cfg: dict, dataset_name: str) -> pd.DataFrame:
    cfg = schema_cfg.get("datasets", {}).get(dataset_name, {})
    unified = schema_cfg.get("unified_schema", {})

    include = cfg.get("include_features") or []
    exclude = cfg.get("exclude_features") or []
    unified_exclude = unified.get("exclude_features") or []
    label_col = cfg.get("label") or cfg.get("target")

    required_cols = set(
        [
            "heart_disease",
            "age_raw",
            "age_group",
            "sex",
            "sex_extended",
            "sex_bin",
            "ethnicity",
            "group_cluster",
            "_dataset_source",
            "_dataset_file",
        ]
    )

    if include:
        keep_cols = set(include) | required_cols
        df = df[[col for col in df.columns if col in keep_cols]].copy()

    drop_cols = set(exclude) | set(unified_exclude)
    if label_col and label_col != "heart_disease":
        drop_cols.add(label_col)

    drop_cols = [col for col in drop_cols if col in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    return df


def _load_domain_config(project_root: Path, pipeline: str) -> dict:
    domain_path = project_root / f"configs/domain/{pipeline}.yaml"
    with open(domain_path, "r") as f:
        return yaml.safe_load(f)


# Canonical feature name -> raw column aliases across all three cardiac datasets.
_CONSTRAINT_ALIASES: dict[str, list[str]] = {
    "age": ["age_raw", "age", "Age"],
    "resting_blood_pressure": ["trestbps", "RestingBP", "ap_hi"],
    "diastolic_blood_pressure": ["ap_lo"],
    "serum_cholesterol": ["chol", "Cholesterol"],
    "max_heart_rate": ["thalach", "MaxHR"],
    "st_depression": ["oldpeak", "Oldpeak"],
    "height_cm": ["height"],
    "weight_kg": ["weight"],
}


def _apply_clinical_constraints(
    df: pd.DataFrame,
    constraints_cfg: dict,
    dataset_name: str,
) -> pd.DataFrame:
    """Drop (or flag) rows that violate physiological validity constraints.

    Constraints are defined in configs/domain/<pipeline>.yaml under
    ``clinical_constraints``. Each key is a canonical feature name resolved to
    an actual column via _CONSTRAINT_ALIASES. Missing features are skipped silently
    so the same config works across all three datasets.
    """
    default_action = constraints_cfg.get("default_action", "drop")
    n_before = len(df)
    total_dropped = 0

    for canonical, rule in constraints_cfg.items():
        if canonical == "default_action" or not isinstance(rule, dict):
            continue

        col = next(
            (alias for alias in _CONSTRAINT_ALIASES.get(canonical, []) if alias in df.columns),
            None,
        )
        if col is None:
            continue  # Feature not present in this dataset -- skip silently

        action = rule.get("action", default_action)
        mask_bad = pd.Series(False, index=df.index)

        if not rule.get("allow_null", True):
            mask_bad |= df[col].isna()
        if not rule.get("allow_zero", True):
            mask_bad |= df[col] == 0
        if "min" in rule:
            mask_bad |= df[col] < rule["min"]
        if "max" in rule:
            mask_bad |= df[col] > rule["max"]

        n_bad = int(mask_bad.sum())
        if n_bad == 0:
            logging.info(f"  [{dataset_name}] {canonical} ({col}): OK")
            continue

        pct = 100.0 * n_bad / len(df)
        logging.warning(
            f"  [{dataset_name}] {canonical} ({col}): {n_bad} row(s) invalid "
            f"({pct:.1f}%) -- action={action}"
        )

        if action == "drop":
            df = df[~mask_bad].copy()
            total_dropped += n_bad
        else:  # flag
            df[f"{col}_invalid"] = mask_bad

    if total_dropped:
        logging.info(
            f"  [{dataset_name}] clinical constraints: "
            f"dropped {total_dropped} / {n_before} rows ({100 * total_dropped / n_before:.1f}%)"
        )
    return df


def _resolve_preprocessor(pipeline: str):
    if pipeline == "cardiac":
        return CardiacPreprocessor
    raise NotImplementedError(
        f"Pipeline '{pipeline}' is not yet supported by common preprocess_data."
    )


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Preprocess pipeline datasets")
    parser.add_argument(
        "--pipeline",
        type=str,
        default="cardiac",
        choices=["cardiac", "dermatology"],
        help="Pipeline name (e.g., cardiac, dermatology)",
    )
    parser.add_argument(
        "--binning-strategy",
        type=str,
        default=None,
        choices=["fixed_10yr", "fixed_5yr", "clinical", "quantile_3", "quantile_5"],
        help="Age binning strategy to apply before split",
    )
    parser.add_argument(
        "--all-binnings", action="store_true", help="Process with all binning strategies"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=10_000,
        # Default 10k: consumer hardware constraint (i5-1135G7 + GTX 1650 Ti Max-Q, 16GB RAM).
        # SVM RBF on full cardio70k (70k rows) allocates ~18GB for the kernel matrix and OOMs.
        # Set to None or a higher value for HPC runs (cardio70k full dataset ~70k rows).
        help="Stratified subsample cap per dataset. Datasets below the cap are unaffected. "
        "Default 10000 targets consumer hardware; use None or higher for HPC.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Optional dataset names to preprocess (CLI override).",
    )
    parser.add_argument(
        "-v", "--verbose", action="count", default=0, help="Verbosity: -v=info, -vv=debug"
    )
    args = parser.parse_args()

    pipeline = args.pipeline
    max_samples = args.max_samples

    # Paths
    project_root = get_project_root(Path(__file__))
    pipeline_cfg = load_pipeline_config(project_root, pipeline)
    data_raw = project_root / pipeline_cfg["paths"]["raw_dir"]
    data_processed_base = project_root / pipeline_cfg["paths"]["processed_dir"]
    run_id = os.getenv("RUN_ID")
    setup_phase_logging(
        project_root,
        "preprocessing.log",
        verbose=args.verbose,
        run_id=run_id,
        stage_name="preprocess",
    )
    if run_id:
        results_fairness = project_root / f"output/{pipeline}/runs/{run_id}/profiling/data_fairness"
    else:
        results_fairness = project_root / f"output/{pipeline}/profiling/data_fairness"

    # Setup
    logging.info("[PHASE] Preprocessing started")
    logging.info(
        "Run context: pipeline=%s run_id=%s data_raw=%s data_processed=%s",
        pipeline,
        run_id or "none",
        data_raw,
        data_processed_base,
    )
    results_fairness.mkdir(parents=True, exist_ok=True)

    # Configuration
    test_size = pipeline_cfg.get("split", {}).get("test_size", 0.3)
    random_state = pipeline_cfg.get("split", {}).get("random_state", 42)

    # Determine which binning strategies to use
    if args.all_binnings:
        binning_strategies = ["fixed_10yr", "fixed_5yr", "clinical", "quantile_3", "quantile_5"]
    elif args.binning_strategy:
        binning_strategies = [args.binning_strategy]
    else:
        binning_strategies = [None]  # No binning, use existing age_group

    logging.info("Configuration:")
    logging.info(f"  Train/Test split: {(1-test_size):.0%}/{test_size:.0%}")
    logging.info(f"  Random state: {random_state}")
    logging.info("  Scaling method: StandardScaler")
    logging.info(f"  Binning strategies: {binning_strategies}")

    # Sensitive/group configuration (age/sex/ethnicity + optional groups)
    sensitive_attrs = preferred_sensitive(
        pipeline_cfg.get("fairness", {}).get("sensitive_attributes")
    )

    schema_cfg = _load_schema_config(project_root / pipeline_cfg["runtime"]["schema_mapping_json"])
    domain_cfg = _load_domain_config(project_root, pipeline)

    # Initialize preprocessor and profiler
    preprocessor_cls = _resolve_preprocessor(pipeline)
    preprocessor = preprocessor_cls(sensitive_attrs=sensitive_attrs)
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

    logging.info(f"Found {len(dataset_files)} datasets to preprocess")

    target_col = pipeline_cfg.get("training", {}).get("target", "heart_disease")

    # Process each dataset with each binning strategy
    for binning_strategy in binning_strategies:
        if binning_strategy:
            logging.info("[BINNING] Strategy=%s", binning_strategy)

        preprocessing_summary = {}

        for filepath in dataset_files:
            dataset_name = filepath.stem.replace("_standardized", "")
            subdir = f"{dataset_name}_{binning_strategy}" if binning_strategy else dataset_name
            data_processed = data_processed_base / subdir
            data_processed.mkdir(parents=True, exist_ok=True)
            logging.info(
                "[DATASET] Preprocessing dataset=%s binning=%s output_dir=%s",
                dataset_name,
                binning_strategy or "default",
                data_processed,
            )

            # Load dataset
            df = pd.read_csv(filepath)
            df = _apply_schema_rules(df, schema_cfg, dataset_name)
            logging.info(f"Loaded: {len(df)} samples, {len(df.columns)} features")
            if df.empty:
                logging.error(f"No rows available after schema rules for {dataset_name}. Skipping.")
                continue

            # Stratified subsample if dataset exceeds max_samples cap.
            if max_samples and len(df) > max_samples:
                original_len = len(df)
                df = (
                    df.groupby(target_col, group_keys=False)
                    .apply(
                        lambda g: g.sample(
                            n=round(max_samples * len(g) / original_len),
                            random_state=random_state,
                        )
                    )
                    .reset_index(drop=True)
                )
                logging.info(
                    f"  Subsampled {dataset_name}: from {original_len} to {len(df)} rows "
                    f"(stratified on '{target_col}')"
                )

            # Normalize age from days to years for datasets that declare age_unit: days.
            # Must happen before any binning strategy is applied (all strategies expect years).
            if "age_raw" in df.columns and get_age_unit(dataset_name) == "days":
                df["age_raw"] = (df["age_raw"] / 365.25).round(2)
                logging.info(
                    f"  age_raw normalized: days to years for {dataset_name} "
                    f"(range now {df['age_raw'].min():.1f}-{df['age_raw'].max():.1f} yrs)"
                )

            # Apply clinical validity constraints (drop physiologically impossible rows).
            # Must run after age normalization so age_raw is already in years.
            constraints_cfg = domain_cfg.get("clinical_constraints", {})
            if constraints_cfg:
                df = _apply_clinical_constraints(df, constraints_cfg, dataset_name)
                if df.empty:
                    logging.error(
                        f"No rows remaining after clinical constraints for {dataset_name}. Skipping."
                    )
                    continue

            # Apply age binning if specified
            if binning_strategy:
                logging.info("Age binning strategy: %s", binning_strategy)
                if "age_raw" not in df.columns:
                    logging.error("'age_raw' column not found. Cannot apply binning.")
                    continue

                # Create binning strategy
                bins, labels = create_binning_strategy(df, binning_strategy)

                # Apply binning (overwrite canonical age_group)
                df = apply_binning(df, bins, labels, col="age_raw", output_col="age_group")
                if df.empty:
                    logging.error(f"No rows available after binning for {dataset_name}. Skipping.")
                    continue

                # Log binning results
                age_dist = df["age_group"].value_counts().sort_index()
                logging.info("Age group distribution:")
                for age_group, count in age_dist.items():
                    pct = count / len(df) * 100
                    logging.info(f"  {age_group}: {count} ({pct:.1f}%)")

            # Step 1: Analyze missing values
            logging.info("Missing value analysis:")
            missing_analysis = preprocessor.analyze_missing_values(df)

            if missing_analysis["total_missing"] == 0:
                logging.info("[SUCCESS] No missing values detected")
            else:
                logging.warning(f"Found {missing_analysis['total_missing']} missing values")
                for col, info in missing_analysis["missing_by_column"].items():
                    logging.warning(
                        f"  {col}: {info['count']} ({info['percentage']:.1f}%) - {info['action']}"
                    )

            # Handle missing values (strategy: drop rows if < 5% missing)
            df_clean, actions = preprocessor.handle_missing_values(df, strategy="drop_rows")
            if df_clean.empty:
                logging.error(
                    f"No rows available after missing-value handling for {dataset_name}. Skipping."
                )
                continue

            # Numeric ordinal encoding of age_group for use as a model feature.
            # Mirrors sex_bin: the string age_group column stays for fairness
            # grouping/plots; age_group_idx is the model-usable numeric form.
            # Bin order is derived from mean age within each bin, so it is correct
            # for every binning strategy without parsing label strings.
            if "age_group" in df_clean.columns and "age_raw" in df_clean.columns:
                age_order = (
                    df_clean.groupby("age_group", observed=True)["age_raw"]
                    .mean()
                    .sort_values()
                    .index.tolist()
                )
                age_rank = {group: idx for idx, group in enumerate(age_order)}
                df_clean["age_group_idx"] = df_clean["age_group"].map(age_rank).astype(int)
                logging.info("  age_group_idx encoded: %s", age_rank)

            # Step 2: Stratified train/test split
            logging.info("Stratified train/test split:")
            train_df, test_df = preprocessor.stratified_split(
                df_clean,
                target=target_col,
                test_size=test_size,
                random_state=random_state,
                context_label=f"dataset={dataset_name}, binning={binning_strategy or 'default'}",
            )

            # Verify split maintains distributions
            verification = preprocessor.verify_split_fairness(train_df, test_df, target=target_col)

            logging.info("Split verification:")
            logging.info("Train target distribution:")
            for label, pct in verification["target_distribution"]["train"].items():
                logging.info(f"  Class {label}: {pct:.2%}")
            logging.info("Test target distribution:")
            for label, pct in verification["target_distribution"]["test"].items():
                logging.info(f"  Class {label}: {pct:.2%}")

            # Step 3: Prepare features and scale
            logging.info("Feature preparation and scaling:")

            X_train, y_train, feature_names = preprocessor.prepare_features(
                train_df, target=target_col
            )
            X_test, y_test, _ = preprocessor.prepare_features(test_df, target=target_col)

            logging.info(f"Features: {len(feature_names)}")
            logging.info(
                f"  {', '.join(feature_names[:10])}{'...' if len(feature_names) > 10 else ''}"
            )

            X_train_scaled, X_test_scaled = preprocessor.scale_features(
                X_train, X_test, method="standard"
            )

            # Step 4: Save processed datasets
            logging.info("Saving processed data:")

            # Save as DataFrames with all columns
            train_processed = train_df.copy()
            test_processed = test_df.copy()

            # Save to CSV (binning strategy is reflected in directory structure)
            train_file = data_processed / f"{dataset_name}_train.csv"
            test_file = data_processed / f"{dataset_name}_test.csv"

            train_processed.to_csv(train_file, index=False)
            test_processed.to_csv(test_file, index=False)

            logging.info(f"[SUCCESS] Train set: {train_file}")
            logging.info(f"[SUCCESS] Test set: {test_file}")

            # Save scaled feature matrices (for modeling) and persist sensitive columns
            train_scaled_file = data_processed / f"{dataset_name}_train_scaled.csv"
            test_scaled_file = data_processed / f"{dataset_name}_test_scaled.csv"

            # Convert scaled arrays to DataFrames
            X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=feature_names)
            X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=feature_names)

            sens_cols_train = available_sensitive(train_df, sensitive_attrs) + [
                c for c in ["sex_extended", "sex_bin", "age_group_idx"] if c in train_df.columns
            ]
            sens_cols_test = available_sensitive(test_df, sensitive_attrs) + [
                c for c in ["sex_extended", "sex_bin", "age_group_idx"] if c in test_df.columns
            ]

            # Avoid duplicate column names in scaled outputs (which pandas later
            # re-labels as ".1", ".2", ... and can trigger noisy warnings).
            sens_cols_train = [
                c
                for c in dict.fromkeys(sens_cols_train)
                if c not in X_train_scaled_df.columns and c != target_col
            ]
            sens_cols_test = [
                c
                for c in dict.fromkeys(sens_cols_test)
                if c not in X_test_scaled_df.columns and c != target_col
            ]

            train_scaled = pd.concat(
                [
                    X_train_scaled_df.reset_index(drop=True),
                    train_df[sens_cols_train + [target_col]].reset_index(drop=True),
                ],
                axis=1,
            )
            test_scaled = pd.concat(
                [
                    X_test_scaled_df.reset_index(drop=True),
                    test_df[sens_cols_test + [target_col]].reset_index(drop=True),
                ],
                axis=1,
            )

            train_scaled.to_csv(train_scaled_file, index=False)
            test_scaled.to_csv(test_scaled_file, index=False)

            logging.info(f"[SUCCESS] Train scaled: {train_scaled_file}")
            logging.info(f"[SUCCESS] Test scaled: {test_scaled_file}")

            # Step 5: Post-preprocessing fairness check
            logging.info("Post-preprocessing fairness assessment:")

            train_profile = profiler.profile_dataset(
                train_processed, target=target_col, dataset_name=f"{dataset_name}_train"
            )
            test_profile = profiler.profile_dataset(
                test_processed, target=target_col, dataset_name=f"{dataset_name}_test"
            )

            # Log key fairness metrics
            for split_name, profile in [("Train", train_profile), ("Test", test_profile)]:
                logging.info(f"{split_name} set:")
                logging.info(f"  Samples: {profile['basic_stats']['n_samples']}")
                logging.info(
                    f"  Disease prevalence: {profile['basic_stats']['target_prevalence']:.2%}"
                )

                for attr, imbalance in profile["label_imbalance_by_group"].items():
                    spd = imbalance["statistical_parity_difference"]
                    logging.info(f"  {attr} - Max parity difference: {spd['max_difference']:.2%}")

            # Save fairness profiles
            train_fairness_file = results_fairness / f"{dataset_name}_train.json"
            test_fairness_file = results_fairness / f"{dataset_name}_test.json"

            with open(train_fairness_file, "w") as f:
                json.dump(train_profile, f, indent=2, default=str)
            with open(test_fairness_file, "w") as f:
                json.dump(test_profile, f, indent=2, default=str)

            logging.info("[SUCCESS] Fairness profiles saved")

            # Save preprocessing summary (stringify to avoid Interval keys)
            preprocessing_summary[dataset_name] = _stringify(
                {
                    "original_samples": len(df),
                    "cleaned_samples": len(df_clean),
                    "train_samples": len(train_df),
                    "test_samples": len(test_df),
                    "n_features": len(feature_names),
                    "missing_value_actions": actions,
                    "split_verification": verification,
                    "binning_strategy": binning_strategy,
                    "output_dir": str(data_processed),
                }
            )

            # Save overall preprocessing metadata for this binning strategy
            metadata_file = data_processed / "preprocessing_metadata.json"
            preprocessor.save_metadata(str(metadata_file))

            # Save summary
            summary_file = data_processed / "preprocessing_summary.json"
            with open(summary_file, "w") as f:
                json.dump(preprocessing_summary, f, indent=2, default=str)
            logging.info(
                "[SUCCESS] Dataset preprocessing complete: dataset=%s binning=%s",
                dataset_name,
                binning_strategy or "default",
            )
            logging.info(f"Processed datasets saved to: {data_processed}")
            logging.info(f"Fairness assessments saved to: {results_fairness}")

    logging.info("[PHASE] Preprocessing complete")
    logging.info(
        "Preprocessing summary: datasets=%d binning_strategies=%d",
        len(dataset_files),
        len(binning_strategies),
    )


if __name__ == "__main__":
    main()
