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

import sys
import logging
import os
from pathlib import Path
import json
import pandas as pd
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from fairxai.data.preprocessors import CardiacPreprocessor
from fairxai.data.profilers import DataProfiler
from fairxai.data.schemas import available_sensitive, preferred_sensitive
from fairxai.experiments.age_binning import create_binning_strategy, apply_binning
from fairxai.cli.runner_base import get_project_root, load_pipeline_config, setup_phase_logging


def _stringify(obj):
    """Recursively stringify dict keys to make JSON-serializable."""
    if isinstance(obj, dict):
        return {str(k): _stringify(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_stringify(v) for v in obj]
    return obj


def _load_schema_config(path: Path) -> dict:
    with open(path, 'r') as f:
        return json.load(f)


def _apply_schema_rules(df: pd.DataFrame, schema_cfg: dict, dataset_name: str) -> pd.DataFrame:
    cfg = schema_cfg.get('datasets', {}).get(dataset_name, {})
    unified = schema_cfg.get('unified_schema', {})

    include = cfg.get('include_features') or []
    exclude = cfg.get('exclude_features') or []
    unified_exclude = unified.get('exclude_features') or []
    label_col = cfg.get('label') or cfg.get('target')

    required_cols = set([
        'heart_disease', 'age_raw', 'age_group', 'sex', 'sex_extended', 'sex_bin',
        'ethnicity', 'group_cluster', '_dataset_source', '_dataset_file'
    ])

    if include:
        keep_cols = set(include) | required_cols
        df = df[[col for col in df.columns if col in keep_cols]].copy()

    drop_cols = set(exclude) | set(unified_exclude)
    if label_col and label_col != 'heart_disease':
        drop_cols.add(label_col)

    drop_cols = [col for col in drop_cols if col in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    return df


def _resolve_preprocessor(pipeline: str):
    if pipeline == "cardiac":
        return CardiacPreprocessor
    raise NotImplementedError(
        f"Pipeline '{pipeline}' is not yet supported by common preprocess_data."
    )


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Preprocess pipeline datasets')
    parser.add_argument(
        '--pipeline',
        type=str,
        default='cardiac',
        choices=['cardiac', 'dermatology'],
        help='Pipeline name (e.g., cardiac, dermatology)'
    )
    parser.add_argument(
        '--binning-strategy',
        type=str,
        default=None,
        choices=['fixed_10yr', 'fixed_5yr', 'clinical', 'quantile_3', 'quantile_5'],
        help='Age binning strategy to apply before split'
    )
    parser.add_argument(
        '--all-binnings',
        action='store_true',
        help='Process with all binning strategies'
    )
    parser.add_argument('-v', '--verbose', action='count', default=0, help='Verbosity: -v=info, -vv=debug')
    args = parser.parse_args()

    pipeline = args.pipeline

    # Paths
    project_root = get_project_root(Path(__file__))
    pipeline_cfg = load_pipeline_config(project_root, pipeline)
    data_raw = project_root / pipeline_cfg['paths']['raw_dir']
    data_processed_base = project_root / pipeline_cfg['paths']['processed_dir']
    log_dir = setup_phase_logging(project_root, 'preprocessing.log', verbose=args.verbose)
    run_id = os.getenv('RUN_ID')
    if run_id:
        results_fairness = project_root / f"output/{pipeline}/runs/{run_id}/profiling/fairness"
    else:
        results_fairness = project_root / f"output/{pipeline}/profiling/fairness"

    # Setup
    logging.info("[PHASE] Preprocessing started")
    results_fairness.mkdir(parents=True, exist_ok=True)

    # Configuration
    test_size = pipeline_cfg.get('split', {}).get('test_size', 0.3)
    random_state = pipeline_cfg.get('split', {}).get('random_state', 42)

    # Determine which binning strategies to use
    if args.all_binnings:
        binning_strategies = ['fixed_10yr', 'fixed_5yr', 'clinical', 'quantile_3', 'quantile_5']
    elif args.binning_strategy:
        binning_strategies = [args.binning_strategy]
    else:
        binning_strategies = [None]  # No binning, use existing age_group

    logging.info(f"Configuration:")
    logging.info(f"  Train/Test split: {(1-test_size):.0%}/{test_size:.0%}")
    logging.info(f"  Random state: {random_state}")
    logging.info(f"  Scaling method: StandardScaler")
    logging.info(f"  Binning strategies: {binning_strategies}")

    # Sensitive/group configuration (age/sex/ethnicity + optional groups)
    sensitive_attrs = preferred_sensitive(
        pipeline_cfg.get('fairness', {}).get('sensitive_attributes')
    )

    schema_cfg = _load_schema_config(project_root / pipeline_cfg['runtime']['schema_mapping_json'])

    # Initialize preprocessor and profiler
    preprocessor_cls = _resolve_preprocessor(pipeline)
    preprocessor = preprocessor_cls(sensitive_attrs=sensitive_attrs)
    profiler = DataProfiler(sensitive_attrs=sensitive_attrs)

    # Find all standardized datasets
    dataset_files = list(data_raw.glob('*_standardized.csv'))

    if not dataset_files:
        logging.error(f"No standardized datasets found in {data_raw}")
        logging.error("Please run scripts/common/load_data.py --pipeline %s first." % pipeline)
        return

    logging.info(f"\nFound {len(dataset_files)} datasets to preprocess")

    target_col = pipeline_cfg.get('training', {}).get('target', 'heart_disease')

    # Process each dataset with each binning strategy
    for binning_strategy in binning_strategies:
        if binning_strategy:
            logging.info(f"\n{'#'*80}")
            logging.info(f"# PROCESSING WITH BINNING STRATEGY: {binning_strategy}")
            logging.info(f"{'#'*80}")

        preprocessing_summary = {}

        for filepath in dataset_files:
            dataset_name = filepath.stem.replace('_standardized', '')
            subdir = f"{dataset_name}_{binning_strategy}" if binning_strategy else dataset_name
            data_processed = data_processed_base / subdir
            data_processed.mkdir(parents=True, exist_ok=True)
            logging.info(f"\n{'='*60}")
            logging.info(f"Preprocessing: {dataset_name}")
            if binning_strategy:
                logging.info(f"Binning: {binning_strategy}")
            logging.info(f"Output directory: {data_processed}")
            logging.info(f"{'='*60}")

            # Load dataset
            df = pd.read_csv(filepath)
            df = _apply_schema_rules(df, schema_cfg, dataset_name)
            logging.info(f"Loaded: {len(df)} samples, {len(df.columns)} features")
            if df.empty:
                logging.error(f"No rows available after schema rules for {dataset_name}. Skipping.")
                continue

            # Apply age binning if specified
            if binning_strategy:
                logging.info(f"\n--- Age Binning: {binning_strategy} ---")
                if 'age_raw' not in df.columns:
                    logging.error("'age_raw' column not found. Cannot apply binning.")
                    continue

                # Create binning strategy
                bins, labels = create_binning_strategy(df, binning_strategy)

                # Apply binning (overwrite canonical age_group)
                df = apply_binning(df, bins, labels, age_col='age_raw', output_col='age_group')
                if df.empty:
                    logging.error(f"No rows available after binning for {dataset_name}. Skipping.")
                    continue

                # Log binning results
                age_dist = df['age_group'].value_counts().sort_index()
                logging.info("Age group distribution:")
                for age_group, count in age_dist.items():
                    pct = count / len(df) * 100
                    logging.info(f"  {age_group}: {count} ({pct:.1f}%)")

            # Step 1: Analyze missing values
            logging.info(f"\n--- Missing Value Analysis ---")
            missing_analysis = preprocessor.analyze_missing_values(df)

            if missing_analysis['total_missing'] == 0:
                logging.info("[SUCCESS] No missing values detected")
            else:
                logging.warning(f"⚠️  Found {missing_analysis['total_missing']} missing values")
                for col, info in missing_analysis['missing_by_column'].items():
                    logging.warning(f"  {col}: {info['count']} ({info['percentage']:.1f}%) - {info['action']}")

            # Handle missing values (strategy: drop rows if < 5% missing)
            df_clean, actions = preprocessor.handle_missing_values(df, strategy='drop_rows')
            if df_clean.empty:
                logging.error(f"No rows available after missing-value handling for {dataset_name}. Skipping.")
                continue

            # Step 2: Stratified train/test split
            logging.info(f"\n--- Stratified Train/Test Split ---")
            train_df, test_df = preprocessor.stratified_split(
                df_clean,
                target=target_col,
                test_size=test_size,
                random_state=random_state
            )

            # Verify split maintains distributions
            verification = preprocessor.verify_split_fairness(train_df, test_df, target=target_col)

            logging.info(f"\n--- Split Verification ---")
            logging.info("Train target distribution:")
            for label, pct in verification['target_distribution']['train'].items():
                logging.info(f"  Class {label}: {pct:.2%}")
            logging.info("Test target distribution:")
            for label, pct in verification['target_distribution']['test'].items():
                logging.info(f"  Class {label}: {pct:.2%}")

            # Step 3: Prepare features and scale
            logging.info(f"\n--- Feature Preparation & Scaling ---")

            X_train, y_train, feature_names = preprocessor.prepare_features(
                train_df, target=target_col
            )
            X_test, y_test, _ = preprocessor.prepare_features(
                test_df, target=target_col
            )

            logging.info(f"Features: {len(feature_names)}")
            logging.info(f"  {', '.join(feature_names[:10])}{'...' if len(feature_names) > 10 else ''}")

            X_train_scaled, X_test_scaled = preprocessor.scale_features(
                X_train, X_test, method='standard'
            )

            # Step 4: Save processed datasets
            logging.info(f"\n--- Saving Processed Data ---")

            # Save as DataFrames with all columns
            train_processed = train_df.copy()
            test_processed = test_df.copy()

            # Save to CSV (binning strategy is reflected in directory structure)
            train_file = data_processed / f'{dataset_name}_train.csv'
            test_file = data_processed / f'{dataset_name}_test.csv'

            train_processed.to_csv(train_file, index=False)
            test_processed.to_csv(test_file, index=False)

            logging.info(f"[SUCCESS] Train set: {train_file}")
            logging.info(f"[SUCCESS] Test set: {test_file}")

            # Save scaled feature matrices (for modeling) and persist sensitive columns
            train_scaled_file = data_processed / f'{dataset_name}_train_scaled.csv'
            test_scaled_file = data_processed / f'{dataset_name}_test_scaled.csv'

            # Convert scaled arrays to DataFrames
            X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=feature_names)
            X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=feature_names)

            sens_cols_train = available_sensitive(train_df, sensitive_attrs) + [c for c in ['sex_extended', 'sex_bin'] if c in train_df.columns]
            sens_cols_test = available_sensitive(test_df, sensitive_attrs) + [c for c in ['sex_extended', 'sex_bin'] if c in test_df.columns]

            train_scaled = pd.concat([
                X_train_scaled_df.reset_index(drop=True),
                train_df[sens_cols_train + [target_col]].reset_index(drop=True)
            ], axis=1)
            test_scaled = pd.concat([
                X_test_scaled_df.reset_index(drop=True),
                test_df[sens_cols_test + [target_col]].reset_index(drop=True)
            ], axis=1)

            train_scaled.to_csv(train_scaled_file, index=False)
            test_scaled.to_csv(test_scaled_file, index=False)

            logging.info(f"[SUCCESS] Train scaled: {train_scaled_file}")
            logging.info(f"[SUCCESS] Test scaled: {test_scaled_file}")

            # Step 5: Post-preprocessing fairness check
            logging.info(f"\n--- Post-Preprocessing Fairness Assessment ---")

            train_profile = profiler.profile_dataset(
                train_processed,
                target=target_col,
                dataset_name=f'{dataset_name}_train'
            )
            test_profile = profiler.profile_dataset(
                test_processed,
                target=target_col,
                dataset_name=f'{dataset_name}_test'
            )

            # Log key fairness metrics
            for split_name, profile in [('Train', train_profile), ('Test', test_profile)]:
                logging.info(f"\n{split_name} Set:")
                logging.info(f"  Samples: {profile['basic_stats']['n_samples']}")
                logging.info(f"  Disease prevalence: {profile['basic_stats']['target_prevalence']:.2%}")

                for attr, imbalance in profile['label_imbalance_by_group'].items():
                    spd = imbalance['statistical_parity_difference']
                    logging.info(f"  {attr} - Max parity difference: {spd['max_difference']:.2%}")

            # Save fairness profiles
            train_fairness_file = results_fairness / f'{dataset_name}_train_fairness.json'
            test_fairness_file = results_fairness / f'{dataset_name}_test_fairness.json'

            with open(train_fairness_file, 'w') as f:
                json.dump(train_profile, f, indent=2, default=str)
            with open(test_fairness_file, 'w') as f:
                json.dump(test_profile, f, indent=2, default=str)

            logging.info("\n[SUCCESS] Fairness profiles saved")

            # Save preprocessing summary (stringify to avoid Interval keys)
            preprocessing_summary[dataset_name] = _stringify({
                'original_samples': len(df),
                'cleaned_samples': len(df_clean),
                'train_samples': len(train_df),
                'test_samples': len(test_df),
                'n_features': len(feature_names),
                'missing_value_actions': actions,
                'split_verification': verification,
                'binning_strategy': binning_strategy,
                'output_dir': str(data_processed)
            })

            # Save overall preprocessing metadata for this binning strategy
            metadata_file = data_processed / 'preprocessing_metadata.json'
            preprocessor.save_metadata(str(metadata_file))

            # Save summary
            summary_file = data_processed / 'preprocessing_summary.json'
            with open(summary_file, 'w') as f:
                json.dump(preprocessing_summary, f, indent=2, default=str)

                logging.info(f"\n{'='*60}")
                logging.info(f"[SUCCESS] Preprocessing complete for {dataset_name} ({binning_strategy or 'default'})")
                logging.info(f"{'='*60}")
            logging.info(f"Processed datasets saved to: {data_processed}")
            logging.info(f"Fairness assessments saved to: {results_fairness}")

            logging.info("[PHASE] Preprocessing complete")


if __name__ == "__main__":
    main()