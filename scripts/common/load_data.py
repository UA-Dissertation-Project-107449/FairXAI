#!/usr/bin/env python3
"""
Load pipeline datasets and perform initial data profiling.

This script:
1. Loads pipeline-relevant datasets
2. Standardizes sensitive attributes and target variables
3. Generates dataset summaries
4. Saves unified datasets to data/raw/{pipeline}/
5. Creates initial data profiling report

Usage:
    python scripts/common/load_data.py --pipeline cardiac
"""

import sys
import logging
import argparse
import os
from pathlib import Path
import json
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from fairxai.data.loaders import CardiacDataLoader, get_dataset_summary
from fairxai.cli.runner_base import get_project_root, load_pipeline_config, setup_phase_logging
from fairxai.cli.runner_utils import resolve_run_id, get_run_root


def _resolve_loader(pipeline: str):
    if pipeline == "cardiac":
        return CardiacDataLoader
    raise NotImplementedError(
        f"Pipeline '{pipeline}' is not yet supported by common load_data."
    )


def main():
    parser = argparse.ArgumentParser(description="Load pipeline datasets")
    parser.add_argument(
        "--pipeline",
        type=str,
        default="cardiac",
        choices=["cardiac", "dermatology"],
        help="Pipeline name (e.g., cardiac, dermatology)"
    )
    parser.add_argument("-v", "--verbose", action="count", default=0, help="Verbosity: -v=info, -vv=debug")
    parser.add_argument(
        "--run-id",
        type=str,
        default=os.getenv("RUN_ID"),
        help="Run identifier (optional, enables run-scoped outputs)",
    )
    args = parser.parse_args()

    pipeline = args.pipeline

    # Paths
    project_root = get_project_root(Path(__file__))
    pipeline_cfg = load_pipeline_config(project_root, pipeline)
    config_path = project_root / pipeline_cfg['runtime']['schema_mapping_json']
    feature_map_path = project_root / f"configs/domain/{pipeline}_feature_map.yaml"
    data_external = project_root / pipeline_cfg['paths']['external_dir']
    data_raw = project_root / pipeline_cfg['paths']['raw_dir']
    run_id = resolve_run_id(args.run_id) if args.run_id else None
    log_dir = setup_phase_logging(
        project_root, 'data_loading.log', verbose=args.verbose,
        run_id=run_id, stage_name='load',
    )
    if run_id:
        results_profiling = get_run_root(project_root / f"output/{pipeline}", run_id) / "profiling"
    else:
        results_profiling = project_root / f"output/{pipeline}/profiling"

    # Setup
    logging.info("[PHASE] Data loading started")
    data_raw.mkdir(parents=True, exist_ok=True)
    results_profiling.mkdir(parents=True, exist_ok=True)

    # Load datasets
    logging.info("Initializing pipeline loader...")
    loader_cls = _resolve_loader(pipeline)
    loader = loader_cls(str(config_path), feature_map_path=str(feature_map_path))

    logging.info(f"Loading datasets from: {data_external}")
    datasets = loader.load_all_cardiac_datasets(str(data_external))

    if not datasets:
        logging.error("[ERROR] No datasets loaded. Exiting.")
        return

    # Process each dataset
    summaries = []
    standardized_datasets = {}

    for dataset_name, df_raw in datasets.items():
        logging.info(f"\n--- Processing: {dataset_name} ---")

        # Generate raw summary
        summary = get_dataset_summary(df_raw, dataset_name)
        summaries.append(summary)

        logging.info(f"Shape: {df_raw.shape}")
        logging.info(f"Missing values: {df_raw.isnull().sum().sum()} total")

        # Datasets returned by loader are already harmonized and standardized
        try:
            # Verify expected columns (cardiac defaults)
            assert 'age_group' in df_raw.columns, "Missing age_group"
            assert 'sex' in df_raw.columns, "Missing sex"
            assert 'heart_disease' in df_raw.columns, "Missing heart_disease"

            standardized_datasets[dataset_name] = df_raw

            # Save to raw/{pipeline}
            output_file = data_raw / f"{dataset_name}_standardized.csv"
            df_raw.to_csv(output_file, index=False)
            logging.info(f"[SUCCESS] Saved standardized dataset to: {output_file}")

            # Quick stats
            logging.info(f"  Age groups: {df_raw['age_group'].value_counts().to_dict()}")
            logging.info(f"  Sex: {df_raw['sex'].value_counts().to_dict()}")
            logging.info(f"  Heart disease: {df_raw['heart_disease'].value_counts().to_dict()}")

        except Exception as e:
            logging.error(f"[ERROR] Failed to verify/save {dataset_name}: {e}")
            continue

    # Save summaries
    summary_file = results_profiling / 'dataset_summaries.json'
    with open(summary_file, 'w') as f:
        json.dump(summaries, f, indent=2)
    logging.info(f"\n[SUCCESS] Dataset summaries saved to: {summary_file}")

    # Generate combined report
    report = {
        'total_datasets': len(datasets),
        'standardized_datasets': len(standardized_datasets),
        'datasets': {}
    }

    for name, df in standardized_datasets.items():
        report['datasets'][name] = {
            'n_samples': len(df),
            'n_features': len(df.columns),
            'age_groups': df['age_group'].value_counts().to_dict(),
            'sex_distribution': df['sex'].value_counts().to_dict(),
            'disease_prevalence': df['heart_disease'].value_counts().to_dict(),
            'missing_values_total': int(df.isnull().sum().sum())
        }

    report_file = results_profiling / 'loading_report.json'
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    logging.info(f"[SUCCESS] Loading report saved to: {report_file}")

    logging.info("[PHASE] Data loading complete")
    logging.info(f"Standardized datasets saved to: {data_raw}")
    logging.info(f"Profiling results saved to: {results_profiling}")


if __name__ == "__main__":
    main()