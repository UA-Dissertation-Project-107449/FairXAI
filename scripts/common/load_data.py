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

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from fairxai.cli.runner_base import get_project_root, load_pipeline_config, setup_phase_logging
from fairxai.cli.runner_utils import get_run_root, resolve_run_id
from fairxai.data.loaders import CardiacDataLoader, DermatologyDataLoader, get_dataset_summary


def _resolve_loader(pipeline: str):
    if pipeline == "cardiac":
        return CardiacDataLoader
    if pipeline == "dermatology":
        return DermatologyDataLoader
    raise NotImplementedError(f"Pipeline '{pipeline}' is not yet supported by common load_data.")


def main():
    parser = argparse.ArgumentParser(description="Load pipeline datasets")
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
        help="Optional dataset names to load (CLI override).",
    )
    args = parser.parse_args()

    pipeline = args.pipeline

    # Paths
    project_root = get_project_root(Path(__file__))
    pipeline_cfg = load_pipeline_config(project_root, pipeline)
    config_path = project_root / pipeline_cfg["runtime"]["schema_mapping_json"]
    feature_map_path = project_root / f"configs/domain/{pipeline}_feature_map.yaml"
    data_external = project_root / pipeline_cfg["paths"]["external_dir"]
    data_raw = project_root / pipeline_cfg["paths"]["raw_dir"]
    run_id = resolve_run_id(args.run_id) if args.run_id else None
    setup_phase_logging(
        project_root,
        "data_loading.log",
        verbose=args.verbose,
        log_subdir=pipeline,
        run_id=run_id,
        stage_name="load",
    )
    if run_id:
        results_profiling = get_run_root(project_root / f"output/{pipeline}", run_id) / "profiling"
    else:
        results_profiling = project_root / f"output/{pipeline}/profiling"

    # Setup
    logging.info("[PHASE] Data loading started")
    logging.info(
        "Run context: pipeline=%s run_id=%s data_external=%s data_raw=%s",
        pipeline,
        run_id or "none",
        data_external,
        data_raw,
    )
    data_raw.mkdir(parents=True, exist_ok=True)
    results_profiling.mkdir(parents=True, exist_ok=True)

    # Load datasets
    logging.info("Initializing pipeline loader...")
    loader_cls = _resolve_loader(pipeline)
    loader = loader_cls(str(config_path), feature_map_path=str(feature_map_path))

    logging.info(f"Loading datasets from: {data_external}")
    selected_datasets = [d.strip() for d in args.datasets] if args.datasets else None
    if selected_datasets:
        datasets = {}
        for dataset_name in selected_datasets:
            try:
                datasets[dataset_name] = loader.load_dataset(dataset_name, str(data_external))
                logging.info(f"[SUCCESS] Loaded {dataset_name}: {len(datasets[dataset_name])} rows")
            except Exception as e:
                logging.error(f"Failed to load {dataset_name}: {e}")
    else:
        if pipeline == "cardiac":
            datasets = loader.load_all_cardiac_datasets(str(data_external))
        else:
            datasets = loader.load_all_dermatology_datasets(str(data_external))

    if not datasets:
        logging.error("No datasets loaded. Exiting.")
        return

    # Process each dataset
    summaries = []
    standardized_datasets = {}

    for dataset_name, df_raw in datasets.items():
        logging.info("[DATASET] Processing dataset=%s", dataset_name)

        # Generate raw summary
        summary = get_dataset_summary(df_raw, dataset_name)
        summaries.append(summary)

        logging.info(f"Shape: {df_raw.shape}")
        logging.info(f"Missing values: {df_raw.isnull().sum().sum()} total")

        # Datasets returned by loader are already harmonized and standardized
        try:
            target_col = pipeline_cfg.get("training", {}).get("target", "heart_disease")
            sensitive_cols = pipeline_cfg.get("fairness", {}).get(
                "sensitive_attributes", ["age_group", "sex"]
            )
            required_cols = [target_col, *sensitive_cols]
            if pipeline_cfg.get("training", {}).get("modality") == "image":
                required_cols.append("image_path")
            missing_cols = [col for col in dict.fromkeys(required_cols) if col not in df_raw.columns]
            if missing_cols:
                raise AssertionError(f"Missing required columns: {missing_cols}")

            standardized_datasets[dataset_name] = df_raw

            # Save to raw/{pipeline}
            output_file = data_raw / f"{dataset_name}_standardized.csv"
            df_raw.to_csv(output_file, index=False)
            logging.info(f"[SUCCESS] Saved standardized dataset to: {output_file}")

            # Quick stats
            for col in sensitive_cols:
                if col in df_raw.columns:
                    logging.info("  %s: %s", col, df_raw[col].value_counts(dropna=False).to_dict())
            logging.info(
                "  %s: %s",
                target_col,
                df_raw[target_col].value_counts(dropna=False).to_dict(),
            )

        except Exception as e:
            logging.error(f"Failed to verify/save {dataset_name}: {e}")
            continue

    # Save summaries
    summary_file = results_profiling / "dataset_summaries.json"
    with open(summary_file, "w") as f:
        json.dump(summaries, f, indent=2)
    logging.info(f"[SUCCESS] Dataset summaries saved to: {summary_file}")

    # Generate combined report
    report = {
        "total_datasets": len(datasets),
        "standardized_datasets": len(standardized_datasets),
        "datasets": {},
    }

    target_col = pipeline_cfg.get("training", {}).get("target", "heart_disease")
    sensitive_cols = pipeline_cfg.get("fairness", {}).get(
        "sensitive_attributes", ["age_group", "sex"]
    )
    for name, df in standardized_datasets.items():
        report["datasets"][name] = {
            "n_samples": len(df),
            "n_features": len(df.columns),
            "sensitive_distributions": {
                col: df[col].value_counts(dropna=False).to_dict()
                for col in sensitive_cols
                if col in df.columns
            },
            "target_distribution": df[target_col].value_counts(dropna=False).to_dict(),
            "missing_values_total": int(df.isnull().sum().sum()),
        }
    if hasattr(loader, "last_image_reports"):
        report["image_validation"] = loader.last_image_reports

    report_file = results_profiling / "loading_report.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)
    logging.info(f"[SUCCESS] Loading report saved to: {report_file}")

    logging.info("[PHASE] Data loading complete")
    logging.info(
        "Data loading summary: loaded=%d standardized=%d",
        len(datasets),
        len(standardized_datasets),
    )
    logging.info(f"Standardized datasets saved to: {data_raw}")
    logging.info(f"Profiling results saved to: {results_profiling}")


if __name__ == "__main__":
    main()
