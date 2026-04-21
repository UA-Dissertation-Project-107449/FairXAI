#!/usr/bin/env python3
"""
Run attribute binning strategies analysis experiment.

This script:
1. Loads standardized raw datasets (with continuous attribute column)
2. Tests multiple binning strategies
3. Computes fairness metrics for each strategy
4. Analyzes sensitive attribute distribution within bins
5. Scores strategies based on sample size, balance, and fairness
6. Generates comprehensive comparison report

Usage:
    python scripts/experiments/run_attribute_binning_analysis.py
    python scripts/experiments/run_attribute_binning_analysis.py --strategies fixed_10yr clinical
    python scripts/experiments/run_attribute_binning_analysis.py --config configs/experiments/age_binning.yaml
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

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
from fairxai.experiments.attribute_binning import (
    analyze_strategy_comprehensive,
    compare_strategies,
    create_binning_strategy,
    generate_summary_report,
)
from fairxai.utils.config import load_yaml_config


def load_dataset_for_binning(
    dataset_name: str, data_dir: Path, sensitive_col: str, target_col: str
):
    """
    Load standardized raw dataset with age_raw column.

    Args:
        dataset_name: Name of dataset ('cleveland' or 'kaggle_heart')
        data_dir: Path to raw data directory

    Returns:
        DataFrame with age_raw, sex, and heart_disease columns
    """
    logging.info(f"Loading dataset: {dataset_name}")

    file_path = data_dir / f"{dataset_name}_standardized.csv"

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    df = pd.read_csv(file_path)

    # Verify required columns
    REQUIRED_COLUMNS = ["age_raw", sensitive_col, target_col]
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    logging.info(f"  Loaded: {len(df)} samples")
    logging.info(f"  Age range: [{df['age_raw'].min()}, {df['age_raw'].max()}]")
    logging.info(f"  Target prevalence: {df[target_col].mean():.2%}")

    return df


def run_strategy_analysis(
    df,
    strategy_name,
    dataset_name,
    sensitive_cols,
    target_col,
    strategy_config=None,
    min_group_size=30,
):
    """
    Analyze a single binning strategy on a dataset.

    Args:
        df: DataFrame with age_raw and required columns
        strategy_name: Strategy to test
        dataset_name: Dataset identifier
        sensitive_cols: List of sensitive attribute columns
        target_col: Target variable column
        strategy_config: Config dict for this strategy (from YAML)

    Returns:
        Analysis result dictionary
    """
    logging.info(f"Analyzing strategy: {strategy_name}")

    try:
        # Create binning strategy (config-driven)
        bins, labels = create_binning_strategy(
            df,
            strategy_name,
            col="age_raw",
            strategy_config=strategy_config,
            min_group_size=min_group_size,
        )

        # Comprehensive analysis (all sensitive attrs)
        result = analyze_strategy_comprehensive(
            df=df,
            strategy_name=strategy_name,
            bins=bins,
            labels=labels,
            dataset_name=dataset_name,
            col="age_raw",
            sensitive_col=sensitive_cols,
            target_col=target_col,
        )

        return result

    except Exception as e:
        logging.error(f"Failed to analyze {strategy_name}: {e}")
        logging.exception(e)
        return None


def run_analysis(
    config_path: str,
    datasets: list = None,
    strategies: list = None,
    output_dir: str = None,
    pipeline: str = "cardiac",
    run_mode: str = "partial",
    archive_previous: bool = True,
    run_id: str = None,
    output_root: str = None,
    verbose: int = 0,
):
    """
    Runs the attribute binning analysis experiment.
    """
    # Paths
    project_root = get_project_root(Path(__file__))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load config
    config_path = project_root / config_path
    if not config_path.exists():
        logging.error(f"Config file not found: {config_path}")
        return

    experiment_cfg = load_yaml_config(str(config_path))
    pipeline_cfg = load_pipeline_config(project_root, pipeline)

    # Validate config
    required_keys = ["data", "binning_strategies"]
    missing = [k for k in required_keys if k not in experiment_cfg]
    if missing:
        logging.error(f"Config missing required keys: {missing}")
        sys.exit(1)

    # Determine datasets and strategies to process
    cfg_datasets = experiment_cfg["data"].get("datasets", "auto")
    sensitive_attrs = experiment_cfg.get("data", {}).get("sensitive_attributes", ["sex"])
    target_col = experiment_cfg.get("data", {}).get("target", "heart_disease")

    # Keep single primary attr for backward-compat helpers (e.g. load_dataset)
    primary_sensitive_col = sensitive_attrs[0] if sensitive_attrs else "sex"

    # Resolve data directory early - needed for auto-detection
    data_dir = project_root / pipeline_cfg["paths"]["raw_dir"]

    if datasets:
        # CLI override takes precedence
        pass
    elif cfg_datasets == "auto" or cfg_datasets is None:
        # Auto-detect: scan for *_standardized.csv files in the data dir
        found = sorted(
            p.stem.replace("_standardized", "") for p in data_dir.glob("*_standardized.csv")
        )
        if not found:
            logging.error(f"No *_standardized.csv files found in {data_dir}")
            sys.exit(1)
        datasets = found
        logging.info(f"Auto-detected datasets: {datasets}")
    else:
        datasets = list(cfg_datasets)

    if strategies:
        strategies = strategies
    else:
        strategies = list(experiment_cfg["binning_strategies"].keys())

    # Strategy config lookup for config-driven binning
    strategy_configs = experiment_cfg.get("binning_strategies", {})

    # Safeguard defaults from config
    safeguards_cfg = experiment_cfg.get("safeguards", {})
    global_min_group_size = safeguards_cfg.get("min_group_size", 30)

    use_run_id = bool(run_id or os.getenv("RUN_ID") or os.getenv("PREFECT__RUNTIME__FLOW_RUN_ID"))
    run_id = resolve_run_id(run_id) if use_run_id else None

    # Determine output directory
    default_output_dir = experiment_cfg.get("output", {}).get("results_dir")
    if run_id:
        base_output = Path(output_root) if output_root else (project_root / f"output/{pipeline}")
    elif output_root:
        base_output = Path(output_root)
    elif default_output_dir:
        base_output = Path(default_output_dir)
        if base_output.parts and base_output.name == "attribute_binning":
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
        "attribute_binning_analysis.log",
        verbose=verbose,
        run_id=run_id,
        stage_name="attribute_binning",
    )
    logger = logging.getLogger(__name__)
    logging.info("[PHASE] Attribute binning analysis started")

    if run_id:
        run_dir = get_run_root(base_output, run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        output_dir = (
            Path(output_dir) if output_dir else run_dir / "experiments" / "attribute_binning"
        )
    else:
        latest_dir = base_output / "latest_run"
        if run_mode == "partial":
            archive_latest_run(base_output, enabled=True, logger=logger)
        else:
            archive_latest_run(base_output, enabled=archive_previous, logger=logger)
        output_dir = Path(output_dir) if output_dir else latest_dir / "attribute_binning"

    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Configuration:")
    logging.info(f"  Datasets: {datasets}")
    logging.info(f"  Strategies: {strategies}")
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
                "phase": "attribute_binning",
                "datasets": datasets,
                "output_dir": str(output_dir),
                "status": "started",
            },
        )

    # Get scoring weights from config
    scoring_cfg = experiment_cfg.get("scoring", {})
    scoring_weights = {
        "sample_size": scoring_cfg.get("sample_size_weight", 0.40),
        "balance": scoring_cfg.get("group_balance_weight", 0.30),
        "fairness": scoring_cfg.get("fairness_sensitivity_weight", 0.30),
    }

    logging.info("Scoring weights:")
    logging.info(f"  Sample size: {scoring_weights['sample_size']:.0%}")
    logging.info(f"  Group balance: {scoring_weights['balance']:.0%}")
    logging.info(f"  Fairness sensitivity: {scoring_weights['fairness']:.0%}")

    # Process each dataset and strategy
    all_results = []

    for dataset_name in datasets:
        logging.info("[DATASET] Processing dataset=%s", dataset_name)

        try:
            # Load dataset
            df = load_dataset_for_binning(dataset_name, data_dir, primary_sensitive_col, target_col)
        except Exception as e:
            logging.error(f"Failed to load {dataset_name}: {e}")
            logging.exception(e)
            continue  # Skip this dataset but continue with others

        # Test each strategy (errors handled within run_strategy_analysis)
        for strategy_name in strategies:
            strategy_cfg = strategy_configs.get(strategy_name)
            result = run_strategy_analysis(
                df,
                strategy_name,
                dataset_name,
                sensitive_cols=sensitive_attrs,
                target_col=target_col,
                strategy_config=strategy_cfg,
                min_group_size=global_min_group_size,
            )
            if result:
                all_results.append(result)

    if not all_results:
        logging.error("No results generated. Exiting.")
        return

    # Compute scores for all results (done once, used by report and recommendations)
    from fairxai.experiments.attribute_binning import compute_strategy_score

    for result in all_results:
        result["score"] = compute_strategy_score(
            result,
            scoring_weights["sample_size"],
            scoring_weights["balance"],
            scoring_weights["fairness"],
        )

    # Generate comparison and reports
    logging.info("[PHASE] Generating comparison report")

    # Comparison table
    comparison_df = compare_strategies(all_results, by_dataset=True)

    # Save results
    csv_file = output_dir / "comparison.csv"
    json_file = output_dir / "analysis.json"
    report_file = output_dir / "report.md"

    comparison_df.to_csv(csv_file, index=False)
    logging.info(f"[SUCCESS] Saved CSV: {csv_file}")

    with open(json_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logging.info(f"[SUCCESS] Saved JSON: {json_file}")

    # Generate markdown report
    generate_summary_report(all_results, report_file, scoring_weights)
    logging.info(f"[SUCCESS] Saved Report: {report_file}")

    # Print summary
    logging.info("Results summary: rows=%d", len(comparison_df))
    logging.debug("Comparison table:\n%s", comparison_df.to_string(index=False))

    # Recommendations
    logging.info("[PHASE] Recommendations")

    # Find top strategies per dataset (using pre-computed scores)
    for dataset in datasets:
        dataset_results = [r for r in all_results if r["dataset"] == dataset]
        if not dataset_results:
            continue

        # Sort by pre-computed score
        dataset_results.sort(key=lambda x: x.get("score", 0), reverse=True)

        logging.info(f"Top 3 strategies for {dataset}:")
        for i, result in enumerate(dataset_results[:3], 1):
            metrics = result["fairness_metrics"]
            logging.info(f"  {i}. {result['strategy']} (score: {result['score']:.3f})")
            logging.info(
                f"     - Groups: {metrics['n_groups']}, "
                f"Min size: {metrics['min_group_size']}, "
                f"Balance CV: {metrics['group_balance_cv']:.3f}, "
                f"SP diff: {metrics['max_sp_difference']:.3f}"
            )

    logging.info("[PHASE] Attribute binning analysis complete")
    logging.info(f"Results saved to: {output_dir}")
    logging.info(f"  - Comparison CSV: {csv_file.name}")
    logging.info(f"  - Detailed JSON: {json_file.name}")
    logging.info(f"  - Summary Report: {report_file.name}")

    if run_id:
        update_latest_pointer(base_output, run_dir, logger)
        append_run_history(
            base_output,
            {
                "run_id": run_id,
                "pipeline": pipeline,
                "mode": run_mode,
                "phase": "attribute_binning",
                "datasets": datasets,
                "output_dir": str(output_dir),
                "status": "completed",
            },
        )


def main():
    parser = argparse.ArgumentParser(description="Run age binning analysis experiment")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiments/age_binning.yaml",
        help="Path to experiment config file",
    )
    parser.add_argument(
        "--datasets", type=str, nargs="+", help="Datasets to process (default: from config)"
    )
    parser.add_argument(
        "--strategies", type=str, nargs="+", help="Strategies to test (default: from config)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory (default: from config or output/{pipeline}/experiments/{run_mode}/latest_run/attribute_binning)",
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
        strategies=args.strategies,
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
