"""Hyperparameter optimisation (HPO) orchestration script.

Runs GridSearchCV / RandomizedSearchCV for each requested model × dataset
combination and saves best params as JSON files consumed by the combinatorial
runner and train_baseline.py.

Usage
-----
# All models on all configured datasets
python scripts/studies/run_hpo.py --pipeline cardiac

# Single model / single dataset
python scripts/studies/run_hpo.py --pipeline cardiac \\
    --model-types logistic_regression random_forest \\
    --datasets cleveland

# Dry-run: print what would run without fitting
python scripts/studies/run_hpo.py --pipeline cardiac --dry-run
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from fairxai.cli.runner_base import get_project_root, setup_study_logging
from fairxai.cli.runner_utils import resolve_run_id, update_output_study_pointer, update_study_pointer
from fairxai.experiments.data_io import (
    default_exclude_columns,
)
from fairxai.experiments.data_io import load_schema_config as load_schema_config_shared
from fairxai.training.grid_search import run_hpo, save_hpo_results
from fairxai.utils.config import load_yaml_config


def _load_processed_data(dataset: str, binning: str, processed_dir: Path):
    data_dir = processed_dir / f"{dataset}_{binning}"
    train_path = data_dir / f"{dataset}_train.csv"
    if not train_path.exists():
        raise FileNotFoundError(
            f"Processed training data not found: {train_path}\n"
            f"Run the preprocess stage first (cardiac_pipeline.sh up to stage 4)."
        )
    return pd.read_csv(train_path)


def main():
    parser = argparse.ArgumentParser(description="FairXAI hyperparameter optimisation")
    parser.add_argument("--pipeline", default="cardiac", help="Pipeline config name")
    parser.add_argument(
        "--config",
        default="configs/experiments/hpo.yaml",
        help="HPO config path (relative to project root)",
    )
    parser.add_argument(
        "--model-types",
        nargs="+",
        default=None,
        help="Models to tune (default: all in hpo.yaml)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Datasets to use (default: all in pipeline config)",
    )
    parser.add_argument(
        "--binning",
        default="fixed_10yr",
        help="Binning strategy for processed data (default: fixed_10yr)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would run without fitting",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    project_root = get_project_root(Path(__file__))
    study_id = resolve_run_id()
    log_subdir = args.pipeline
    setup_study_logging(
        project_root,
        "hpo",
        study_id,
        "hpo.log",
        verbose=args.verbose,
        log_subdir=log_subdir,
    )
    logger = logging.getLogger(__name__)
    update_study_pointer(
        project_root / "logs" / log_subdir,
        "hpo",
        study_id,
        logger,
    )

    hpo_cfg_path = project_root / args.config
    hpo_cfg = load_yaml_config(str(hpo_cfg_path))

    pipeline_cfg = load_yaml_config(str(project_root / f"configs/pipelines/{args.pipeline}.yaml"))
    schema_cfg = load_schema_config_shared(
        project_root / pipeline_cfg["runtime"]["schema_mapping_json"]
    )
    target_col = pipeline_cfg.get("training", {}).get("target", "heart_disease")
    processed_dir = project_root / pipeline_cfg["paths"]["processed_dir"]
    output_dir = project_root / hpo_cfg.get("output_dir", f"output/{args.pipeline}/hpo")
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = args.datasets or pipeline_cfg["runtime"]["datasets"]
    model_types = args.model_types or list(hpo_cfg.get("grids", {}).keys())
    binning = args.binning

    scoring = hpo_cfg.get("scoring", "f1")
    cv_folds = int(hpo_cfg.get("cv_folds", 5))
    recall_floor = float(hpo_cfg.get("recall_hard_floor", 0.60))

    logger.info("=" * 70)
    logger.info("FAIRXAI HYPERPARAMETER OPTIMISATION")
    logger.info("=" * 70)
    logger.info(f"Pipeline : {args.pipeline}")
    logger.info(f"Datasets : {datasets}")
    logger.info(f"Models   : {model_types}")
    logger.info(f"Binning  : {binning}")
    logger.info(f"Scoring  : {scoring}")
    logger.info(f"CV folds : {cv_folds}")
    logger.info(f"Output   : {output_dir}")

    grids_cfg = hpo_cfg.get("grids", {})

    for dataset in datasets:
        logger.info(f"\n[DATASET] {dataset}")

        train_df = _load_processed_data(dataset, binning, processed_dir)
        exclude_cols = default_exclude_columns(schema_cfg, train_df.columns.tolist())
        feature_cols = [c for c in train_df.columns if c not in exclude_cols and c != target_col]

        if target_col not in train_df.columns:
            logger.error(f"Target column '{target_col}' not found in {dataset}. Skipping.")
            continue

        X_train = train_df[feature_cols]
        y_train = train_df[target_col]
        n_rows = len(X_train)
        logger.info(f"  n_train={n_rows}, n_features={len(feature_cols)}")

        for model_type in model_types:
            if model_type not in grids_cfg:
                logger.warning(f"  No HPO grid defined for '{model_type}'. Skipping.")
                continue

            out_path = output_dir / f"best_params_{dataset}_{model_type}.json"

            grid_entry = grids_cfg[model_type]
            param_grid = dict(grid_entry.get("params", {}))
            search_mode = grid_entry.get("search", "grid")
            n_iter = int(grid_entry.get("n_iter", 20))

            if args.dry_run:
                logger.info(f"  [DRY-RUN] Would search {model_type} on {dataset}: {param_grid}")
                continue

            logger.info(f"\n  [HPO] {model_type} on {dataset}")
            try:
                result = run_hpo(
                    model_type=model_type,
                    X_train=X_train,
                    y_train=y_train,
                    param_grid=param_grid,
                    base_params={"random_state": 42, "n_jobs": -1},
                    search=search_mode,
                    cv=cv_folds,
                    scoring=scoring,
                    n_iter=n_iter,
                    n_jobs=-1,
                    recall_hard_floor=recall_floor,
                )
                result["dataset"] = dataset
                save_hpo_results(result, out_path)
                logger.info(
                    f"  [OK] {model_type}/{dataset}: "
                    f"best_{scoring}={result['best_score']:.4f} → {out_path.name}"
                )
            except Exception as exc:
                logger.error(f"  [FAIL] {model_type}/{dataset}: {exc}")

    logger.info("\n[DONE] HPO complete.")
    update_output_study_pointer(
        project_root / f"output/{args.pipeline}",
        "hpo",
        study_id,
    )


if __name__ == "__main__":
    main()
