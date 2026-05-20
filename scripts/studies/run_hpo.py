"""Hyperparameter optimisation (HPO) orchestration script.

Runs GridSearchCV / RandomizedSearchCV for each requested model x dataset
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

from fairxai.cli.memory_utils import safe_n_jobs, warn_if_large_dataset
from fairxai.cli.runner_base import get_project_root, setup_study_logging
from fairxai.cli.runner_utils import (
    resolve_run_id,
    update_output_study_pointer,
    update_study_pointer,
)
from fairxai.experiments.data_io import (
    default_exclude_columns,
)
from fairxai.experiments.data_io import load_schema_config as load_schema_config_shared
from fairxai.experiments.data_io import (
    resolve_dataset_dir,
    resolve_default_binning,
)
from fairxai.training.grid_search import run_hpo, save_hpo_results
from fairxai.utils.config import load_yaml_config


def _load_processed_data(dataset: str, binning: str, processed_dir: Path):
    data_dir = resolve_dataset_dir(processed_dir, dataset, binning)
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
        default=None,
        help="Binning strategy for processed data (default: runtime.default_binning in pipeline config)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would run without fitting",
    )
    parser.add_argument(
        "--search-n-jobs",
        type=int,
        default=None,
        help="Parallel workers for GridSearch/RandomizedSearch (default: config or -1).",
    )
    parser.add_argument(
        "--model-n-jobs",
        type=int,
        default=None,
        help="n_jobs passed to underlying model when supported (default: config or 1).",
    )
    parser.add_argument(
        "--max-rows-for-rbf-svm",
        type=int,
        default=None,
        help="Maximum rows to keep RBF kernel in SVM HPO grid (default: hpo.yaml).",
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
    schema_cfg = load_schema_config_shared(project_root, args.pipeline)
    target_col = pipeline_cfg.get("training", {}).get("target", "heart_disease")
    processed_dir = project_root / pipeline_cfg["paths"]["processed_dir"]
    hpo_output_root = project_root / hpo_cfg.get(
        "output_dir", f"output/{args.pipeline}/studies/hpo"
    )
    output_dir = hpo_output_root / study_id
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = args.datasets or pipeline_cfg["runtime"]["datasets"]
    model_types = args.model_types or list(hpo_cfg.get("grids", {}).keys())
    binning = args.binning or resolve_default_binning(pipeline_cfg)

    scoring = hpo_cfg.get("scoring", "f1")
    cv_folds = int(hpo_cfg.get("cv_folds", 5))
    recall_floor = float(hpo_cfg.get("recall_hard_floor", 0.60))

    sched_cfg = pipeline_cfg.get("scheduling") or {}
    warn_rows_threshold = int(sched_cfg.get("warn_rows_threshold", 50_000))
    max_memory_fraction = float(sched_cfg.get("max_memory_fraction", 0.80))

    search_n_jobs = (
        int(args.search_n_jobs)
        if args.search_n_jobs is not None
        else int(hpo_cfg.get("search_n_jobs", -1))
    )
    model_n_jobs = (
        int(args.model_n_jobs)
        if args.model_n_jobs is not None
        else int(hpo_cfg.get("model_n_jobs", 1))
    )
    max_rows_for_rbf_svm = (
        int(args.max_rows_for_rbf_svm)
        if args.max_rows_for_rbf_svm is not None
        else int(hpo_cfg.get("max_rows_for_rbf_svm", 5000))
    )

    if search_n_jobs <= 0 and search_n_jobs != -1:
        logger.warning(
            "search_n_jobs=%d is invalid; using 1 (only -1 or positive integers are supported)",
            search_n_jobs,
        )
        search_n_jobs = 1
    if model_n_jobs <= 0 and model_n_jobs != -1:
        logger.warning(
            "model_n_jobs=%d is invalid; using 1 (only -1 or positive integers are supported)",
            model_n_jobs,
        )
        model_n_jobs = 1
    if max_rows_for_rbf_svm <= 0:
        logger.warning("max_rows_for_rbf_svm<=0 is invalid; using 5000")
        max_rows_for_rbf_svm = 5000

    logger.info("[PHASE] HPO study started")
    logger.info(
        f"[RUN_CONTEXT] pipeline={args.pipeline} study_id={study_id} datasets={datasets} "
        f"models={model_types} binning={binning} scoring={scoring} cv_folds={cv_folds} "
        f"search_n_jobs={search_n_jobs} model_n_jobs={model_n_jobs} "
        f"rbf_max_rows={max_rows_for_rbf_svm} output_dir={output_dir} dry_run={args.dry_run}"
    )

    grids_cfg = hpo_cfg.get("grids", {})

    for dataset in datasets:
        logger.info(f"[DATASET] name={dataset}")

        train_df = _load_processed_data(dataset, binning, processed_dir)
        exclude_cols = default_exclude_columns(
            schema_cfg,
            dataset,
            target=target_col,
            sensitive_attrs=pipeline_cfg.get("fairness", {}).get("sensitive_attributes", []),
        )
        feature_cols = [c for c in train_df.columns if c not in exclude_cols and c != target_col]

        if target_col not in train_df.columns:
            logger.error(f"Target column '{target_col}' not found in {dataset}. Skipping.")
            continue

        X_train = train_df[feature_cols]
        y_train = train_df[target_col]
        n_rows = len(X_train)
        n_cols = len(feature_cols)
        logger.info(f"  n_train={n_rows}, n_features={n_cols}")

        warn_if_large_dataset(n_rows, warn_rows_threshold, context=f"dataset={dataset}")
        effective_search_n_jobs = safe_n_jobs(
            n_rows, n_cols, search_n_jobs, cv_folds, max_memory_fraction
        )

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
                logger.info(
                    f"  [DRY_RUN] dataset={dataset} model={model_type} "
                    f"search={search_mode} params={param_grid}"
                )
                continue

            logger.info(
                f"  [HPO] dataset={dataset} model={model_type} "
                f"search={search_mode} n_iter={n_iter}"
            )
            try:
                result = run_hpo(
                    model_type=model_type,
                    X_train=X_train,
                    y_train=y_train,
                    param_grid=param_grid,
                    base_params={"random_state": 42, "n_jobs": model_n_jobs},
                    search=search_mode,
                    cv=cv_folds,
                    scoring=scoring,
                    n_iter=n_iter,
                    n_jobs=effective_search_n_jobs,
                    recall_hard_floor=recall_floor,
                    max_rows_for_rbf_svm=max_rows_for_rbf_svm,
                )
                result["dataset"] = dataset
                save_hpo_results(result, out_path)
                logger.info(
                    f"  [SUCCESS] dataset={dataset} model={model_type} "
                    f"best_{scoring}={result['best_score']:.4f} output={out_path.name}"
                )
            except Exception as exc:
                logger.error(f"  [FAILED] dataset={dataset} model={model_type} error={exc}")

    logger.info("[PHASE] HPO study complete")
    default_hpo_root = project_root / f"output/{args.pipeline}/studies/hpo"
    if hpo_output_root == default_hpo_root:
        update_output_study_pointer(
            project_root / f"output/{args.pipeline}",
            "hpo",
            study_id,
        )
    else:
        hpo_output_root.mkdir(parents=True, exist_ok=True)
        (hpo_output_root / "latest.txt").write_text(study_id, encoding="utf-8")
        logger.info("Updated latest HPO pointer: %s", hpo_output_root / "latest.txt")


if __name__ == "__main__":
    main()
