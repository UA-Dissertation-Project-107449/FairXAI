#!/usr/bin/env python3
"""
Run clustering and similarity subgroup discovery (stage 4b).

Discovers latent patient subgroups via unsupervised clustering and evaluates
per-cluster fairness.  Writes ``group_cluster`` back to the processed CSV so
downstream stages (train, assess) treat clusters as a sensitive attribute.

Usage:
    python scripts/experiments/run_grouping_analysis.py --run-id latest
    python scripts/experiments/run_grouping_analysis.py --run-id 2026-04-10_run_01 \\
        --datasets cleveland --methods kmeans hierarchical
    RUN_GROUPING=false bash scripts/cardiac/cardiac_pipeline.sh  # skip entirely
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd

# Add src to path (matches pattern used in other scripts)
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from fairxai.cli.runner_base import get_project_root, setup_phase_logging
from fairxai.cli.runner_utils import get_run_root, resolve_run_id
from fairxai.clustering import (
    ClusteringEngine,
    ClusteringError,
    ClusterProfiler,
    FairnessPerCluster,
)
from fairxai.similarity import SimilarityEngine, ViolationDensityMapper
from fairxai.utils.config import load_yaml_config

logger = logging.getLogger(__name__)

_ROOT = get_project_root(Path(__file__))
_PROCESSED_DIR = _ROOT / "data" / "processed" / "cardiac"
_CLUSTERING_CONFIG = _ROOT / "configs" / "experiments" / "clustering.yaml"

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _resolve_datasets(cli_datasets: list[str], config: dict) -> list[str]:
    """Flag → config → discover from processed dir."""
    if cli_datasets:
        return cli_datasets
    from_config = (config.get("data", {}) or {}).get("datasets", [])
    if from_config:
        return list(from_config)
    # Auto-discover: processed files + split files in both flat and sub-dir layouts.
    found = {p.stem.replace("_processed", "") for p in _PROCESSED_DIR.glob("*_processed.csv")}
    found |= {p.stem.replace("_train", "") for p in _PROCESSED_DIR.glob("*_train.csv")}
    for dataset_dir in _PROCESSED_DIR.iterdir():
        if not dataset_dir.is_dir():
            continue
        candidate = dataset_dir / f"{dataset_dir.name}_train.csv"
        if candidate.exists():
            found.add(dataset_dir.name)
    if found:
        discovered = sorted(found)
        logger.info("Auto-discovered datasets: %s", discovered)
        return discovered
    return []


def _resolve_methods(cli_methods: list[str], config: dict) -> list[str]:
    """Flag → config keys → all four."""
    if cli_methods:
        return [m.lower() for m in cli_methods]
    from_config = list((config.get("clustering_methods", {}) or {}).keys())
    return from_config or ["kmeans", "hierarchical", "dbscan", "gaussian_mixture"]


def _load_predictions(run_root: Path, dataset: str) -> pd.DataFrame | None:
    """Load baseline predictions for one model, preferring train+test pair files."""
    results_dir = run_root / "baseline" / "results"
    if not results_dir.exists():
        return None

    def _model_key(path: Path, split_suffix: str) -> str | None:
        stem = path.stem
        prefix = f"{dataset}_"
        if not stem.startswith(prefix) or not stem.endswith(split_suffix):
            return None
        return stem[len(prefix) : -len(split_suffix)]

    train_files = sorted(results_dir.glob(f"{dataset}_*_train_predictions.csv"))
    test_files = sorted(results_dir.glob(f"{dataset}_*_test_predictions.csv"))

    train_by_model = {
        model: path
        for path in train_files
        if (model := _model_key(path, "_train_predictions")) is not None
    }
    test_by_model = {
        model: path
        for path in test_files
        if (model := _model_key(path, "_test_predictions")) is not None
    }

    common_models = sorted(set(train_by_model) & set(test_by_model))
    if common_models:
        model = common_models[0]
        train_df = pd.read_csv(train_by_model[model])
        test_df = pd.read_csv(test_by_model[model])
        logger.info(
            "  Loaded baseline predictions for model=%s (train=%d, test=%d)",
            model,
            len(train_df),
            len(test_df),
        )
        return pd.concat([train_df, test_df], ignore_index=True)

    merged_candidates = [
        p
        for p in sorted(results_dir.glob(f"{dataset}_*_predictions.csv"))
        if "_train_predictions" not in p.name and "_test_predictions" not in p.name
    ]
    if merged_candidates:
        logger.info("  Loaded merged prediction file: %s", merged_candidates[0].name)
        return pd.read_csv(merged_candidates[0])

    return None


def _load_grouping_dataframe(dataset: str) -> tuple[pd.DataFrame, Path, dict[str, Any]]:
    """Resolve grouping input from processed file or from train/test split files."""
    processed_path = _PROCESSED_DIR / f"{dataset}_processed.csv"
    if processed_path.exists():
        logger.info("  Using processed input: %s", processed_path)
        return pd.read_csv(processed_path), processed_path, {"source": "processed"}

    split_candidates = [
        (
            _PROCESSED_DIR / dataset / f"{dataset}_train.csv",
            _PROCESSED_DIR / dataset / f"{dataset}_test.csv",
        ),
        (
            _PROCESSED_DIR / f"{dataset}_train.csv",
            _PROCESSED_DIR / f"{dataset}_test.csv",
        ),
    ]

    for train_path, test_path in split_candidates:
        if not train_path.exists() or not test_path.exists():
            continue
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        logger.info(
            "  Using split inputs: train=%s, test=%s",
            train_path,
            test_path,
        )
        merged_df = pd.concat([train_df, test_df], ignore_index=True)
        return (
            merged_df,
            processed_path,
            {
                "source": "splits",
                "train_path": train_path,
                "test_path": test_path,
                "train_rows": len(train_df),
            },
        )

    raise FileNotFoundError(
        f"No processed or split files found for dataset '{dataset}' under {_PROCESSED_DIR}"
    )


def _persist_group_cluster(
    df: pd.DataFrame,
    processed_path: Path,
    source_meta: dict[str, Any],
) -> None:
    """Persist group_cluster to canonical processed file and source split files when relevant."""
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_path, index=False)
    logger.info("[SUCCESS] group_cluster written to %s", processed_path.name)

    if source_meta.get("source") != "splits":
        return

    train_path = source_meta["train_path"]
    test_path = source_meta["test_path"]
    train_rows = int(source_meta["train_rows"])

    df.iloc[:train_rows].to_csv(train_path, index=False)
    df.iloc[train_rows:].to_csv(test_path, index=False)
    logger.info(
        "[SUCCESS] group_cluster updated split files: %s and %s", train_path.name, test_path.name
    )


def _report(label: str, result) -> None:
    if result is not None:
        logger.info("[SUCCESS] %s", label)
    else:
        logger.info("[INFO] %s: skipped (no output produced)", label)


# ------------------------------------------------------------------
# Per-dataset pipeline
# ------------------------------------------------------------------


def run_dataset(
    dataset: str,
    run_root: Path,
    output_dir: Path,
    config: dict,
    methods: list[str],
) -> None:
    logger.info("[PHASE] grouping — dataset: %s", dataset)

    try:
        df, processed_path, source_meta = _load_grouping_dataframe(dataset)
    except FileNotFoundError as exc:
        logger.warning("[WARNING] %s", exc)
        return

    logger.info("  Loaded %d samples from %s", len(df), processed_path.name)

    ds_out = output_dir / dataset
    ds_out.mkdir(parents=True, exist_ok=True)

    # Feature selection from config
    feat_exclude = list(
        (config.get("data", {}) or {})
        .get("feature_selection", {})
        .get("exclude", ["heart_disease", "age_group", "sex", "ethnicity", "group_cluster"])
    )

    # Build method config — restrict to requested methods
    method_cfg = {
        k: v for k, v in (config.get("clustering_methods", {}) or {}).items() if k in methods
    }

    # -- 1. Clustering engine ------------------------------------------
    logger.info("[PHASE] clustering")
    try:
        engine = ClusteringEngine(config=method_cfg, feature_exclude=feat_exclude)
        cluster_result = engine.fit(df)
        engine.save_diagnostics(cluster_result, ds_out)

        # Write cluster_assignments.csv
        assignments = cluster_result.to_assignments_df()
        assignments.to_csv(ds_out / "cluster_assignments.csv")
        logger.info("[SUCCESS] cluster_assignments.csv — %d clusters", cluster_result.n_clusters)

        # Persist group_cluster to canonical processed CSV and split sources.
        df["group_cluster"] = cluster_result.group_cluster.values
        _persist_group_cluster(df, processed_path, source_meta)

    except ClusteringError as exc:
        logger.error("clustering failed for %s: %s", dataset, exc)
        return  # Can't proceed without cluster labels

    # -- 2. Per-cluster fairness (requires predictions) ----------------
    logger.info("[PHASE] cluster fairness")
    pred_df = _load_predictions(run_root, dataset)
    if pred_df is not None:
        # Merge cluster labels into predictions for fairness analysis
        merged = pred_df.copy()
        if "group_cluster" not in merged.columns:
            # Align by index position (predictions have same rows as processed)
            if len(merged) == len(df):
                merged["group_cluster"] = df["group_cluster"].values
            else:
                logger.warning(
                    "[WARNING] cluster fairness: row count mismatch (%d vs %d), skipping",
                    len(merged),
                    len(df),
                )
                pred_df = None

    if pred_df is not None:
        sensitive_attrs = [
            c for c in ["age_group", "age_group_cat", "sex", "sex_cat"] if c in merged.columns
        ]
        fpc = FairnessPerCluster(sensitive_attrs=sensitive_attrs)
        fairness_df = fpc.compute(merged, cluster_col="group_cluster")
        _report("fairness_by_cluster", fairness_df if not fairness_df.empty else None)
        fairness_df.to_csv(ds_out / "fairness_by_cluster.csv", index=False)

        corr_df = fpc.cramers_v_matrix(merged, cluster_col="group_cluster")
        _report("correlation_matrix (Cramér's V)", corr_df if not corr_df.empty else None)
        corr_df.to_csv(ds_out / "correlation_matrix.csv", index=False)
    else:
        logger.info(
            "[INFO] cluster fairness: no compatible prediction CSV found for %s "
            "(run stage 5 first). Skipping per-cluster metrics.",
            dataset,
        )

    # -- 3. Cluster profiles -------------------------------------------
    logger.info("[PHASE] cluster profiles")
    target_col = "heart_disease"
    if target_col not in df.columns:
        # Try to find target col from config or fallback
        target_col = next(
            (c for c in df.columns if "disease" in c.lower() or "target" in c.lower()),
            df.columns[-1],
        )
    profiler = ClusterProfiler(target_col=target_col)
    report = profiler.compute(df, cluster_col="group_cluster")
    profiler.save_report(report, ds_out / "subgroup_profiles.md")

    # -- 4. Similarity (k-NN individual fairness) ----------------------
    logger.info("[PHASE] similarity")
    k_values = list(
        (config.get("fairness_analysis", {}) or {})
        .get("similarity_based_fairness", {})
        .get("parameters", {})
        .get("k", [5, 10, 20])
    )
    numeric_cols = [
        c
        for c in df.select_dtypes(include="number").columns
        if c not in feat_exclude and c != "group_cluster"
    ]

    pred_col = "y_pred"
    sim_df = (
        df
        if pred_col in df.columns
        else (merged if (pred_df is not None and pred_col in merged.columns) else None)
    )

    if sim_df is not None and pred_col in sim_df.columns:
        sim_engine = SimilarityEngine(k_values=k_values, pred_col=pred_col)
        sim_result = sim_engine.compute(sim_df, feature_cols=numeric_cols)
        _report("similarity_fairness_scores", sim_result.rows if sim_result.rows else None)
        sim_engine.save_scores(sim_result, ds_out)

        # Violation density map (requires predictions + features)
        mapper = ViolationDensityMapper(k=min(k_values))
        map_result = mapper.compute(
            sim_df,
            feature_cols=numeric_cols,
            pred_col=pred_col,
            output_file=ds_out / "violation_density_map.png",
            similarity_engine=sim_engine,
        )
        _report("violation_density_map", map_result.output_file)
    else:
        logger.info(
            "[INFO] similarity: no %r column found (run stage 5 first). Skipping.", pred_col
        )

    logger.info("[SUCCESS] grouping complete — %s", ds_out)


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Grouping & similarity subgroup discovery (stage 4b)")
    p.add_argument("--run-id", default=None, help="Run ID or 'latest'")
    p.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Dataset names (CLI override). Precedence: flag > config > auto-discover.",
    )
    p.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="Clustering methods to use: kmeans hierarchical dbscan gaussian_mixture",
    )
    p.add_argument("--config", default=None, help="Path to clustering.yaml (optional override)")
    p.add_argument("-v", action="store_true", help="Verbose logging")
    p.add_argument("-vv", action="store_true", help="Debug logging")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    verbose = 2 if args.vv else (1 if args.v else 0)
    setup_phase_logging(
        _ROOT,
        "grouping.log",
        verbose=verbose,
        run_id=args.run_id,
        stage_name="preprocess",
        sub_stage="grouping",
    )

    # Load config
    config_path = Path(args.config) if args.config else _CLUSTERING_CONFIG
    if not config_path.exists():
        logger.warning("[WARNING] clustering.yaml not found at %s; using defaults", config_path)
        config = {}
    else:
        config = load_yaml_config(str(config_path))

    # Resolve run root
    run_id = resolve_run_id(args.run_id)
    run_root = get_run_root(_ROOT / "output" / "cardiac", run_id)
    output_dir = run_root / "grouping"

    datasets = _resolve_datasets(args.datasets or [], config)
    methods = _resolve_methods(args.methods or [], config)

    if not datasets:
        logger.error("No datasets found. Pass --datasets or ensure processed CSVs exist.")
        sys.exit(1)

    logger.info(
        "[PHASE] grouping analysis — run_id=%s datasets=%s methods=%s", run_id, datasets, methods
    )

    for dataset in datasets:
        try:
            run_dataset(dataset, run_root, output_dir, config, methods)
        except Exception as exc:
            logger.error("dataset %s failed: %s", dataset, exc, exc_info=True)

    logger.info("[SUCCESS] done — artifacts in %s", output_dir)


if __name__ == "__main__":
    main()
