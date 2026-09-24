#!/usr/bin/env python3
"""
Run clustering and similarity subgroup discovery (standalone study).

Discovers latent patient subgroups via unsupervised clustering and evaluates
per-cluster fairness.  Writes ``group_cluster`` back to the processed CSV so
subsequent pipeline runs treat clusters as a sensitive attribute.

Output: output/<pipeline>/studies/grouping/<study_id>/

Usage:
    python scripts/studies/run_grouping_analysis.py
    python scripts/studies/run_grouping_analysis.py \\
        --datasets cleveland --methods kmeans hierarchical
    # Optionally load baseline predictions for per-cluster fairness:
    python scripts/studies/run_grouping_analysis.py --run-id <pipeline_run_id>
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd

# Add src to path (matches pattern used in other scripts)
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from fairxai.cli.runner_base import get_project_root, setup_study_logging
from fairxai.cli.runner_utils import (
    get_run_root,
    get_study_root,
    resolve_latest_run_dir,
    resolve_run_id,
    update_output_study_pointer,
)
from fairxai.clustering import (
    ClusteringEngine,
    ClusteringError,
    ClusterProfiler,
    FairnessPerCluster,
)
from fairxai.experiments.data_io import resolve_dataset_dir, resolve_default_binning
from fairxai.similarity import run_similarity_for_predictions
from fairxai.utils.config import load_yaml_config

logger = logging.getLogger(__name__)

_ROOT = get_project_root(Path(__file__))
_PROCESSED_DIR = _ROOT / "data" / "processed" / "cardiac"
_CLUSTERING_CONFIG = _ROOT / "configs" / "experiments" / "clustering.yaml"

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _resolve_datasets(cli_datasets: list[str], config: dict) -> list[str]:
    """Flag to config to discover from processed dir."""
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
    """Flag to config keys to all four."""
    if cli_methods:
        return [m.lower() for m in cli_methods]
    from_config = list((config.get("clustering_methods", {}) or {}).keys())
    return from_config or ["kmeans", "hierarchical", "dbscan", "gaussian_mixture"]


def _load_predictions(run_root: Path, dataset: str) -> pd.DataFrame | None:
    """Load baseline predictions for one model, preferring train+test pair files.

    Checks both ``baseline/results/predictions/`` (current layout) and the flat
    ``baseline/results/`` directory (legacy fallback) to remain compatible with
    older runs.  Filenames follow the pattern ``{dataset}_{model}_{train|test}.csv``
    (current) or ``{dataset}_{model}_{train|test}_predictions.csv`` (legacy).
    """

    def _model_key(path: Path, split_suffix: str) -> str | None:
        stem = path.stem
        prefix = f"{dataset}_"
        if not stem.startswith(prefix) or not stem.endswith(split_suffix):
            return None
        return stem[len(prefix) : -len(split_suffix)]

    def _try_load(results_dir: Path, train_suffix: str, test_suffix: str) -> pd.DataFrame | None:
        if not results_dir.exists():
            return None
        train_files = sorted(results_dir.glob(f"{dataset}_*{train_suffix}.csv"))
        test_files = sorted(results_dir.glob(f"{dataset}_*{test_suffix}.csv"))
        train_by_model = {
            model: path
            for path in train_files
            if (model := _model_key(path, train_suffix)) is not None
        }
        test_by_model = {
            model: path
            for path in test_files
            if (model := _model_key(path, test_suffix)) is not None
        }
        common_models = sorted(set(train_by_model) & set(test_by_model))
        if common_models:
            model = common_models[0]
            train_df = pd.read_csv(train_by_model[model])
            test_df = pd.read_csv(test_by_model[model])
            logger.info(
                "  Loaded baseline predictions for model=%s (train=%d, test=%d) from %s",
                model,
                len(train_df),
                len(test_df),
                results_dir.name,
            )
            return pd.concat([train_df, test_df], ignore_index=True)
        return None

    # Current layout: baseline/results/predictions/{dataset}_{model}_{train|test}.csv
    result = _try_load(run_root / "baseline" / "results" / "predictions", "_train", "_test")
    if result is not None:
        return result

    # Legacy layout: baseline/results/{dataset}_{model}_{train|test}_predictions.csv
    result = _try_load(run_root / "baseline" / "results", "_train_predictions", "_test_predictions")
    if result is not None:
        return result

    # Last-resort: merged prediction file
    for results_dir in [
        run_root / "baseline" / "results" / "predictions",
        run_root / "baseline" / "results",
    ]:
        if not results_dir.exists():
            continue
        merged_candidates = [
            p
            for p in sorted(results_dir.glob(f"{dataset}_*_predictions.csv"))
            if "_train_predictions" not in p.name and "_test_predictions" not in p.name
        ]
        if merged_candidates:
            logger.info("  Loaded merged prediction file: %s", merged_candidates[0].name)
            return pd.read_csv(merged_candidates[0])

    return None


def _load_grouping_dataframe(
    dataset: str, binning: str
) -> tuple[pd.DataFrame, Path, dict[str, Any]]:
    """Resolve grouping input, preferring the canonical train/test splits.

    Splits are preferred over a flat ``{dataset}_processed.csv`` so the clustered
    row set matches the baseline predictions used for per-cluster fairness (same
    rows, same train-then-test order).  ``{dataset}_processed.csv`` is a
    last-resort fallback and may be a stale, pre-cleaning row set.
    """
    processed_path = _PROCESSED_DIR / f"{dataset}_processed.csv"

    canonical_dir = resolve_dataset_dir(_PROCESSED_DIR, dataset, binning)
    train_path = canonical_dir / f"{dataset}_train.csv"
    test_path = canonical_dir / f"{dataset}_test.csv"

    if train_path.exists() and test_path.exists():
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        logger.info("  Using split inputs: train=%s, test=%s", train_path, test_path)
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

    if processed_path.exists():
        logger.info("  Using processed input (fallback): %s", processed_path)
        return pd.read_csv(processed_path), processed_path, {"source": "processed"}

    raise FileNotFoundError(
        f"No split or processed files found for dataset '{dataset}' under {_PROCESSED_DIR}"
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


def _expand_exclusions(df: pd.DataFrame, feat_exclude: list[str]) -> list[str]:
    """Add the encoded forms of each excluded column to the exclusion list.

    The exclusion list is written in terms of the source attributes -- ``sex``,
    ``age_group`` -- but preprocessing keeps a string column for fairness
    grouping and adds a model-usable numeric encoding beside it (``sex_bin``,
    ``sex_extended``, ``age_group_idx``). Only the string column matches the
    list by name, so the numeric encoding of every sensitive attribute survived
    the filter and was clustered on. That undercuts what the clusters are for:
    subgroups that the declared sensitive attributes do not already describe.

    The convention across preprocessing is ``<attr>_<encoding>``, so an exact
    match or a match up to an underscore catches the encodings without catching
    an unrelated column that merely starts with the same letters.
    """
    expanded = set(feat_exclude)
    for col in df.columns:
        for name in feat_exclude:
            if col == name or col.startswith(f"{name}_"):
                expanded.add(col)
                break
    return sorted(expanded)


def _training_feature_space(
    dataset: str, binning: str, df: pd.DataFrame, feat_exclude: list[str]
) -> tuple[list[str], dict[str, Any]]:
    """Columns to cluster on: the ones the models were actually trained on.

    The grouping study reads the *unscaled* splits, because cluster profiles have
    to be readable in clinical units. Those splits still carry the raw UCI
    missingness that preprocessing resolved, and preprocessing resolved it two
    ways: it dropped the columns that were missing too often to keep, and imputed
    what remained. Clustering a feature space the models never saw is wrong on its
    own terms -- the clusters are meant to explain model behaviour -- and on
    four_site_uci it was also fatal, because ``ca`` (609/918 missing), ``thal``,
    ``slope`` and ``chol`` survive only in the unscaled file and every sklearn
    method rejects NaN. That dataset produced no clusters at all until this.

    The scaled split is the record of what training kept, so it is the reference
    here rather than a hard-coded list: the set is per dataset (cleveland_uci
    keeps ``ca``/``thal``/``slope``, four_site_uci does not) and it moves whenever
    preprocessing changes.
    """
    scaled_dir = resolve_dataset_dir(_PROCESSED_DIR, dataset, binning)
    scaled_path = scaled_dir / f"{dataset}_train_scaled.csv"
    if not scaled_path.exists():
        logger.warning(
            "  No scaled split at %s; clustering on every numeric column instead.",
            scaled_path,
        )
        return [], {"aligned": False}

    trained_cols = set(pd.read_csv(scaled_path, nrows=0).columns)
    candidates = ClusteringEngine(feature_exclude=feat_exclude)._resolve_feature_cols(df, None)
    aligned = [c for c in candidates if c in trained_cols]
    dropped = [c for c in candidates if c not in trained_cols]
    if not aligned:
        logger.warning(
            "  No clustering feature survived alignment to %s; falling back.", scaled_path.name
        )
        return [], {"aligned": False}
    if dropped:
        logger.info("  Dropped %d column(s) absent from training: %s", len(dropped), dropped)
    return aligned, {"aligned": True, "dropped": dropped, "features": aligned}


def _impute_like_training(df: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, int]:
    """Return a copy with median-filled NaN in the clustering features.

    Preprocessing imputed these same columns before training -- the scaled split
    has no missing values -- so leaving them NaN here would cluster a different
    row set than the baseline predictions this study pairs clusters against.
    Dropping the rows instead would break that row-for-row correspondence, which
    is the whole reason the splits are preferred over the flat processed file.

    The fill goes on a copy and never on the caller's frame. This study writes
    its ``group_cluster`` labels back into the processed file and the source
    splits, so imputing in place silently replaced observed missingness in the
    inputs with fitted values -- a study recording its own preprocessing into the
    data every other stage reads. The cluster profiles are also built from the
    caller's frame, and they should describe what was measured.
    """
    resid = df[feature_cols].isna()
    n_rows = int(resid.any(axis=1).sum())
    if not n_rows:
        return df, 0
    filled = df.copy()
    for col in feature_cols:
        if resid[col].any():
            filled[col] = filled[col].fillna(filled[col].median())
    logger.info("  Median-imputed residual NaN in %d row(s) before clustering.", n_rows)
    return filled, n_rows


def run_dataset(
    dataset: str,
    run_root: Path,
    output_dir: Path,
    config: dict,
    methods: list[str],
    binning: str,
) -> None:
    logger.info("[DATASET] grouping dataset=%s", dataset)

    try:
        df, processed_path, source_meta = _load_grouping_dataframe(dataset, binning)
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
    # Both phases below are meant to work on the non-sensitive feature space, so
    # the encoded forms of the excluded attributes have to go too.
    feat_exclude = _expand_exclusions(df, feat_exclude)

    # Build method config - restrict to requested methods
    method_cfg = {
        k: v for k, v in (config.get("clustering_methods", {}) or {}).items() if k in methods
    }

    # -- 1. Clustering engine ------------------------------------------
    # Idempotency guard: when group_cluster is already present (written by the
    # pre-train cluster step, fit on TRAIN only), do NOT re-cluster here. This
    # study fits on the concatenated train+test frame, which would overwrite the
    # leakage-safe pre-train labels with leaky ones.
    if "group_cluster" in df.columns:
        logger.info(
            "[PHASE] clustering — SKIPPED (group_cluster already present; "
            "reusing pre-train labels to avoid a leaky re-fit)"
        )
    else:
        logger.info("[PHASE] clustering")
        try:
            feature_cols, align_meta = _training_feature_space(dataset, binning, df, feat_exclude)
            cluster_input = df
            if feature_cols:
                cluster_input, imputed_rows = _impute_like_training(df, feature_cols)
                align_meta["imputed_rows"] = imputed_rows
            # The feature space is a result, not an implementation detail: cluster
            # profiles are only interpretable against the columns they were built
            # from, and which columns survived alignment differs per dataset.
            (ds_out / "clustering_features.json").write_text(json.dumps(align_meta, indent=2))
            engine = ClusteringEngine(config=method_cfg, feature_exclude=feat_exclude)
            cluster_result = engine.fit(cluster_input, feature_cols=feature_cols or None)
            engine.save_diagnostics(cluster_result, ds_out)

            # Record what won and how much of the cohort it set aside as noise.
            # The silhouette alone does not say whether a solution covers the
            # cohort or only the part of it that clustered cleanly.
            align_meta["selected_clustering"] = {
                "method": cluster_result.method,
                "n_clusters": cluster_result.n_clusters,
                "silhouette": round(cluster_result.silhouette, 4),
                "n_noise": cluster_result.n_noise,
                "noise_fraction": round(cluster_result.noise_fraction, 4),
            }
            (ds_out / "clustering_features.json").write_text(json.dumps(align_meta, indent=2))

            # Write cluster_assignments.csv
            assignments = cluster_result.to_assignments_df()
            assignments.to_csv(ds_out / "cluster_assignments.csv")
            logger.info("[SUCCESS] cluster_assignments.csv clusters=%d", cluster_result.n_clusters)

            # Persist group_cluster to canonical processed CSV and split sources.
            df["group_cluster"] = cluster_result.group_cluster.values
            _persist_group_cluster(df, processed_path, source_meta)

        except ClusteringError as exc:
            # Diagnostics-on-failure: even when no valid solution emerges, persist
            # the per-candidate diagnostics so the failed run is still auditable.
            if getattr(exc, "diagnostics", None):
                diag_path = ds_out / "cluster_diagnostics.csv"
                pd.DataFrame([d.to_dict() for d in exc.diagnostics]).to_csv(diag_path, index=False)
                logger.info("[SUCCESS] cluster_diagnostics saved to %s", diag_path)
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
        _report("correlation_matrix (Cramers V)", corr_df if not corr_df.empty else None)
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
    # Delegates to the shared post-assess core so the study and the gated
    # pipeline step (scripts/cardiac/similarity_analysis.py) stay in lockstep
    # (scaled distance, per-sensitive-group breakdown, density map).
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
        sensitive_attrs = [
            c
            for c in ["age_group", "age_group_cat", "sex", "sex_cat", "ethnicity", "group_cluster"]
            if c in sim_df.columns
        ]
        summary = run_similarity_for_predictions(
            sim_df,
            feature_cols=numeric_cols,
            sensitive_attrs=sensitive_attrs,
            k_values=k_values,
            out_dir=ds_out,
            pred_col=pred_col,
        )
        _report("similarity_fairness_scores", summary)
    else:
        logger.info(
            "[INFO] similarity: no %r column found (run stage 5 first). Skipping.", pred_col
        )

    logger.info("[SUCCESS] grouping complete: output_dir=%s", ds_out)


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Grouping & similarity subgroup discovery (standalone study)"
    )
    p.add_argument(
        "--run-id",
        default=None,
        help="Optional: pipeline run ID to load baseline predictions for per-cluster fairness",
    )
    p.add_argument(
        "--pipeline",
        default="cardiac",
        help="Pipeline name (default: cardiac)",
    )
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
    p.add_argument(
        "--study-id",
        default=None,
        help="Override auto-generated study ID (useful for tests and reproducible runs)",
    )
    p.add_argument("-v", action="store_true", help="Verbose logging")
    p.add_argument("-vv", action="store_true", help="Debug logging")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    verbose = 2 if args.vv else (1 if args.v else 0)
    pipeline = args.pipeline
    study_id = args.study_id if args.study_id else resolve_run_id()
    setup_study_logging(
        _ROOT,
        "grouping",
        study_id,
        "grouping.log",
        verbose=verbose,
        log_subdir=pipeline,
    )

    # Load config
    config_path = Path(args.config) if args.config else _CLUSTERING_CONFIG
    if not config_path.exists():
        logger.warning("[WARNING] clustering.yaml not found at %s; using defaults", config_path)
        config = {}
    else:
        config = load_yaml_config(str(config_path))

    # Study output dir (not tied to a pipeline run)
    base_results = _ROOT / "output" / pipeline
    output_dir = get_study_root(base_results, "grouping", study_id)

    # Resolve a pipeline run root for baseline predictions.
    # Explicit --run-id takes precedence; otherwise auto-resolve the latest run.
    run_root: Path | None = None
    if args.run_id:
        run_root = get_run_root(base_results, args.run_id)
    else:
        run_root = resolve_latest_run_dir(base_results)
        if run_root:
            logger.info("[INFO] Auto-resolved run root: %s", run_root.name)
        else:
            logger.warning(
                "[WARNING] No pipeline run found under %s; per-cluster fairness will be skipped. "
                "Pass --run-id to specify a run explicitly.",
                base_results / "runs",
            )

    datasets = _resolve_datasets(args.datasets or [], config)
    methods = _resolve_methods(args.methods or [], config)

    # Canonical processed-layout binning (matches train/mitigation/hpo loaders),
    # so clustered rows match the baseline predictions for per-cluster fairness.
    pipeline_cfg_path = _ROOT / "configs" / "pipelines" / f"{pipeline}.yaml"
    default_binning = (
        resolve_default_binning(load_yaml_config(str(pipeline_cfg_path)))
        if pipeline_cfg_path.exists()
        else "fixed_10yr"
    )

    if not datasets:
        logger.error("No datasets found. Pass --datasets or ensure processed CSVs exist.")
        sys.exit(1)

    logger.info("[PHASE] grouping study started")
    logger.info(
        "[RUN_CONTEXT] pipeline=%s study_id=%s datasets=%s methods=%s output_dir=%s run_root=%s",
        pipeline,
        study_id,
        datasets,
        methods,
        output_dir,
        run_root if run_root else "none",
    )

    for dataset in datasets:
        try:
            run_dataset(
                dataset, run_root or output_dir, output_dir, config, methods, default_binning
            )
        except Exception as exc:
            logger.error("dataset %s failed: %s", dataset, exc, exc_info=True)

    update_output_study_pointer(base_results, "grouping", study_id)
    logger.info("[SUCCESS] grouping study complete: output_dir=%s", output_dir)


if __name__ == "__main__":
    main()
