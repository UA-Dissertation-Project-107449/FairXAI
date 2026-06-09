#!/usr/bin/env python3
"""Pre-train clustering step: discover subgroups and write ``group_cluster``.

Runs BEFORE training (injected between preprocess and train in
``cardiac_pipeline.sh``).  Fits clustering on each dataset's TRAIN split only
and assigns the test split by nearest centroid (leakage guard), then persists
``group_cluster`` into the canonical train/test split CSVs.  The trainer and
mitigation already consume ``fairness.sensitive_attributes`` from the pipeline
config, so once ``group_cluster`` is in the splits it flows into CV
stratification and per-attribute mitigation automatically.

Idempotent: skips any dataset whose splits already carry ``group_cluster``.

Usage:
    python scripts/cardiac/cluster_subgroups.py --pipeline cardiac
    python scripts/cardiac/cluster_subgroups.py --datasets cleveland \\
        --methods kmeans hierarchical
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path (matches the pattern used by sibling scripts).
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from fairxai.cli.runner_base import get_project_root, setup_phase_logging
from fairxai.clustering.grouping_pipeline import DEFAULT_FEATURE_EXCLUDE, cluster_and_persist
from fairxai.experiments.data_io import resolve_default_binning
from fairxai.utils.config import load_yaml_config

logger = logging.getLogger(__name__)

_ROOT = get_project_root(Path(__file__))


def _resolve_methods(cli_methods, config) -> list[str]:
    if cli_methods:
        return [m.lower() for m in cli_methods]
    from_config = list((config.get("clustering_methods", {}) or {}).keys())
    return from_config or ["kmeans", "hierarchical", "dbscan", "gaussian_mixture"]


def _resolve_datasets(cli_datasets, pipeline_cfg) -> list[str]:
    if cli_datasets:
        return cli_datasets
    return list((pipeline_cfg.get("runtime", {}) or {}).get("datasets", []))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pre-train clustering subgroup discovery")
    p.add_argument("--pipeline", default="cardiac", help="Pipeline name (default: cardiac)")
    p.add_argument("--datasets", nargs="+", default=None, help="Dataset names (CLI override)")
    p.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="Clustering methods: kmeans hierarchical dbscan gaussian_mixture",
    )
    p.add_argument("--config", default=None, help="Path to clustering.yaml (optional override)")
    p.add_argument("-v", "--verbose", action="count", default=0, help="-v=info, -vv=debug")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pipeline = args.pipeline

    setup_phase_logging(
        _ROOT,
        "cluster_subgroups.log",
        verbose=args.verbose,
        stage_name="cluster",
    )

    pipeline_cfg_path = _ROOT / "configs" / "pipelines" / f"{pipeline}.yaml"
    pipeline_cfg = load_yaml_config(str(pipeline_cfg_path)) if pipeline_cfg_path.exists() else {}
    binning = resolve_default_binning(pipeline_cfg)
    processed_dir = _ROOT / "data" / "processed" / pipeline

    config_path = (
        Path(args.config)
        if args.config
        else (_ROOT / "configs" / "experiments" / "clustering.yaml")
    )
    config = load_yaml_config(str(config_path)) if config_path.exists() else {}

    methods = _resolve_methods(args.methods, config)
    method_cfg = {
        k: v for k, v in (config.get("clustering_methods", {}) or {}).items() if k in methods
    } or {k: {} for k in methods}

    feature_exclude = list(
        (config.get("data", {}) or {})
        .get("feature_selection", {})
        .get("exclude", DEFAULT_FEATURE_EXCLUDE)
    )

    datasets = _resolve_datasets(args.datasets, pipeline_cfg)
    if not datasets:
        logger.error("No datasets resolved. Pass --datasets or set runtime.datasets in the config.")
        sys.exit(1)

    # Validity gate (from the pipeline config grouping block). Rejects degenerate
    # clusterings; defaults are a no-op if unset.
    grouping_cfg = pipeline_cfg.get("grouping", {}) or {}
    min_clusters = int(grouping_cfg.get("min_clusters", 2))
    min_cluster_size_abs = int(grouping_cfg.get("min_cluster_size_abs", 1))
    min_cluster_size_frac = float(grouping_cfg.get("min_cluster_size_frac", 0.0))

    out_base = _ROOT / "output" / pipeline / "studies" / "grouping_pretrain"

    logger.info(
        "[PHASE] pre-train clustering started pipeline=%s datasets=%s methods=%s binning=%s",
        pipeline,
        datasets,
        methods,
        binning,
    )

    for dataset in datasets:
        try:
            cluster_and_persist(
                dataset=dataset,
                processed_dir=processed_dir,
                binning=binning,
                method_cfg=method_cfg,
                feature_exclude=feature_exclude,
                out_dir=out_base / dataset,
                min_clusters=min_clusters,
                min_cluster_size_abs=min_cluster_size_abs,
                min_cluster_size_frac=min_cluster_size_frac,
            )
        except Exception as exc:  # noqa: BLE001 — isolate per-dataset failures
            logger.error("[ERROR] cluster: dataset %s failed: %s", dataset, exc, exc_info=True)

    logger.info("[SUCCESS] pre-train clustering complete")


if __name__ == "__main__":
    main()
