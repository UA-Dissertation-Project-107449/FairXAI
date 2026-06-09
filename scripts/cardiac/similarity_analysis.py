#!/usr/bin/env python3
"""Post-assess similarity step: per-model individual-fairness analysis.

Runs AFTER training/assess (injected after stage 8 in ``cardiac_pipeline.sh``).
For every model of each dataset it computes k-NN prediction consistency (scaled
distance), a per-sensitive-group breakdown, and a violation-density map, writing
to ``<run>/baseline/individual_fairness/<dataset>/<model>/``.

Analysis only — it does not change training (unlike the pre-train cluster step).

Usage:
    python scripts/cardiac/similarity_analysis.py --pipeline cardiac
    python scripts/cardiac/similarity_analysis.py --datasets cleveland --run-id <id>
"""

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from fairxai.cli.runner_base import get_project_root, setup_phase_logging
from fairxai.cli.runner_utils import get_run_root, resolve_latest_run_dir
from fairxai.similarity import run_similarity
from fairxai.utils.config import load_yaml_config

logger = logging.getLogger(__name__)

_ROOT = get_project_root(Path(__file__))


def _resolve_datasets(cli_datasets, pipeline_cfg) -> list[str]:
    if cli_datasets:
        return cli_datasets
    return list((pipeline_cfg.get("runtime", {}) or {}).get("datasets", []))


def _expand_sensitive(configured: list[str]) -> list[str]:
    """Include decoded ``_cat`` siblings so prediction CSV columns are matched."""
    out: list[str] = []
    for s in configured:
        out.append(s)
        if not s.endswith("_cat"):
            out.append(f"{s}_cat")
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Post-assess similarity (individual fairness)")
    p.add_argument("--pipeline", default="cardiac", help="Pipeline name (default: cardiac)")
    p.add_argument("--datasets", nargs="+", default=None, help="Dataset names (CLI override)")
    p.add_argument("--run-id", default=None, help="Pipeline run ID (default: RUN_ID env / latest)")
    p.add_argument("--config", default=None, help="Path to pipeline yaml (optional override)")
    p.add_argument("-v", "--verbose", action="count", default=0, help="-v=info, -vv=debug")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pipeline = args.pipeline

    setup_phase_logging(
        _ROOT, "similarity_analysis.log", verbose=args.verbose, stage_name="similar"
    )

    cfg_path = (
        Path(args.config) if args.config else (_ROOT / "configs" / "pipelines" / f"{pipeline}.yaml")
    )
    pipeline_cfg = load_yaml_config(str(cfg_path)) if cfg_path.exists() else {}

    sim_cfg = pipeline_cfg.get("similarity", {}) or {}
    k_values = list(sim_cfg.get("k", [5, 10, 20]))
    configured_sensitive = list(
        (pipeline_cfg.get("fairness", {}) or {}).get("sensitive_attributes", ["age_group", "sex"])
    )
    sensitive_attrs = _expand_sensitive(configured_sensitive)

    base_results = _ROOT / "output" / pipeline
    run_id = args.run_id or os.environ.get("RUN_ID")
    run_root = (
        get_run_root(base_results, run_id) if run_id else resolve_latest_run_dir(base_results)
    )
    if not run_root:
        logger.error("No pipeline run found under %s. Pass --run-id or set RUN_ID.", base_results)
        sys.exit(1)

    datasets = _resolve_datasets(args.datasets, pipeline_cfg)
    if not datasets:
        logger.error("No datasets resolved. Pass --datasets or set runtime.datasets in the config.")
        sys.exit(1)

    out_base = run_root / "baseline" / "individual_fairness"
    logger.info(
        "[PHASE] similarity started run=%s datasets=%s k=%s sensitive=%s",
        run_root.name,
        datasets,
        k_values,
        sensitive_attrs,
    )

    for dataset in datasets:
        try:
            summary = run_similarity(
                run_root=run_root,
                dataset=dataset,
                sensitive_attrs=sensitive_attrs,
                k_values=k_values,
                out_base=out_base,
            )
            if summary is None:
                logger.info("[INFO] similarity: no predictions for %s; skipped", dataset)
        except Exception as exc:  # noqa: BLE001 — isolate per-dataset failures
            logger.error("[ERROR] similarity: dataset %s failed: %s", dataset, exc, exc_info=True)

    logger.info("[SUCCESS] similarity analysis complete: %s", out_base)


if __name__ == "__main__":
    main()
