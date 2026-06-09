#!/usr/bin/env python3
"""Post-mitigation age-binning fairness sensitivity sweep.

Runs AFTER mitigation (so the "after" regime exists) but degrades gracefully to
baseline-only when no mitigation predictions are present. For each dataset:

* **Axis B** — per baseline model, recompute per-age-bin fairness under several
  binning strategies → ``<run>/baseline/age_binning_sensitivity/<dataset>/<model>/``.
* **Axis A** — pair baseline ("before") vs each mitigated set ("after") per bin →
  ``<run>/baseline/age_binning_sensitivity/<dataset>/before_after/<regime>/``.

Analysis only — predictions are independent of the age binning, so nothing here
changes training. See the root design note ``AGE_BINNING_FAIRNESS_SENSITIVITY.md``.

Usage:
    python scripts/cardiac/age_binning_analysis.py --pipeline cardiac
    python scripts/cardiac/age_binning_analysis.py --datasets cleveland --run-id <id>
"""

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from fairxai.cli.runner_base import get_project_root, setup_phase_logging
from fairxai.cli.runner_utils import get_run_root, resolve_latest_run_dir
from fairxai.experiments import run_age_binning
from fairxai.utils.config import load_yaml_config

logger = logging.getLogger(__name__)

_ROOT = get_project_root(Path(__file__))

_DEFAULT_STRATEGIES = ["quantile_3", "fixed_5yr", "fixed_10yr", "equal_width_4"]


def _resolve_datasets(cli_datasets, pipeline_cfg) -> list[str]:
    if cli_datasets:
        return cli_datasets
    return list((pipeline_cfg.get("runtime", {}) or {}).get("datasets", []))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Post-mitigation age-binning fairness sweep")
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
        _ROOT, "age_binning_analysis.log", verbose=args.verbose, stage_name="agebin"
    )

    cfg_path = (
        Path(args.config) if args.config else (_ROOT / "configs" / "pipelines" / f"{pipeline}.yaml")
    )
    pipeline_cfg = load_yaml_config(str(cfg_path)) if cfg_path.exists() else {}

    sweep_cfg = pipeline_cfg.get("age_binning_sensitivity", {}) or {}
    strategies = list(sweep_cfg.get("strategies", _DEFAULT_STRATEGIES))
    k = int(sweep_cfg.get("k", 5))
    min_bin_size = int(sweep_cfg.get("min_bin_size", 20))

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

    out_base = run_root / "baseline" / "age_binning_sensitivity"
    logger.info(
        "[PHASE] age-binning sweep started run=%s datasets=%s strategies=%s k=%s",
        run_root.name,
        datasets,
        strategies,
        k,
    )

    for dataset in datasets:
        try:
            summary = run_age_binning(
                run_root=run_root,
                dataset=dataset,
                strategies=strategies,
                out_base=out_base,
                k=k,
                min_bin_size=min_bin_size,
            )
            if summary is None:
                logger.info(
                    "[INFO] age-binning sweep: no baseline predictions for %s; skipped", dataset
                )
            else:
                logger.info(
                    "[SUCCESS] age-binning %s: axis_b_models=%s axis_a_regimes=%s",
                    dataset,
                    summary.get("axis_b_models"),
                    summary.get("axis_a_regimes"),
                )
        except Exception as exc:  # noqa: BLE001 — isolate per-dataset failures
            logger.error("[ERROR] age-binning: dataset %s failed: %s", dataset, exc, exc_info=True)

    logger.info("[SUCCESS] age-binning sweep complete: %s", out_base)


if __name__ == "__main__":
    main()
