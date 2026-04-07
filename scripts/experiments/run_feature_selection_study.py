"""Feature selection study — sensitive-attribute ablation.

Runs train_baseline.py for each combination of feature-selection mode × dataset × model,
collecting results under output/<pipeline>/feature_selection_study/.

Usage
-----
# All modes, all models, all configured datasets
python scripts/experiments/run_feature_selection_study.py --pipeline cardiac

# Single mode
python scripts/experiments/run_feature_selection_study.py --pipeline cardiac \
    --modes exclude_sensitive include_all_sensitive

# Dry-run: print commands without executing
python scripts/experiments/run_feature_selection_study.py --pipeline cardiac --dry-run
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from fairxai.cli.runner_base import get_project_root, setup_phase_logging
from fairxai.utils.config import load_yaml_config

logger = logging.getLogger(__name__)

DEFAULT_MODES = [
    "exclude_sensitive",
    "include_all_sensitive",
    "include_sex_only",
    "include_age_only",
    "include_ethnicity_only",
    "rfe_top_k",
]


def _run_one(
    project_root: Path,
    pipeline: str,
    model_type: str,
    mode: str,
    rfe_top_k: int,
    verbose: bool,
    dry_run: bool,
) -> dict:
    """Run train_baseline.py for one (mode, model_type) combo across all datasets.

    train_baseline.py has no --datasets flag — it reads dataset list from the pipeline
    config. One subprocess call per (mode, model_type) covers all configured datasets.
    Output is routed to output/<pipeline>/runs/fs_study_<mode>/ via RUN_ID.
    Returns a status dict with timing and exit code.
    """
    run_id = f"fs_study_{mode}"

    cmd = [
        sys.executable,
        str(project_root / "scripts" / "common" / "train_baseline.py"),
        "--pipeline", pipeline,
        "--model-types", model_type,
        "--feature-selection-mode", mode,
        "--rfe-top-k", str(rfe_top_k),
    ]
    if verbose:
        cmd.append("-v")

    env = {**os.environ, "RUN_ID": run_id}
    run_dir = project_root / f"output/{pipeline}/runs/{run_id}"

    logger.info(f"  [RUN] mode={mode} model={model_type} (RUN_ID={run_id})")
    if dry_run:
        logger.info(f"  [DRY-RUN] RUN_ID={run_id} {' '.join(cmd)}")
        return {"mode": mode, "model": model_type, "status": "dry_run", "duration_s": 0}

    t0 = time.monotonic()
    try:
        result = subprocess.run(
            cmd,
            cwd=str(project_root),
            env=env,
            capture_output=not verbose,
            text=True,
            timeout=600,
        )
        duration = time.monotonic() - t0
        if result.returncode != 0:
            logger.error(
                f"  [FAIL] mode={mode} model={model_type} "
                f"(exit {result.returncode}) — see {run_dir}"
            )
            if not verbose and result.stderr:
                logger.debug(result.stderr[-2000:])
            return {"mode": mode, "model": model_type,
                    "status": "failed", "exit_code": result.returncode, "duration_s": duration}
        logger.info(f"  [OK]   mode={mode} model={model_type} ({duration:.1f}s)")
        return {"mode": mode, "model": model_type, "status": "success", "duration_s": duration}
    except subprocess.TimeoutExpired:
        logger.error(f"  [TIMEOUT] mode={mode} model={model_type} (>600s)")
        return {"mode": mode, "model": model_type, "status": "timeout", "duration_s": 600}
    except Exception as exc:
        logger.error(f"  [ERROR] {exc}")
        return {"mode": mode, "model": model_type, "status": "error", "error": str(exc), "duration_s": 0}


def main():
    parser = argparse.ArgumentParser(
        description="FairXAI feature selection study — sensitive-attribute ablation"
    )
    parser.add_argument("--pipeline", default="cardiac", help="Pipeline config name")
    parser.add_argument(
        "--config",
        default="configs/experiments/feature_selection_study.yaml",
        help="Study config path (relative to project root)",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=None,
        help="Feature selection modes to run (default: all in study config)",
    )
    parser.add_argument(
        "--model-types",
        nargs="+",
        default=None,
        help="Model types to run (default: all in study config)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    project_root = get_project_root(Path(__file__))
    setup_phase_logging(
        project_root, "feature_selection_study.log",
        verbose=args.verbose, stage_name="feature_selection_study",
    )

    study_cfg_path = project_root / args.config
    study_cfg = load_yaml_config(str(study_cfg_path))

    modes = args.modes or study_cfg.get("feature_selection_modes", DEFAULT_MODES)
    datasets = study_cfg.get("datasets", ["cleveland", "kaggle_heart"])  # informational only
    model_types = args.model_types or study_cfg.get("models", ["logistic_regression"])
    rfe_top_k = int(study_cfg.get("rfe_top_k", 10))

    # Summary JSON goes here; actual training outputs go to
    # output/<pipeline>/runs/fs_study_<mode>/ via RUN_ID.
    summary_dir = project_root / study_cfg.get("output_dir", f"output/{args.pipeline}/feature_selection_study")
    summary_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("FAIRXAI FEATURE SELECTION STUDY")
    logger.info("=" * 70)
    logger.info(f"Pipeline : {args.pipeline}")
    logger.info(f"Datasets : {datasets} (all processed per subprocess call)")
    logger.info(f"Models   : {model_types}")
    logger.info(f"Modes    : {modes}")
    logger.info(f"rfe_top_k: {rfe_top_k}")
    logger.info(f"Summary  : {summary_dir}")

    # One subprocess per (mode, model_type) — each call covers all configured datasets.
    total = len(modes) * len(model_types)
    logger.info(f"Total runs: {total}\n")

    results = []
    for mode in modes:
        logger.info(f"\n[MODE] {mode}")
        for model_type in model_types:
            status = _run_one(
                project_root=project_root,
                pipeline=args.pipeline,
                model_type=model_type,
                mode=mode,
                rfe_top_k=rfe_top_k,
                verbose=args.verbose,
                dry_run=args.dry_run,
            )
            results.append(status)

    # Write summary
    summary_path = summary_dir / "study_summary.json"
    if not args.dry_run:
        with open(summary_path, "w") as fh:
            json.dump(
                {
                    "pipeline": args.pipeline,
                    "modes": modes,
                    "datasets_from_pipeline_config": datasets,
                    "models": model_types,
                    "rfe_top_k": rfe_top_k,
                    "total": total,
                    "succeeded": sum(1 for r in results if r["status"] == "success"),
                    "failed": sum(1 for r in results if r["status"] not in ("success", "dry_run")),
                    "runs": results,
                },
                fh,
                indent=2,
            )
        logger.info(f"\n[DONE] Summary written to {summary_path}")
    else:
        logger.info("\n[DRY-RUN] No runs executed.")

    succeeded = sum(1 for r in results if r["status"] == "success")
    failed = sum(1 for r in results if r["status"] not in ("success", "dry_run"))
    logger.info(f"Results: {succeeded}/{total} succeeded, {failed} failed")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
