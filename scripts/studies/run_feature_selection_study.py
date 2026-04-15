"""Feature selection study — sensitive-attribute ablation.

Runs train_baseline.py for each combination of feature-selection mode × dataset × model,
collecting results under output/<pipeline>/studies/feature_selection/<study_id>/.

Training sub-runs land at:
  output/<pipeline>/studies/feature_selection/<study_id>/runs/fs_<mode>__<model>/baseline/

Study-level summary/manifest land at:
  output/<pipeline>/studies/feature_selection/<study_id>/

Usage
-----
# All modes, all models, all configured datasets
python scripts/studies/run_feature_selection_study.py --pipeline cardiac

# Single mode
python scripts/studies/run_feature_selection_study.py --pipeline cardiac \
    --modes exclude_sensitive include_all_sensitive

# Dry-run: print commands without executing
python scripts/studies/run_feature_selection_study.py --pipeline cardiac --dry-run
"""

import argparse
import concurrent.futures as cf
import json
import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from fairxai.cli.runner_base import get_project_root, setup_study_logging
from fairxai.cli.runner_utils import resolve_run_id, update_output_study_pointer, update_study_pointer
from fairxai.utils.config import load_yaml_config

logger = logging.getLogger(__name__)
_ACTIVE_PROCESSES: set[subprocess.Popen] = set()
_ACTIVE_PROCESS_LOCK = threading.Lock()

DEFAULT_MODES = [
    "exclude_sensitive",
    "include_all_sensitive",
    "include_sex_only",
    "include_age_only",
    "include_ethnicity_only",
    "rfe_top_k",
]


def _build_sub_run_key(mode: str, model_type: str) -> str:
    return f"fs_{mode}__{model_type}"


def _register_process(process: subprocess.Popen) -> None:
    with _ACTIVE_PROCESS_LOCK:
        _ACTIVE_PROCESSES.add(process)


def _unregister_process(process: subprocess.Popen) -> None:
    with _ACTIVE_PROCESS_LOCK:
        _ACTIVE_PROCESSES.discard(process)


def _terminate_active_processes() -> None:
    with _ACTIVE_PROCESS_LOCK:
        active = list(_ACTIVE_PROCESSES)
    for process in active:
        if process.poll() is None:
            process.terminate()


def _stop_process(process: subprocess.Popen) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def _run_one(
    project_root: Path,
    pipeline: str,
    model_type: str,
    mode: str,
    study_id: str,
    rfe_top_k: int,
    verbose: bool,
    dry_run: bool,
    stop_event: threading.Event,
    model_n_jobs: int = -1,
    threads_per_worker: int = 0,
) -> dict:
    """Run train_baseline.py for one (mode, model_type) combo across all datasets.

    train_baseline.py has no --datasets flag — it reads dataset list from the pipeline
    config. One subprocess call per (mode, model_type) covers all configured datasets.
    Output is routed via --output-dir to:
      output/<pipeline>/studies/feature_selection/<study_id>/runs/fs_<mode>__<model>/baseline/
    Returns a status dict with timing and exit code.
    """
    sub_key = _build_sub_run_key(mode, model_type)
    baseline_root = (
        project_root
        / f"output/{pipeline}/studies/feature_selection/{study_id}/runs/{sub_key}/baseline"
    )
    results_dir = baseline_root / "results"
    models_dir = baseline_root / "models"

    cmd = [
        sys.executable,
        "-u",
        str(project_root / "scripts" / "common" / "train_baseline.py"),
        "--pipeline",
        pipeline,
        "--output-dir",
        str(baseline_root),
        "--model-types",
        model_type,
        "--feature-selection-mode",
        mode,
        "--rfe-top-k",
        str(rfe_top_k),
        "--model-n-jobs",
        str(model_n_jobs),
        "--cv-n-jobs",
        "1",
    ]
    if verbose:
        cmd.append("-v")

    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
    }
    if threads_per_worker > 0:
        cap = str(threads_per_worker)
        env.update(
            {
                "OMP_NUM_THREADS": cap,
                "MKL_NUM_THREADS": cap,
                "OPENBLAS_NUM_THREADS": cap,
                "NUMEXPR_NUM_THREADS": cap,
            }
        )

    logger.info(f"[PHASE] RUN mode={mode} model={model_type}")
    logger.info(f"[PHASE] OUTPUT results={results_dir}")
    logger.info(f"[PHASE] OUTPUT models={models_dir}")
    if dry_run:
        logger.info(f"[PHASE] DRY-RUN {' '.join(cmd)}")
        return {"mode": mode, "model": model_type, "status": "dry_run", "duration_s": 0}

    if stop_event.is_set():
        return {"mode": mode, "model": model_type, "status": "interrupted", "duration_s": 0}

    t0 = time.monotonic()
    process = None
    try:
        process = subprocess.Popen(cmd, cwd=str(project_root), env=env)
        _register_process(process)
        deadline = time.monotonic() + 600
        while True:
            if stop_event.is_set():
                raise KeyboardInterrupt
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(cmd, timeout=600)
            try:
                process.wait(timeout=min(1.0, remaining))
                break
            except subprocess.TimeoutExpired:
                continue
        duration = time.monotonic() - t0
        if process.returncode != 0:
            logger.error(
                f"RUN mode={mode} model={model_type} "
                f"(exit {process.returncode}) — see {baseline_root.parent}"
            )
            return {
                "mode": mode,
                "model": model_type,
                "status": "failed",
                "exit_code": process.returncode,
                "duration_s": duration,
            }
        logger.info(f"[SUCCESS] RUN mode={mode} model={model_type} ({duration:.1f}s)")
        return {"mode": mode, "model": model_type, "status": "success", "duration_s": duration}
    except subprocess.TimeoutExpired:
        if process is not None:
            _stop_process(process)
        logger.error(f"TIMEOUT mode={mode} model={model_type} (>600s)")
        return {"mode": mode, "model": model_type, "status": "timeout", "duration_s": 600}
    except KeyboardInterrupt:
        if process is not None:
            _stop_process(process)
        logger.warning(f"[PHASE] INTERRUPTED mode={mode} model={model_type}")
        return {
            "mode": mode,
            "model": model_type,
            "status": "interrupted",
            "duration_s": time.monotonic() - t0,
        }
    except Exception as exc:
        if process is not None and process.poll() is None:
            _stop_process(process)
        logger.error(f"{exc}")
        return {
            "mode": mode,
            "model": model_type,
            "status": "error",
            "error": str(exc),
            "duration_s": 0,
        }
    finally:
        if process is not None:
            _unregister_process(process)


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
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of study jobs to run in parallel (default: 1)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    project_root = get_project_root(Path(__file__))
    study_id = resolve_run_id()
    log_subdir = args.pipeline
    setup_study_logging(
        project_root,
        "feature_selection",
        study_id,
        "study.log",
        verbose=args.verbose,
        log_subdir=log_subdir,
    )
    update_study_pointer(
        project_root / "logs" / log_subdir,
        "feature_selection",
        study_id,
        logger,
    )

    study_cfg_path = project_root / args.config
    study_cfg = load_yaml_config(str(study_cfg_path))

    modes = args.modes or study_cfg.get("feature_selection_modes", DEFAULT_MODES)
    datasets = study_cfg.get("datasets", ["cleveland", "kaggle_heart"])  # informational only
    model_types = args.model_types or study_cfg.get("models", ["logistic_regression"])
    rfe_top_k = int(study_cfg.get("rfe_top_k", 10))

    # Study output: summary/manifest + training sub-runs all under studies/feature_selection/<study_id>/
    study_base = project_root / study_cfg.get(
        "output_dir", f"output/{args.pipeline}/studies/feature_selection"
    )
    summary_dir = study_base / study_id
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / "study_summary.json"
    manifest_path = summary_dir / "study_manifest.json"

    cpu_count = os.cpu_count() or 1
    if args.jobs > 1:
        model_n_jobs = 1
        threads_per_worker = max(1, cpu_count // args.jobs)
    else:
        model_n_jobs = -1
        threads_per_worker = 0

    logger.info("[PHASE] FAIRXAI FEATURE SELECTION STUDY")
    logger.info(f"[PHASE] Pipeline: {args.pipeline}")
    logger.info(f"[PHASE] Datasets: {datasets} (all processed per subprocess call)")
    logger.info(f"[PHASE] Models: {model_types}")
    logger.info(f"[PHASE] Modes: {modes}")
    logger.info(f"[PHASE] rfe_top_k: {rfe_top_k}")
    logger.info(
        f"[PHASE] Jobs: {args.jobs}  model_n_jobs={model_n_jobs}  threads_per_worker={threads_per_worker or 'uncapped'}"
    )
    logger.info(f"[PHASE] Summary path: {summary_path}")
    logger.info(f"[PHASE] Manifest path: {manifest_path}")

    # One subprocess per (mode, model_type) — each call covers all configured datasets.
    total = len(modes) * len(model_types)
    logger.info(f"[PHASE] Total runs: {total}")

    results = []
    stop_event = threading.Event()
    try:
        tasks = []
        for mode in modes:
            logger.info(f"[PHASE] MODE {mode}")
            for model_type in model_types:
                tasks.append((mode, model_type))

        if args.jobs == 1:
            for mode, model_type in tasks:
                status = _run_one(
                    project_root=project_root,
                    pipeline=args.pipeline,
                    model_type=model_type,
                    mode=mode,
                    study_id=study_id,
                    rfe_top_k=rfe_top_k,
                    verbose=args.verbose,
                    dry_run=args.dry_run,
                    stop_event=stop_event,
                    model_n_jobs=model_n_jobs,
                    threads_per_worker=threads_per_worker,
                )
                results.append(status)
                if status["status"] == "interrupted":
                    raise KeyboardInterrupt
        else:
            logger.info(f"[PHASE] PARALLEL up to {args.jobs} study jobs at a time")
            with cf.ThreadPoolExecutor(max_workers=args.jobs) as executor:
                future_map = {
                    executor.submit(
                        _run_one,
                        project_root,
                        args.pipeline,
                        model_type,
                        mode,
                        study_id,
                        rfe_top_k,
                        args.verbose,
                        args.dry_run,
                        stop_event,
                        model_n_jobs,
                        threads_per_worker,
                    ): (mode, model_type)
                    for mode, model_type in tasks
                }
                for future in cf.as_completed(future_map):
                    mode, model_type = future_map[future]
                    try:
                        status = future.result()
                    except Exception as exc:
                        logger.error(f"RUN mode={mode} model={model_type} crashed: {exc}")
                        stop_event.set()
                        _terminate_active_processes()
                        status = {
                            "mode": mode,
                            "model": model_type,
                            "status": "error",
                            "error": str(exc),
                            "duration_s": 0,
                        }
                    results.append(status)
                    if status["status"] == "interrupted":
                        stop_event.set()
                        _terminate_active_processes()
                        raise KeyboardInterrupt
    except KeyboardInterrupt:
        stop_event.set()
        _terminate_active_processes()
        logger.warning("[PHASE] INTERRUPTED Feature selection study stopped by user")

    # Write summary
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
    else:
        logger.info("[SUCCESS] DRY-RUN no runs executed")

    manifest = {
        "pipeline": args.pipeline,
        "study_id": study_id,
        "config": str(study_cfg_path),
        "modes": modes,
        "datasets_from_pipeline_config": datasets,
        "models": model_types,
        "rfe_top_k": rfe_top_k,
        "jobs": args.jobs,
        "summary_dir": str(summary_dir),
        "summary_path": str(summary_path),
        "runs": [
            {
                **run,
                "sub_key": _build_sub_run_key(run["mode"], run["model"]),
                "baseline_root": str(
                    project_root
                    / f"output/{args.pipeline}/studies/feature_selection/{study_id}"
                    f"/runs/{_build_sub_run_key(run['mode'], run['model'])}/baseline"
                ),
                "results_dir": str(
                    project_root
                    / f"output/{args.pipeline}/studies/feature_selection/{study_id}"
                    f"/runs/{_build_sub_run_key(run['mode'], run['model'])}/baseline/results"
                ),
                "models_dir": str(
                    project_root
                    / f"output/{args.pipeline}/studies/feature_selection/{study_id}"
                    f"/runs/{_build_sub_run_key(run['mode'], run['model'])}/baseline/models"
                ),
            }
            for run in results
        ],
    }
    if not args.dry_run:
        with open(manifest_path, "w") as fh:
            json.dump(manifest, fh, indent=2)

    succeeded = sum(1 for r in results if r["status"] == "success")
    failed = sum(1 for r in results if r["status"] not in ("success", "dry_run"))
    logger.info(f"[PHASE] Results: {succeeded}/{total} succeeded, {failed} failed")
    if args.dry_run:
        logger.info(f"[PHASE] Study summary will be written to: {summary_path}")
        logger.info(f"[PHASE] Study manifest will be written to: {manifest_path}")
    else:
        logger.info(f"[SUCCESS] Study summary written: {summary_path}")
        logger.info(f"[SUCCESS] Study manifest written: {manifest_path}")
        update_output_study_pointer(
            project_root / f"output/{args.pipeline}",
            "feature_selection",
            study_id,
        )
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
