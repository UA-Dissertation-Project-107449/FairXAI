#!/usr/bin/env python3
"""
Prefect flow for the cardiac pipeline.
"""

from prefect import flow, task, get_run_logger
import sys
import os
import subprocess
from pathlib import Path

# Add the src directory to the path
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR / "src"))

from fairxai.cli.runner_utils import resolve_run_id


def _run_script(script_path: Path, args: list, env: dict) -> None:
    cmd = [sys.executable, str(script_path)] + args
    subprocess.run(cmd, env=env, check=True, cwd=str(ROOT_DIR))


@task
def load_data(run_id: str, verbose: bool = False):
    logger = get_run_logger()
    logger.info("[PHASE 1/8] Loading cardiac datasets (standardization + profiling)")
    script = ROOT_DIR / "scripts" / "cardiac" / "load_data.py"
    args = ["-v"] if verbose else []
    env = os.environ.copy()
    env["RUN_ID"] = run_id
    _run_script(script, args, env)


@task
def preprocess_data(run_id: str, all_binnings: bool = False, verbose: bool = False):
    logger = get_run_logger()
    logger.info("[PHASE 2/8] Preprocessing datasets (split + scale + fairness profiles)")
    script = ROOT_DIR / "scripts" / "cardiac" / "preprocess.py"
    args = []
    if all_binnings:
        args.append("--all-binnings")
    if verbose:
        args.append("-v")
    env = os.environ.copy()
    env["RUN_ID"] = run_id
    _run_script(script, args, env)


@task
def train_baseline_model(run_id: str, verbose: bool = False):
    logger = get_run_logger()
    logger.info("[PHASE 3/8] Training baseline model(s)")
    script = ROOT_DIR / "scripts" / "cardiac" / "train_baseline.py"
    args = ["-v"] if verbose else []
    env = os.environ.copy()
    env["RUN_ID"] = run_id
    _run_script(script, args, env)


@task
def assess_predictions(run_id: str, verbose: bool = False):
    logger = get_run_logger()
    logger.info("[PHASE 4/8] Assessing post-prediction fairness")
    script = ROOT_DIR / "scripts" / "cardiac" / "assess_predictions.py"
    args = ["-v"] if verbose else []
    env = os.environ.copy()
    env["RUN_ID"] = run_id
    _run_script(script, args, env)


@task
def analyze_age_binning(run_id: str, verbose: bool = False):
    """Analyzes age binning strategies."""
    logger = get_run_logger()
    logger.info("[PHASE 5/8] Age binning strategies analysis")
    script = ROOT_DIR / "scripts" / "cardiac" / "age_binning.py"
    args = ["--config", "configs/experiments/age_binning.yaml", "--run-mode", "full", "--run-id", run_id]
    if verbose:
        args.append("-v")
    _run_script(script, args, os.environ.copy())


@task
def compare_mitigation_techniques(run_id: str, verbose: bool = False):
    """Compares mitigation techniques."""
    logger = get_run_logger()
    logger.info("[PHASE 6/8] Mitigation techniques comparison")
    script = ROOT_DIR / "scripts" / "cardiac" / "mitigation.py"
    args = ["--config", "configs/experiments/mitigation.yaml", "--run-mode", "full", "--run-id", run_id]
    if verbose:
        args.append("-v")
    _run_script(script, args, os.environ.copy())


@task
def run_combinatorial_experiments(run_id: str, verbose: bool = False):
    """Runs combinatorial experiments."""
    logger = get_run_logger()
    logger.info("[PHASE 7/8] Combinatorial experiments")
    script = ROOT_DIR / "scripts" / "cardiac" / "combinatorial.py"
    args = ["--config", "configs/experiments/combinatorial.yaml", "--run-id", run_id]
    if verbose:
        args.append("-v")
    _run_script(script, args, os.environ.copy())


@task
def compare_experiments(run_id: str, verbose: bool = False):
    """Compares experiments."""
    logger = get_run_logger()
    logger.info("[PHASE 8/8] Experiment comparison")
    script = ROOT_DIR / "scripts" / "cardiac" / "compare.py"
    args = ["--pipeline", "cardiac", "--run-id", run_id]
    if verbose:
        args.append("-v")
    _run_script(script, args, os.environ.copy())


@flow(name="Cardiac Fairness Pipeline")
def cardiac_pipeline(
    run_age_binning: bool = True,
    run_mitigation: bool = True,
    run_combinatorial: bool = True,
    run_comparison: bool = True,
    verbose: bool = False,
):
    """
    The main pipeline flow for the cardiac fairness analysis.
    """
    logger = get_run_logger()
    run_id = resolve_run_id()
    os.environ["RUN_ID"] = run_id

    logger.info("======================================================================")
    logger.info("CARDIAC FAIRNESS PIPELINE")
    logger.info("======================================================================")
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Age binning: {run_age_binning}")
    logger.info(f"Mitigation: {run_mitigation}")
    logger.info(f"Combinatorial: {run_combinatorial}")
    logger.info(f"Comparison: {run_comparison}")

    load_data_task = load_data.submit(run_id, verbose)
    preprocess_data_task = preprocess_data.submit(run_id, run_combinatorial, verbose, wait_for=[load_data_task])
    train_baseline_model_task = train_baseline_model.submit(run_id, verbose, wait_for=[preprocess_data_task])
    assess_predictions_task = assess_predictions.submit(run_id, verbose, wait_for=[train_baseline_model_task])

    age_task = None
    mitigation_task = None
    combinatorial_task = None
    comparison_task = None

    if run_age_binning:
        age_task = analyze_age_binning.submit(run_id, verbose, wait_for=[assess_predictions_task])

    if run_mitigation:
        mitigation_task = compare_mitigation_techniques.submit(run_id, verbose, wait_for=[assess_predictions_task])

    if run_combinatorial:
        combinatorial_task = run_combinatorial_experiments.submit(run_id, verbose, wait_for=[assess_predictions_task])
    
    if run_comparison:
        wait_for_task = [combinatorial_task] if combinatorial_task else [assess_predictions_task]
        comparison_task = compare_experiments.submit(run_id, verbose, wait_for=wait_for_task)

    load_data_task.result()
    preprocess_data_task.result()
    train_baseline_model_task.result()
    assess_predictions_task.result()
    if age_task:
        age_task.result()
    if mitigation_task:
        mitigation_task.result()
    if combinatorial_task:
        combinatorial_task.result()
    if comparison_task:
        comparison_task.result()

    logger.info("======================================================================")
    logger.info("PIPELINE COMPLETE")
    logger.info("======================================================================")
    logger.info("Results saved to:")
    logger.info(f"  - Raw data:           {ROOT_DIR}/data/raw/cardiac")
    logger.info(f"  - Processed data:     {ROOT_DIR}/data/processed/cardiac")
    logger.info(f"  - Baseline results:   {ROOT_DIR}/results/cardiac/baseline")
    logger.info(f"  - Baseline models:    {ROOT_DIR}/results/cardiac/baseline/models")
    if run_age_binning:
        logger.info(f"  - Age binning:        {ROOT_DIR}/results/cardiac/runs/{run_id}/experiments/full/age_binning")
    if run_mitigation:
        logger.info(f"  - Mitigation:         {ROOT_DIR}/results/cardiac/runs/{run_id}/experiments/full/mitigation")
    if run_combinatorial:
        logger.info(f"  - Combinatorial:      {ROOT_DIR}/results/cardiac/runs/{run_id}/experiments/full")
    if run_comparison:
        logger.info(f"  - Comparison:         {ROOT_DIR}/results/cardiac/runs/{run_id}/experiments/full/comparisons")

if __name__ == "__main__":
    cardiac_pipeline()
