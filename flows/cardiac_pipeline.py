#!/usr/bin/env python3
"""
Prefect flow for the cardiac pipeline.

Supports partial execution via ``--resume-from`` and ``--go-until``.
Run ``python flows/cardiac_pipeline.py --help`` for details.
"""

from prefect import flow, task, get_run_logger
import argparse
import sys
import os
import subprocess
from pathlib import Path
from typing import Optional

# Add the src directory to the path
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR / "src"))

from fairxai.cli.runner_utils import (
    resolve_run_id,
    get_run_root,
    resolve_latest_run_dir,
)
from fairxai.pipeline.stages import (
    STAGES,
    PipelineStage,
    resolve_stage,
    get_stage_range,
    validate_prior_stages,
    mark_stage_complete,
)


def _run_script(script_path: Path, args: list, env: dict) -> None:
    cmd = [sys.executable, str(script_path)] + args
    subprocess.run(cmd, env=env, check=True, cwd=str(ROOT_DIR))


def _verbose_flags(level: int) -> list[str]:
    """Convert a verbosity int (0/1/2) into CLI flags."""
    if level >= 2:
        return ["-vv"]
    if level >= 1:
        return ["-v"]
    return []


@task
def load_data(run_id: str, verbose: int = 0):
    logger = get_run_logger()
    logger.info("[PHASE 1/10] Loading cardiac datasets (standardization)")
    script = ROOT_DIR / "scripts" / "cardiac" / "load_data.py"
    args = _verbose_flags(verbose)
    env = os.environ.copy()
    env["RUN_ID"] = run_id
    _run_script(script, args, env)


@task
def profile_data(run_id: str, verbose: int = 0):
    logger = get_run_logger()
    logger.info("[PHASE 2/10] Profiling datasets (complexity + fairness)")
    script = ROOT_DIR / "scripts" / "cardiac" / "profile_data.py"
    args = _verbose_flags(verbose)
    env = os.environ.copy()
    env["RUN_ID"] = run_id
    _run_script(script, args, env)


@task
def generate_recommendations(run_id: str, verbose: int = 0):
    """Generate fairness triage recommendations (Phase 3)."""
    logger = get_run_logger()
    logger.info("[PHASE 3/10] Generating fairness triage recommendations")
    script = ROOT_DIR / "scripts" / "cardiac" / "generate_recommendations.py"
    args = ["--run-id", run_id] + _verbose_flags(verbose)
    env = os.environ.copy()
    env["RUN_ID"] = run_id
    _run_script(script, args, env)


@task
def preprocess_data(run_id: str, all_binnings: bool = False, verbose: int = 0):
    logger = get_run_logger()
    logger.info("[PHASE 4/10] Preprocessing datasets (split + scale + fairness profiles)")
    script = ROOT_DIR / "scripts" / "cardiac" / "preprocess.py"
    args = []
    if all_binnings:
        args.append("--all-binnings")
    args.extend(_verbose_flags(verbose))
    env = os.environ.copy()
    env["RUN_ID"] = run_id
    _run_script(script, args, env)


@task
def train_baseline_model(run_id: str, verbose: int = 0):
    logger = get_run_logger()
    logger.info("[PHASE 5/10] Training baseline model(s)")
    script = ROOT_DIR / "scripts" / "cardiac" / "train_baseline.py"
    args = _verbose_flags(verbose)
    env = os.environ.copy()
    env["RUN_ID"] = run_id
    _run_script(script, args, env)


@task
def assess_predictions(run_id: str, verbose: int = 0):
    logger = get_run_logger()
    logger.info("[PHASE 6/10] Assessing post-prediction fairness")
    script = ROOT_DIR / "scripts" / "cardiac" / "assess_predictions.py"
    args = _verbose_flags(verbose)
    env = os.environ.copy()
    env["RUN_ID"] = run_id
    _run_script(script, args, env)


@task
def analyze_age_binning(run_id: str, verbose: int = 0):
    """Analyzes age binning strategies."""
    logger = get_run_logger()
    logger.info("[PHASE 7/10] Age binning strategies analysis")
    script = ROOT_DIR / "scripts" / "cardiac" / "age_binning.py"
    args = ["--config", "configs/experiments/age_binning.yaml", "--run-mode", "full", "--run-id", run_id]
    args.extend(_verbose_flags(verbose))
    _run_script(script, args, os.environ.copy())


@task
def compare_mitigation_techniques(run_id: str, verbose: int = 0):
    """Compares mitigation techniques."""
    logger = get_run_logger()
    logger.info("[PHASE 8/10] Mitigation techniques comparison")
    script = ROOT_DIR / "scripts" / "cardiac" / "mitigation.py"
    args = ["--config", "configs/experiments/mitigation.yaml", "--run-mode", "full", "--run-id", run_id]
    args.extend(_verbose_flags(verbose))
    _run_script(script, args, os.environ.copy())


@task
def run_combinatorial_experiments(run_id: str, verbose: int = 0):
    """Runs combinatorial experiments."""
    logger = get_run_logger()
    logger.info("[PHASE 9/10] Combinatorial experiments")
    script = ROOT_DIR / "scripts" / "cardiac" / "combinatorial.py"
    args = ["--config", "configs/experiments/combinatorial.yaml", "--run-id", run_id]
    args.extend(_verbose_flags(verbose))
    _run_script(script, args, os.environ.copy())


@task
def compare_experiments(run_id: str, verbose: int = 0):
    """Compares experiments."""
    logger = get_run_logger()
    logger.info("[PHASE 10/10] Experiment comparison")
    script = ROOT_DIR / "scripts" / "cardiac" / "compare.py"
    args = ["--pipeline", "cardiac", "--run-id", run_id]
    args.extend(_verbose_flags(verbose))
    _run_script(script, args, os.environ.copy())


@flow(name="Cardiac Fairness Pipeline")
def cardiac_pipeline(
    run_age_binning: bool = True,
    run_mitigation: bool = True,
    run_combinatorial: bool = True,
    run_comparison: bool = True,
    verbose: int = 0,
    resume_from: Optional[str] = None,
    go_until: Optional[str] = None,
    run_id_override: Optional[str] = None,
):
    """
    The main pipeline flow for the cardiac fairness analysis.

    Flow-control flags
    ------------------
    resume_from : stage name/number to resume from (inclusive).
    go_until    : stage name/number to stop after (inclusive).
    run_id_override : explicit run ID; on resume, defaults to latest run.
    """
    logger = get_run_logger()

    # --- Resolve stage range ------------------------------------------------
    active_stages = get_stage_range(resume_from, go_until)
    active_nums = {s.number for s in active_stages}

    def _should_run(stage_number: int) -> bool:
        return stage_number in active_nums

    # --- Resolve run ID -----------------------------------------------------
    base_results = ROOT_DIR / "results" / "cardiac"
    if resume_from:
        # Re-use an existing run
        if run_id_override:
            run_id = resolve_run_id(run_id_override)
        else:
            latest_dir = resolve_latest_run_dir(base_results)
            if latest_dir is None:
                raise RuntimeError(
                    "No --run-id provided and no latest run found under "
                    f"{base_results}. Cannot resume."
                )
            run_id = latest_dir.name
            logger.info(f"Auto-resolved run ID from latest run: {run_id}")
    else:
        run_id = resolve_run_id(run_id_override)

    os.environ["RUN_ID"] = run_id
    run_root = get_run_root(base_results, run_id)

    # --- Validate prior stages on resume ------------------------------------
    if resume_from:
        first_stage = resolve_stage(resume_from)
        validate_prior_stages(run_root, first_stage, ROOT_DIR)
        logger.info(
            f"Resume validation passed — prior stages through "
            f"{first_stage.number - 1} are complete."
        )

    # --- Banner -------------------------------------------------------------
    first = active_stages[0]
    last = active_stages[-1]
    logger.info("======================================================================")
    logger.info("CARDIAC FAIRNESS PIPELINE")
    logger.info("======================================================================")
    logger.info(f"Run ID:       {run_id}")
    logger.info(f"Stages:       {first.number}..{last.number}  ({first.name} → {last.name})")
    logger.info(f"Age binning:  {run_age_binning}")
    logger.info(f"Mitigation:   {run_mitigation}")
    logger.info(f"Combinatorial:{run_combinatorial}")
    logger.info(f"Comparison:   {run_comparison}")

    # --- Helper: checkpoint after a successful task -------------------------
    def _checkpoint(stage_num: int, future):
        """Wait for a task future, then write a checkpoint marker."""
        future.result()  # raises on failure
        mark_stage_complete(run_root, STAGES[stage_num - 1])

    # --- Submit tasks, gated by active range --------------------------------
    load_data_task = None
    profile_task = None
    recommendations_task = None
    preprocess_data_task = None
    train_baseline_model_task = None
    assess_predictions_task = None
    age_task = None
    mitigation_task = None
    combinatorial_task = None
    comparison_task = None

    # Stage 1 — Load
    if _should_run(1):
        load_data_task = load_data.submit(run_id, verbose)
    else:
        logger.info("[1/10] load — SKIPPED (outside active range)")

    # Stage 2 — Profile
    if _should_run(2):
        wait = [load_data_task] if load_data_task else []
        profile_task = profile_data.submit(run_id, verbose, wait_for=wait)
    else:
        logger.info("[2/10] profile — SKIPPED (outside active range)")

    # Stage 3 — Recommendations
    if _should_run(3):
        wait = [profile_task] if profile_task else []
        recommendations_task = generate_recommendations.submit(
            run_id, verbose, wait_for=wait
        )
    else:
        logger.info("[3/10] recommend — SKIPPED (outside active range)")

    # Stage 4 — Preprocess
    if _should_run(4):
        wait = [profile_task] if profile_task else []
        preprocess_data_task = preprocess_data.submit(
            run_id, run_combinatorial, verbose, wait_for=wait
        )
    else:
        logger.info("[4/10] preprocess — SKIPPED (outside active range)")

    # Stage 5 — Train baseline
    if _should_run(5):
        wait = [preprocess_data_task] if preprocess_data_task else []
        train_baseline_model_task = train_baseline_model.submit(
            run_id, verbose, wait_for=wait
        )
    else:
        logger.info("[5/10] train — SKIPPED (outside active range)")

    # Stage 6 — Assess fairness
    if _should_run(6):
        wait = [train_baseline_model_task] if train_baseline_model_task else []
        assess_predictions_task = assess_predictions.submit(
            run_id, verbose, wait_for=wait
        )
    else:
        logger.info("[6/10] assess — SKIPPED (outside active range)")

    # Stage 7 — Age binning (optional + gated)
    if _should_run(7) and run_age_binning:
        wait = [assess_predictions_task] if assess_predictions_task else []
        age_task = analyze_age_binning.submit(run_id, verbose, wait_for=wait)
    else:
        reason = "disabled" if not run_age_binning else "outside active range"
        logger.info(f"[7/10] age_binning — SKIPPED ({reason})")

    # Stage 8 — Mitigation (optional + gated)
    if _should_run(8) and run_mitigation:
        wait = [assess_predictions_task] if assess_predictions_task else []
        mitigation_task = compare_mitigation_techniques.submit(
            run_id, verbose, wait_for=wait
        )
    else:
        reason = "disabled" if not run_mitigation else "outside active range"
        logger.info(f"[8/10] mitigation — SKIPPED ({reason})")

    # Stage 9 — Combinatorial (optional + gated)
    if _should_run(9) and run_combinatorial:
        wait = [assess_predictions_task] if assess_predictions_task else []
        combinatorial_task = run_combinatorial_experiments.submit(
            run_id, verbose, wait_for=wait
        )
    else:
        reason = "disabled" if not run_combinatorial else "outside active range"
        logger.info(f"[9/10] combinatorial — SKIPPED ({reason})")

    # Stage 10 — Comparison (optional + gated)
    if _should_run(10) and run_comparison:
        wait = [combinatorial_task] if combinatorial_task else [assess_predictions_task] if assess_predictions_task else []
        comparison_task = compare_experiments.submit(
            run_id, verbose, wait_for=wait
        )
    else:
        reason = "disabled" if not run_comparison else "outside active range"
        logger.info(f"[10/10] compare — SKIPPED ({reason})")

    # --- Collect results & write checkpoints --------------------------------
    task_map = {
        1: load_data_task,
        2: profile_task,
        3: recommendations_task,
        4: preprocess_data_task,
        5: train_baseline_model_task,
        6: assess_predictions_task,
        7: age_task,
        8: mitigation_task,
        9: combinatorial_task,
        10: comparison_task,
    }
    for stage_num in sorted(task_map):
        future = task_map[stage_num]
        if future is not None:
            _checkpoint(stage_num, future)

    # --- Summary ------------------------------------------------------------
    logger.info("======================================================================")
    logger.info("PIPELINE COMPLETE")
    logger.info("======================================================================")
    logger.info(f"Stages executed: {first.name} → {last.name}")
    logger.info("Results saved to:")
    logger.info(f"  - Run root:           {run_root}")
    if _should_run(1):
        logger.info(f"  - Raw data:           {ROOT_DIR}/data/raw/cardiac")
    if _should_run(4):
        logger.info(f"  - Processed data:     {ROOT_DIR}/data/processed/cardiac")
    if _should_run(2):
        logger.info(f"  - Profiling results:  {run_root}/profiling")
    if _should_run(3):
        logger.info(f"  - Recommendations:    {run_root}/recommendations")
    if _should_run(5):
        logger.info(f"  - Baseline results:   {run_root}/baseline")
    if age_task:
        logger.info(f"  - Age binning:        {run_root}/experiments/full/age_binning")
    if mitigation_task:
        logger.info(f"  - Mitigation:         {run_root}/experiments/full/mitigation")
    if combinatorial_task:
        logger.info(f"  - Combinatorial:      {run_root}/experiments/full")
    if comparison_task:
        logger.info(f"  - Comparison:         {run_root}/experiments/full/comparisons")


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run the cardiac fairness pipeline (Prefect flow).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Stage names (number or name accepted):
  1=load  2=profile  3=recommend  4=preprocess  5=train
  6=assess  7=age_binning  8=mitigation  9=combinatorial  10=compare

Examples:
  # Run only through profiling
  %(prog)s --go-until profile

  # Resume a failed run from preprocessing
  %(prog)s --resume-from preprocess --run-id run_20260224_143000_12345_abc

  # Resume from latest run, stop after training
  %(prog)s --resume-from preprocess --go-until train
""",
    )
    p.add_argument("--resume-from", default=None,
                    help="Stage to resume from (inclusive). Accepts name or number.")
    p.add_argument("--go-until", default=None,
                    help="Last stage to execute (inclusive). Accepts name or number.")
    p.add_argument("--run-id", default=None,
                    help="Explicit run ID. On resume, defaults to latest run.")
    p.add_argument("--no-age-binning", action="store_true", help="Skip age binning stage.")
    p.add_argument("--no-mitigation", action="store_true", help="Skip mitigation stage.")
    p.add_argument("--no-combinatorial", action="store_true", help="Skip combinatorial stage.")
    p.add_argument("--no-comparison", action="store_true", help="Skip comparison stage.")
    p.add_argument("-v", "--verbose", action="count", default=0,
                    help="Verbosity: -v=info, -vv=debug")
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    cardiac_pipeline(
        run_age_binning=not args.no_age_binning,
        run_mitigation=not args.no_mitigation,
        run_combinatorial=not args.no_combinatorial,
        run_comparison=not args.no_comparison,
        verbose=args.verbose,
        resume_from=args.resume_from,
        go_until=args.go_until,
        run_id_override=args.run_id,
    )
