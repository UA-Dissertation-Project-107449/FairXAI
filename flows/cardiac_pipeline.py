#!/usr/bin/env python3
"""
Prefect flow for the cardiac pipeline.

Supports partial execution via ``--resume-from`` and ``--go-until``.
Run ``python flows/cardiac_pipeline.py --help`` for details.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

from prefect import flow, get_run_logger, task

# Add the src directory to the path
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR / "src"))

from fairxai.cli.runner_utils import (  # noqa: E402
    get_run_root,
    resolve_latest_run_dir,
    resolve_run_id,
    update_log_latest_pointer,
)
from fairxai.pipeline.stages import (  # noqa: E402
    STAGES,
    get_stage_range,
    mark_stage_complete,
    resolve_stage,
    validate_prior_stages,
)
from fairxai.utils.config import load_yaml_config  # noqa: E402
from fairxai.utils.logging_utils import summarize_run_logs  # noqa: E402


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


def _study_verbose_flags(level: int) -> list[str]:
    """Convert verbosity for study scripts (only supports -v)."""
    return ["-v"] if level >= 1 else []


@task
def load_data(run_id: str, datasets: Optional[list[str]] = None, verbose: int = 0):
    logger = get_run_logger()
    logger.info("[PHASE 1/12] Loading cardiac datasets (standardization)")
    script = ROOT_DIR / "scripts" / "cardiac" / "load_data.py"
    args = []
    if datasets:
        args.extend(["--datasets", *datasets])
    args.extend(_verbose_flags(verbose))
    env = os.environ.copy()
    env["RUN_ID"] = run_id
    _run_script(script, args, env)


@task
def profile_data(run_id: str, datasets: Optional[list[str]] = None, verbose: int = 0):
    logger = get_run_logger()
    logger.info("[PHASE 2/12] Profiling datasets (complexity + fairness)")
    script = ROOT_DIR / "scripts" / "cardiac" / "profile_data.py"
    args = []
    if datasets:
        args.extend(["--datasets", *datasets])
    args.extend(_verbose_flags(verbose))
    env = os.environ.copy()
    env["RUN_ID"] = run_id
    _run_script(script, args, env)


@task
def generate_recommendations(run_id: str, verbose: int = 0):
    """Generate fairness triage recommendations (Phase 3)."""
    logger = get_run_logger()
    logger.info("[PHASE 3/12] Generating fairness triage recommendations")
    script = ROOT_DIR / "scripts" / "cardiac" / "generate_recommendations.py"
    args = ["--run-id", run_id] + _verbose_flags(verbose)
    env = os.environ.copy()
    env["RUN_ID"] = run_id
    _run_script(script, args, env)


@task
def preprocess_data(
    run_id: str,
    all_binnings: bool = False,
    datasets: Optional[list[str]] = None,
    verbose: int = 0,
):
    logger = get_run_logger()
    logger.info("[PHASE 4/12] Preprocessing datasets (split + scale + fairness profiles)")
    script = ROOT_DIR / "scripts" / "cardiac" / "preprocess.py"
    args = []
    if all_binnings:
        args.append("--all-binnings")
    if datasets:
        args.extend(["--datasets", *datasets])
    args.extend(_verbose_flags(verbose))
    env = os.environ.copy()
    env["RUN_ID"] = run_id
    _run_script(script, args, env)


@task
def run_hpo_study(
    datasets: Optional[list[str]] = None,
    model_types: Optional[list[str]] = None,
    verbose: int = 0,
):
    """Runs HPO study before baseline/experiments."""
    logger = get_run_logger()
    logger.info("[PHASE 5/12] Hyperparameter optimisation study")
    script = ROOT_DIR / "scripts" / "studies" / "run_hpo.py"
    args = ["--pipeline", "cardiac", "--config", "configs/experiments/hpo.yaml"]
    if datasets:
        args.extend(["--datasets", *datasets])
    if model_types:
        args.extend(["--model-types", *model_types])
    args.extend(_study_verbose_flags(verbose))
    _run_script(script, args, os.environ.copy())


@task
def run_feature_selection_study(
    datasets: Optional[list[str]] = None,
    model_types: Optional[list[str]] = None,
    verbose: int = 0,
):
    """Runs feature-selection study before baseline/experiments."""
    logger = get_run_logger()
    logger.info("[PHASE 6/12] Feature-selection ablation study")
    script = ROOT_DIR / "scripts" / "studies" / "run_feature_selection_study.py"
    args = [
        "--pipeline",
        "cardiac",
        "--config",
        "configs/experiments/feature_selection_study.yaml",
    ]
    if datasets:
        args.extend(["--datasets", *datasets])
    if model_types:
        args.extend(["--model-types", *model_types])
    args.extend(_study_verbose_flags(verbose))
    _run_script(script, args, os.environ.copy())


@task
def build_selector_contract(
    run_id: str,
    datasets: Optional[list[str]] = None,
    model_types: Optional[list[str]] = None,
    verbose: int = 0,
) -> str:
    logger = get_run_logger()
    logger.info("[WIRING] Building selector contract from study artifacts")
    script = ROOT_DIR / "scripts" / "studies" / "build_selector_contract.py"
    args = ["--pipeline", "cardiac", "--run-id", run_id]
    if datasets:
        args.extend(["--datasets", *datasets])
    if model_types:
        args.extend(["--model-types", *model_types])
    args.extend(_study_verbose_flags(verbose))
    _run_script(script, args, os.environ.copy())

    contract_path = (
        ROOT_DIR
        / "output"
        / "cardiac"
        / "runs"
        / run_id
        / "recommendations"
        / "selector_contract.json"
    )
    if not contract_path.exists():
        logger.warning("Selector contract was not created at expected path: %s", contract_path)
    return str(contract_path)


@task
def train_baseline_model(
    run_id: str,
    datasets: Optional[list[str]] = None,
    model_types: Optional[list[str]] = None,
    selector_contract_path: Optional[str] = None,
    verbose: int = 0,
):
    logger = get_run_logger()
    logger.info("[PHASE 7/12] Training baseline model(s)")
    script = ROOT_DIR / "scripts" / "cardiac" / "train_baseline.py"
    args = []
    if datasets:
        args.extend(["--datasets", *datasets])
    if model_types:
        args.extend(["--model-types", *model_types])
    if selector_contract_path:
        args.extend(["--selector-contract", selector_contract_path])
    args.extend(_verbose_flags(verbose))
    env = os.environ.copy()
    env["RUN_ID"] = run_id
    _run_script(script, args, env)


@task
def assess_predictions(
    run_id: str,
    datasets: Optional[list[str]] = None,
    model_types: Optional[list[str]] = None,
    verbose: int = 0,
):
    logger = get_run_logger()
    logger.info("[PHASE 8/12] Assessing post-prediction fairness")
    script = ROOT_DIR / "scripts" / "cardiac" / "assess_predictions.py"
    args = []
    if datasets:
        args.extend(["--datasets", *datasets])
    if model_types:
        args.extend(["--model-types", *model_types])
    args.extend(_verbose_flags(verbose))
    env = os.environ.copy()
    env["RUN_ID"] = run_id
    _run_script(script, args, env)


@task
def analyze_attribute_binning(run_id: str, datasets: Optional[list[str]] = None, verbose: int = 0):
    """Analyzes attribute binning strategies."""
    logger = get_run_logger()
    logger.info("[PHASE 9/12] Attribute binning strategies analysis")
    script = ROOT_DIR / "scripts" / "experiments" / "run_attribute_binning_analysis.py"
    args = [
        "--config",
        "configs/experiments/age_binning.yaml",
        "--run-mode",
        "full",
        "--run-id",
        run_id,
        "--pipeline",
        "cardiac",
    ]
    if datasets:
        args.extend(["--datasets", *datasets])
    args.extend(_verbose_flags(verbose))
    _run_script(script, args, os.environ.copy())


@task
def compare_mitigation_techniques(
    run_id: str, datasets: Optional[list[str]] = None, verbose: int = 0
):
    """Compares mitigation techniques."""
    logger = get_run_logger()
    logger.info("[PHASE 10/12] Mitigation techniques comparison")
    script = ROOT_DIR / "scripts" / "cardiac" / "mitigation.py"
    args = [
        "--config",
        "configs/experiments/mitigation.yaml",
        "--run-mode",
        "full",
        "--run-id",
        run_id,
    ]
    if datasets:
        args.extend(["--datasets", *datasets])
    args.extend(_verbose_flags(verbose))
    _run_script(script, args, os.environ.copy())


@task
def run_combinatorial_experiments(
    run_id: str,
    datasets: Optional[list[str]] = None,
    model_types: Optional[list[str]] = None,
    selector_contract_path: Optional[str] = None,
    verbose: int = 0,
):
    """Runs combinatorial experiments."""
    logger = get_run_logger()
    logger.info("[PHASE 11/12] Combinatorial experiments")
    script = ROOT_DIR / "scripts" / "cardiac" / "combinatorial.py"
    args = ["--config", "configs/experiments/combinatorial.yaml", "--run-id", run_id]
    if datasets:
        args.extend(["--datasets", *datasets])
    if model_types:
        args.extend(["--model-types", *model_types])
    if selector_contract_path:
        args.extend(["--selector-contract", selector_contract_path])
    args.extend(_verbose_flags(verbose))
    _run_script(script, args, os.environ.copy())


@task
def compare_experiments(run_id: str, verbose: int = 0):
    """Compares experiments."""
    logger = get_run_logger()
    logger.info("[PHASE 12/12] Experiment comparison and dissertation plots")
    script = ROOT_DIR / "scripts" / "cardiac" / "compare.py"
    args = ["--pipeline", "cardiac", "--run-id", run_id]
    args.extend(_verbose_flags(verbose))
    _run_script(script, args, os.environ.copy())

    plots_script = ROOT_DIR / "scripts" / "studies" / "generate_dissertation_plots.py"
    plots_args = ["--run-id", run_id]
    _run_script(plots_script, plots_args, os.environ.copy())


@flow(name="Cardiac Fairness Pipeline")
def cardiac_pipeline(
    run_hpo_study_enabled: bool = True,
    run_feature_selection_study_enabled: bool = True,
    skip_studies: Optional[bool] = None,
    run_attribute_binning: bool = True,
    run_mitigation: bool = True,
    run_combinatorial: bool = True,
    run_comparison: bool = True,
    verbose: int = 0,
    resume_from: Optional[str] = None,
    go_until: Optional[str] = None,
    run_id_override: Optional[str] = None,
    datasets: Optional[list[str]] = None,
    model_types: Optional[list[str]] = None,
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

    skip_studies_cfg = False
    if skip_studies is None:
        cfg_path = ROOT_DIR / "configs" / "pipelines" / "cardiac.yaml"
        try:
            pipeline_cfg = load_yaml_config(str(cfg_path))
            skip_studies_cfg = bool((pipeline_cfg.get("studies") or {}).get("skip", False))
        except Exception as exc:
            logger.warning("Could not read studies.skip from %s: %s", cfg_path, exc)
    resolved_skip_studies = skip_studies if skip_studies is not None else skip_studies_cfg

    if resolved_skip_studies:
        run_hpo_study_enabled = False
        run_feature_selection_study_enabled = False

    # --- Resolve stage range ------------------------------------------------
    active_stages = get_stage_range(resume_from, go_until)
    active_nums = {s.number for s in active_stages}

    def _should_run(stage_number: int) -> bool:
        return stage_number in active_nums

    # --- Resolve run ID -----------------------------------------------------
    base_results = ROOT_DIR / "output" / "cardiac"
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

    # Point logs/cardiac/latest_run at this run's log directory
    import logging as _logging

    update_log_latest_pointer(ROOT_DIR, run_id, _logging.getLogger(__name__))

    # --- Validate prior stages on resume ------------------------------------
    if resume_from:
        first_stage = resolve_stage(resume_from)
        validate_prior_stages(run_root, first_stage, ROOT_DIR)
        logger.info(
            f"Resume validation passed - prior stages through "
            f"{first_stage.number - 1} are complete."
        )

    # --- Banner -------------------------------------------------------------
    first = active_stages[0]
    last = active_stages[-1]
    logger.info("[PHASE] Cardiac fairness pipeline started")
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Stage window: {first.number}..{last.number} ({first.name} to {last.name})")
    logger.info(f"Skip studies: {resolved_skip_studies}")
    logger.info(f"HPO study enabled: {run_hpo_study_enabled}")
    logger.info(f"Feature-selection study enabled: {run_feature_selection_study_enabled}")
    logger.info(f"Attribute binning enabled: {run_attribute_binning}")
    logger.info(f"Mitigation enabled: {run_mitigation}")
    logger.info(f"Combinatorial enabled: {run_combinatorial}")
    logger.info(f"Comparison enabled: {run_comparison}")
    logger.info(f"Datasets override: {datasets if datasets else 'config/default'}")
    logger.info(f"Model types override: {model_types if model_types else 'config/default'}")

    # --- Helper: checkpoint after a successful task -------------------------
    def _checkpoint(stage_num: int, future):
        """Wait for a task future, then write a checkpoint marker."""
        future.result()  # raises on failure
        mark_stage_complete(run_root, STAGES[stage_num - 1])

    def _mark_skipped(stage_num: int, reason: str) -> None:
        """Write a checkpoint marker for an intentionally skipped stage."""
        stage = STAGES[stage_num - 1]
        logger.info(f"[{stage_num}/12] {stage.name} - checkpointed as skipped ({reason})")
        mark_stage_complete(run_root, stage)

    # --- Submit tasks, gated by active range --------------------------------
    load_data_task = None
    profile_task = None
    recommendations_task = None
    preprocess_data_task = None
    hpo_study_task = None
    feature_selection_study_task = None
    selector_contract_task = None
    train_baseline_model_task = None
    assess_predictions_task = None
    age_task = None
    mitigation_task = None
    combinatorial_task = None
    comparison_task = None

    # Stage 1 - Load
    if _should_run(1):
        load_data_task = load_data.submit(run_id, datasets, verbose)
    else:
        logger.info("[1/12] load - skipped (outside active range)")

    # Stage 2 - Profile
    if _should_run(2):
        wait = [load_data_task] if load_data_task else []
        profile_task = profile_data.submit(run_id, datasets, verbose, wait_for=wait)
    else:
        logger.info("[2/12] profile - skipped (outside active range)")

    # Stage 3 - Recommendations
    if _should_run(3):
        wait = [profile_task] if profile_task else []
        recommendations_task = generate_recommendations.submit(run_id, verbose, wait_for=wait)
    else:
        logger.info("[3/12] recommend - skipped (outside active range)")

    # Stage 4 - Preprocess
    if _should_run(4):
        wait = [profile_task] if profile_task else []
        preprocess_data_task = preprocess_data.submit(
            run_id, run_combinatorial, datasets, verbose, wait_for=wait
        )
    else:
        logger.info("[4/12] preprocess - skipped (outside active range)")

    # Stage 5 - HPO study (optional + gated)
    if _should_run(5):
        if run_hpo_study_enabled:
            wait = [preprocess_data_task] if preprocess_data_task else []
            hpo_study_task = run_hpo_study.submit(datasets, model_types, verbose, wait_for=wait)
        else:
            logger.info("[5/12] hpo_study - skipped (disabled)")
            _mark_skipped(5, "disabled")
    else:
        logger.info("[5/12] hpo_study - skipped (outside active range)")

    # Stage 6 - Feature-selection study (optional + gated)
    if _should_run(6):
        if run_feature_selection_study_enabled:
            if hpo_study_task:
                wait = [hpo_study_task]
            elif preprocess_data_task:
                wait = [preprocess_data_task]
            else:
                wait = []
            feature_selection_study_task = run_feature_selection_study.submit(
                datasets,
                model_types,
                verbose,
                wait_for=wait,
            )
        else:
            logger.info("[6/12] feature_selection_study - skipped (disabled)")
            _mark_skipped(6, "disabled")
    else:
        logger.info("[6/12] feature_selection_study - skipped (outside active range)")

    # Wiring - Selector contract (internal helper for stages 7/11)
    if _should_run(7) or (_should_run(11) and run_combinatorial):
        if feature_selection_study_task:
            wait = [feature_selection_study_task]
        elif hpo_study_task:
            wait = [hpo_study_task]
        elif preprocess_data_task:
            wait = [preprocess_data_task]
        else:
            wait = []

        selector_contract_task = build_selector_contract.submit(
            run_id,
            datasets,
            model_types,
            verbose,
            wait_for=wait,
        )
    else:
        logger.info("[WIRING] selector_contract - skipped (downstream stages not active)")

    # Stage 7 - Train baseline
    if _should_run(7):
        if selector_contract_task:
            wait = [selector_contract_task]
        elif feature_selection_study_task:
            wait = [feature_selection_study_task]
        elif hpo_study_task:
            wait = [hpo_study_task]
        elif preprocess_data_task:
            wait = [preprocess_data_task]
        else:
            wait = []
        train_baseline_model_task = train_baseline_model.submit(
            run_id,
            datasets,
            model_types,
            selector_contract_task,
            verbose,
            wait_for=wait,
        )
    else:
        logger.info("[7/12] train - skipped (outside active range)")

    # Stage 8 - Assess fairness
    if _should_run(8):
        wait = [train_baseline_model_task] if train_baseline_model_task else []
        assess_predictions_task = assess_predictions.submit(
            run_id, datasets, model_types, verbose, wait_for=wait
        )
    else:
        logger.info("[8/12] assess - skipped (outside active range)")

    # Stage 9 - Attribute binning (optional + gated)
    if _should_run(9):
        if run_attribute_binning:
            wait = [assess_predictions_task] if assess_predictions_task else []
            age_task = analyze_attribute_binning.submit(run_id, datasets, verbose, wait_for=wait)
        else:
            logger.info("[9/12] attribute_binning - skipped (disabled)")
            _mark_skipped(9, "disabled")
    else:
        logger.info("[9/12] attribute_binning - skipped (outside active range)")

    # Stage 10 - Mitigation (optional + gated)
    if _should_run(10):
        if run_mitigation:
            wait = [assess_predictions_task] if assess_predictions_task else []
            mitigation_task = compare_mitigation_techniques.submit(
                run_id, datasets, verbose, wait_for=wait
            )
        else:
            logger.info("[10/12] mitigation - skipped (disabled)")
            _mark_skipped(10, "disabled")
    else:
        logger.info("[10/12] mitigation - skipped (outside active range)")

    # Stage 11 - Combinatorial (optional + gated)
    if _should_run(11):
        if run_combinatorial:
            wait = [assess_predictions_task] if assess_predictions_task else []
            combinatorial_task = run_combinatorial_experiments.submit(
                run_id,
                datasets,
                model_types,
                selector_contract_task,
                verbose,
                wait_for=wait,
            )
        else:
            logger.info("[11/12] combinatorial - skipped (disabled)")
            _mark_skipped(11, "disabled")
    else:
        logger.info("[11/12] combinatorial - skipped (outside active range)")

    # Stage 12 - Comparison (optional + gated)
    if _should_run(12):
        if run_comparison:
            wait = (
                [combinatorial_task]
                if combinatorial_task
                else [assess_predictions_task] if assess_predictions_task else []
            )
            comparison_task = compare_experiments.submit(run_id, verbose, wait_for=wait)
        else:
            logger.info("[12/12] compare - skipped (disabled)")
            _mark_skipped(12, "disabled")
    else:
        logger.info("[12/12] compare - skipped (outside active range)")

    # --- Collect results & write checkpoints --------------------------------
    task_map = {
        1: load_data_task,
        2: profile_task,
        3: recommendations_task,
        4: preprocess_data_task,
        5: hpo_study_task,
        6: feature_selection_study_task,
        7: train_baseline_model_task,
        8: assess_predictions_task,
        9: age_task,
        10: mitigation_task,
        11: combinatorial_task,
        12: comparison_task,
    }
    for stage_num in sorted(task_map):
        future = task_map[stage_num]
        if future is not None:
            _checkpoint(stage_num, future)

    # --- Log summary --------------------------------------------------------
    run_log_dir = ROOT_DIR / "logs" / "cardiac" / "runs" / run_id
    log_summary = summarize_run_logs(run_log_dir)
    if log_summary["total_warnings"] or log_summary["total_errors"]:
        logger.info(
            f"Log summary: {log_summary['total_warnings']} warning(s), "
            f"{log_summary['total_errors']} error(s) - see {run_log_dir / 'run_summary.json'}"
        )

    # --- Summary ------------------------------------------------------------
    logger.info("[PHASE] Cardiac fairness pipeline complete")
    logger.info(f"Stages executed: {first.name} to {last.name}")
    logger.info("Output paths:")
    logger.info(f"  - Run root:           {run_root}")
    if _should_run(1):
        logger.info(f"  - Raw data:           {ROOT_DIR}/data/raw/cardiac")
    if _should_run(4):
        logger.info(f"  - Processed data:     {ROOT_DIR}/data/processed/cardiac")
    if hpo_study_task:
        logger.info(f"  - HPO study:          {ROOT_DIR}/output/cardiac/studies/hpo")
    if feature_selection_study_task:
        logger.info(f"  - FS study:           {ROOT_DIR}/output/cardiac/studies/feature_selection")
    if _should_run(2):
        logger.info(f"  - Profiling:          {run_root}/profiling")
    if _should_run(3):
        logger.info(f"  - Recommendations:    {run_root}/recommendations")
    if selector_contract_task:
        logger.info(f"  - Selector contract:  {run_root}/recommendations/selector_contract.json")
    if _should_run(7):
        logger.info(f"  - Baseline:           {run_root}/baseline")
    if age_task:
        logger.info(f"  - Attr binning:       {run_root}/experiments/attribute_binning")
    if mitigation_task:
        logger.info(f"  - Mitigation:         {run_root}/experiments/mitigation")
    if combinatorial_task:
        logger.info(f"  - Combinatorial:      {run_root}/experiments")
    if comparison_task:
        logger.info(f"  - Comparison:         {run_root}/experiments/comparisons")


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run the cardiac fairness pipeline (Prefect flow).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Stage names (number or name accepted):
    1=load  2=profile  3=recommend  4=preprocess
    5=hpo_study  6=feature_selection_study
    7=train  8=assess  9=attribute_binning
    10=mitigation  11=combinatorial  12=compare

Examples:
  # Run only through profiling
  %(prog)s --go-until profile

  # Resume a failed run from preprocessing
  %(prog)s --resume-from preprocess --run-id run_20260224_143000_12345_abc

  # Resume from latest run, stop after training
  %(prog)s --resume-from preprocess --go-until train
""",
    )
    p.add_argument(
        "--resume-from",
        default=None,
        help="Stage to resume from (inclusive). Accepts name or number.",
    )
    p.add_argument(
        "--go-until",
        default=None,
        help="Last stage to execute (inclusive). Accepts name or number.",
    )
    p.add_argument(
        "--run-id", default=None, help="Explicit run ID. On resume, defaults to latest run."
    )
    skip_group = p.add_mutually_exclusive_group()
    skip_group.add_argument(
        "--skip-studies",
        dest="skip_studies",
        action="store_true",
        help="Skip both HPO and feature-selection study stages.",
    )
    skip_group.add_argument(
        "--no-skip-studies",
        dest="skip_studies",
        action="store_false",
        help="Force studies enabled even if config has studies.skip=true.",
    )
    p.set_defaults(skip_studies=None)
    p.add_argument("--no-hpo-study", action="store_true", help="Skip HPO study stage.")
    p.add_argument(
        "--no-feature-selection-study",
        action="store_true",
        help="Skip feature-selection study stage.",
    )
    p.add_argument(
        "--no-attribute-binning", action="store_true", help="Skip attribute binning stage."
    )
    p.add_argument("--no-mitigation", action="store_true", help="Skip mitigation stage.")
    p.add_argument("--no-combinatorial", action="store_true", help="Skip combinatorial stage.")
    p.add_argument("--no-comparison", action="store_true", help="Skip comparison stage.")
    p.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Optional dataset override passed to stages (CLI > config > defaults).",
    )
    p.add_argument(
        "--model-types",
        nargs="+",
        default=None,
        help="Optional model types override for baseline/combinatorial stages.",
    )
    p.add_argument(
        "-v", "--verbose", action="count", default=0, help="Verbosity: -v=info, -vv=debug"
    )
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    cardiac_pipeline(
        run_hpo_study_enabled=not args.no_hpo_study,
        run_feature_selection_study_enabled=not args.no_feature_selection_study,
        skip_studies=args.skip_studies,
        run_attribute_binning=not args.no_attribute_binning,
        run_mitigation=not args.no_mitigation,
        run_combinatorial=not args.no_combinatorial,
        run_comparison=not args.no_comparison,
        verbose=args.verbose,
        resume_from=args.resume_from,
        go_until=args.go_until,
        run_id_override=args.run_id,
        datasets=args.datasets,
        model_types=args.model_types,
    )
