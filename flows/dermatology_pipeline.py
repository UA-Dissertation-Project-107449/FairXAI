#!/usr/bin/env python3
"""
Prefect flow for the dermatology pipeline.

Mirrors ``scripts/dermatology/dermatology_pipeline.sh`` stage for stage, including
its sparse numbering: this domain has no stage 5 or 6, so the stage numbers here
are 1-4 and 7-11 and must not be renumbered -- checkpoint markers are shared with
the bash pipeline, and a run started by one can be resumed by the other.

Supports partial execution via ``--resume-from`` and ``--go-until``.
Run ``python3 flows/dermatology_pipeline.py --help`` for details.
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

from prefect import flow, get_run_logger, task

# Add the src directory to the path
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR / "src"))
PIPELINE_CONFIG = "configs/pipelines/dermatology.yaml"
SCRIPTS_DIR = ROOT_DIR / "scripts" / "dermatology"

from fairxai.cli.runner_utils import (  # noqa: E402
    get_run_root,
    resolve_latest_run_dir,
    resolve_run_id,
    update_log_latest_pointer,
)
from fairxai.pipeline.stages import (  # noqa: E402
    DERMATOLOGY_STAGE_BY_NUMBER,
    get_stage_range,
    mark_stage_complete,
    resolve_stage,
    validate_prior_stages,
)
from fairxai.utils.config import load_yaml_config  # noqa: E402
from fairxai.utils.logging_utils import summarize_run_logs  # noqa: E402

# These three helpers are deliberate small duplicates of the cardiac flow's.
# Each flow is a standalone entry point for a different domain; sharing them
# would couple the two so that a change made for one domain can break the other.


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


def _toggle_flags(value: Optional[bool], on: str, off: str) -> list[str]:
    """Render a tri-state toggle the way bash's ``*_ARGS`` arrays do.

    ``None`` means "no flag" so the stage script falls back to its config value;
    the bash pipeline expresses the same thing with an empty array.
    """
    if value is None:
        return []
    return [on] if value else [off]


def _stage_env(run_id: str) -> dict:
    """Stages 7-11 resolve the run from ``RUN_ID``, exactly as under bash."""
    env = os.environ.copy()
    env["RUN_ID"] = run_id
    return env


@task
def load_data(run_id: str, datasets: Optional[list[str]] = None, verbose: int = 0):
    """Stage 1: load and standardize the raw dermatology datasets."""
    logger = get_run_logger()
    logger.info("[PHASE 1] Loading dermatology datasets")
    args = []
    if datasets:
        args.extend(["--datasets", *datasets])
    args.extend(_verbose_flags(verbose))
    _run_script(SCRIPTS_DIR / "load_data.py", args, _stage_env(run_id))


@task
def profile_data(run_id: str, datasets: Optional[list[str]] = None, verbose: int = 0):
    """Stage 2: profile the standardized datasets."""
    logger = get_run_logger()
    logger.info("[PHASE 2] Profiling dermatology datasets")
    args = []
    if datasets:
        args.extend(["--datasets", *datasets])
    args.extend(["--run-id", run_id])
    args.extend(_verbose_flags(verbose))
    _run_script(SCRIPTS_DIR / "profile_data.py", args, _stage_env(run_id))


@task
def generate_recommendations(run_id: str, datasets: Optional[list[str]] = None, verbose: int = 0):
    """Stage 3: generate fairness triage recommendations."""
    logger = get_run_logger()
    logger.info("[PHASE 3] Generating recommendations")
    args = []
    if datasets:
        args.extend(["--datasets", *datasets])
    args.extend(["--run-id", run_id])
    args.extend(_verbose_flags(verbose))
    _run_script(SCRIPTS_DIR / "generate_recommendations.py", args, _stage_env(run_id))


@task
def preprocess_data(
    run_id: str,
    datasets: Optional[list[str]] = None,
    figures: Optional[bool] = None,
    verbose: int = 0,
):
    """Stage 4: prepare image metadata and the dataset splits."""
    logger = get_run_logger()
    logger.info("[PHASE 4] Preprocessing dermatology datasets")
    args = []
    if datasets:
        args.extend(["--datasets", *datasets])
    args.extend(_toggle_flags(figures, "--figures", "--no-figures"))
    args.extend(_verbose_flags(verbose))
    _run_script(SCRIPTS_DIR / "preprocess.py", args, _stage_env(run_id))


@task
def train_baseline_model(
    run_id: str,
    datasets: Optional[list[str]] = None,
    model_types: Optional[list[str]] = None,
    device: Optional[str] = None,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    pretrained: Optional[bool] = None,
    augmentation: Optional[bool] = None,
    verbose: int = 0,
):
    """Stage 7: train the image baseline for each requested model family."""
    logger = get_run_logger()
    logger.info("[PHASE 7] Training image baseline")
    args = []
    if datasets:
        args.extend(["--datasets", *datasets])
    if model_types:
        args.extend(["--model-types", *model_types])
    if device:
        args.extend(["--device", device])
    if epochs is not None:
        args.extend(["--epochs", str(epochs)])
    if batch_size is not None:
        args.extend(["--batch-size", str(batch_size)])
    args.extend(_toggle_flags(pretrained, "--pretrained", "--no-pretrained"))
    args.extend(_toggle_flags(augmentation, "--augmentation", "--no-augmentation"))
    args.extend(_verbose_flags(verbose))
    _run_script(SCRIPTS_DIR / "train_baseline.py", args, _stage_env(run_id))


@task
def assess_predictions(
    run_id: str,
    datasets: Optional[list[str]] = None,
    model_types: Optional[list[str]] = None,
    group_views: Optional[bool] = None,
    figures: Optional[bool] = None,
    verbose: int = 0,
):
    """Stage 8: assess post-prediction subgroup fairness."""
    logger = get_run_logger()
    logger.info("[PHASE 8] Assessing post-prediction fairness")
    args = []
    if datasets:
        args.extend(["--datasets", *datasets])
    if model_types:
        args.extend(["--model-types", *model_types])
    args.extend(_toggle_flags(group_views, "--group-views", "--no-group-views"))
    args.extend(_toggle_flags(figures, "--figures", "--no-figures"))
    args.extend(_verbose_flags(verbose))
    _run_script(SCRIPTS_DIR / "assess_predictions.py", args, _stage_env(run_id))


@task
def compare_models(
    run_id: str,
    datasets: Optional[list[str]] = None,
    model_types: Optional[list[str]] = None,
    figures: Optional[bool] = None,
    verbose: int = 0,
):
    """Stage 9: collate per-model metrics and the fairness report into one table."""
    logger = get_run_logger()
    logger.info("[PHASE 9] Comparing baseline models")
    args = []
    if datasets:
        args.extend(["--datasets", *datasets])
    if model_types:
        args.extend(["--model-types", *model_types])
    args.extend(_toggle_flags(figures, "--figures", "--no-figures"))
    args.extend(_verbose_flags(verbose))
    _run_script(SCRIPTS_DIR / "compare.py", args, _stage_env(run_id))


@task
def explain_models(
    run_id: str,
    datasets: Optional[list[str]] = None,
    model_types: Optional[list[str]] = None,
    verbose: int = 0,
):
    """Stage 10: post-hoc SHAP / LIME / Grad-CAM saliency for each trained model."""
    logger = get_run_logger()
    logger.info("[PHASE 10] Explaining baseline models (XAI)")
    args = []
    if datasets:
        args.extend(["--datasets", *datasets])
    if model_types:
        args.extend(["--model-types", *model_types])
    args.extend(_verbose_flags(verbose))
    _run_script(SCRIPTS_DIR / "explain.py", args, _stage_env(run_id))


@task
def mitigate_predictions(
    run_id: str,
    datasets: Optional[list[str]] = None,
    model_types: Optional[list[str]] = None,
    figures: Optional[bool] = None,
    verbose: int = 0,
):
    """Stage 11: post-processing mitigation over the baseline predictions."""
    logger = get_run_logger()
    logger.info("[PHASE 11] Post-processing fairness mitigation")
    args = []
    if datasets:
        args.extend(["--datasets", *datasets])
    if model_types:
        args.extend(["--model-types", *model_types])
    args.extend(_toggle_flags(figures, "--figures", "--no-figures"))
    args.extend(_verbose_flags(verbose))
    _run_script(SCRIPTS_DIR / "mitigate.py", args, _stage_env(run_id))


@flow(name="Dermatology Fairness Pipeline")
def dermatology_pipeline(
    run_recommendations: bool = True,
    run_explain: Optional[bool] = None,
    verbose: int = 0,
    resume_from: Optional[str] = None,
    go_until: Optional[str] = None,
    run_id_override: Optional[str] = None,
    datasets: Optional[list[str]] = None,
    model_types: Optional[list[str]] = None,
    device: Optional[str] = None,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    pretrained: Optional[bool] = None,
    augmentation: Optional[bool] = None,
    figures: Optional[bool] = None,
    group_views: Optional[bool] = None,
):
    """
    The dermatology baseline pipeline flow.

    Flow-control flags
    ------------------
    resume_from : stage name/number to resume from (inclusive).
    go_until    : stage name/number to stop after (inclusive).
    run_id_override : explicit run ID; on resume, defaults to latest run.
    run_explain : ``None`` defers to ``RUN_EXPLAIN`` then ``xai.enabled``.
    augmentation : ``None`` defers to ``training.image.use_augmentation``.
        Enabling it disables frozen-feature caching, so an augmented run pays
        the pixels->features cost every epoch.
    """
    flow_logger = get_run_logger()

    cfg_path = ROOT_DIR / PIPELINE_CONFIG
    pipeline_cfg: dict = {}
    try:
        pipeline_cfg = load_yaml_config(str(cfg_path))
    except Exception as exc:
        flow_logger.warning("Could not read pipeline config from %s: %s", cfg_path, exc)

    # Scope to runtime.datasets when no explicit override, so every stage task
    # receives the same explicit set the stage scripts would default to.
    if not datasets:
        cfg_datasets = (pipeline_cfg.get("runtime", {}) or {}).get("datasets")
        if cfg_datasets:
            datasets = [str(d).strip() for d in cfg_datasets]

    # Stage 10 is the most expensive stage of a cached-feature run and feeds none
    # of the fairness numbers. Precedence: explicit param > env RUN_EXPLAIN >
    # config xai.enabled. Matches the bash pipeline's resolution order.
    if run_explain is not None:
        run_explain_enabled = bool(run_explain)
    else:
        env_explain = os.getenv("RUN_EXPLAIN")
        if env_explain is not None:
            run_explain_enabled = env_explain.strip().lower() in ("1", "true", "yes", "on")
        else:
            run_explain_enabled = bool((pipeline_cfg.get("xai") or {}).get("enabled", False))

    # --- Resolve stage range ------------------------------------------------
    active_stages = get_stage_range(resume_from, go_until, domain="dermatology")
    active_nums = {s.number for s in active_stages}

    def _should_run(stage_number: int) -> bool:
        return stage_number in active_nums

    # --- Resolve run ID -----------------------------------------------------
    base_results = ROOT_DIR / "output" / "dermatology"
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
            flow_logger.info(f"Auto-resolved run ID from latest run: {run_id}")
    else:
        run_id = resolve_run_id(run_id_override)

    os.environ["RUN_ID"] = run_id
    run_root = get_run_root(base_results, run_id)

    # Point logs/dermatology/latest_run at this run's log directory
    update_log_latest_pointer(
        ROOT_DIR, run_id, logging.getLogger(__name__), log_subdir="dermatology"
    )

    # --- Validate prior stages on resume ------------------------------------
    if resume_from:
        first_stage = resolve_stage(resume_from, domain="dermatology")
        validate_prior_stages(run_root, first_stage, ROOT_DIR, domain="dermatology")
        flow_logger.info(
            f"Resume validation passed - prior stages before " f"{first_stage.number} are complete."
        )

    # --- Banner -------------------------------------------------------------
    first = active_stages[0]
    last = active_stages[-1]
    flow_logger.info("[PHASE] Dermatology baseline pipeline started")
    flow_logger.info(f"Run ID: {run_id}")
    flow_logger.info(f"Stage window: {first.number}..{last.number} ({first.name} to {last.name})")
    flow_logger.info(f"Effective datasets: {datasets if datasets else 'config/default'}")
    flow_logger.info(f"Model types override: {model_types if model_types else 'config/default'}")
    flow_logger.info(f"Device: {device if device else 'config/default'}")
    flow_logger.info(f"Recommendations enabled: {run_recommendations}")
    flow_logger.info(f"Explain (XAI) enabled: {run_explain_enabled}")

    # --- Helper: checkpoint after a successful task -------------------------
    def _checkpoint(stage_num: int, future):
        """Wait for a task future, then write a checkpoint marker."""
        future.result()  # raises on failure
        mark_stage_complete(run_root, DERMATOLOGY_STAGE_BY_NUMBER[stage_num])

    def _mark_skipped(stage_num: int, reason: str) -> None:
        """Write a checkpoint marker for an intentionally skipped stage."""
        stage = DERMATOLOGY_STAGE_BY_NUMBER[stage_num]
        flow_logger.info(f"{stage} - checkpointed as skipped ({reason})")
        mark_stage_complete(run_root, stage)

    def _out_of_range(stage_num: int) -> None:
        flow_logger.info(f"{DERMATOLOGY_STAGE_BY_NUMBER[stage_num]} - skipped (outside range)")

    # --- Submit tasks, gated by active range --------------------------------
    # Task submissions use keyword arguments throughout: these tasks take several
    # same-typed optional parameters, and a positional call would silently rebind
    # them all if one were ever inserted in the middle.
    load_task = None
    profile_task = None
    recommendations_task = None
    preprocess_task = None
    train_task = None
    assess_task = None
    compare_task = None
    explain_task = None
    mitigate_task = None

    if _should_run(1):
        load_task = load_data.submit(run_id=run_id, datasets=datasets, verbose=verbose)
    else:
        _out_of_range(1)

    if _should_run(2):
        profile_task = profile_data.submit(
            run_id=run_id,
            datasets=datasets,
            verbose=verbose,
            wait_for=[load_task] if load_task else [],
        )
    else:
        _out_of_range(2)

    # Stage 3 and stage 4 both hang off profiling; preprocessing does not read the
    # recommendations, so the two are free to overlap (as in the cardiac flow).
    if _should_run(3):
        if run_recommendations:
            recommendations_task = generate_recommendations.submit(
                run_id=run_id,
                datasets=datasets,
                verbose=verbose,
                wait_for=[profile_task] if profile_task else [],
            )
        else:
            _mark_skipped(3, "disabled")
    else:
        _out_of_range(3)

    if _should_run(4):
        preprocess_task = preprocess_data.submit(
            run_id=run_id,
            datasets=datasets,
            figures=figures,
            verbose=verbose,
            wait_for=[profile_task] if profile_task else [],
        )
    else:
        _out_of_range(4)

    if _should_run(7):
        train_task = train_baseline_model.submit(
            run_id=run_id,
            datasets=datasets,
            model_types=model_types,
            device=device,
            epochs=epochs,
            batch_size=batch_size,
            pretrained=pretrained,
            augmentation=augmentation,
            verbose=verbose,
            wait_for=[preprocess_task] if preprocess_task else [],
        )
    else:
        _out_of_range(7)

    if _should_run(8):
        assess_task = assess_predictions.submit(
            run_id=run_id,
            datasets=datasets,
            model_types=model_types,
            group_views=group_views,
            figures=figures,
            verbose=verbose,
            wait_for=[train_task] if train_task else [],
        )
    else:
        _out_of_range(8)

    # Stages 9-11 run strictly one after another, as under bash. Explain reloads
    # every trained model onto the accelerator, so overlapping it with a
    # neighbouring stage would put two stages on the same device memory at once.
    previous = assess_task

    if _should_run(9):
        compare_task = compare_models.submit(
            run_id=run_id,
            datasets=datasets,
            model_types=model_types,
            figures=figures,
            verbose=verbose,
            wait_for=[previous] if previous else [],
        )
        previous = compare_task
    else:
        _out_of_range(9)

    if _should_run(10):
        if run_explain_enabled:
            explain_task = explain_models.submit(
                run_id=run_id,
                datasets=datasets,
                model_types=model_types,
                verbose=verbose,
                wait_for=[previous] if previous else [],
            )
            previous = explain_task
        else:
            _mark_skipped(10, "disabled")
    else:
        _out_of_range(10)

    if _should_run(11):
        mitigate_task = mitigate_predictions.submit(
            run_id=run_id,
            datasets=datasets,
            model_types=model_types,
            figures=figures,
            verbose=verbose,
            wait_for=[previous] if previous else [],
        )
    else:
        _out_of_range(11)

    # --- Collect results & write checkpoints --------------------------------
    task_map = {
        1: load_task,
        2: profile_task,
        3: recommendations_task,
        4: preprocess_task,
        7: train_task,
        8: assess_task,
        9: compare_task,
        10: explain_task,
        11: mitigate_task,
    }
    for stage_num in sorted(task_map):
        future = task_map[stage_num]
        if future is not None:
            _checkpoint(stage_num, future)

    # --- Log summary --------------------------------------------------------
    run_log_dir = ROOT_DIR / "logs" / "dermatology" / "runs" / run_id
    log_summary = summarize_run_logs(run_log_dir)
    if log_summary["total_warnings"] or log_summary["total_errors"]:
        flow_logger.info(
            f"Log summary: {log_summary['total_warnings']} warning(s), "
            f"{log_summary['total_errors']} error(s) - see {run_log_dir / 'run_summary.json'}"
        )

    # --- Summary ------------------------------------------------------------
    flow_logger.info("[PHASE] Dermatology baseline pipeline complete")
    flow_logger.info(f"Stages executed: {first.name} to {last.name}")
    flow_logger.info("Output paths:")
    flow_logger.info(f"  - Run root:           {run_root}")
    if load_task:
        flow_logger.info(f"  - Raw data:           {ROOT_DIR}/data/raw/dermatology")
    if profile_task:
        flow_logger.info(f"  - Profiling:          {run_root}/profiling")
    if recommendations_task:
        flow_logger.info(f"  - Recommendations:    {run_root}/recommendations")
    if preprocess_task:
        flow_logger.info(f"  - Processed data:     {ROOT_DIR}/data/processed/dermatology")
    if train_task:
        flow_logger.info(f"  - Baseline:           {run_root}/baseline")
    if assess_task:
        flow_logger.info(f"  - Fairness:           {run_root}/baseline/fairness")
    if compare_task:
        flow_logger.info(f"  - Comparison:         {run_root}/baseline/comparison")
    if explain_task:
        flow_logger.info(f"  - Explainability:     {run_root}/baseline/explainability")
    if mitigate_task:
        flow_logger.info(f"  - Mitigation:         {run_root}/experiments/mitigation")


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run the dermatology baseline pipeline (Prefect flow).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Stage names (number or name accepted). Numbering is sparse - this domain has
no stage 5 or 6:
    1=load  2=profile  3=recommend  4=preprocess
    7=train  8=assess  9=compare  10=explain  11=mitigate

Examples:
  # Run only through profiling
  %(prog)s --go-until profile

  # Full baseline without the expensive saliency stage
  %(prog)s --no-explain

  # Resume a failed run from training, stop after the fairness assessment
  %(prog)s --resume-from train --go-until assess

  # Augmentation on/off comparison (aug-off keeps the frozen-feature cache)
  %(prog)s --no-augmentation --run-id derm_aug_off
  %(prog)s --augmentation --run-id derm_aug_on
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
        help="Optional model-family override for the train, assess, compare, "
        "explain and mitigate stages.",
    )
    p.add_argument("--device", default=None, help="Torch device override (e.g. cpu, cuda).")
    p.add_argument("--epochs", type=int, default=None, help="Epoch cap override for training.")
    p.add_argument("--batch-size", type=int, default=None, help="Batch size override.")
    pretrained_group = p.add_mutually_exclusive_group()
    pretrained_group.add_argument(
        "--pretrained",
        dest="pretrained",
        action="store_true",
        help="Start from pretrained backbone weights.",
    )
    pretrained_group.add_argument(
        "--no-pretrained",
        dest="pretrained",
        action="store_false",
        help="Train the backbone from scratch.",
    )
    p.set_defaults(pretrained=None)
    augmentation_group = p.add_mutually_exclusive_group()
    augmentation_group.add_argument(
        "--augmentation",
        dest="augmentation",
        action="store_true",
        help="Enable train-only image augmentation. Disables frozen-feature caching.",
    )
    augmentation_group.add_argument(
        "--no-augmentation",
        dest="augmentation",
        action="store_false",
        help="Disable image augmentation, keeping the frozen-feature cache usable.",
    )
    p.set_defaults(augmentation=None)
    figures_group = p.add_mutually_exclusive_group()
    figures_group.add_argument(
        "--figures",
        dest="figures",
        action="store_true",
        help="Render stage figures.",
    )
    figures_group.add_argument(
        "--no-figures",
        dest="figures",
        action="store_false",
        help="Skip stage figures.",
    )
    p.set_defaults(figures=None)
    group_views_group = p.add_mutually_exclusive_group()
    group_views_group.add_argument(
        "--group-views",
        dest="group_views",
        action="store_true",
        help="Emit per-sensitive-group views in the fairness assessment.",
    )
    group_views_group.add_argument(
        "--no-group-views",
        dest="group_views",
        action="store_false",
        help="Skip per-sensitive-group views.",
    )
    p.set_defaults(group_views=None)
    p.add_argument(
        "--no-recommendations", action="store_true", help="Skip the recommendations stage."
    )
    explain_group = p.add_mutually_exclusive_group()
    explain_group.add_argument(
        "--explain",
        dest="run_explain",
        action="store_true",
        help="Force the explainability stage on even if xai.enabled=false in config.",
    )
    explain_group.add_argument(
        "--no-explain",
        dest="run_explain",
        action="store_false",
        help="Skip the explainability stage (the most expensive stage of a cached-feature run).",
    )
    p.set_defaults(run_explain=None)
    p.add_argument(
        "-v", "--verbose", action="count", default=0, help="Verbosity: -v=info, -vv=debug"
    )
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    dermatology_pipeline(
        run_recommendations=not args.no_recommendations,
        run_explain=args.run_explain,
        verbose=args.verbose,
        resume_from=args.resume_from,
        go_until=args.go_until,
        run_id_override=args.run_id,
        datasets=args.datasets,
        model_types=args.model_types,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        pretrained=args.pretrained,
        augmentation=args.augmentation,
        figures=args.figures,
        group_views=args.group_views,
    )
