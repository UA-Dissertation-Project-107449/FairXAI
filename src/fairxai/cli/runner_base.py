"""Shared helpers for script runners."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional, Union

from fairxai.utils.config import load_yaml_config
from fairxai.utils.logging_utils import setup_logging


def get_project_root(current_file: Path) -> Path:
    """Return repo root given a script path under scripts/"""
    return current_file.resolve().parents[2]


def resolve_project_root(
    current_file: Path,
    *,
    cli_project_root: Optional[str] = None,
    env_var_name: str = "FAIRXAI_PROJECT_ROOT",
) -> Path:
    """Resolve project root with explicit override precedence.

    Precedence:
    1) ``cli_project_root`` argument, when provided.
    2) Environment variable named by ``env_var_name``.
    3) Default root inferred from script location.
    """
    if cli_project_root:
        return Path(cli_project_root).expanduser().resolve()

    env_root = os.getenv(env_var_name)
    if env_root:
        return Path(env_root).expanduser().resolve()

    return get_project_root(current_file)


def load_pipeline_config(root: Path, pipeline: str = "cardiac") -> Dict:
    return load_yaml_config(str(root / f"configs/pipelines/{pipeline}.yaml"))


def setup_phase_logging(
    root: Path,
    log_name: str,
    verbose: Union[bool, int] = 0,
    log_subdir: str = "cardiac",
    *,
    run_id: Optional[str] = None,
    stage_name: Optional[str] = None,
    sub_stage: Optional[str] = None,
    stage_number: Optional[int] = None,
) -> Path:
    """Configure per-phase logging.

    When *run_id* **and** *stage_name* are both supplied, logs are written to a
    numbered phase directory under the run::

        logs/{log_subdir}/runs/{run_id}/{NN}_{stage_name}/{stage_name}.log

    When *sub_stage* is also given the log files are placed inside the parent
    stage's numbered directory but named after *sub_stage*::

        logs/{log_subdir}/runs/{run_id}/{NN}_{stage_name}/{sub_stage}.log

    ``stage_number`` can override the shared cardiac stage registry number for
    domain-specific runners whose lean stages do not match the full registry.

    An aggregate ``run.log`` is appended to whenever *run_id* is provided,
    collecting records from every phase in a single file::

        logs/{log_subdir}/runs/{run_id}/run.log

    Otherwise the legacy flat layout is used::

        logs/{log_subdir}/{log_name}
    """
    aggregate_log: Optional[Path] = None

    if run_id and stage_name:
        from fairxai.pipeline.stages import STAGE_BY_NAME  # local to avoid circular

        stage = STAGE_BY_NAME.get(stage_name.lower()) if stage_number is None else None
        if stage:
            phase_dir = f"{stage.number:02d}_{stage.name}"
            log_dir = root / "logs" / log_subdir / "runs" / run_id / phase_dir
            log_file = log_dir / (f"{sub_stage}.log" if sub_stage else f"{stage.name}.log")
        elif stage_number is not None:
            phase_dir = f"{stage_number:02d}_{stage_name}"
            log_dir = root / "logs" / log_subdir / "runs" / run_id / phase_dir
            log_file = log_dir / (f"{sub_stage}.log" if sub_stage else f"{stage_name}.log")
        else:
            # Unknown stage — fall back to named directory
            log_dir = root / "logs" / log_subdir / "runs" / run_id / stage_name
            log_file = log_dir / f"{stage_name}.log"
        aggregate_log = root / "logs" / log_subdir / "runs" / run_id / "run.log"
    else:
        log_dir = root / "logs" / log_subdir
        log_file = log_dir / log_name

    setup_logging(log_file, verbose=verbose, aggregate_log=aggregate_log)
    return log_dir


def setup_study_logging(
    root: Path,
    study_type: str,
    study_id: str,
    log_name: str,
    verbose: Union[bool, int] = 0,
    log_subdir: str = "cardiac",
) -> Path:
    """Configure logging for an experiment study (feature selection, HPO, etc.).

    Logs are written under::

        logs/{log_subdir}/studies/{study_type}/{study_id}/{log_name}

    Parameters
    ----------
    root : Path
        Project root.
    study_type : str
        Study category, e.g. ``"feature_selection"`` or ``"hpo"``.
    study_id : str
        Unique identifier for this study run (use :func:`resolve_run_id`).
    log_name : str
        Log filename, e.g. ``"study.log"``.
    verbose : bool | int
        Verbosity level passed to :func:`setup_logging`.
    log_subdir : str
        Pipeline name, e.g. ``"cardiac"``.
    """
    log_dir = root / "logs" / log_subdir / "studies" / study_type / study_id
    log_file = log_dir / log_name
    setup_logging(log_file, verbose=verbose)
    return log_dir
