"""CLI helper package for FairXAI."""

from .runner_base import (
    get_project_root,
    load_pipeline_config,
    setup_phase_logging,
    setup_study_logging,
)
from .runner_utils import (
    append_run_history,
    archive_latest_run,
    get_run_root,
    resolve_latest_run_dir,
    resolve_run_id,
    update_latest_pointer,
)

__all__ = [
    "get_project_root",
    "load_pipeline_config",
    "setup_phase_logging",
    "setup_study_logging",
    "append_run_history",
    "archive_latest_run",
    "get_run_root",
    "resolve_latest_run_dir",
    "resolve_run_id",
    "update_latest_pointer",
]
