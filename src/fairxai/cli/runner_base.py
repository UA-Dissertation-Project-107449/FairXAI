"""Shared helpers for script runners."""

from pathlib import Path
from typing import Dict, Optional

from fairxai.utils.config import load_yaml_config
from fairxai.utils.logging_utils import setup_logging


def get_project_root(current_file: Path) -> Path:
    """Return repo root given a script path under scripts/"""
    return current_file.resolve().parents[2]


def load_pipeline_config(root: Path, pipeline: str = "cardiac") -> Dict:
    return load_yaml_config(str(root / f"configs/pipelines/{pipeline}.yaml"))


def setup_phase_logging(
    root: Path,
    log_name: str,
    verbose: bool = False,
    log_subdir: str = "cardiac",
) -> Path:
    log_dir = root / "logs" / log_subdir
    setup_logging(log_dir / log_name, verbose=verbose)
    return log_dir
