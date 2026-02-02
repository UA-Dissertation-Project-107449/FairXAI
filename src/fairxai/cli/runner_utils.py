"""Shared utilities for script runners."""

from pathlib import Path
import logging
import shutil
from datetime import datetime


def archive_latest_run(base_dir: Path, enabled: bool, logger: logging.Logger) -> None:
    if not enabled:
        return
    latest_dir = base_dir / 'latest_run'
    archives_dir = base_dir / 'archived_runs'
    archives_dir.mkdir(parents=True, exist_ok=True)

    has_files = latest_dir.exists() and any(p.is_file() for p in latest_dir.rglob('*'))
    if not has_files:
        return

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    archive_path = archives_dir / f'run_{timestamp}'
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if archive_path.exists():
        shutil.rmtree(archive_path)
    shutil.move(str(latest_dir), str(archive_path))
    latest_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Archived previous latest_run to: {archive_path}")
