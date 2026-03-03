"""Shared utilities for script runners."""

from __future__ import annotations

from pathlib import Path
import json
import logging
import os
import shutil
import time
import uuid
from datetime import datetime
from typing import Any, Dict, Optional


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


def _sanitize_run_id(run_id: str) -> str:
    safe = ''.join(c if c.isalnum() or c in ('-', '_') else '_' for c in run_id)
    return safe.strip('_') or 'run'


def resolve_run_id(explicit: Optional[str] = None) -> str:
    if explicit:
        safe = _sanitize_run_id(str(explicit))
        return safe if safe.startswith('run_') else f"run_{safe}"

    env_run_id = (
        os.getenv('RUN_ID')
        or os.getenv('PREFECT__RUNTIME__FLOW_RUN_ID')
        or os.getenv('PREFECT_FLOW_RUN_ID')
    )

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
    if env_run_id:
        safe = _sanitize_run_id(str(env_run_id))
        base = safe if safe.startswith('run_') else f"run_{safe}"
        return f"{base}_{timestamp}"

    pid = os.getpid()
    suffix = uuid.uuid4().hex[:6]
    return f"run_{timestamp}_{pid}_{suffix}"


def get_run_root(base_results: Path, run_id: str) -> Path:
    return base_results / 'runs' / _sanitize_run_id(run_id)


def resolve_latest_run_dir(base_results: Path) -> Optional[Path]:
    latest_link = base_results / 'latest_run'
    latest_txt = base_results / 'latest_run.txt'

    if latest_link.exists() and latest_link.is_symlink():
        try:
            target = latest_link.resolve()
            return target
        except OSError:
            return None

    if latest_txt.exists():
        try:
            raw = latest_txt.read_text().strip()
            if not raw:
                return None
            candidate = Path(raw)
            if not candidate.is_absolute():
                candidate = base_results / 'runs' / candidate
            return candidate
        except OSError:
            return None

    if latest_link.exists() and latest_link.is_dir():
        return latest_link

    return None


def _acquire_lock(lock_path: Path, timeout: float = 15.0) -> Optional[int]:
    start = time.time()
    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_RDWR)
            return fd
        except FileExistsError:
            if time.time() - start > timeout:
                return None
            time.sleep(0.1)


def _release_lock(lock_path: Path, fd: Optional[int]) -> None:
    if fd is not None:
        try:
            os.close(fd)
        finally:
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass


def update_latest_pointer(base_results: Path, run_dir: Path, logger: logging.Logger) -> None:
    base_results.mkdir(parents=True, exist_ok=True)
    lock_path = base_results / '.latest_run.lock'
    fd = _acquire_lock(lock_path)
    if fd is None:
        logger.warning("Could not acquire latest pointer lock; skipping update.")
        return

    def _extract_run_id(path: Path) -> Optional[str]:
        parts = list(path.parts)
        if 'runs' in parts:
            idx = parts.index('runs')
            if idx + 1 < len(parts):
                return parts[idx + 1]
        return path.name if path.name else None

    try:
        run_id = _extract_run_id(run_dir)
        if not run_id:
            logger.warning("Could not resolve run_id from run_dir; skipping update.")
            return

        latest_link = base_results / 'latest_run'
        latest_txt = base_results / 'latest_run.txt'

        if latest_link.exists() or latest_link.is_symlink():
            if latest_link.is_symlink():
                latest_link.unlink()
            elif latest_link.is_dir():
                logger.info("latest_run directory exists; leaving intact and updating pointer file instead.")

        if not latest_link.exists():
            try:
                os.symlink(Path('runs') / run_id, latest_link)
                logger.info(f"Updated latest_run symlink -> runs/{run_id}")
            except OSError:
                logger.info("Symlink not supported; using latest_run.txt pointer.")

        latest_txt.write_text(run_id)
    finally:
        _release_lock(lock_path, fd)


def append_run_history(base_results: Path, record: Dict[str, Any]) -> None:
    history_path = base_results / 'run_history.jsonl'
    history_path.parent.mkdir(parents=True, exist_ok=True)
    record = dict(record)
    record.setdefault('timestamp', datetime.now().isoformat())
    with open(history_path, 'a', encoding='utf-8') as handle:
        handle.write(json.dumps(record, default=str))
        handle.write('\n')


def update_log_latest_pointer(
    project_root: Path,
    run_id: str,
    logger: logging.Logger,
    log_subdir: str = "cardiac",
) -> None:
    """Point ``logs/{log_subdir}/latest_run`` at the current run's log dir.

    Re-uses the same atomic-symlink + .txt fallback logic as
    :func:`update_latest_pointer`.
    """
    base_logs = project_root / "logs" / log_subdir
    run_log_dir = base_logs / "runs" / run_id
    run_log_dir.mkdir(parents=True, exist_ok=True)
    update_latest_pointer(base_logs, run_log_dir, logger)
