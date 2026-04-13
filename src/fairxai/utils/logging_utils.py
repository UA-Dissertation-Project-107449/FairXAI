"""Shared logging utilities for scripts.

Verbosity levels
----------------
0 (default) — quiet: console shows [PHASE]/[SUCCESS]/[ERROR] tags,
              plus any WARNING+ messages.
1 (-v)      — verbose: console shows all INFO+ messages.
2 (-vv)     — debug: console shows all DEBUG+ messages.
"""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Union


class _PhaseFilter(logging.Filter):
    """Pass only phase-marker lines and WARNING+ records."""

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno >= logging.WARNING:
            return True
        msg = record.getMessage()
        return msg.startswith("[PHASE]") or msg.startswith("[SUCCESS]") or msg.startswith("[ERROR]")


class _WarningFormatter(logging.Formatter):
    """Formatter that prefixes warning records with their Python category type.

    Produces: ``... - WARNING - [UserWarning] <message>``
    Falls back gracefully for non-warning records.
    """

    def format(self, record: logging.LogRecord) -> str:
        category = getattr(record, "warning_category", None)
        if category:
            saved_msg, saved_args = record.msg, record.args
            record.msg = f"[{category}] {record.getMessage()}"
            record.args = ()
            result = super().format(record)
            record.msg, record.args = saved_msg, saved_args
            return result
        return super().format(record)


class _ErrorFormatter(logging.Formatter):
    """Formatter that prefixes error records with the exception class name.

    Produces: ``... - ERROR - [ValueError] <message>``
    Falls back gracefully when no exc_info is attached.
    """

    def format(self, record: logging.LogRecord) -> str:
        if record.exc_info and record.exc_info[0] is not None:
            exc_name = record.exc_info[0].__name__
            saved_msg, saved_args = record.msg, record.args
            record.msg = f"[{exc_name}] {record.getMessage()}"
            record.args = ()
            result = super().format(record)
            record.msg, record.args = saved_msg, saved_args
            return result
        return super().format(record)


def _normalise_verbosity(verbose: Union[bool, int]) -> int:
    """Accept legacy ``bool`` (``True`` → 1) or an ``int`` level."""
    if isinstance(verbose, bool):
        return int(verbose)
    return max(0, min(int(verbose), 2))


def setup_logging(
    log_file: Path,
    verbose: Union[bool, int] = 0,
    *,
    aggregate_log: Optional[Path] = None,
) -> logging.Logger:
    """Configure logging to file (full) and console (filtered by verbosity).

    Parameters
    ----------
    log_file : Path
        Main log-file path.  ``_warnings.log`` and ``_errors.log``
        siblings are created alongside it.
    verbose : bool | int
        0 = quiet (phase tags + WARNING+),
        1 = INFO+,
        2 = DEBUG+.
        Legacy ``True``/``False`` still accepted (mapped to 1/0).
    aggregate_log : Path, optional
        When provided, a 6th handler appends every record (DEBUG+) from all
        library loggers to this single file.  Opened in ``"a"`` mode so
        successive pipeline phases accumulate without overwriting.
        Format includes the logger name for traceability.
    """
    level = _normalise_verbosity(verbose)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # --- File handler (always full DEBUG) -----------------------------------
    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # --- Console handler (verbosity-dependent) ------------------------------
    console_handler = logging.StreamHandler()
    if level >= 2:
        console_handler.setLevel(logging.DEBUG)
    elif level == 1:
        console_handler.setLevel(logging.INFO)
    else:
        console_handler.setLevel(logging.DEBUG)
        console_handler.addFilter(_PhaseFilter())
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # --- Dedicated warning / error files ------------------------------------
    warn_path = log_file.with_name(f"{log_file.stem}_warnings.log")
    err_path = log_file.with_name(f"{log_file.stem}_errors.log")

    warn_formatter = _WarningFormatter("%(asctime)s - %(levelname)s - %(message)s")
    warning_handler = logging.FileHandler(warn_path, mode="w")
    warning_handler.setLevel(logging.WARNING)
    warning_handler.addFilter(lambda r: r.levelno == logging.WARNING)
    warning_handler.setFormatter(warn_formatter)

    err_formatter = _ErrorFormatter("%(asctime)s - %(levelname)s - %(message)s")
    error_handler = logging.FileHandler(err_path, mode="w")
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(err_formatter)

    logger.addHandler(warning_handler)
    logger.addHandler(error_handler)

    # --- Aggregate run log (append mode, spans all phases) ------------------
    if aggregate_log is not None:
        aggregate_log.parent.mkdir(parents=True, exist_ok=True)
        agg_formatter = logging.Formatter("%(asctime)s [%(name)s] %(levelname)s - %(message)s")
        agg_handler = logging.FileHandler(aggregate_log, mode="a")
        agg_handler.setLevel(logging.DEBUG)
        agg_handler.setFormatter(agg_formatter)
        logger.addHandler(agg_handler)

    # --- Capture Python warnings into the logging system --------------------
    logging.captureWarnings(True)
    warnings.simplefilter("default")

    # Override the showwarning installed by captureWarnings so that the
    # warning category type is surfaced as a structured field on the record.
    # The _WarningFormatter on the warning_handler picks this up as [Category].
    _py_warn_logger = logging.getLogger("py.warnings")

    def _showwarning(  # noqa: WPS430
        message: Warning,
        category: type,
        filename: str,
        lineno: int,
        file: Any = None,
        line: Optional[str] = None,
    ) -> None:
        text = warnings.formatwarning(message, category, filename, lineno, line)
        _py_warn_logger.warning(
            text.rstrip("\n"),
            extra={"warning_category": category.__name__},
        )

    warnings.showwarning = _showwarning
    return logger


# ---------------------------------------------------------------------------
# Post-run summary
# ---------------------------------------------------------------------------


def _count_log_lines(path: Path) -> int:
    """Count non-empty lines in a log file — one line ≈ one record."""
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text().splitlines() if line.strip())


def summarize_run_logs(run_log_dir: Path) -> Dict[str, Any]:
    """Produce a per-phase warning/error tally for a structured run log dir.

    Parameters
    ----------
    run_log_dir : Path
        E.g. ``logs/cardiac/runs/<run_id>``.  Expected to contain numbered
        phase directories such as ``01_load/``, ``02_profile/``, etc.

    Returns
    -------
    dict
        ``{"phases": {<dir_name>: {"warnings": N, "errors": N}}, ...}``
        plus ``total_warnings`` and ``total_errors`` roll-ups.
        The dict is also written as ``run_summary.json`` inside *run_log_dir*.
    """
    summary: Dict[str, Any] = {
        "phases": {},
        "total_warnings": 0,
        "total_errors": 0,
    }

    if not run_log_dir.is_dir():
        return summary

    for phase_dir in sorted(run_log_dir.iterdir()):
        if not phase_dir.is_dir():
            continue

        warn_count = sum(_count_log_lines(f) for f in phase_dir.glob("*_warnings.log"))
        err_count = sum(_count_log_lines(f) for f in phase_dir.glob("*_errors.log"))

        summary["phases"][phase_dir.name] = {
            "warnings": warn_count,
            "errors": err_count,
        }
        summary["total_warnings"] += warn_count
        summary["total_errors"] += err_count

    # Persist alongside the phase directories
    out_path = run_log_dir / "run_summary.json"
    out_path.write_text(json.dumps(summary, indent=2) + "\n")

    return summary
