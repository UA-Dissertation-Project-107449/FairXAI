"""Shared logging utilities for scripts.

Verbosity levels
----------------
0 (default) — quiet: console shows [PHASE]/[SUCCESS]/[ERROR] tags,
              plus any WARNING+ messages.
1 (-v)      — verbose: console shows all INFO+ messages.
2 (-vv)     — debug: console shows all DEBUG+ messages.
"""

from pathlib import Path
from typing import Union
import logging
import warnings


class _PhaseFilter(logging.Filter):
    """Pass only phase-marker lines and WARNING+ records."""

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno >= logging.WARNING:
            return True
        msg = record.getMessage()
        return (
            msg.startswith("[PHASE]")
            or msg.startswith("[SUCCESS]")
            or msg.startswith("[ERROR]")
        )


def _normalise_verbosity(verbose: Union[bool, int]) -> int:
    """Accept legacy ``bool`` (``True`` → 1) or an ``int`` level."""
    if isinstance(verbose, bool):
        return int(verbose)
    return max(0, min(int(verbose), 2))


def setup_logging(
    log_file: Path, verbose: Union[bool, int] = 0
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

    warning_handler = logging.FileHandler(warn_path, mode="w")
    warning_handler.setLevel(logging.WARNING)
    warning_handler.addFilter(lambda r: r.levelno == logging.WARNING)
    warning_handler.setFormatter(formatter)

    error_handler = logging.FileHandler(err_path, mode="w")
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)

    logger.addHandler(warning_handler)
    logger.addHandler(error_handler)

    # --- Capture Python warnings into the logging system --------------------
    logging.captureWarnings(True)
    warnings.simplefilter("default")
    return logger
