"""Shared logging utilities for scripts."""

from pathlib import Path
import logging
import warnings


class _PhaseFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return (
            record.levelno >= logging.ERROR
            or msg.startswith("[PHASE]")
            or msg.startswith("[SUCCESS]")
            or msg.startswith("[ERROR]")
        )


def setup_logging(log_file: Path, verbose: bool = False) -> logging.Logger:
    """Configure logging to file (full) and console (quiet by default)."""
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    if not verbose:
        console_handler.addFilter(_PhaseFilter())

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

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

    logging.captureWarnings(True)
    warnings.simplefilter("default")
    return logger
