"""Shared logging utilities for scripts."""

from pathlib import Path
import logging


class _PhaseFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return (
            record.levelno >= logging.WARNING
            or msg.startswith("[PHASE]")
            or msg.startswith("[SUCCESS]")
            or msg.startswith("[ERROR]")
        )


def setup_logging(log_file: Path, verbose: bool = False) -> logging.Logger:
    """Configure logging to file (full) and console (quiet by default)."""
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    if not verbose:
        console_handler.addFilter(_PhaseFilter())

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger
