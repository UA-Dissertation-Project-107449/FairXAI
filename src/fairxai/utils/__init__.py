"""Utilities package public API.

Exports shared helpers for logging and configuration used across the codebase.
"""

from .gpu import detect_accelerator
from .logging_utils import setup_logging

__all__ = [
    "detect_accelerator",
    "setup_logging",
]
