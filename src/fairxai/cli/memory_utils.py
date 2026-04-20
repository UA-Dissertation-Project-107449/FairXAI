"""Memory-aware worker budgeting utilities for study runners."""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# Bytes per numeric element (float64)
_BYTES_PER_ELEMENT = 8

# Number of in-memory copies estimated per CV fold per job:
# train split + test split + predictions ≈ 3 copies
_CV_COPY_FACTOR = 3


def safe_n_jobs(
    n_rows: int,
    n_cols: int,
    n_requested: int,
    cv_folds: int = 5,
    max_memory_fraction: float = 0.80,
) -> int:
    """Return a job count that keeps estimated CV memory within budget.

    Estimates peak memory as:
        n_rows × n_cols × bytes_per_element × cv_copy_factor × cv_folds × n_jobs

    Falls back to a normalized requested value if psutil is unavailable or
    n_rows/n_cols are non-positive.

    Only ``-1`` is treated as "all CPUs". Any other non-positive value is
    considered invalid and clamped to ``1``.
    """
    cpu_total = os.cpu_count() or 1
    if n_requested == -1:
        resolved_requested = cpu_total
    elif n_requested <= 0:
        logger.warning(
            "Invalid n_jobs=%d; using 1 (only -1 or positive integers are supported).",
            n_requested,
        )
        resolved_requested = 1
    else:
        resolved_requested = n_requested

    if n_rows <= 0 or n_cols <= 0:
        return resolved_requested

    try:
        import psutil

        available_bytes = psutil.virtual_memory().available
    except Exception:
        logger.debug("psutil unavailable — skipping memory cap on n_jobs")
        return resolved_requested

    bytes_per_job = n_rows * n_cols * _BYTES_PER_ELEMENT * _CV_COPY_FACTOR * cv_folds
    if bytes_per_job <= 0:
        return resolved_requested

    budget = int(available_bytes * max_memory_fraction)
    safe = max(1, budget // bytes_per_job)

    capped = min(safe, resolved_requested)
    if capped < resolved_requested:
        logger.warning(
            "Memory cap applied: requested n_jobs=%d → safe n_jobs=%d "
            "(n_rows=%d n_cols=%d cv_folds=%d available_ram=%.1fGB budget=%.1fGB)",
            resolved_requested,
            capped,
            n_rows,
            n_cols,
            cv_folds,
            available_bytes / 1e9,
            budget / 1e9,
        )
    return capped


def warn_if_large_dataset(
    n_rows: int,
    threshold: int = 50_000,
    context: Optional[str] = None,
) -> None:
    """Log a warning when dataset exceeds the row threshold."""
    if n_rows >= threshold:
        ctx = f" ({context})" if context else ""
        logger.warning(
            "Large dataset%s: n_rows=%d >= warn_rows_threshold=%d. "
            "Memory usage and parallelism may need adjustment.",
            ctx,
            n_rows,
            threshold,
        )
