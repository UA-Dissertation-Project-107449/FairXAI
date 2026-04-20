from __future__ import annotations

import os
import sys
from types import SimpleNamespace

from fairxai.cli.memory_utils import safe_n_jobs


def test_safe_n_jobs_clamps_zero_and_negative_values() -> None:
    assert safe_n_jobs(n_rows=100, n_cols=10, n_requested=0) == 1
    assert safe_n_jobs(n_rows=100, n_cols=10, n_requested=-2) == 1


def test_safe_n_jobs_uses_cpu_count_for_auto_when_shape_unknown(monkeypatch) -> None:
    monkeypatch.setattr(os, "cpu_count", lambda: 6)

    assert safe_n_jobs(n_rows=0, n_cols=0, n_requested=-1) == 6


def test_safe_n_jobs_caps_auto_request_by_memory_budget(monkeypatch) -> None:
    fake_psutil = SimpleNamespace(virtual_memory=lambda: SimpleNamespace(available=200_000_000))
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
    monkeypatch.setattr(os, "cpu_count", lambda: 8)

    capped = safe_n_jobs(
        n_rows=1_000,
        n_cols=1_000,
        n_requested=-1,
        cv_folds=5,
        max_memory_fraction=0.80,
    )

    assert capped == 1
