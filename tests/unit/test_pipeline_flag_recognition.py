"""Dry-run flag recognition tests for pipeline orchestrators.

These tests validate that new scope flags are recognized by CLI parsers without
running full pipeline workloads.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PREFECT_FLOW = ROOT / "flows" / "cardiac_pipeline.py"
COMBINATORIAL = ROOT / "scripts" / "experiments" / "run_combinatorial_experiments.py"
BASH_PIPELINE = ROOT / "scripts" / "cardiac" / "cardiac_pipeline.sh"
DERM_BASH_PIPELINE = ROOT / "scripts" / "dermatology" / "dermatology_pipeline.sh"


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        timeout=60,
    )


def test_prefect_help_lists_scope_flags() -> None:
    result = _run([sys.executable, str(PREFECT_FLOW), "--help"])
    assert result.returncode == 0
    output = (result.stdout or "") + (result.stderr or "")
    assert "--datasets" in output
    assert "--model-types" in output
    assert "--study-mode" in output
    assert "--parallel-studies" in output
    assert "--parallel-experiments" in output
    assert "--max-cores" in output
    assert "--hpo-search-n-jobs" in output


def test_combinatorial_help_lists_scope_flags() -> None:
    result = _run([sys.executable, str(COMBINATORIAL), "--help"])
    assert result.returncode == 0
    output = (result.stdout or "") + (result.stderr or "")
    assert "--datasets" in output
    assert "--model-types" in output


def test_bash_parser_accepts_scope_flags_before_stage_validation() -> None:
    # Use an invalid stage so execution stops during stage resolution,
    # proving parser accepted the new flags without launching stage scripts.
    result = _run(
        [
            "bash",
            str(BASH_PIPELINE),
            "--datasets",
            "cleveland",
            "--model-types",
            "logistic_regression",
            "--study-mode",
            "auto_safe",
            "--parallel-studies",
            "--parallel-experiments",
            "--max-cores",
            "4",
            "--hpo-search-n-jobs",
            "2",
            "--go-until",
            "invalid_stage_name",
        ]
    )

    combined = (result.stdout or "") + (result.stderr or "")
    assert result.returncode != 0
    assert "Unknown stage 'invalid_stage_name'" in combined
    assert "Unknown argument '--datasets'" not in combined
    assert "Unknown argument '--model-types'" not in combined
    assert "Unknown argument '--study-mode'" not in combined
    assert "Unknown argument '--parallel-studies'" not in combined
    assert "Unknown argument '--parallel-experiments'" not in combined
    assert "Unknown argument '--max-cores'" not in combined
    assert "Unknown argument '--hpo-search-n-jobs'" not in combined


def test_dermatology_bash_parser_accepts_baseline_flags() -> None:
    result = _run(
        [
            "bash",
            str(DERM_BASH_PIPELINE),
            "--datasets",
            "pad_ufes_20",
            "--model-types",
            "resnet18",
            "--device",
            "cpu",
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--no-pretrained",
            "--go-until",
            "invalid_stage_name",
        ]
    )

    combined = (result.stdout or "") + (result.stderr or "")
    assert result.returncode != 0
    assert "Unknown stage 'invalid_stage_name'" in combined
    assert "Unknown argument '--datasets'" not in combined
    assert "Unknown argument '--model-types'" not in combined
    assert "Unknown argument '--device'" not in combined
    assert "Unknown argument '--epochs'" not in combined
    assert "Unknown argument '--batch-size'" not in combined
    assert "Unknown argument '--no-pretrained'" not in combined
