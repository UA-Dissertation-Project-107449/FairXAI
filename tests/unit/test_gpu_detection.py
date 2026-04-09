"""Unit tests for fairxai.utils.gpu.detect_accelerator."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from fairxai.utils.gpu import detect_accelerator

VALID_DEVICES = {"cpu", "cuda"}


def test_force_cpu_always_returns_cpu():
    assert detect_accelerator("cpu") == "cpu"


def test_force_cuda_returns_valid_device():
    # On CI/laptops without CUDA, "cuda" mode may still return "cpu" as fallback.
    result = detect_accelerator("cuda")
    assert result in VALID_DEVICES


def test_auto_returns_valid_device():
    result = detect_accelerator("auto")
    assert result in VALID_DEVICES


def test_auto_never_raises():
    """detect_accelerator("auto") must not raise even without nvidia-smi."""
    try:
        detect_accelerator("auto")
    except Exception as exc:
        pytest.fail(f"detect_accelerator('auto') raised unexpectedly: {exc}")


def test_unknown_accelerator_falls_back_to_cpu():
    """An unrecognised accelerator string should not crash; expect cpu fallback."""
    result = detect_accelerator("rocm")  # AMD not supported
    assert result in VALID_DEVICES
