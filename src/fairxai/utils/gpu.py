"""GPU/accelerator detection utilities."""

from __future__ import annotations

import subprocess


def detect_accelerator(requested: str = "auto") -> str:
    """Resolve the compute accelerator to use.

    Args:
        requested: One of ``'auto'``, ``'cuda'``, or ``'cpu'``.
            ``'auto'`` probes for an NVIDIA GPU via ``nvidia-smi``; returns
            ``'cuda'`` when one is found and ``'cpu'`` otherwise.
            AMD/ROCm is not supported and always resolves to ``'cpu'``.

    Returns:
        ``'cuda'`` or ``'cpu'``.
    """
    val = str(requested).strip().lower()
    if val == "cpu":
        return "cpu"
    if val == "cuda":
        return "cuda"
    # auto: probe for NVIDIA GPU (no heavy import needed)
    try:
        result = subprocess.run(
            ["nvidia-smi", "--list-gpus"],
            capture_output=True,
            timeout=3,
        )
        if result.returncode == 0 and result.stdout.strip():
            return "cuda"
    except Exception:
        pass
    return "cpu"
