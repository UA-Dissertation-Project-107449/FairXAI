"""GPU/accelerator detection utilities."""

from __future__ import annotations

import subprocess


def _torch_device_available(kind: str) -> bool:
    try:
        import torch  # type: ignore
    except Exception:
        return False

    if kind == "cuda":
        return bool(torch.cuda.is_available() and not getattr(torch.version, "hip", None))
    if kind == "rocm":
        return bool(torch.cuda.is_available() and getattr(torch.version, "hip", None))
    return False


def detect_accelerator(requested: str = "auto") -> str:
    """Resolve the compute accelerator to use.

    Args:
        requested: One of ``'auto'``, ``'cuda'``, ``'rocm'``, or ``'cpu'``.
            ``'auto'`` probes CUDA, then ROCm, then CPU.

    Returns:
        ``'cuda'``, ``'rocm'``, or ``'cpu'``.
    """
    val = str(requested).strip().lower()
    if val == "cpu":
        return "cpu"
    if val == "cuda":
        return "cuda" if _torch_device_available("cuda") else "cpu"
    if val == "rocm":
        return "rocm" if _torch_device_available("rocm") else "cpu"

    if _torch_device_available("cuda"):
        return "cuda"
    if _torch_device_available("rocm"):
        return "rocm"

    # Lightweight fallback for environments without torch installed.
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
