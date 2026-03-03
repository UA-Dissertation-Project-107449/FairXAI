"""Configuration loader for the profiling/complexity module.

Follows the same pattern as ``recommendations.config``: a typed config
class backed by a YAML file, with sensible built-in defaults so that
callers that pass no config continue to work identically.
"""

from __future__ import annotations

import logging
from pathlib import Path

from ..utils.config import load_yaml_config

logger = logging.getLogger(__name__)

# Default config path relative to project root.
_DEFAULT_CONFIG_REL = "configs/profiling/complexity.yaml"

# ── built-in defaults (match the YAML file) ──────────────────────────
_BUILTIN_DEFAULTS: dict[str, object] = {
    "max_samples": 1000,
    "t1_max_samples": 600,
    "raug_k": 5,
    "random_seed": 42,
    "logistic_regression": {
        "solver": "liblinear",
        "max_iter": 1000,
    },
    "default_target": "heart_disease",
}


class ComplexityConfig:
    """Typed, attribute-based access to complexity metric tunables.

    Parameters
    ----------
    raw : dict
        Parsed YAML content (or empty dict for built-in defaults).
    """

    def __init__(self, raw: dict[str, object] | None = None) -> None:
        raw = raw or {}

        self.max_samples: int = int(raw.get("max_samples", _BUILTIN_DEFAULTS["max_samples"]))
        self.t1_max_samples: int = int(raw.get("t1_max_samples", _BUILTIN_DEFAULTS["t1_max_samples"]))
        self.raug_k: int = int(raw.get("raug_k", _BUILTIN_DEFAULTS["raug_k"]))
        self.random_seed: int = int(raw.get("random_seed", _BUILTIN_DEFAULTS["random_seed"]))
        self.default_target: str = str(raw.get("default_target", _BUILTIN_DEFAULTS["default_target"]))

        lr = raw.get("logistic_regression", {})
        lr_defaults = _BUILTIN_DEFAULTS["logistic_regression"]
        self.lr_solver: str = str(lr.get("solver", lr_defaults["solver"]))
        self.lr_max_iter: int = int(lr.get("max_iter", lr_defaults["max_iter"]))

    # Convenience: allow ``dict(cfg)`` for callers that need plain dicts.
    def to_dict(self) -> dict[str, object]:
        return {
            "max_samples": self.max_samples,
            "t1_max_samples": self.t1_max_samples,
            "raug_k": self.raug_k,
            "random_seed": self.random_seed,
            "default_target": self.default_target,
            "logistic_regression": {
                "solver": self.lr_solver,
                "max_iter": self.lr_max_iter,
            },
        }


def load_complexity_config(
    config_path: str | None = None,
    project_root: Path | None = None,
) -> ComplexityConfig:
    """Load and return a :class:`ComplexityConfig`.

    Parameters
    ----------
    config_path : str, optional
        Explicit YAML path.  When *None*, ``<project_root>/configs/profiling/complexity.yaml``
        is tried; if that also fails, built-in defaults are returned.
    project_root : Path, optional
        Repository root used to locate the default YAML.
    """
    if config_path:
        raw = load_yaml_config(config_path)
    elif project_root:
        default_path = project_root / _DEFAULT_CONFIG_REL
        if default_path.exists():
            raw = load_yaml_config(str(default_path))
        else:
            logger.debug(
                "Profiling config not found at %s; using built-in defaults.",
                default_path,
            )
            raw = {}
    else:
        logger.debug("No config path or project root supplied; using built-in defaults.")
        raw = {}

    return ComplexityConfig(raw)
