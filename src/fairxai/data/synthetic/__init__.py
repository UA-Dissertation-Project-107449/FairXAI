"""Synthetic dataset generation for the profiling-sensitivity study.

Public API::

    from fairxai.data.synthetic import (
        SyntheticConfig, GroundTruthColumn,
        generate_abstract, generate_healthcare, generate,
        inject_missingness,
        write_dataset, write_grid_manifest,
        build_grid, build_smoke_grid,
    )
"""

from __future__ import annotations

import pandas as pd

from .config import GroundTruthColumn, SyntheticConfig
from .duplication import inject_duplicates
from .generators import generate_abstract, generate_healthcare
from .grid import build_grid, build_smoke_grid
from .manifest import write_dataset, write_grid_manifest
from .missingness import inject_missingness

__all__ = [
    "SyntheticConfig",
    "GroundTruthColumn",
    "generate_abstract",
    "generate_healthcare",
    "generate",
    "inject_missingness",
    "inject_duplicates",
    "write_dataset",
    "write_grid_manifest",
    "build_grid",
    "build_smoke_grid",
]


def generate(cfg: SyntheticConfig) -> tuple[pd.DataFrame, list[GroundTruthColumn]]:
    """Dispatch to the generator for ``cfg.tier``."""
    if cfg.tier == "abstract":
        return generate_abstract(cfg)
    if cfg.tier == "healthcare":
        return generate_healthcare(cfg)
    raise ValueError(f"Unknown tier: {cfg.tier!r}")
