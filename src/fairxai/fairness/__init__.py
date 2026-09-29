"""Fairness package public API.

Exports fairness metrics and mitigation entry points used by scripts and
experiments.
"""

from .metrics import FairnessMetrics
from .mitigation import (
    InProcessingMitigation,
    MitigationEngine,
    PostProcessingMitigation,
    PreProcessingMitigation,
)
from .uncertainty import (
    BootstrapResult,
    PairedEffectResult,
    adaptive_bootstrap_replicates,
    bootstrap_fairness_metrics,
    paired_arm_differences,
)

__all__ = [
    "FairnessMetrics",
    "BootstrapResult",
    "bootstrap_fairness_metrics",
    "adaptive_bootstrap_replicates",
    "PairedEffectResult",
    "paired_arm_differences",
    "PreProcessingMitigation",
    "InProcessingMitigation",
    "PostProcessingMitigation",
    "MitigationEngine",
]
