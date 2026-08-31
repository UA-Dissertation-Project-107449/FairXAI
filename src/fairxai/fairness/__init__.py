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
    adaptive_bootstrap_replicates,
    bootstrap_fairness_metrics,
)

__all__ = [
    "FairnessMetrics",
    "BootstrapResult",
    "bootstrap_fairness_metrics",
    "adaptive_bootstrap_replicates",
    "PreProcessingMitigation",
    "InProcessingMitigation",
    "PostProcessingMitigation",
    "MitigationEngine",
]
