"""Fairness package public API.

Exports fairness metrics and mitigation entry points used by scripts and
experiments.
"""

from .metrics import FairnessMetrics
from .mitigation import (
    PreProcessingMitigation,
    InProcessingMitigation,
    PostProcessingMitigation,
    MitigationEngine
)

__all__ = [
    'FairnessMetrics',
    'PreProcessingMitigation',
    'InProcessingMitigation',
    'PostProcessingMitigation',
    'MitigationEngine'
]
