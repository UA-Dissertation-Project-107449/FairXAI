"""Models package public API.

Exports baseline model wrappers and cross-validation training utilities used by
scripts and experiment orchestrators.
"""

from .baseline import BaselineLogisticRegression, generate_predictions_with_metadata
from .cv_trainer import CVTrainer

__all__ = ['BaselineLogisticRegression', 'generate_predictions_with_metadata', 'CVTrainer']
