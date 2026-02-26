"""Configuration loader for the recommendation engine."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..utils.config import load_yaml_config

logger = logging.getLogger(__name__)

# Default config path relative to project root
_DEFAULT_CONFIG_REL = "configs/recommendations/thresholds.yaml"


class TriageConfig:
    """Typed access to recommendation thresholds loaded from YAML."""

    def __init__(self, raw: Dict[str, Any]):
        self._raw = raw

        # Representation
        rep = raw.get("representation", {})
        self.size_ratio_warning: float = rep.get("size_ratio_warning", 3.0)
        self.min_group_samples: int = rep.get("min_group_samples", 50)
        self.statistical_parity_warning: float = rep.get("statistical_parity_warning", 0.15)
        self.intersectional_min_samples: int = rep.get("intersectional_min_samples", 30)

        # Complexity
        comp = raw.get("complexity", {})
        self.high_overlap_percentile: int = comp.get("high_overlap_percentile", 75)
        self.elevated_metrics: List[str] = comp.get(
            "elevated_metrics", ["F2", "F3", "F4", "N2", "N3", "N4", "Raug", "T1"]
        )
        self.group_divergence_threshold: float = comp.get("group_divergence_threshold", 0.20)

        # Explainability
        exp = raw.get("explainability", {})
        self.linear_complexity_metrics: List[str] = exp.get(
            "linear_complexity_metrics", ["L1", "L2", "L3"]
        )
        self.structural_overlap_metric: str = exp.get("structural_overlap_metric", "T1")
        self.explainability_high_threshold: float = exp.get("high_threshold", 0.5)

        # Readiness
        rdy = raw.get("readiness", {})
        self.p0_makes_not_ready: bool = rdy.get("p0_makes_not_ready", True)
        self.p1_caveat_threshold: int = rdy.get("p1_caveat_threshold", 1)

        # Fairness
        fair = raw.get("fairness", {})
        self.max_fairness_violation: float = fair.get("max_fairness_violation", 0.10)
        self.min_recall: float = fair.get("min_recall", 0.70)

        # Task framing
        tf = raw.get("task_framing", {})
        self.multiclass_minority_support: int = tf.get("multiclass_minority_support", 20)
        self.complexity_warning_metrics: List[str] = tf.get(
            "complexity_warning_metrics", ["N3", "N4", "T1", "F4"]
        )
        self.complexity_high_threshold: float = tf.get("complexity_high_threshold", 0.5)

        # Sensitive adequacy
        sa = raw.get("sensitive_adequacy", {})
        self.max_null_fraction: float = sa.get("max_null_fraction", 0.10)
        self.min_unique_groups: int = sa.get("min_unique_groups", 2)

        # Reference
        ref = raw.get("reference", {})
        self.use_historical: bool = ref.get("use_historical", True)
        self.fallback_to_defaults: bool = ref.get("fallback_to_defaults", True)


def load_triage_config(
    config_path: Optional[str] = None,
    project_root: Optional[Path] = None,
) -> TriageConfig:
    """Load and return a ``TriageConfig``.

    Parameters
    ----------
    config_path : str, optional
        Explicit path to the thresholds YAML.  If *None*, the default
        location ``<project_root>/configs/recommendations/thresholds.yaml``
        is used.
    project_root : Path, optional
        Repository root (needed when *config_path* is not given).
    """
    if config_path:
        raw = load_yaml_config(config_path)
    elif project_root:
        raw = load_yaml_config(str(project_root / _DEFAULT_CONFIG_REL))
    else:
        logger.warning("No config path or project root supplied; using built-in defaults.")
        raw = {}

    return TriageConfig(raw)
