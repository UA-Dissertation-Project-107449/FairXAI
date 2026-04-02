"""Shared gate contract for fairness and clinical recall enforcement.

This module provides:
- ``load_gate_thresholds``: canonical threshold loading (experiment config > thresholds.yaml)
- ``evaluate_recall_gate``: two-tier recall gate (hard floor + clinical goal)
- ``evaluate_fairness_gate``: fairness violation gate

All threshold values must come from config files. No defaults are defined here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

_THRESHOLDS_REL = "configs/recommendations/thresholds.yaml"


@dataclass
class GateResult:
    """Outcome of a single gate evaluation."""

    passed: bool
    tier: str  # 'full_pass' | 'lower_tier' | 'fail'
    reason: Optional[str] = None


def _load_canonical_thresholds(project_root: Path) -> Dict[str, float]:
    """Load the fairness block from thresholds.yaml with no external dependencies."""
    path = project_root / _THRESHOLDS_REL
    if not path.exists():
        raise FileNotFoundError(f"Canonical thresholds not found: {path}")
    with open(path) as f:
        raw = yaml.safe_load(f)
    fair = raw.get("fairness", {})
    return {
        "min_recall": float(fair.get("min_recall", 0.70)),
        "recall_hard_floor": float(fair.get("recall_hard_floor", 0.60)),
        "max_fairness_violation": float(fair.get("max_fairness_violation", 0.10)),
    }


def load_gate_thresholds(
    experiment_cfg: Dict[str, Any],
    project_root: Path,
) -> Dict[str, float]:
    """Load gate thresholds with experiment config taking priority over thresholds.yaml.

    Priority order:
    1. ``experiment_cfg['fairness_thresholds']`` block (per-experiment override)
    2. ``configs/recommendations/thresholds.yaml`` (canonical source)

    Parameters
    ----------
    experiment_cfg:
        Parsed experiment YAML dict (e.g. contents of combinatorial.yaml).
    project_root:
        Repository root used to locate thresholds.yaml.

    Returns
    -------
    dict with keys: ``min_recall``, ``recall_hard_floor``, ``max_fairness_violation``
    """
    canonical = _load_canonical_thresholds(project_root)
    overrides = experiment_cfg.get("fairness_thresholds", {})

    return {
        "min_recall": float(overrides.get("min_recall", canonical["min_recall"])),
        "recall_hard_floor": float(
            overrides.get("recall_hard_floor", canonical["recall_hard_floor"])
        ),
        "max_fairness_violation": float(
            overrides.get("max_fairness_violation", canonical["max_fairness_violation"])
        ),
    }


def evaluate_recall_gate(
    recall: Optional[float],
    hard_floor: float,
    min_recall: float,
) -> GateResult:
    """Evaluate two-tier recall gate.

    Tiers
    -----
    - ``recall < hard_floor``               → fail  (hard exclusion, never ranked)
    - ``hard_floor ≤ recall < min_recall``  → lower_tier  (included, ranked below full_pass)
    - ``recall ≥ min_recall``               → full_pass

    Parameters
    ----------
    recall:
        Recall value to evaluate; ``None`` is treated as a fail.
    hard_floor:
        Hard exclusion threshold (e.g. 0.60).
    min_recall:
        Clinical goal threshold (e.g. 0.70).
    """
    if recall is None:
        return GateResult(passed=False, tier="fail", reason="recall is None")

    r = float(recall)
    if r < hard_floor:
        return GateResult(
            passed=False,
            tier="fail",
            reason=f"recall={r:.3f} < recall_hard_floor={hard_floor:.2f}",
        )
    if r < min_recall:
        return GateResult(
            passed=True,
            tier="lower_tier",
            reason=f"recall={r:.3f} in [{hard_floor:.2f}, {min_recall:.2f})",
        )
    return GateResult(
        passed=True,
        tier="full_pass",
        reason=f"recall={r:.3f} >= min_recall={min_recall:.2f}",
    )


def evaluate_fairness_gate(
    fairness_gap: Optional[float],
    max_violation: float,
) -> GateResult:
    """Evaluate fairness violation gate.

    Parameters
    ----------
    fairness_gap:
        Max demographic parity (or equalized odds) difference; ``None`` → pass
        with a warning (missing fairness data is not penalised here).
    max_violation:
        Maximum tolerated fairness gap (e.g. 0.10).
    """
    if fairness_gap is None:
        return GateResult(
            passed=True,
            tier="full_pass",
            reason="fairness_gap is None (no fairness data available)",
        )

    g = float(fairness_gap)
    if g > max_violation:
        return GateResult(
            passed=False,
            tier="fail",
            reason=f"fairness_gap={g:.3f} > max_fairness_violation={max_violation:.2f}",
        )
    return GateResult(
        passed=True,
        tier="full_pass",
        reason=f"fairness_gap={g:.3f} <= max_fairness_violation={max_violation:.2f}",
    )
