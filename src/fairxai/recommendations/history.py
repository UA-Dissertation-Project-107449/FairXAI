"""Historical reference loader for triage recommendation evidence.

Scans prior experiment runs to build reference distributions (median, IQR,
min, max) for complexity and fairness metrics.  When no history is available,
falls back to sensible literature-based defaults.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Literature-based defaults for complexity metrics
# These are approximate ranges from the data-complexity literature and our
# own cardiac experiment runs.  They are used when no run history exists.
# ---------------------------------------------------------------------------

_LITERATURE_DEFAULTS: Dict[str, Dict[str, float]] = {
    "F2": {"min": 0.0, "p25": 0.02, "median": 0.05, "p75": 0.15, "max": 1.0},
    "F3": {"min": 0.0, "p25": 0.01, "median": 0.04, "p75": 0.10, "max": 1.0},
    "F4": {"min": 0.0, "p25": 0.00, "median": 0.02, "p75": 0.08, "max": 1.0},
    "N2": {"min": 0.0, "p25": 0.50, "median": 0.70, "p75": 0.85, "max": 1.0},
    "N3": {"min": 0.0, "p25": 0.15, "median": 0.25, "p75": 0.35, "max": 1.0},
    "N4": {"min": 0.0, "p25": 0.10, "median": 0.20, "p75": 0.30, "max": 1.0},
    "Raug": {"min": 0.0, "p25": 0.20, "median": 0.35, "p75": 0.50, "max": 1.0},
    "L1": {"min": 0.0, "p25": 0.10, "median": 0.20, "p75": 0.35, "max": 1.0},
    "L2": {"min": 0.0, "p25": 0.10, "median": 0.20, "p75": 0.35, "max": 1.0},
    "L3": {"min": 0.0, "p25": 0.10, "median": 0.20, "p75": 0.35, "max": 1.0},
    "T1": {"min": 0.0, "p25": 0.05, "median": 0.15, "p75": 0.30, "max": 1.0},
    "BayesImbalance": {"min": 0.0, "p25": 0.05, "median": 0.15, "p75": 0.40, "max": 1.0},
}

# Fairness metric defaults (from acceptable/violation ranges)
_FAIRNESS_DEFAULTS: Dict[str, Dict[str, float]] = {
    "demographic_parity_difference": {
        "min": 0.0,
        "p25": 0.03,
        "median": 0.08,
        "p75": 0.15,
        "max": 0.50,
    },
    "equalized_odds_tpr_diff": {"min": 0.0, "p25": 0.03, "median": 0.08, "p75": 0.15, "max": 0.50},
    "equalized_odds_fpr_diff": {"min": 0.0, "p25": 0.02, "median": 0.06, "p75": 0.12, "max": 0.40},
    "equal_opportunity_difference": {
        "min": 0.0,
        "p25": 0.03,
        "median": 0.08,
        "p75": 0.15,
        "max": 0.50,
    },
    "predictive_parity_difference": {
        "min": 0.0,
        "p25": 0.03,
        "median": 0.08,
        "p75": 0.15,
        "max": 0.50,
    },
    "statistical_parity_difference": {
        "min": 0.0,
        "p25": 0.05,
        "median": 0.10,
        "p75": 0.20,
        "max": 0.60,
    },
}


# ---------------------------------------------------------------------------
# Reference stats container
# ---------------------------------------------------------------------------


class ReferenceStats:
    """Simple container for a metric's reference distribution."""

    def __init__(self, values: Dict[str, float]):
        self.min = values.get("min", 0.0)
        self.p25 = values.get("p25", 0.0)
        self.median = values.get("median", 0.0)
        self.p75 = values.get("p75", 0.0)
        self.max = values.get("max", 1.0)
        self.n_observations = values.get("n_observations", 0)

    def to_dict(self) -> Dict[str, float]:
        return {
            "min": self.min,
            "p25": self.p25,
            "median": self.median,
            "p75": self.p75,
            "max": self.max,
            "n_observations": self.n_observations,
        }


# ---------------------------------------------------------------------------
# Historical reference builder
# ---------------------------------------------------------------------------


class HistoricalReference:
    """Load and query historical run data for metric reference distributions.

    Typical base_path: ``output/cardiac/`` (contains ``run_history.jsonl``,
    ``archived_runs/``, ``runs/``).
    """

    def __init__(
        self,
        base_path: Optional[str] = None,
        use_defaults: bool = True,
    ):
        self._base = Path(base_path) if base_path else None
        self._use_defaults = use_defaults

        # Cached distributions: metric_name → list of observed values
        self._complexity_values: Dict[str, List[float]] = {}
        self._fairness_values: Dict[str, List[float]] = {}

        if self._base:
            self._scan_history()

    # ------------------------------------------------------------------
    # Scanning
    # ------------------------------------------------------------------

    def _scan_history(self) -> None:
        """Walk archived and current runs collecting profiling + fairness JSONs."""
        if not self._base or not self._base.exists():
            return

        profile_jsons: List[Path] = []
        fairness_jsons: List[Path] = []

        # Look in runs/ and archived_runs/
        for search_root in (self._base / "runs", self._base / "archived_runs"):
            if not search_root.exists():
                continue
            for p in search_root.rglob("*_data_profile.json"):
                profile_jsons.append(p)
            for p in search_root.rglob("*_fairness_assessment.json"):
                fairness_jsons.append(p)

        # Also check non-run-scoped profiling (baseline results)
        profiling_dir = self._base / "profiling"
        if profiling_dir.exists():
            for p in profiling_dir.rglob("*_data_profile.json"):
                profile_jsons.append(p)

        baseline_fairness = self._base / "baseline" / "fairness"
        if baseline_fairness.exists():
            for p in baseline_fairness.rglob("*_fairness_assessment.json"):
                fairness_jsons.append(p)

        # Deduplicate by resolved path
        profile_jsons = list({p.resolve(): p for p in profile_jsons}.values())
        fairness_jsons = list({p.resolve(): p for p in fairness_jsons}.values())

        logger.info(
            "Historical scan: %d profile JSONs, %d fairness JSONs",
            len(profile_jsons),
            len(fairness_jsons),
        )

        self._extract_complexity(profile_jsons)
        self._extract_fairness(fairness_jsons)

    def _extract_complexity(self, paths: List[Path]) -> None:
        for p in paths:
            try:
                with open(p, "r") as fh:
                    data = json.load(fh)
                metrics = data.get("complexity_metrics", {})
                for name, val in metrics.items():
                    if val is not None and isinstance(val, (int, float)):
                        self._complexity_values.setdefault(name, []).append(float(val))
            except Exception:
                logger.debug("Could not read profile %s", p, exc_info=True)

    def _extract_fairness(self, paths: List[Path]) -> None:
        for p in paths:
            try:
                with open(p, "r") as fh:
                    data = json.load(fh)
                gf = data.get("group_fairness", {})
                for attr_data in gf.values():
                    dp = attr_data.get("demographic_parity", {})
                    if "max_difference" in dp:
                        self._fairness_values.setdefault(
                            "demographic_parity_difference", []
                        ).append(float(dp["max_difference"]))

                    eo = attr_data.get("equalized_odds", {})
                    if "tpr_max_difference" in eo:
                        self._fairness_values.setdefault("equalized_odds_tpr_diff", []).append(
                            float(eo["tpr_max_difference"])
                        )
                    if "fpr_max_difference" in eo:
                        self._fairness_values.setdefault("equalized_odds_fpr_diff", []).append(
                            float(eo["fpr_max_difference"])
                        )

                    eop = attr_data.get("equal_opportunity", {})
                    if "max_difference" in eop:
                        self._fairness_values.setdefault("equal_opportunity_difference", []).append(
                            float(eop["max_difference"])
                        )

                    pp = attr_data.get("predictive_parity", {})
                    if "max_difference" in pp:
                        self._fairness_values.setdefault("predictive_parity_difference", []).append(
                            float(pp["max_difference"])
                        )
            except Exception:
                logger.debug("Could not read fairness %s", p, exc_info=True)

    # ------------------------------------------------------------------
    # Query interface
    # ------------------------------------------------------------------

    def get_complexity_reference(self, metric: str) -> Optional[ReferenceStats]:
        """Return reference distribution for a complexity metric."""
        values = self._complexity_values.get(metric)
        if values and len(values) >= 2:
            return self._stats_from_values(values)
        if self._use_defaults and metric in _LITERATURE_DEFAULTS:
            return ReferenceStats(_LITERATURE_DEFAULTS[metric])
        return None

    def get_fairness_reference(self, metric: str) -> Optional[ReferenceStats]:
        """Return reference distribution for a fairness metric."""
        values = self._fairness_values.get(metric)
        if values and len(values) >= 2:
            return self._stats_from_values(values)
        if self._use_defaults and metric in _FAIRNESS_DEFAULTS:
            return ReferenceStats(_FAIRNESS_DEFAULTS[metric])
        return None

    @property
    def has_history(self) -> bool:
        return bool(self._complexity_values or self._fairness_values)

    @staticmethod
    def _stats_from_values(values: List[float]) -> ReferenceStats:
        arr = np.array(values, dtype=float)
        return ReferenceStats(
            {
                "min": float(np.nanmin(arr)),
                "p25": float(np.nanpercentile(arr, 25)),
                "median": float(np.nanmedian(arr)),
                "p75": float(np.nanpercentile(arr, 75)),
                "max": float(np.nanmax(arr)),
                "n_observations": len(arr),
            }
        )
