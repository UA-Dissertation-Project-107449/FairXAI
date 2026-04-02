"""Recommendation engine orchestrator.

Ties together ingestion, profiling, configuration, historical references,
and the rule engine to produce a ``TriageReport``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from ..data.profilers import DataProfiler
from .config import TriageConfig, load_triage_config
from .history import HistoricalReference
from .ingestion import DatasetIngestor, confirm_ingestion, ingestion_from_schema
from .models import (
    DatasetIngestion,
    Priority,
    ReadinessStatus,
    Recommendation,
    TriageReport,
)
from .rules import run_all_checks

logger = logging.getLogger(__name__)


class RecommendationEngine:
    """High-level API for generating fairness triage recommendations.

    Typical usage::

        engine = RecommendationEngine(project_root="/path/to/FairXAI")
        ingestion = engine.ingest("/data/my_dataset.csv",
                                   label_column="target",
                                   sensitive_columns=["sex", "age_group"])
        report = engine.generate(ingestion)
        print(report.readiness_status)
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        project_root: Optional[str] = None,
        history_base_path: Optional[str] = None,
    ):
        root = Path(project_root) if project_root else None
        self.project_root = root
        self.config = load_triage_config(config_path=config_path, project_root=root)

        # Historical reference (optional)
        self.history: Optional[HistoricalReference] = None
        if self.config.use_historical:
            h_path = history_base_path or (str(root / "output" / "cardiac") if root else None)
            if h_path and Path(h_path).exists():
                self.history = HistoricalReference(
                    base_path=h_path,
                    use_defaults=self.config.fallback_to_defaults,
                )
            elif self.config.fallback_to_defaults:
                self.history = HistoricalReference(
                    base_path=None,
                    use_defaults=True,
                )

    # ------------------------------------------------------------------
    # Ingestion entry points
    # ------------------------------------------------------------------

    def ingest(
        self,
        csv_path: str,
        *,
        label_column: Optional[str] = None,
        sensitive_columns: Optional[List[str]] = None,
        identifier_columns: Optional[List[str]] = None,
        dataset_name: Optional[str] = None,
    ) -> DatasetIngestion:
        """Auto-detect column metadata from a CSV and return an ingestion descriptor."""
        ingestor = DatasetIngestor()
        return ingestor.ingest(
            csv_path,
            label_column=label_column,
            sensitive_columns=sensitive_columns,
            identifier_columns=identifier_columns,
            dataset_name=dataset_name,
        )

    def ingest_from_schema(
        self,
        schema_path: str,
        dataset_key: str,
        data_dir: Optional[str] = None,
    ) -> DatasetIngestion:
        """Build ingestion from an existing JSON schema (happy path)."""
        return ingestion_from_schema(schema_path, dataset_key, data_dir=data_dir)

    # ------------------------------------------------------------------
    # Core generation
    # ------------------------------------------------------------------

    def generate(
        self,
        ingestion: DatasetIngestion,
        profile: Optional[Dict] = None,
    ) -> TriageReport:
        """Run the full triage pipeline and return a ``TriageReport``.

        Parameters
        ----------
        ingestion : DatasetIngestion
            Describes the dataset, columns, roles, and file location.
        profile : dict, optional
            Pre-computed profiling dict (from ``DataProfiler``).  If *None*,
            the engine loads the CSV and profiles it automatically.
        """
        # --- Step 1: load data if no profile supplied ---
        if profile is None:
            profile = self._profile_dataset(ingestion)

        # --- Step 2: run all triage rules ---
        recommendations = run_all_checks(
            profile=profile,
            ingestion=ingestion,
            config=self.config,
            ref=self.history,
        )

        # --- Step 3: derive readiness status ---
        readiness_rec = next((r for r in recommendations if r.category.value == "F"), None)
        if readiness_rec:
            status_str = readiness_rec.evidence.get("readiness_status", "")
            try:
                readiness = ReadinessStatus(status_str)
            except ValueError:
                readiness = ReadinessStatus.READY_WITH_CAVEATS
        else:
            readiness = ReadinessStatus.READY

        # --- Step 4: build limitations ---
        limitations = self._collect_limitations(ingestion, profile)

        # --- Step 5: dataset summary ---
        summary = {
            "name": ingestion.dataset_name or "unknown",
            "filepath": ingestion.filepath,
            "n_rows": ingestion.n_rows or profile.get("basic_stats", {}).get("n_samples"),
            "n_cols": ingestion.n_cols or profile.get("basic_stats", {}).get("n_features"),
            "label_column": ingestion.label_column,
            "sensitive_columns": ingestion.sensitive_columns,
            "n_classes": len(profile.get("target_distribution", {}).get("counts", {})),
            "has_historical_reference": self.history is not None and self.history.has_history,
        }

        # --- Step 6: visual panel references ---
        visual_panels = self._build_visual_panel_refs(profile, ingestion)

        feature_type_summary = self._build_feature_type_summary(ingestion)
        feature_metadata = [column.to_dict() for column in ingestion.columns]
        columns_with_quality_issues = self._identify_quality_issues(ingestion, profile)

        return TriageReport(
            readiness_status=readiness,
            recommendations=recommendations,
            visual_panels=visual_panels,
            limitations=limitations,
            dataset_summary=summary,
            feature_type_summary=feature_type_summary,
            feature_metadata=feature_metadata,
            columns_with_quality_issues=columns_with_quality_issues,
        )

    # ------------------------------------------------------------------
    # Convenience: generate from CSV path directly
    # ------------------------------------------------------------------

    def generate_from_csv(
        self,
        csv_path: str,
        *,
        label_column: Optional[str] = None,
        sensitive_columns: Optional[List[str]] = None,
        dataset_name: Optional[str] = None,
    ) -> TriageReport:
        """One-shot: ingest + generate in a single call."""
        ingestion = self.ingest(
            csv_path,
            label_column=label_column,
            sensitive_columns=sensitive_columns,
            dataset_name=dataset_name,
        )
        return self.generate(ingestion)

    def generate_from_profile(
        self,
        profile: Dict,
        ingestion: DatasetIngestion,
    ) -> TriageReport:
        """Generate recommendations from a pre-computed profile dict."""
        return self.generate(ingestion, profile=profile)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _profile_dataset(self, ingestion: DatasetIngestion) -> Dict:
        """Load CSV and run DataProfiler."""
        logger.info("Loading %s for profiling…", ingestion.filepath)
        df = pd.read_csv(
            ingestion.filepath,
            sep=ingestion.separator,
            header=0 if ingestion.has_header else None,
            low_memory=False,
        )

        # Update row/col counts
        ingestion.n_rows, ingestion.n_cols = df.shape

        target = ingestion.label_column or "target"
        sensitive = ingestion.sensitive_columns or []

        profiler = DataProfiler(
            sensitive_attrs=sensitive if sensitive else None,
            min_group_samples=self.config.min_group_samples,
        )

        profile = profiler.profile_dataset(
            df,
            target=target,
            dataset_name=ingestion.dataset_name or "unknown",
        )
        return profile

    @staticmethod
    def _collect_limitations(
        ingestion: DatasetIngestion,
        profile: Dict,
    ) -> List[str]:
        """Enumerate known limitations of this triage run."""
        limitations: List[str] = []

        if not ingestion.sensitive_columns:
            limitations.append(
                "No sensitive attributes were declared or detected; "
                "fairness analysis is not possible."
            )

        missing = profile.get("missing_value_analysis", {}).get("total_missing", 0)
        if missing > 0:
            limitations.append(
                f"The dataset contains {missing} missing values. Some profiling "
                "metrics may be affected."
            )

        unconfirmed = [c.name for c in ingestion.columns if not c.user_confirmed]
        if unconfirmed:
            limitations.append(
                f"{len(unconfirmed)} column(s) have auto-detected roles/types "
                "that have not been confirmed by the user."
            )

        return limitations

    @staticmethod
    def _build_visual_panel_refs(
        profile: Dict,
        ingestion: DatasetIngestion,
    ) -> List[Dict]:
        """Return metadata references for TRIAGE_PLAN §5 visuals."""
        panels = []

        panels.append(
            {
                "id": "triage_scorecard",
                "type": "table",
                "description": "Fairness triage scorecard",
            }
        )

        if ingestion.sensitive_columns:
            panels.append(
                {
                    "id": "subgroup_support_heatmap",
                    "type": "heatmap",
                    "description": "Sensitive group × class sample counts",
                    "sensitive_columns": ingestion.sensitive_columns,
                }
            )

        if profile.get("complexity_metrics"):
            panels.append(
                {
                    "id": "complexity_radar",
                    "type": "radar",
                    "description": "Complexity family metrics vs. reference",
                }
            )

        panels.append(
            {
                "id": "readiness_gauge",
                "type": "gauge",
                "description": "Overall readiness status",
            }
        )

        if profile.get("intersection_complexity_metrics"):
            panels.append(
                {
                    "id": "intersectional_risk_map",
                    "type": "matrix",
                    "description": "Intersectional subgroup support + class prevalence",
                }
            )

        return panels

    @staticmethod
    def _build_feature_type_summary(ingestion: DatasetIngestion) -> Dict[str, int]:
        summary: Dict[str, int] = {}
        for column in ingestion.columns:
            detected_type = column.detected_type.value
            summary[detected_type] = summary.get(detected_type, 0) + 1
        return summary

    @staticmethod
    def _identify_quality_issues(
        ingestion: DatasetIngestion,
        profile: Dict,
    ) -> Dict[str, List[str]]:
        issues: Dict[str, List[str]] = {
            "high_missing": [],
            "unconfirmed_role": [],
            "low_unique": [],
        }

        n_rows = ingestion.n_rows or profile.get("basic_stats", {}).get("n_samples") or 0
        high_missing_threshold = max(1, int(n_rows * 0.20)) if n_rows else 0

        for column in ingestion.columns:
            if not column.user_confirmed:
                issues["unconfirmed_role"].append(column.name)

            if column.n_unique is not None and column.n_unique <= 1:
                issues["low_unique"].append(column.name)

            if (
                high_missing_threshold
                and column.n_missing is not None
                and column.n_missing >= high_missing_threshold
            ):
                issues["high_missing"].append(column.name)

        return {key: values for key, values in issues.items() if values}
