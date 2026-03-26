"""Data models for the fairness triage recommendation engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Column metadata
# ---------------------------------------------------------------------------

class ColumnType(str, Enum):
    """Detected or declared column data type."""
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    BINARY = "binary"
    IDENTIFIER = "identifier"
    DATETIME = "datetime"
    TEXT = "text"
    UNKNOWN = "unknown"


class ColumnRole(str, Enum):
    """Semantic role a column plays in the dataset."""
    FEATURE = "feature"
    LABEL = "label"
    SENSITIVE = "sensitive"
    IDENTIFIER = "identifier"
    EXCLUDE = "exclude"


@dataclass
class ColumnMeta:
    """Metadata about a single column in a dataset."""
    name: str
    detected_type: ColumnType = ColumnType.UNKNOWN
    role: ColumnRole = ColumnRole.FEATURE
    user_confirmed: bool = False
    n_unique: Optional[int] = None
    n_missing: Optional[int] = None
    sample_values: Optional[List[Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "detected_type": self.detected_type.value,
            "role": self.role.value,
            "user_confirmed": self.user_confirmed,
            "n_unique": self.n_unique,
            "n_missing": self.n_missing,
            "sample_values": self.sample_values,
        }


# ---------------------------------------------------------------------------
# Dataset ingestion descriptor
# ---------------------------------------------------------------------------

@dataclass
class DatasetIngestion:
    """Fully described dataset ready for profiling and recommendation."""
    filepath: str
    columns: List[ColumnMeta] = field(default_factory=list)
    label_column: Optional[str] = None
    sensitive_columns: List[str] = field(default_factory=list)
    identifier_columns: List[str] = field(default_factory=list)
    has_header: bool = True
    separator: str = ","
    n_rows: Optional[int] = None
    n_cols: Optional[int] = None
    dataset_name: Optional[str] = None

    # Convenience helpers -------------------------------------------------------

    @property
    def feature_columns(self) -> List[str]:
        return [c.name for c in self.columns if c.role == ColumnRole.FEATURE]

    @property
    def exclude_columns(self) -> List[str]:
        return [c.name for c in self.columns if c.role == ColumnRole.EXCLUDE]

    def get_column(self, name: str) -> Optional[ColumnMeta]:
        for c in self.columns:
            if c.name == name:
                return c
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "filepath": self.filepath,
            "columns": [c.to_dict() for c in self.columns],
            "label_column": self.label_column,
            "sensitive_columns": self.sensitive_columns,
            "identifier_columns": self.identifier_columns,
            "has_header": self.has_header,
            "separator": self.separator,
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "dataset_name": self.dataset_name,
        }


# ---------------------------------------------------------------------------
# Triage recommendations
# ---------------------------------------------------------------------------

class Priority(str, Enum):
    """Recommendation priority / severity."""
    P0 = "P0"  # Critical – fairness assessment validity at risk
    P1 = "P1"  # High – strong overlap/imbalance distorts fairness interpretation
    P2 = "P2"  # Medium – explainability reliability concerns
    P3 = "P3"  # Info – documentation and monitoring suggestions


class Confidence(str, Enum):
    """Confidence in a recommendation based on evidence quality."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class TriageCategory(str, Enum):
    """Categories from the TRIAGE_PLAN."""
    A_TASK_FRAMING = "A"          # Task framing readiness
    B_SENSITIVE_ADEQUACY = "B"    # Sensitive-attribute adequacy
    C_REPRESENTATION = "C"        # Representation and subgroup support risk
    D_OVERLAP_AMBIGUITY = "D"     # Overlap and local ambiguity risk
    E_EXPLAINABILITY = "E"        # Explainability suitability
    F_READINESS = "F"             # Fairness benchmark readiness status


@dataclass
class Recommendation:
    """A single triage recommendation with evidence and action."""
    category: TriageCategory
    priority: Priority
    evidence: Dict[str, Any]
    fairness_relevance: str
    explainability_relevance: str
    action: str
    expected_outcome: str
    confidence: Confidence = Confidence.MEDIUM
    title: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category.value,
            "priority": self.priority.value,
            "title": self.title or f"Category {self.category.value}",
            "evidence": self.evidence,
            "fairness_relevance": self.fairness_relevance,
            "explainability_relevance": self.explainability_relevance,
            "action": self.action,
            "expected_outcome": self.expected_outcome,
            "confidence": self.confidence.value,
        }


# ---------------------------------------------------------------------------
# Readiness status
# ---------------------------------------------------------------------------

class ReadinessStatus(str, Enum):
    """Overall fairness benchmark readiness."""
    READY = "Ready"
    READY_WITH_CAVEATS = "Ready with caveats"
    NOT_READY = "Not ready"


# ---------------------------------------------------------------------------
# Triage report (top-level output)
# ---------------------------------------------------------------------------

@dataclass
class TriageReport:
    """Complete triage output: readiness + recommendations + metadata."""
    readiness_status: ReadinessStatus
    recommendations: List[Recommendation] = field(default_factory=list)
    visual_panels: List[Dict[str, Any]] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    dataset_summary: Dict[str, Any] = field(default_factory=dict)
    feature_type_summary: Dict[str, int] = field(default_factory=dict)
    feature_metadata: List[Dict[str, Any]] = field(default_factory=list)
    columns_with_quality_issues: Dict[str, List[str]] = field(default_factory=dict)
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    # Convenience filters -------------------------------------------------------

    def by_priority(self, priority: Priority) -> List[Recommendation]:
        return [r for r in self.recommendations if r.priority == priority]

    def by_category(self, category: TriageCategory) -> List[Recommendation]:
        return [r for r in self.recommendations if r.category == category]

    @property
    def critical_count(self) -> int:
        return len(self.by_priority(Priority.P0))

    @property
    def high_count(self) -> int:
        return len(self.by_priority(Priority.P1))

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "readiness_status": self.readiness_status.value,
            "recommendations": [r.to_dict() for r in self.recommendations],
            "visual_panels": self.visual_panels,
            "limitations": self.limitations,
            "dataset_summary": self.dataset_summary,
            "generated_at": self.generated_at,
        }

        if self.feature_type_summary:
            payload["feature_type_summary"] = self.feature_type_summary
        if self.feature_metadata:
            payload["feature_metadata"] = self.feature_metadata
        if self.columns_with_quality_issues:
            payload["columns_with_quality_issues"] = self.columns_with_quality_issues

        return payload
