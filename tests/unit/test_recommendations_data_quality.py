"""Tests for the data-quality triage category and readiness hard restriction."""

from __future__ import annotations

from fairxai.recommendations.config import load_triage_config
from fairxai.recommendations.models import (
    ColumnMeta,
    DatasetIngestion,
    Priority,
    ReadinessStatus,
    Recommendation,
    TriageCategory,
    TriageReport,
)
from fairxai.recommendations.rules import run_all_checks


def _profile(missing=None, duplicate_count=0, index_duplicate_count=0):
    """Minimal but complete-enough profile dict (A–E rules find nothing)."""
    missing = missing or {}
    return {
        "basic_stats": {"n_samples": 200, "n_features": 3},
        "target_distribution": {"counts": {"0": 100, "1": 100}, "imbalance_ratio": 1.0},
        "sensitive_attr_distribution": {
            "sex": {"counts": {"M": 100, "F": 100}, "proportions": {"M": 0.5, "F": 0.5}}
        },
        "representation_balance": {"sex": {"size_ratio": 1.0, "min_group_size": 100}},
        "label_imbalance_by_group": {},
        "group_statistics": {},
        "complexity_metrics": {},
        "group_complexity_metrics": {},
        "intersection_complexity_metrics": {},
        "missing_value_analysis": {
            "total_missing": sum(missing.values()),
            "columns_with_missing": missing,
        },
        "duplicate_analysis": {
            "duplicate_count": duplicate_count,
            "index_duplicate_count": index_duplicate_count,
        },
    }


def _ingestion(label="target", sensitive=("sex",), identifier=None):
    cols = [ColumnMeta(name=n) for n in ("feat_0", "sex", "target")]
    return DatasetIngestion(
        filepath="x.csv",
        columns=cols,
        label_column=label,
        sensitive_columns=list(sensitive),
        identifier_columns=list(identifier or []),
    )


def _config():
    return load_triage_config()


def _status(recs):
    readiness = next(r for r in recs if r.category == TriageCategory.F_READINESS)
    return readiness.evidence["readiness_status"]


def _data_quality(recs):
    return [r for r in recs if r.category == TriageCategory.G_DATA_QUALITY]


def test_clean_dataset_is_ready():
    recs = run_all_checks(_profile(), _ingestion(), _config())
    assert _status(recs) == ReadinessStatus.READY.value
    assert _data_quality(recs) == []


def test_feature_missing_caps_at_caveats():
    recs = run_all_checks(_profile(missing={"feat_0": 10}), _ingestion(), _config())
    assert _status(recs) == ReadinessStatus.READY_WITH_CAVEATS.value
    dq = _data_quality(recs)
    assert any(r.priority == Priority.P1 and "feat_0" in r.title for r in dq)


def test_row_duplicates_cap_at_caveats():
    recs = run_all_checks(_profile(duplicate_count=5), _ingestion(), _config())
    assert _status(recs) == ReadinessStatus.READY_WITH_CAVEATS.value
    dq = _data_quality(recs)
    assert any(r.priority == Priority.P1 and "5 duplicate" in r.title for r in dq)


def test_target_missing_is_not_ready():
    recs = run_all_checks(_profile(missing={"target": 3}), _ingestion(), _config())
    assert _status(recs) == ReadinessStatus.NOT_READY.value
    dq = _data_quality(recs)
    assert any(r.priority == Priority.P0 and "target" in r.title for r in dq)


def test_duplicate_index_is_not_ready():
    recs = run_all_checks(
        _profile(index_duplicate_count=2),
        _ingestion(identifier=["id"]),
        _config(),
    )
    assert _status(recs) == ReadinessStatus.NOT_READY.value
    dq = _data_quality(recs)
    assert any(r.priority == Priority.P0 and "index" in r.title.lower() for r in dq)


def test_to_dict_includes_category_descriptions():
    report = TriageReport(
        readiness_status=ReadinessStatus.READY_WITH_CAVEATS,
        recommendations=[
            Recommendation(
                category=TriageCategory.G_DATA_QUALITY,
                priority=Priority.P1,
                evidence={},
                fairness_relevance="x",
                explainability_relevance="y",
                action="z",
                expected_outcome="w",
            )
        ],
    )
    payload = report.to_dict()
    assert "category_descriptions" in payload
    assert "G" in payload["category_descriptions"]
    assert payload["category_descriptions"]["G"]
