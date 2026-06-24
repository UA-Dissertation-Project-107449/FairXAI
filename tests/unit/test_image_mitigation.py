"""Unit tests for post-processing dermatology mitigation (stage 11).

Synthetic prediction CSVs only — no torch, no model load. Exercises the
fit-on-train/apply-on-test threshold optimization, per-attribute isolation,
the side-by-side constraints, and the run-level discovery/writing.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from fairxai.fairness.image_mitigation import (
    DEFAULT_CONSTRAINTS,
    _PrecomputedScoreEstimator,
    mitigate_predictions_frame,
    mitigate_run,
)


def _biased_frame(n: int, seed: int) -> pd.DataFrame:
    """Two sex groups with deliberately group-dependent score/label coupling."""
    rng = np.random.default_rng(seed)
    sex = np.array([0] * (n // 2) + [1] * (n - n // 2))
    # group 0 well-calibrated, group 1 scores shifted -> threshold unfairness
    proba = np.where(sex == 0, rng.uniform(0.0, 1.0, n), rng.uniform(0.0, 0.6, n))
    y_true = (proba + rng.normal(0, 0.1, n) >= 0.5).astype(int)
    y_pred = (proba >= 0.5).astype(int)
    return pd.DataFrame(
        {
            "sex": sex,
            "fitzpatrick_group": np.where(sex == 0, "I-II", "III-IV"),
            "y_true": y_true,
            "y_pred": y_pred,
            "y_proba": proba,
        }
    )


def test_precomputed_estimator_returns_two_column_proba() -> None:
    est = _PrecomputedScoreEstimator()
    x = np.array([[0.2], [0.9]])
    proba = est.predict_proba(x)
    assert proba.shape == (2, 2)
    np.testing.assert_allclose(proba[:, 1], [0.2, 0.9])
    assert est.predict(x).tolist() == [0, 1]


def test_mitigate_frame_reports_each_constraint_with_before_after() -> None:
    train = _biased_frame(400, seed=1)
    test = _biased_frame(400, seed=2)
    report = mitigate_predictions_frame(
        train,
        test,
        ["sex"],
        constraints=["demographic_parity", "equalized_odds"],
        min_group_samples=20,
    )
    sex = report["sensitive_attributes"]["sex"]
    assert set(sex["constraints"]) == {"demographic_parity", "equalized_odds"}
    for cr in sex["constraints"].values():
        # either a clean before/after pair, or a recorded error (never silent)
        assert ("summary_deltas" in cr) or ("error" in cr)


def test_mitigate_frame_excludes_degenerate_group_from_fit() -> None:
    # Two healthy sex groups plus a tiny single-class "unknown" (-1) group that
    # fairlearn cannot fit. It must be dropped from eligible_groups, and the
    # constraints must still produce before/after deltas for the fittable groups.
    base = _biased_frame(400, seed=3)
    unknown = pd.DataFrame(
        {
            "sex": [-1] * 5,
            "fitzpatrick_group": ["unknown"] * 5,
            "y_true": [0] * 5,
            "y_pred": [0] * 5,
            "y_proba": [0.1] * 5,
        }
    )
    train = pd.concat([base, unknown], ignore_index=True)
    test = pd.concat([_biased_frame(400, seed=4), unknown], ignore_index=True)
    report = mitigate_predictions_frame(
        train, test, ["sex"], constraints=["equalized_odds"], min_group_samples=20
    )
    sex = report["sensitive_attributes"]["sex"]
    assert "Unknown" not in sex["eligible_groups"]
    assert set(sex["eligible_groups"]) == {"Female", "Male"}
    assert "summary_deltas" in sex["constraints"]["equalized_odds"]


def test_mitigate_frame_notes_when_fewer_than_two_eligible_groups() -> None:
    df = _biased_frame(200, seed=5)
    # min support far above any group -> nothing eligible.
    report = mitigate_predictions_frame(
        df, df, ["sex"], constraints=["equalized_odds"], min_group_samples=10_000
    )
    sex = report["sensitive_attributes"]["sex"]
    assert sex["eligible_groups"] == []
    assert "note" in sex
    assert sex["constraints"] == {}


def test_mitigate_frame_skips_absent_attribute() -> None:
    train = _biased_frame(100, seed=1)
    test = _biased_frame(100, seed=2)
    report = mitigate_predictions_frame(
        train, test, ["nonexistent"], constraints=["demographic_parity"]
    )
    assert report["sensitive_attributes"] == {}


def _write_pair(results: Path, key: str, train: pd.DataFrame, test: pd.DataFrame) -> None:
    preds = results / "predictions"
    preds.mkdir(parents=True, exist_ok=True)
    train_path = preds / f"{key}_train.csv"
    test_path = preds / f"{key}_test.csv"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    (results / f"{key}_metrics.json").write_text(
        json.dumps({"train_predictions": str(train_path), "test_predictions": str(test_path)})
    )


def test_mitigate_run_writes_outputs(tmp_path: Path) -> None:
    run_root = tmp_path / "runs" / "r1"
    results = run_root / "baseline" / "results"
    _write_pair(results, "pad_ufes_20_resnet18", _biased_frame(300, 1), _biased_frame(300, 2))

    reports = mitigate_run(
        run_root, ["sex"], constraints=["demographic_parity"], min_group_samples=20
    )

    assert "pad_ufes_20_resnet18" in reports
    out_dir = run_root / "baseline" / "mitigation"
    assert (out_dir / "mitigation_report.json").exists()
    assert (out_dir / "mitigation_report.md").exists()
    csv = pd.read_csv(out_dir / "mitigation_groups.csv")
    assert {"run_key", "attr", "constraint"} <= set(csv.columns)


def test_default_constraints_are_fairlearn_valid() -> None:
    assert "equalized_odds" in DEFAULT_CONSTRAINTS
    assert "true_positive_rate_parity" in DEFAULT_CONSTRAINTS
