"""Stage 10's cross-validation protocol.

The headline mitigation numbers for the two small cardiac cohorts come from
pooled out-of-fold predictions rather than one 61-row test split. These tests
cover the wiring that makes that reportable: every patient scored exactly once,
every arm scoring the same rows as its baseline (which is what keeps the paired
comparison valid), and per-fold spread recorded alongside the pooled metrics.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]

TECHNIQUES = {
    "reweighting": {"stage": "pre-processing", "method": "sample_reweighting"},
}


@pytest.fixture(scope="module")
def stage_module():
    spec = importlib.util.spec_from_file_location(
        "run_mitigation_comparison",
        ROOT / "scripts" / "experiments" / "run_mitigation_comparison.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def cohort():
    """A small cohort with both sensitive groups present in every stratum."""
    rng = np.random.RandomState(42)
    n = 120
    sex = rng.randint(0, 2, n)
    y = rng.randint(0, 2, n)
    X = pd.DataFrame(
        {
            "chol": rng.normal(240, 40, n),
            "thalach": rng.normal(150, 20, n),
            # One informative feature so the folds are not pure noise.
            "oldpeak": y * 1.5 + rng.normal(0, 0.5, n),
        }
    )
    sensitive = pd.DataFrame({"sex": sex})
    return X, pd.Series(y, name="heart_disease"), sensitive


@pytest.fixture(scope="module")
def cv_run(stage_module, cohort, tmp_path_factory):
    X, y, sensitive = cohort
    predictions_dir = tmp_path_factory.mktemp("cv") / "predictions_cv"
    index = []
    results = stage_module.run_cv_protocol(
        "cleveland_uci",
        X,
        y,
        sensitive,
        X,  # raw siblings: refit impute/scale inside every fold
        TECHNIQUES,
        "logistic_regression",
        {},
        "all",
        predictions_dir,
        index,
        n_folds=5,
        random_seed=42,
    )
    return results, index, predictions_dir


def test_every_patient_scored_once_out_of_fold(cv_run, cohort):
    """Pooled table has one row per patient, so no row is scored twice or missed."""
    _, index, predictions_dir = cv_run
    X, _, _ = cohort
    for row in index:
        pooled = pd.read_csv(predictions_dir / row["file"])
        assert len(pooled) == len(X)
        assert sorted(pooled["sample_idx"]) == list(range(len(X)))
        assert pooled["fold"].nunique() == 5


def test_arms_stay_paired_with_their_baseline(cv_run):
    """Each arm scores the same rows in the same fold as the baseline arm."""
    _, index, predictions_dir = cv_run
    by_technique = {row["technique"]: row["file"] for row in index}
    assert "baseline" in by_technique and "reweighting" in by_technique

    baseline = pd.read_csv(predictions_dir / by_technique["baseline"])
    arm = pd.read_csv(predictions_dir / by_technique["reweighting"])
    pd.testing.assert_series_equal(baseline["sample_idx"], arm["sample_idx"])
    pd.testing.assert_series_equal(baseline["fold"], arm["fold"])
    pd.testing.assert_series_equal(baseline["y_true"], arm["y_true"])


def test_results_report_pooled_metrics_and_fold_spread(cv_run):
    """Rows are labelled as CV and carry the fold std the chapter quotes."""
    results, _, _ = cv_run
    assert results
    for result in results:
        assert result["split"] == "cv"
        assert result["metadata"]["n_folds"] == 5
        assert len(result["metadata"]["f1_folds"]) == 5
        assert result["metadata"]["f1_fold_std"] >= 0.0
        for metric in ("accuracy", "precision", "recall", "f1_score"):
            assert 0.0 <= result["test_metrics"][metric] <= 1.0

    techniques = {result["technique"] for result in results}
    assert techniques == {"baseline", "reweighting"}
    # The baseline arm imposes no constraint; the mitigated arm records which
    # attribute it was made fair about.
    baseline = next(r for r in results if r["technique"] == "baseline")
    assert baseline["constraint_attr"] == ""
    arm = next(r for r in results if r["technique"] == "reweighting")
    assert arm["constraint_attr"] == "sex"


def test_comparison_table_labels_the_protocol(stage_module, cv_run):
    """`split` distinguishes CV rows from holdout rows in the summary tables."""
    results, _, _ = cv_run
    table = stage_module.create_comparison_table(results)
    assert set(table["split"]) == {"cv"}

    holdout = [dict(r, split="holdout") for r in results]
    del holdout[0]["split"]  # rows written before this change carry no split
    assert set(stage_module.create_comparison_table(holdout)["split"]) == {"holdout"}
