"""SHAP fallback, per-model status records, and the run-level report.

The September runs lost every XGBoost explanation to a caught exception that
left only a log line. These tests pin the three pieces that make that visible:
the tree-path-dependent fallback, the status record each caught call leaves,
and the report (and ``--strict`` exit code) over a run's status files.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_FAIRXAI_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_FAIRXAI_ROOT / "src"))
sys.path.insert(0, str(_FAIRXAI_ROOT / "scripts" / "common"))

from fairxai.explainability.tabular import shap_explain_tabular  # noqa: E402
from fairxai.explainability.tabular import (  # noqa: E402
    collect_shap_status,
    format_shap_status_report,
    has_problems,
    overall_status,
    shap_status_record,
    write_shap_status,
)

shap = pytest.importorskip("shap")

FEATURES = ["chol", "thalach", "oldpeak", "age"]


def _data(n: int = 200, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, len(FEATURES))), columns=FEATURES)
    y = (X["chol"] + X["age"] + rng.normal(0, 0.3, n) > 0).astype(int)
    return X, y


@pytest.mark.xgboost_model
def test_xgboost_gets_interventional_shap_in_this_environment():
    """Guards the xgboost/shap pin: this pair must not need the fallback.

    shap 0.52 with xgboost >= 3.3 fails here, which is what emptied the
    September XGBoost SHAP outputs.
    """
    xgboost = pytest.importorskip("xgboost")
    X, y = _data()
    model = xgboost.XGBClassifier(n_estimators=20, max_depth=3).fit(X, y)

    explanation = shap_explain_tabular(model, X, max_samples=100)

    assert explanation.shap_values.shape == (100, len(FEATURES))
    assert explanation.explainer == "TreeExplainer"
    assert explanation.feature_perturbation == "interventional"
    assert explanation.fallback_reason is None
    assert shap_status_record("holdout_global", explanation=explanation)["status"] == "ok"


def test_interventional_failure_falls_back_to_tree_path_dependent(monkeypatch):
    from sklearn.ensemble import RandomForestClassifier

    X, y = _data()
    model = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=0).fit(X, y)
    real_tree_explainer = shap.TreeExplainer

    class _BrokenInterventional:
        feature_perturbation = "interventional"

        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, *args, **kwargs):
            raise NotImplementedError("Categorical split is not yet supported.")

    monkeypatch.setattr(shap, "Explainer", _BrokenInterventional)
    monkeypatch.setattr(shap, "TreeExplainer", real_tree_explainer)

    with pytest.warns(RuntimeWarning, match="tree_path_dependent"):
        explanation = shap_explain_tabular(model, X, max_samples=50)

    assert explanation.shap_values.shape == (50, len(FEATURES))
    assert explanation.feature_perturbation == "tree_path_dependent"
    assert "Categorical split" in explanation.fallback_reason
    record = shap_status_record("holdout_global", explanation=explanation)
    assert record["status"] == "fallback"
    assert record["feature_perturbation"] == "tree_path_dependent"


def test_not_implemented_outside_the_interventional_tree_path_still_raises(monkeypatch):
    class _BrokenOther:
        feature_perturbation = None

        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, *args, **kwargs):
            raise NotImplementedError("something else")

    monkeypatch.setattr(shap, "Explainer", _BrokenOther)
    X, _ = _data()
    with pytest.raises(NotImplementedError, match="something else"):
        shap_explain_tabular(object(), X, max_samples=20)


def test_records_and_overall_status():
    failed = shap_status_record("cv_fold0_global", error=ValueError("boom"))
    skipped = shap_status_record("holdout_global", skipped_reason="xai.skip_shap_model_types")

    assert failed == {
        "scope": "cv_fold0_global",
        "status": "failed",
        "message": "ValueError: boom",
    }
    assert skipped["status"] == "skipped"
    assert overall_status([skipped]) == "skipped"
    assert overall_status([{"status": "ok"}, {"status": "fallback"}]) == "fallback"
    assert overall_status([{"status": "fallback"}, failed]) == "failed"
    assert overall_status([]) == "skipped"


def test_report_lists_only_problem_models(tmp_path):
    write_shap_status(tmp_path / "a__logistic_regression", "a__lr", [{"status": "ok"}])
    write_shap_status(
        tmp_path / "a__xgboost",
        "a__xgboost",
        [shap_status_record("holdout_global", error=NotImplementedError("Categorical split"))],
    )
    write_shap_status(
        tmp_path / "sweep" / "shap", "exp_7", [{"status": "ok", "scope": "g"}], prefix="exp_7_"
    )

    entries = collect_shap_status(tmp_path)
    report = format_shap_status_report(entries)

    assert len(entries) == 3
    assert (tmp_path / "sweep" / "shap" / "exp_7_shap_status.json").exists()
    assert has_problems(entries)
    assert "2 ok, 1 failed" in report
    assert "[FAILED] a__xgboost" in report
    assert "holdout_global: NotImplementedError: Categorical split" in report
    assert "a__lr" not in report


def test_report_without_problems_is_one_line(tmp_path):
    write_shap_status(tmp_path / "m", "m", [{"status": "ok"}])
    entries = collect_shap_status(tmp_path)
    assert not has_problems(entries)
    assert format_shap_status_report(entries) == "SHAP status: 1 model(s) - 1 ok"


def test_unreadable_status_file_counts_as_failed(tmp_path):
    (tmp_path / "m").mkdir()
    (tmp_path / "m" / "shap_status.json").write_text("{not json", encoding="utf-8")
    entries = collect_shap_status(tmp_path)
    assert entries[0]["overall"] == "failed"


def test_strict_cli_exit_code(tmp_path):
    from report_shap_status import main

    write_shap_status(tmp_path / "ok", "ok", [{"status": "ok"}])
    assert main(["--run-root", str(tmp_path), "--strict"]) == 0

    write_shap_status(tmp_path / "fb", "fb", [{"status": "fallback", "scope": "g"}])
    assert main(["--run-root", str(tmp_path)]) == 0
    assert main(["--run-root", str(tmp_path), "--strict"]) == 1


def test_save_xai_outputs_returns_records_for_skipped_and_failed_families(tmp_path):
    from sklearn.linear_model import LogisticRegression
    from train_baseline import save_xai_outputs

    X, y = _data(120)

    class _Wrapped:
        def __init__(self, estimator):
            self.model = estimator

    model = _Wrapped(LogisticRegression(max_iter=200).fit(X, y))
    common = dict(X_ref=X, X_lime=X.head(5), output_dir=tmp_path, X_global=X)

    ok = save_xai_outputs(model, "logistic_regression", dataset_name="d_ok", **common)
    assert [r["status"] for r in ok] == ["ok"]

    skipped = save_xai_outputs(
        model,
        "svm",
        dataset_name="d_svm",
        xai_cfg={"skip_shap_model_types": ["svm"], "lime_instances": 0},
        **common,
    )
    assert [r["status"] for r in skipped] == ["skipped"]

    class _Unexplainable:
        model = object()

    failed = save_xai_outputs(
        _Unexplainable(),
        "mystery",
        dataset_name="d_bad",
        xai_cfg={"lime_instances": 0},
        **common,
    )
    assert [r["status"] for r in failed] == ["failed"]
    json.dumps(failed)  # records must stay JSON-serialisable
