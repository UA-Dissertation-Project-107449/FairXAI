"""Stage 10 (mitigate) multi-model wiring."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def stage_module():
    spec = importlib.util.spec_from_file_location(
        "run_mitigation_comparison",
        ROOT / "scripts" / "experiments" / "run_mitigation_comparison.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_model_types_win(stage_module):
    assert stage_module._resolve_model_types(
        ["random_forest", "svm"], {"model_types": ["logistic_regression"]}
    ) == ["random_forest", "svm"]


def test_config_model_types_used_when_no_cli(stage_module):
    assert stage_module._resolve_model_types(
        None, {"model_types": ["logistic_regression", "xgboost"]}
    ) == ["logistic_regression", "xgboost"]


def test_default_is_logistic_regression(stage_module):
    assert stage_module._resolve_model_types(None, {}) == ["logistic_regression"]


def test_model_params_come_from_the_family_config(stage_module):
    params = stage_module._load_model_params(ROOT, "random_forest")
    assert params["n_estimators"] == 200
    assert "max_depth" in params


def _result_row(model_type, technique, dataset="cleveland_uci", dp=0.2):
    return {
        "dataset": dataset,
        "model_type": model_type,
        "technique": technique,
        "constraint_attr": "sex",
        "stage": "none" if technique == "baseline" else "pre-processing",
        "test_metrics": {
            "accuracy": 0.8,
            "precision": 0.8,
            "recall": 0.8,
            "f1_score": 0.8,
            "auc_roc": 0.8,
        },
        "fairness": {
            "group_fairness": {
                "sex": {
                    "demographic_parity": {"max_difference": dp},
                    "equalized_odds": {"tpr_max_difference": dp, "fpr_max_difference": dp},
                }
            }
        },
        "metadata": {},
    }


def test_comparison_table_carries_model_type(stage_module):
    df = stage_module.create_comparison_table(
        [_result_row("logistic_regression", "baseline"), _result_row("random_forest", "baseline")]
    )
    assert set(df["model_type"]) == {"logistic_regression", "random_forest"}


def test_fairness_gain_is_measured_against_the_same_family(stage_module):
    """An RF mitigation row must not be scored against the LR baseline."""
    rows = [
        _result_row("logistic_regression", "baseline", dp=0.10),
        _result_row("random_forest", "baseline", dp=0.40),
        _result_row("random_forest", "smote", dp=0.20),
    ]
    df = stage_module.create_comparison_table(rows)
    rf_smote = df[(df["model_type"] == "random_forest") & (df["technique"] == "smote")].iloc[0]
    assert rf_smote["baseline_fairness_gap"] == pytest.approx(0.40)
    assert rf_smote["fairness_gain"] == pytest.approx(0.20)


def test_multi_model_mitigation_produces_one_row_per_family(stage_module):
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(3)
    n = 60
    X = pd.DataFrame({"f1": rng.normal(size=n), "f2": rng.normal(size=n)})
    y = pd.Series((X["f1"] > 0).astype(int), name="heart_disease")
    sensitive = pd.DataFrame({"sex": rng.integers(0, 2, size=n)})
    techniques = {"reweighting": {"stage": "pre-processing"}}

    rows = []
    for model_type, params in (
        ("logistic_regression", {}),
        ("random_forest", {"n_estimators": 5, "max_depth": 2}),
    ):
        rows.extend(
            stage_module.apply_mitigation_techniques(
                X,
                y,
                X,
                y,
                sensitive,
                sensitive,
                "cleveland_uci",
                None,
                techniques,
                model_type=model_type,
                model_params=params,
                constraint_attrs=["sex"],
            )
        )
    assert {r["model_type"] for r in rows} == {"logistic_regression", "random_forest"}


def test_prediction_index_records_family_and_file(stage_module, tmp_path):
    """B5 reads this index; the filename cannot be parsed back by splitting on '_'."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(5)
    n = 60
    X = pd.DataFrame({"f1": rng.normal(size=n), "f2": rng.normal(size=n)})
    y = pd.Series((X["f1"] > 0).astype(int), name="heart_disease")
    sensitive = pd.DataFrame({"sex": rng.integers(0, 2, size=n)})

    index = []
    stage_module.apply_mitigation_techniques(
        X,
        y,
        X,
        y,
        sensitive,
        sensitive,
        "cleveland_uci",
        None,
        {"reweighting": {"stage": "pre-processing"}},
        model_type="random_forest",
        model_params={"n_estimators": 5, "max_depth": 2},
        constraint_attrs=["sex"],
        predictions_dir=tmp_path / "predictions",
        prediction_index=index,
    )
    assert index == [
        {
            "file": "cleveland_uci_random_forest_reweighting_sex.csv",
            "dataset": "cleveland_uci",
            "model_type": "random_forest",
            "technique": "reweighting",
            "constraint_attr": "sex",
        }
    ]
    assert (tmp_path / "predictions" / index[0]["file"]).exists()
