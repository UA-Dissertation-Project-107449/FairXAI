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
