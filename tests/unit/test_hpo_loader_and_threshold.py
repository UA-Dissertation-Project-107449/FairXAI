from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from fairxai.training.grid_search import load_hpo_params, run_hpo


def test_load_hpo_params_supports_legacy_flat_layout(tmp_path: Path) -> None:
    hpo_dir = tmp_path / "hpo"
    hpo_dir.mkdir(parents=True)
    payload = {"best_params": {"C": 1.0, "kernel": "linear"}}
    target = hpo_dir / "best_params_cleveland_svm.json"
    target.write_text(json.dumps(payload), encoding="utf-8")

    params = load_hpo_params(hpo_dir, "cleveland", "svm")

    assert params == payload["best_params"]


def test_load_hpo_params_supports_run_scoped_latest_layout(tmp_path: Path) -> None:
    hpo_dir = tmp_path / "hpo"
    study_id = "run_20260420_000000"
    study_dir = hpo_dir / study_id
    study_dir.mkdir(parents=True)
    (hpo_dir / "latest.txt").write_text(study_id, encoding="utf-8")

    payload = {"best_params": {"C": 0.1, "kernel": "linear"}}
    target = study_dir / "best_params_cleveland_svm.json"
    target.write_text(json.dumps(payload), encoding="utf-8")

    params = load_hpo_params(hpo_dir, "cleveland", "svm")

    assert params == payload["best_params"]


def test_run_hpo_respects_rbf_row_threshold() -> None:
    X_train = pd.DataFrame(
        {
            "f1": [0.1, 0.3, 0.5, 0.7, 0.9],
            "f2": [1.0, 0.8, 0.6, 0.4, 0.2],
        }
    )
    y_train = pd.Series([0, 0, 1, 1, 1])

    result = run_hpo(
        model_type="svm",
        X_train=X_train,
        y_train=y_train,
        param_grid={"C": [0.1, 1.0], "kernel": ["linear", "rbf"], "gamma": ["scale"]},
        base_params={"random_state": 42, "n_jobs": 1},
        search="grid",
        cv=2,
        scoring="f1",
        n_jobs=1,
        max_rows_for_rbf_svm=1,
    )

    assert result["best_params"]["kernel"] == "linear"


def test_run_hpo_preprocess_in_cv_handles_nan_and_strips_prefix() -> None:
    # preprocess_in_cv wraps the estimator in an impute+scale pipeline so RAW
    # features (with NaN) are handled inside every search fold. The returned
    # best_params must be plain estimator keys, never the internal `model__` step.
    rng = np.random.RandomState(0)
    X_train = pd.DataFrame({"a": rng.rand(60), "b": rng.rand(60)})
    X_train.loc[0:5, "a"] = np.nan  # NaN would break a bare estimator
    y_train = pd.Series((X_train["b"] > 0.5).astype(int))

    result = run_hpo(
        model_type="logistic_regression",
        X_train=X_train,
        y_train=y_train,
        param_grid={"C": [0.5, 1.0]},
        base_params={"random_state": 42},
        search="grid",
        cv=3,
        scoring="f1",
        n_jobs=1,
        preprocess_in_cv=True,
    )

    assert result["preprocess_in_cv"] is True
    assert result["preprocessing_mode"] == "fold_safe_pipeline"
    assert "C" in result["best_params"]
    assert all("__" not in key for key in result["best_params"])


def test_run_hpo_records_prescaled_mode_by_default() -> None:
    X_train = pd.DataFrame({"a": [0.1, 0.4, 0.6, 0.9], "b": [0.2, 0.5, 0.7, 0.3]})
    y_train = pd.Series([0, 0, 1, 1])

    result = run_hpo(
        model_type="logistic_regression",
        X_train=X_train,
        y_train=y_train,
        param_grid={"C": [1.0]},
        base_params={"random_state": 42},
        cv=2,
        n_jobs=1,
    )

    assert result["preprocess_in_cv"] is False
    assert result["preprocessing_mode"] == "prescaled_panel"
