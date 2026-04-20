from __future__ import annotations

import json
from pathlib import Path

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
