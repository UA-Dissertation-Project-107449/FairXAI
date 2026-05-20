"""Integration test: train_baseline.py trains all 4 model types and saves prediction CSVs."""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

FAIRXAI_ROOT = Path(__file__).parent.parent.parent
TRAIN_SCRIPT = FAIRXAI_ROOT / "scripts" / "common" / "train_baseline.py"
TEST_CONFIG_ROOT = FAIRXAI_ROOT / "tests" / "fixtures" / "configs"


@pytest.fixture
def tiny_cardiac_processed(tmp_path):
    """Write minimal scaled train/test CSVs to a temp processed dir."""
    rng = np.random.default_rng(7)
    n_train, n_test = 40, 20

    def _make(n):
        return pd.DataFrame(
            {
                "age_group": rng.uniform(-1, 1, n),
                "sex": rng.choice([0.0, 1.0], n),
                "trestbps": rng.standard_normal(n),
                "chol": rng.standard_normal(n),
                "thalach": rng.standard_normal(n),
                "oldpeak": rng.standard_normal(n),
                "ca": rng.standard_normal(n),
                "heart_disease": rng.integers(0, 2, n),
            }
        )

    processed = tmp_path / "processed"
    dataset_dir = processed / "cleveland"
    dataset_dir.mkdir(parents=True)
    _make(n_train).to_csv(dataset_dir / "cleveland_train_scaled.csv", index=False)
    _make(n_test).to_csv(dataset_dir / "cleveland_test_scaled.csv", index=False)
    return processed


@pytest.fixture
def tiny_pipeline_config(tmp_path, tiny_cardiac_processed):
    """Write a minimal pipeline YAML and model YAMLs to an isolated temp root."""
    cfg = yaml.safe_load((TEST_CONFIG_ROOT / "pipelines" / "cardiac_base.yaml").read_text())

    schema_json = {"datasets": {}, "columns": {}}
    schema_path = tmp_path / "configs" / "schema" / "cardiac.json"
    schema_path.parent.mkdir(parents=True, exist_ok=True)
    schema_path.write_text(json.dumps(schema_json))

    cfg["paths"]["processed_dir"] = str(tiny_cardiac_processed.relative_to(tmp_path))
    cfg["paths"]["experiments_dir"] = "baseline/results"
    cfg["paths"]["models_dir"] = "baseline/models"
    cfg["paths"]["results_fairness_dir"] = "baseline/prediction_fairness"
    cfg["paths"]["results_binning_dir"] = "binning"
    cfg["runtime"]["schema_mapping_json"] = str(schema_path.relative_to(tmp_path))

    config_dir = tmp_path / "configs" / "pipelines"
    config_dir.mkdir(parents=True)
    config_path = config_dir / "cardiac.yaml"
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

    model_cfgs_dir = tmp_path / "configs" / "models"
    model_cfgs_dir.mkdir(parents=True)
    for model in ["logistic_regression", "random_forest", "svm", "xgboost"]:
        src = TEST_CONFIG_ROOT / "models" / f"{model}_test.yaml"
        (model_cfgs_dir / f"{model}.yaml").write_text(src.read_text())

    return tmp_path


@pytest.mark.slow
@pytest.mark.parametrize(
    "model_subset",
    [
        pytest.param(["logistic_regression", "random_forest"], id="lr_rf"),
        pytest.param(["svm", "xgboost"], id="svm_xgb", marks=pytest.mark.xgboost_model),
    ],
)
def test_all_four_models_produce_prediction_csvs(tiny_pipeline_config, model_subset):
    """train_baseline.py must output train+test prediction CSVs for each requested model subset."""
    results_dir = tiny_pipeline_config / "baseline" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--project-root",
        str(tiny_pipeline_config),
        "--pipeline",
        "cardiac",
        "--output-dir",
        str(tiny_pipeline_config / "baseline"),
        "--model-types",
        *model_subset,
    ]

    result = subprocess.run(
        cmd,
        cwd=str(tiny_pipeline_config),
        capture_output=True,
        text=True,
        timeout=240,
        env={
            **os.environ,
            "PYTHONPATH": str(FAIRXAI_ROOT / "src"),
        },
    )

    # The script may warn/skip some models but should not hard-crash
    assert result.returncode == 0, (
        "train_baseline.py failed:\n"
        f"STDOUT:\n{result.stdout[-3000:]}\n"
        f"STDERR:\n{result.stderr[-3000:]}"
    )

    predictions_dir = results_dir / "predictions"
    required_cols = {
        "y_true",
        "y_pred",
        "y_proba",
        "threshold",
        "confidence",
        "near_threshold",
        "age_group",
        "sex",
    }
    for model_type in model_subset:
        train_csv = predictions_dir / f"cleveland_{model_type}_train.csv"
        test_csv = predictions_dir / f"cleveland_{model_type}_test.csv"
        assert train_csv.exists(), f"Missing train predictions for {model_type}"
        assert test_csv.exists(), f"Missing test predictions for {model_type}"

        train_df = pd.read_csv(train_csv)
        test_df = pd.read_csv(test_csv)
        assert not train_df.empty, f"Empty train predictions for {model_type}"
        assert not test_df.empty, f"Empty test predictions for {model_type}"
        assert required_cols.issubset(train_df.columns), (
            f"Train predictions missing columns for {model_type}: "
            f"{required_cols - set(train_df.columns)}"
        )
        assert required_cols.issubset(test_df.columns), (
            f"Test predictions missing columns for {model_type}: "
            f"{required_cols - set(test_df.columns)}"
        )
