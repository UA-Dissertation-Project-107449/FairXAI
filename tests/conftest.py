"""Shared test fixtures for FairXAI unit and integration tests."""

import json

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_cardiac_df():
    """50-row cardiac-like DataFrame with required columns for fairness tests."""
    rng = np.random.default_rng(42)
    n = 50
    age_groups = rng.choice(["<40", "40-49", "50-59", "60-69", "70+"], size=n)
    sex = rng.choice(["Female", "Male"], size=n)
    heart_disease = rng.integers(0, 2, size=n)
    return pd.DataFrame(
        {
            "age_group": age_groups,
            "sex": sex,
            "trestbps": rng.integers(90, 180, size=n).astype(float),
            "chol": rng.integers(150, 350, size=n).astype(float),
            "thalach": rng.integers(100, 200, size=n).astype(float),
            "oldpeak": rng.uniform(0, 4, size=n),
            "ca": rng.integers(0, 4, size=n).astype(float),
            "heart_disease": heart_disease,
        }
    )


@pytest.fixture
def synthetic_predictions_df(synthetic_cardiac_df):
    """DataFrame with y_true / y_pred / y_proba columns + sensitive attrs."""
    rng = np.random.default_rng(99)
    df = synthetic_cardiac_df.copy()
    df["y_true"] = df["heart_disease"]
    df["y_proba"] = rng.uniform(0.2, 0.8, size=len(df))
    df["y_pred"] = (df["y_proba"] >= 0.5).astype(int)
    df["age_group_cat"] = df["age_group"]
    df["sex_cat"] = df["sex"]
    return df


@pytest.fixture
def minimal_fairness_metrics_dict():
    """A minimal fairness_metrics dict as returned by FairnessMetrics.calculate_all_metrics()."""
    return {
        "group_fairness": {
            "age_group_cat": {
                "demographic_parity": {
                    "group_rates": {
                        "40-49": {"positive_rate": 0.4, "count": 10},
                        "50-59": {"positive_rate": 0.6, "count": 10},
                    },
                    "overall_rate": 0.5,
                    "max_difference": 0.2,
                    "is_fair": "False",
                },
                "equalized_odds": {
                    "group_metrics": {
                        "40-49": {"tpr": 0.7, "fpr": 0.2, "count": 10},
                        "50-59": {"tpr": 0.9, "fpr": 0.3, "count": 10},
                    },
                    "tpr_max_difference": 0.2,
                    "fpr_max_difference": 0.1,
                    "is_fair": "True",
                },
                "equal_opportunity": {
                    "group_tpr": {"40-49": 0.7, "50-59": 0.9},
                    "max_difference": 0.2,
                    "is_fair": "True",
                },
                "predictive_parity": {
                    "group_precision": {"40-49": 0.65, "50-59": 0.75},
                    "max_difference": 0.1,
                    "is_fair": "True",
                },
            },
            "sex_cat": {
                "demographic_parity": {
                    "group_rates": {
                        "Female": {"positive_rate": 0.45, "count": 25},
                        "Male": {"positive_rate": 0.55, "count": 25},
                    },
                    "overall_rate": 0.5,
                    "max_difference": 0.1,
                    "is_fair": "True",
                },
            },
        },
        "individual_fairness": {
            "mean_consistency": 0.85,
            "median_consistency": 0.90,
        },
    }


@pytest.fixture
def tmp_run_root(tmp_path):
    """Minimal run directory structure mirroring a real pipeline run."""
    run_root = tmp_path / "output" / "cardiac" / "runs" / "run_test"
    (run_root / "baseline" / "fairness").mkdir(parents=True)
    (run_root / "experiments" / "full" / "manifests").mkdir(parents=True)
    (run_root / "experiments" / "full" / "results").mkdir(parents=True)
    return run_root


@pytest.fixture
def sample_baseline_fairness_json(tmp_run_root, minimal_fairness_metrics_dict):
    """Write a realistic stage-6 baseline fairness JSON and return its path."""
    payload = {
        "dataset": "cleveland_logistic_regression",
        "train_metrics": minimal_fairness_metrics_dict,
        "test_metrics": minimal_fairness_metrics_dict,
        "comparison": {},
    }
    json_path = (
        tmp_run_root
        / "baseline"
        / "fairness"
        / "cleveland_logistic_regression_fairness_assessment.json"
    )
    json_path.write_text(json.dumps(payload))
    return json_path
