"""Tests for dissertation study orchestration helpers."""

import matplotlib

matplotlib.use("Agg")

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.studies.generate_dissertation_plots import _generate_model_stability_plots


def test_baseline_model_comparison_merges_overfit_by_dataset_and_model(tmp_path):
    run_dir = tmp_path / "run"
    overfit_dir = run_dir / "baseline" / "prediction_fairness"
    overfit_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {"dataset": "cleveland", "model": "logistic_regression", "overfit_risk": "low"},
            {"dataset": "kaggle_heart", "model": "logistic_regression", "overfit_risk": "medium"},
        ]
    ).to_csv(overfit_dir / "overfit_gap_table.csv", index=False)

    full_df = pd.DataFrame(
        [
            {
                "dataset": "cleveland",
                "model_type": "logistic_regression",
                "mitigation_technique": "baseline",
                "f1_value": 0.80,
                "recall_value": 0.81,
                "precision_value": 0.82,
                "auc_value": 0.90,
                "fairness_gap": 0.20,
            },
            {
                "dataset": "kaggle_heart",
                "model_type": "logistic_regression",
                "mitigation_technique": "baseline",
                "f1_value": 0.85,
                "recall_value": 0.86,
                "precision_value": 0.87,
                "auc_value": 0.92,
                "fairness_gap": 0.25,
            },
        ]
    )

    out_dir = tmp_path / "figures"
    _generate_model_stability_plots(run_dir, full_df, out_dir, {})

    table = pd.read_csv(out_dir / "baseline_model_comparison.csv")
    assert len(table) == 2
    risks = dict(zip(table["dataset"], table["overfit_risk"]))
    assert risks == {"cleveland": "low", "kaggle_heart": "medium"}
