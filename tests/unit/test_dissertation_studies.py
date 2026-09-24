"""Tests for dissertation study orchestration helpers."""

import matplotlib

matplotlib.use("Agg")

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.studies.generate_dissertation_plots import (  # noqa: E402
    _generate_model_stability_plots,
)


def _write_run(tmp_path, training_results=None):
    """A run directory with the two files the stability section reads."""
    run_dir = tmp_path / "run"
    overfit_dir = run_dir / "baseline" / "prediction_fairness"
    overfit_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {"dataset": "cleveland", "model": "logistic_regression", "overfit_risk": "low"},
            {"dataset": "kaggle_heart", "model": "logistic_regression", "overfit_risk": "medium"},
        ]
    ).to_csv(overfit_dir / "overfit_gap_table.csv", index=False)

    if training_results is not None:
        results_dir = run_dir / "baseline" / "results"
        results_dir.mkdir(parents=True)
        (results_dir / "training_results.json").write_text(json.dumps(training_results))
    return run_dir


def _training_results():
    return {
        "cleveland": {
            "logistic_regression": {
                "test_metrics": {
                    "f1_score": 0.776,
                    "recall": 0.74,
                    "precision": 0.81,
                    "auc_roc": 0.88,
                    "accuracy": 0.79,
                },
                "cv_results": {
                    "metrics": {"f1_score": {"mean": 0.762, "std": 0.031}},
                    "n_folds": 5,
                },
            }
        },
        "kaggle_heart": {
            "logistic_regression": {
                "test_metrics": {"f1_score": 0.70, "recall": 0.68, "precision": 0.72}
            }
        },
    }


def _sweep_frame():
    """Baseline cells from both training arms, the CV ones deliberately weaker."""
    rows = []
    for training_method, f1 in (("single_split", 0.820), ("kfold_cv", 0.745)):
        rows.append(
            {
                "dataset": "cleveland",
                "model_type": "logistic_regression",
                "mitigation_technique": "baseline",
                "training_method": training_method,
                "binning_strategy": "fixed_10yr",
                "f1_value": f1,
                "f1_score_mean": f1 if training_method == "kfold_cv" else None,
                "recall_value": 0.80,
                "fairness_gap": 0.20,
            }
        )
    rows.append(
        {
            "dataset": "cleveland",
            "model_type": "logistic_regression",
            "mitigation_technique": "baseline",
            "training_method": "kfold_cv",
            "binning_strategy": "quantile_3",
            "f1_value": 0.731,
            "f1_score_mean": 0.731,
            "recall_value": 0.78,
            "fairness_gap": 0.18,
        }
    )
    return pd.DataFrame(rows)


class TestBaselineModelComparison:
    """The table is the stage-7 baseline, not the best cell of the sweep."""

    def test_table_comes_from_training_results(self, tmp_path):
        run_dir = _write_run(tmp_path, _training_results())
        out_dir = tmp_path / "figures"

        _generate_model_stability_plots(run_dir, _sweep_frame(), out_dir, {})

        table = pd.read_csv(out_dir / "baseline_model_comparison.csv")
        cleveland = table[table["dataset"] == "cleveland"].iloc[0]
        # 0.820 is the best sweep cell on the test split; 0.776 is the baseline.
        assert cleveland["f1_value"] == pytest.approx(0.776)
        assert cleveland["cv_f1_mean"] == pytest.approx(0.762)
        assert cleveland["source"] == "stage_7_training_results"
        assert set(table["dataset"]) == {"cleveland", "kaggle_heart"}

    def test_overfit_risk_is_merged_by_dataset_and_model(self, tmp_path):
        run_dir = _write_run(tmp_path, _training_results())
        out_dir = tmp_path / "figures"

        _generate_model_stability_plots(run_dir, _sweep_frame(), out_dir, {})

        table = pd.read_csv(out_dir / "baseline_model_comparison.csv")
        risks = dict(zip(table["dataset"], table["overfit_risk"]))
        assert risks == {"cleveland": "low", "kaggle_heart": "medium"}

    def test_no_training_results_means_no_table(self, tmp_path):
        run_dir = _write_run(tmp_path, training_results=None)
        out_dir = tmp_path / "figures"

        _generate_model_stability_plots(run_dir, _sweep_frame(), out_dir, {})

        # Better absent than filled from the sweep under the same name.
        assert not (out_dir / "baseline_model_comparison.csv").exists()


class TestBestSweepBaselineByCv:
    """The sweep-derived table keeps its own name and selects on CV."""

    def test_selection_runs_on_the_cv_mean(self, tmp_path):
        run_dir = _write_run(tmp_path, _training_results())
        out_dir = tmp_path / "figures"

        _generate_model_stability_plots(run_dir, _sweep_frame(), out_dir, {})

        table = pd.read_csv(out_dir / "best_sweep_baseline_by_cv.csv")
        assert len(table) == 1
        row = table.iloc[0]
        assert row["f1_score_mean"] == pytest.approx(0.745)
        assert row["binning_strategy"] == "fixed_10yr"
        assert row["selected_on"] == "f1_score_mean"

    def test_single_split_cells_are_not_eligible(self, tmp_path):
        run_dir = _write_run(tmp_path, _training_results())
        out_dir = tmp_path / "figures"
        holdout_only = _sweep_frame()
        holdout_only = holdout_only[holdout_only["training_method"] == "single_split"]

        _generate_model_stability_plots(run_dir, holdout_only, out_dir, {})

        assert not (out_dir / "best_sweep_baseline_by_cv.csv").exists()
        # The stage-7 table is unaffected by the sweep having nothing to offer.
        assert (out_dir / "baseline_model_comparison.csv").exists()
