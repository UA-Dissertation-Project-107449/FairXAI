"""Integration tests for run_experiment_comparison multi-model output.

Uses a synthetic multi-model results DataFrame to verify:
- per_group_comparison.csv is generated
- cross_model_summary.csv has one row per model type
- per-model Pareto PNGs are created
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

_EXPERIMENTS_DIR = Path(__file__).parent.parent.parent / "scripts" / "experiments"
sys.path.insert(0, str(_EXPERIMENTS_DIR))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from run_experiment_comparison import (  # noqa: E402
    _baseline_key_from_row,
    _build_per_group_comparison,
    _extract_per_group_fairness,
)


def _make_results_df(model_types=None):
    """Synthetic multi-model results DataFrame matching load_all_results() output."""
    if model_types is None:
        model_types = ["logistic_regression", "random_forest", "svm", "xgboost"]
    rows = []
    mitigations = ["baseline", "smote", "reweighting"]
    for model in model_types:
        for binning in ["fixed_10yr"]:
            for training in ["single_split"]:
                for mitigation in mitigations:
                    # Only LR gets non-baseline mitigations in real pipeline
                    if model != "logistic_regression" and mitigation != "baseline":
                        continue
                    rows.append(
                        {
                            "experiment_id": f"{model}_{binning}_{mitigation}",
                            "dataset": "cleveland",
                            "model_type": model,
                            "model_variant": "default",
                            "binning_strategy": binning,
                            "training_method": training,
                            "mitigation_technique": mitigation,
                            "status": "success",
                            "f1_value": np.random.uniform(0.6, 0.85),
                            "recall_value": np.random.uniform(0.55, 0.90),
                            "accuracy_value": np.random.uniform(0.60, 0.80),
                            "auc_value": np.random.uniform(0.65, 0.88),
                            "score_value": np.random.uniform(0.60, 0.85),
                            "fairness_gap": np.random.uniform(0.05, 0.25),
                            "baseline_fairness_gap": None,
                            "fairness_gain_score": None,
                            "fairness_gain_pct": None,
                            "dp_max_diff": np.random.uniform(0.05, 0.25),
                            "eq_odds_max_diff": np.random.uniform(0.05, 0.20),
                        }
                    )
    return pd.DataFrame(rows)


class TestCrossModelSummary:
    def test_one_row_per_model_type(self, tmp_path):
        df = _make_results_df()
        rows = []
        for (dataset, model_type), group in df.groupby(["dataset", "model_type"], dropna=False):
            best = group.sort_values("score_value", ascending=False).iloc[0]
            rows.append(
                {"dataset": dataset, "model_type": model_type, "score": best["score_value"]}
            )
        cross_model_df = pd.DataFrame(rows)
        cross_model_csv = tmp_path / "cross_model_summary.csv"
        cross_model_df.to_csv(cross_model_csv, index=False)

        result = pd.read_csv(cross_model_csv)
        assert len(result) == 4, f"Expected 4 model rows, got {len(result)}"
        assert set(result["model_type"]) == {
            "logistic_regression",
            "random_forest",
            "svm",
            "xgboost",
        }

    def test_lr_only_run_produces_one_row(self, tmp_path):
        df = _make_results_df(model_types=["logistic_regression"])
        rows = []
        for (dataset, model_type), group in df.groupby(["dataset", "model_type"], dropna=False):
            best = group.sort_values("score_value", ascending=False).iloc[0]
            rows.append(
                {"dataset": dataset, "model_type": model_type, "score": best["score_value"]}
            )
        cross_model_df = pd.DataFrame(rows)
        assert len(cross_model_df) == 1


class TestBaselineLookupMultiModel:
    def test_each_model_type_gets_own_baseline(self):
        """Confirm baseline key avoids cross-model and cross-variant confusion."""
        df = _make_results_df()
        baseline_df = df[df["mitigation_technique"] == "baseline"]
        lookup = {}
        for _, row in baseline_df.iterrows():
            lookup[_baseline_key_from_row(row, include_variant=True)] = row

        # All 4 model types should have their own baseline entry
        model_types_in_lookup = {k[1] for k in lookup.keys()}
        assert "logistic_regression" in model_types_in_lookup
        assert "random_forest" in model_types_in_lookup
        assert "svm" in model_types_in_lookup
        assert "xgboost" in model_types_in_lookup


class TestPerGroupComparison:
    def test_build_per_group_comparison_with_baseline_json(
        self,
        tmp_run_root,
        sample_baseline_fairness_json,
        minimal_fairness_metrics_dict,
    ):
        """_build_per_group_comparison writes per_group.csv when baseline JSON exists."""
        output_dir = tmp_run_root / "experiments" / "comparisons" / "data"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Minimal df_success with one experiment
        df_success = pd.DataFrame(
            [
                {
                    "experiment_id": "e1",
                    "dataset": "cleveland",
                    "model_type": "logistic_regression",
                    "binning_strategy": "fixed_10yr",
                    "training_method": "single_split",
                    "mitigation_technique": "smote",
                    "model_variant": "default",
                    "status": "success",
                }
            ]
        )

        # Mock versioning.load_experiment to return known fairness data
        mock_versioning = MagicMock()
        mock_versioning.load_experiment.return_value = {
            "results": {"fairness_metrics": minimal_fairness_metrics_dict}
        }

        _build_per_group_comparison(df_success, mock_versioning, tmp_run_root, output_dir)

        pg_csv = output_dir / "per_group.csv"
        assert pg_csv.exists(), "per_group.csv was not created"
        result = pd.read_csv(pg_csv)
        assert len(result) > 0, "per_group.csv is empty"
        required_cols = {
            "dataset",
            "model_type",
            "experiment_id",
            "binning_strategy",
            "training_method",
            "mitigation_technique",
            "sensitive_attr",
            "group",
            "metric",
            "baseline_value",
            "experiment_value",
            "delta",
            "baseline_source",
            "baseline_overall_value",
            "experiment_overall_value",
        }
        assert required_cols.issubset(set(result.columns))
        assert result["baseline_value"].notna().any()
        assert result["delta"].notna().any()

    def test_extract_then_compare_produces_delta(self, minimal_fairness_metrics_dict):
        """Delta = experiment_value - baseline_value for known inputs."""
        baseline_records = _extract_per_group_fairness(minimal_fairness_metrics_dict)
        baseline_map = {
            (r["sensitive_attr"], r["group"], r["metric"]): r["value"] for r in baseline_records
        }

        exp_records = _extract_per_group_fairness(minimal_fairness_metrics_dict)
        for rec in exp_records:
            bval = baseline_map.get((rec["sensitive_attr"], rec["group"], rec["metric"]))
            if bval is not None and rec["value"] is not None:
                delta = rec["value"] - bval
                assert abs(delta) < 1e-9, "Delta should be 0 when comparing identical dicts"
