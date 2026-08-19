"""Per-model fairness evidence for stage 12."""

import pandas as pd

from fairxai.comparison.config import DEFAULT_COMPARISON_CONFIG
from fairxai.comparison.metric_tables import (
    build_fairness_evidence_summary,
    build_fairness_evidence_summary_by_model,
)


def _full_df():
    rows = []
    for model in ("logistic_regression", "random_forest"):
        for technique, gap in (("baseline", 0.0), ("smote", 0.05), ("reweighting", 0.09)):
            rows.append(
                {
                    "experiment_id": f"{model}_{technique}",
                    "dataset": "cleveland_uci",
                    "model_type": model,
                    "model_variant": "default",
                    "binning_strategy": "fixed_10yr",
                    "training_method": "single_split",
                    "mitigation_technique": technique,
                    "delta_fairness_gap": gap,
                    "delta_recall": 0.0,
                    "delta_f1": 0.0,
                    "delta_precision": 0.0,
                    "delta_auc": 0.01,
                    "delta_accuracy": 0.0,
                    "delta_dp_gap": gap,
                    "delta_eq_tpr_gap": gap,
                    "delta_eq_fpr_gap": gap,
                }
            )
    return pd.DataFrame(rows)


def test_primary_summary_still_lr_only():
    """The dissertation headline table keeps its primary-model-only contract."""
    out = build_fairness_evidence_summary(_full_df(), None, DEFAULT_COMPARISON_CONFIG)
    assert not out.empty
    assert set(out["model_type"]) == {"logistic_regression"}


def test_by_model_summary_covers_every_family():
    out = build_fairness_evidence_summary_by_model(_full_df(), None, DEFAULT_COMPARISON_CONFIG)
    assert set(out["model_type"]) == {"logistic_regression", "random_forest"}
    assert (out.groupby("model_type").size() > 0).all()


def test_by_model_summary_ignores_the_primary_dataset_filter_only_for_models():
    """Per-family view still honours every non-model selection rule."""
    config = {
        **DEFAULT_COMPARISON_CONFIG,
        "selection": {
            **DEFAULT_COMPARISON_CONFIG["selection"],
            "primary_dataset": "not_a_dataset",
        },
    }
    assert build_fairness_evidence_summary_by_model(_full_df(), None, config).empty


def test_by_model_summary_empty_without_model_column():
    df = _full_df().drop(columns=["model_type"])
    assert build_fairness_evidence_summary_by_model(df, None, DEFAULT_COMPARISON_CONFIG).empty


def test_experiment_index_flags_mitigation_that_did_nothing():
    """A reweighting row whose weights were dropped is a baseline in disguise."""
    from fairxai.comparison.metric_tables import build_experiment_index

    df = pd.DataFrame(
        [
            {
                "experiment_id": "rf_base",
                "dataset": "cleveland_uci",
                "model_type": "random_forest",
                "model_variant": "default",
                "binning_strategy": "fixed_10yr",
                "training_method": "single_split",
                "mitigation_technique": "baseline",
                "status": "success",
                "sample_weight_applied": None,
            },
            {
                "experiment_id": "rf_rw",
                "dataset": "cleveland_uci",
                "model_type": "random_forest",
                "model_variant": "default",
                "binning_strategy": "fixed_10yr",
                "training_method": "single_split",
                "mitigation_technique": "reweighting",
                "status": "success",
                "sample_weight_applied": False,
            },
            {
                "experiment_id": "lr_rw",
                "dataset": "cleveland_uci",
                "model_type": "logistic_regression",
                "model_variant": "default",
                "binning_strategy": "fixed_10yr",
                "training_method": "single_split",
                "mitigation_technique": "reweighting",
                "status": "success",
                "sample_weight_applied": True,
            },
        ]
    )
    index = build_experiment_index(df).set_index("experiment_id")
    assert bool(index.loc["rf_rw", "mitigation_degraded"]) is True
    assert bool(index.loc["lr_rw", "mitigation_degraded"]) is False
    assert bool(index.loc["rf_base", "mitigation_degraded"]) is False


def test_experiment_index_without_the_flag_reports_no_degradation():
    """Results written before the flag existed must not read as degraded."""
    from fairxai.comparison.metric_tables import build_experiment_index

    df = pd.DataFrame(
        [
            {
                "experiment_id": "old_run",
                "dataset": "cleveland_uci",
                "model_type": "logistic_regression",
                "model_variant": "default",
                "binning_strategy": "fixed_10yr",
                "training_method": "single_split",
                "mitigation_technique": "reweighting",
                "status": "success",
            }
        ]
    )
    index = build_experiment_index(df)
    assert bool(index.loc[0, "mitigation_degraded"]) is False
    assert index.loc[0, "sample_weight_applied"] is None


def test_manifest_warns_when_a_mitigation_row_was_degraded(tmp_path):
    """The warning must reach the manifest; the log line is long gone by then."""
    import json

    from fairxai.comparison.metric_tables import write_canonical_comparison_outputs

    df = pd.DataFrame(
        [
            {
                "experiment_id": "rf_base",
                "dataset": "cleveland_uci",
                "model_type": "random_forest",
                "model_variant": "default",
                "binning_strategy": "fixed_10yr",
                "training_method": "single_split",
                "mitigation_technique": "baseline",
                "status": "success",
                "f1_value": 0.7,
                "sample_weight_applied": None,
            },
            {
                "experiment_id": "rf_rw",
                "dataset": "cleveland_uci",
                "model_type": "random_forest",
                "model_variant": "default",
                "binning_strategy": "fixed_10yr",
                "training_method": "single_split",
                "mitigation_technique": "reweighting",
                "status": "success",
                "f1_value": 0.7,
                "sample_weight_applied": False,
            },
        ]
    )
    write_canonical_comparison_outputs(df, None, tmp_path, DEFAULT_COMPARISON_CONFIG)
    manifest = json.loads((tmp_path / "comparison_manifest.json").read_text())
    warning = " ".join(manifest["warnings"])
    assert "sample weight" in warning.lower()
    assert "random_forest" in warning
