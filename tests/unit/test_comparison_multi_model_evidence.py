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
