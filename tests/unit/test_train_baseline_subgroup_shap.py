"""Integration tests for the subgroup SHAP outputs written by the baseline trainer.

Covers the wiring rather than the statistics (those live in
``test_subgroup_shap.py``): that the files land next to the global summary, that
the sensitive frame is realigned onto the rows SHAP actually explained, and that
the config toggle is honoured.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_FAIRXAI_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_FAIRXAI_ROOT / "src"))
sys.path.insert(0, str(_FAIRXAI_ROOT / "scripts" / "common"))

from train_baseline import save_xai_outputs  # noqa: E402

pytest.importorskip("shap")

FEATURES = ["chol", "thalach", "oldpeak"]


class _WrappedModel:
    """Minimal stand-in for a fairxai model wrapper (exposes ``.model``)."""

    def __init__(self, estimator):
        self.model = estimator


def _fitted(n: int = 120, seed: int = 0):
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, len(FEATURES))), columns=FEATURES)
    y = (X["chol"] + rng.normal(0, 0.3, n) > 0).astype(int)
    estimator = LogisticRegression(max_iter=200).fit(X, y)
    return _WrappedModel(estimator), X


def test_writes_subgroup_files_beside_the_global_summary(tmp_path):
    model, X = _fitted()
    sensitive = pd.DataFrame({"sex": ["f"] * 60 + ["m"] * 60}, index=X.index)

    save_xai_outputs(
        model,
        "logistic_regression",
        X,
        X,
        tmp_path,
        "cleveland__logistic_regression",
        X_global=X,
        xai_cfg={"lime_instances": 0, "subgroup_min_size": 10},
        sensitive_global=sensitive,
    )

    shap_dir = tmp_path / "cleveland__logistic_regression" / "holdout" / "shap"
    assert (shap_dir / "summary.csv").exists()
    per_group = pd.read_csv(shap_dir / "subgroup_summary.csv")
    assert set(per_group["group"]) == {"f", "m"}
    assert set(per_group["feature"]) == set(FEATURES)
    assert (shap_dir / "subgroup_disparity.csv").exists()
    assert (shap_dir / "subgroup_agreement.csv").exists()


def test_groups_follow_row_labels_not_positions(tmp_path):
    """SHAP subsamples; the sensitive frame must be realigned by index label.

    The frame is given a shuffled, non-contiguous index and a group assignment
    that is a function of that label, so a positional join would mislabel rows
    and the recovered group sizes would not match.
    """
    model, X = _fitted(n=200)
    X.index = pd.Index(np.arange(1000, 1200)[::-1])
    sensitive = pd.DataFrame({"sex": np.where(X.index % 2 == 0, "even", "odd")}, index=X.index)

    save_xai_outputs(
        model,
        "logistic_regression",
        X,
        X,
        tmp_path,
        "cleveland__logistic_regression",
        X_global=X,
        # Forces SHAP to explain a strict subset of the rows.
        xai_cfg={"lime_instances": 0, "global_max_samples": 80, "subgroup_min_size": 5},
        sensitive_global=sensitive,
    )

    shap_dir = tmp_path / "cleveland__logistic_regression" / "holdout" / "shap"
    per_group = pd.read_csv(shap_dir / "subgroup_summary.csv")
    sizes = per_group.groupby("group")["n"].max()

    explained = X.sample(n=80, random_state=42).index
    expected = sensitive.loc[explained, "sex"].value_counts()
    assert sizes.to_dict() == expected.to_dict()
    assert sizes.sum() == 80


def test_toggle_off_writes_only_the_global_summary(tmp_path):
    model, X = _fitted()
    sensitive = pd.DataFrame({"sex": ["f"] * 60 + ["m"] * 60}, index=X.index)

    save_xai_outputs(
        model,
        "logistic_regression",
        X,
        X,
        tmp_path,
        "cleveland__logistic_regression",
        X_global=X,
        xai_cfg={"lime_instances": 0, "subgroup_shap": False},
        sensitive_global=sensitive,
    )

    shap_dir = tmp_path / "cleveland__logistic_regression" / "holdout" / "shap"
    assert (shap_dir / "summary.csv").exists()
    assert not (shap_dir / "subgroup_summary.csv").exists()


def test_absent_sensitive_frame_leaves_previous_behaviour_untouched(tmp_path):
    model, X = _fitted()

    save_xai_outputs(
        model,
        "logistic_regression",
        X,
        X,
        tmp_path,
        "cleveland__logistic_regression",
        X_global=X,
        xai_cfg={"lime_instances": 0},
    )

    shap_dir = tmp_path / "cleveland__logistic_regression" / "holdout" / "shap"
    assert (shap_dir / "summary.csv").exists()
    assert not (shap_dir / "subgroup_summary.csv").exists()
