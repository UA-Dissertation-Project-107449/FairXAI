"""Per-arm subgroup SHAP written by stage 10 (mitigate).

Stage 7 writes the subgroup tables for the baseline only, which cannot answer
RO4 ("does mitigating change how the model explains each group?"). These tests
cover the wiring: which arms get explained, where the files land, and which arms
are deliberately skipped. The statistics live in ``test_subgroup_shap.py``.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("shap")

ROOT = Path(__file__).resolve().parents[2]
TABLES = ("subgroup_summary.csv", "subgroup_disparity.csv", "subgroup_agreement.csv")
FEATURES = ["chol", "thalach", "oldpeak"]


@pytest.fixture(scope="module")
def stage_module():
    spec = importlib.util.spec_from_file_location(
        "run_mitigation_comparison",
        ROOT / "scripts" / "experiments" / "run_mitigation_comparison.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Wrapped:
    """A fairxai model wrapper: the sklearn estimator hangs off ``.model``."""

    def __init__(self, estimator):
        self.model = estimator


class _Randomised:
    """``ExponentiatedGradient``: predicts by drawing from ``predictors_`` per row."""

    def __init__(self, predictors):
        self.predictors_ = predictors
        self.weights_ = np.full(len(predictors), 1.0 / len(predictors))


class _GridSearch:
    """``GridSearch``: fits a grid, then predicts with ``predictors_[best_idx_]``."""

    def __init__(self, predictors, best_idx):
        self.predictors_ = predictors
        self.best_idx_ = best_idx


def _fitted(n: int = 120, seed: int = 0):
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, len(FEATURES))), columns=FEATURES)
    y = (X["chol"] + rng.normal(0, 0.3, n) > 0).astype(int)
    return LogisticRegression(max_iter=200).fit(X, y), X


def _cfg(tmp_path, **over):
    cfg = {"enabled": True, "min_group_size": 10, "max_samples": 1000, "dir": tmp_path / "sg"}
    cfg.update(over)
    return cfg


def _sensitive(X):
    half = len(X) // 2
    return pd.DataFrame({"sex": ["f"] * half + ["m"] * (len(X) - half)}, index=X.index)


def test_writes_one_directory_per_arm(stage_module, tmp_path):
    estimator, X = _fitted()

    stage_module._persist_arm_subgroup_shap(
        _Wrapped(estimator),
        X,
        _sensitive(X),
        _cfg(tmp_path),
        "cleveland_uci",
        "logistic_regression",
        "smote",
        "sex",
    )

    arm_dir = tmp_path / "sg" / "cleveland_uci_logistic_regression_smote_sex"
    for name in TABLES:
        assert (arm_dir / name).exists(), name
    summary = pd.read_csv(arm_dir / "subgroup_summary.csv")
    assert set(summary["group"]) == {"f", "m"}


def test_baseline_and_mitigated_arms_do_not_collide(stage_module, tmp_path):
    """RO4 needs both halves of the pair side by side, not one overwriting the other."""
    estimator, X = _fitted()
    cfg = _cfg(tmp_path)

    for technique, attr in (("baseline", "none"), ("reweighting", "sex")):
        stage_module._persist_arm_subgroup_shap(
            _Wrapped(estimator),
            X,
            _sensitive(X),
            cfg,
            "cleveland_uci",
            "logistic_regression",
            technique,
            attr,
        )

    written = sorted(p.name for p in (tmp_path / "sg").iterdir())
    assert written == [
        "cleveland_uci_logistic_regression_baseline_none",
        "cleveland_uci_logistic_regression_reweighting_sex",
    ]


def test_arm_without_a_single_estimator_is_skipped(stage_module, tmp_path, caplog):
    """ExponentiatedGradient has no one model whose attributions to report."""
    estimator, X = _fitted()

    with caplog.at_level("INFO"):
        stage_module._persist_arm_subgroup_shap(
            _Randomised([estimator]),
            X,
            _sensitive(X),
            _cfg(tmp_path),
            "cleveland_uci",
            "logistic_regression",
            "exponentiated_gradient",
            "sex",
        )

    assert not (tmp_path / "sg").exists()
    assert "no single fitted estimator" in caplog.text


def test_family_not_selected_writes_nothing(stage_module, tmp_path):
    """The runner passes ``None`` for families outside the RO4 list."""
    estimator, X = _fitted()

    stage_module._persist_arm_subgroup_shap(
        _Wrapped(estimator),
        X,
        _sensitive(X),
        None,
        "cleveland_uci",
        "svm",
        "smote",
        "sex",
    )

    assert not (tmp_path / "sg").exists()


def test_toggle_off_writes_nothing(stage_module, tmp_path):
    estimator, X = _fitted()

    stage_module._persist_arm_subgroup_shap(
        _Wrapped(estimator),
        X,
        _sensitive(X),
        _cfg(tmp_path, enabled=False),
        "cleveland_uci",
        "logistic_regression",
        "smote",
        "sex",
    )

    assert not (tmp_path / "sg").exists()


def test_grid_search_arm_is_explained_through_its_chosen_member(stage_module, tmp_path):
    """GridSearch predicts with one grid member, so that member is the arm's model."""
    chosen, X = _fitted(seed=1)
    other, _ = _fitted(seed=2)

    stage_module._persist_arm_subgroup_shap(
        _GridSearch([other, chosen], best_idx=1),
        X,
        _sensitive(X),
        _cfg(tmp_path),
        "cleveland_uci",
        "logistic_regression",
        "grid_search",
        "sex",
    )

    arm_dir = tmp_path / "sg" / "cleveland_uci_logistic_regression_grid_search_sex"
    for name in TABLES:
        assert (arm_dir / name).exists(), name
