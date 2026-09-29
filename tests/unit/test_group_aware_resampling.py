"""Group-aware resamplers balance (group x label) cells, not just the label.

SMOTE, ADASYN, ROS and RUS balance the label alone, so on a near-balanced cohort
they leave the training set essentially unchanged whichever attribute the arm is
reported under. They stay in stage 10 as label-balancing controls; these two
techniques are the ones that target the constraint attribute.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from fairxai.fairness.mitigation import MitigationEngine, PreProcessingMitigation  # noqa: E402


def _imbalanced_cohort(n: int = 200, seed: int = 0):
    """Balanced label overall, but the label depends strongly on sex."""
    rng = np.random.default_rng(seed)
    sex = np.array(["F"] * (n // 2) + ["M"] * (n // 2))
    # 80% of men are positive, 20% of women: cells are 80/20/20/80.
    positive_rate = np.where(sex == "M", 0.8, 0.2)
    y = (rng.random(n) < positive_rate).astype(int)
    X = pd.DataFrame({"f1": rng.normal(size=n), "f2": rng.normal(size=n), "f3": rng.normal(size=n)})
    return X, pd.Series(y, name="target"), pd.DataFrame({"sex": sex})


def _cell_sizes(y: pd.Series, groups) -> dict:
    frame = pd.DataFrame({"group": np.asarray(groups), "label": np.asarray(y)})
    return frame.value_counts().to_dict()


class TestGroupUniformSampling:
    def test_every_cell_ends_the_same_size(self) -> None:
        X, y, sensitive = _imbalanced_cohort()
        cells = _cell_sizes(y, sensitive["sex"])
        X_res, y_res = PreProcessingMitigation.apply_group_uniform_sampling(X, y, sensitive, "sex")
        # Four (group x label) cells, all raised to the largest one.
        assert len(cells) == 4
        assert len(X_res) == len(y_res) == 4 * max(cells.values())
        assert y_res.value_counts().nunique() == 1

    def test_label_dtype_and_name_survive(self) -> None:
        X, y, sensitive = _imbalanced_cohort()
        _, y_res = PreProcessingMitigation.apply_group_uniform_sampling(X, y, sensitive, "sex")
        assert y_res.name == y.name
        assert y_res.dtype == y.dtype
        assert set(y_res.unique()) == {0, 1}

    def test_label_only_resampling_barely_moves_this_cohort(self) -> None:
        """The control: the label is already near-balanced, so SMOTE adds little."""
        X, y, sensitive = _imbalanced_cohort()
        _, y_smote = PreProcessingMitigation.apply_smote(X, y, random_state=42)
        _, y_group = PreProcessingMitigation.apply_group_uniform_sampling(X, y, sensitive, "sex")
        assert len(y_smote) < 1.2 * len(y)
        assert len(y_group) > 1.5 * len(y)

    def test_missing_constraint_column_raises(self) -> None:
        X, y, sensitive = _imbalanced_cohort()
        with pytest.raises(ValueError, match="age_group"):
            PreProcessingMitigation.apply_group_uniform_sampling(X, y, sensitive, "age_group")


class TestGroupAwareSmote:
    def test_synthesises_rows_until_cells_match(self) -> None:
        X, y, sensitive = _imbalanced_cohort()
        X_res, y_res = PreProcessingMitigation.apply_smote_group(X, y, sensitive, "sex")
        assert len(X_res) > len(X)
        assert y_res.value_counts().nunique() == 1

    def test_tiny_cell_falls_back_instead_of_raising(self) -> None:
        X = pd.DataFrame({"f1": [0.0, 1.0, 2.0, 3.0], "f2": [1.0, 0.0, 1.0, 0.0]})
        y = pd.Series([0, 1, 0, 1], name="target")
        sensitive = pd.DataFrame({"sex": ["F", "F", "M", "M"]})
        X_res, y_res = PreProcessingMitigation.apply_smote_group(X, y, sensitive, "sex")
        assert len(X_res) == len(X)
        assert list(y_res) == list(y)


class TestEngineAndConfigWiring:
    def test_engine_accepts_both_as_preprocessing(self) -> None:
        assert "uniform_sampling" in MitigationEngine.VALID_PREPROCESSING
        assert "smote_group" in MitigationEngine.VALID_PREPROCESSING

    def test_stage_ten_config_declares_both(self) -> None:
        with open(ROOT / "configs" / "experiments" / "mitigation.yaml") as handle:
            strategies = yaml.safe_load(handle)["mitigation_strategies"]
        for name in ("uniform_sampling", "smote_group"):
            assert strategies[name]["stage"] == "pre-processing"

    def test_engine_runs_the_technique_end_to_end(self) -> None:
        X, y, sensitive = _imbalanced_cohort(n=120)
        engine = MitigationEngine(random_state=42, model_type="logistic_regression")
        result = engine.apply_technique(
            technique_name="uniform_sampling",
            stage="pre-processing",
            X_train=X,
            y_train=y,
            X_test=X,
            y_test=y,
            sensitive_train=sensitive,
            sensitive_test=sensitive,
            sensitive_attr="sex",
        )
        assert result["metadata"]["technique"] == "uniform_sampling"
        assert result["metadata"]["samples_after"] > result["metadata"]["samples_before"]
