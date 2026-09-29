"""The sweep's constraint attribute is a configured axis, not a hard-coded pick.

Before this, every mitigated cell constrained on the first non-age sensitive
column, which is always ``sex``. Changing the binning strategy then changed only
how the age gap was measured, never what the technique optimised.
"""

import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "experiments"))
sys.path.insert(0, str(ROOT / "src"))

from run_combinatorial_experiments import _SIGNATURE_FIELDS, _experiment_signature  # noqa: E402

from fairxai.data.schemas import resolve_constraint_attribute  # noqa: E402


class TestResolveConstraintAttribute:
    def test_requested_attribute_wins(self) -> None:
        assert (
            resolve_constraint_attribute(["sex", "age_group"], "age_group", ["age_group", "sex"])
            == "age_group"
        )

    def test_missing_requested_attribute_raises(self) -> None:
        with pytest.raises(ValueError, match="group_cluster"):
            resolve_constraint_attribute(["sex", "age_group"], "group_cluster")

    def test_without_request_skips_age_group(self) -> None:
        assert (
            resolve_constraint_attribute(["age_group", "sex"], None, ["age_group", "sex"]) == "sex"
        )

    def test_without_request_falls_back_to_any_column(self) -> None:
        assert resolve_constraint_attribute(["age_group"], None, ["age_group"]) == "age_group"

    def test_no_columns_returns_none(self) -> None:
        assert resolve_constraint_attribute([], None, ["sex"]) is None


class TestConstraintAttributeIsACellCoordinate:
    def test_signature_includes_constraint_attribute(self) -> None:
        assert "constraint_attribute" in _SIGNATURE_FIELDS

    def test_two_constraints_are_different_cells(self) -> None:
        base = {
            "dataset": "cleveland",
            "binning_strategy": "clinical",
            "mitigation_technique": "reweighting",
            "training_method": "cv",
            "model_type": "logistic_regression",
            "model_variant": "tuned",
        }
        sex_cell = _experiment_signature({**base, "constraint_attribute": "sex"})
        age_cell = _experiment_signature({**base, "constraint_attribute": "age_group"})
        assert sex_cell != age_cell


class TestShippedConfig:
    def test_combinatorial_declares_the_axis(self) -> None:
        with open(ROOT / "configs" / "experiments" / "combinatorial.yaml") as handle:
            config = yaml.safe_load(handle)
        attrs = config["constraint_attributes"]
        assert attrs == ["sex", "age_group"]
        # Every constraint attribute has to be one of the measured axes.
        assert set(attrs) <= set(config["sensitive_attributes"])
