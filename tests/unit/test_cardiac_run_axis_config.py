"""Guards on the cardiac run axes declared in the shipped configs.

Two axes are pinned here: which columns fairness is measured across, and
which model families mitigation actually runs for.

``ethnicity`` was dropped as a fairness run axis: no cardiac dataset in the
study actually carries the column, so every run spent its fairness budget on
an attribute that was never present. The removal is deliberately scoped to
the *run-axis* configs — the domain vocabulary still describes the column,
and the loaders/engines keep their drop-if-present guards so a future dataset
that does carry it still behaves.
"""

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "configs"


def _yaml(relative_path: str) -> dict:
    with open(CONFIGS / relative_path) as handle:
        return yaml.safe_load(handle)


class TestEthnicityNotARunAxis:
    """The four configs that turn a column into a per-run fairness axis."""

    def test_cardiac_pipeline_config(self) -> None:
        attrs = _yaml("pipelines/cardiac.yaml")["fairness"]["sensitive_attributes"]
        assert "ethnicity" not in attrs
        assert ["age_group", "sex", "group_cluster"] == list(attrs)

    def test_combinatorial_experiment_config(self) -> None:
        attrs = _yaml("experiments/combinatorial.yaml")["sensitive_attributes"]
        assert "ethnicity" not in attrs

    def test_mitigation_experiment_config(self) -> None:
        attrs = _yaml("experiments/mitigation.yaml")["data"]["sensitive_attributes"]
        assert "ethnicity" not in attrs

    def test_unified_schema(self) -> None:
        with open(CONFIGS / "schema" / "cardiac.json") as handle:
            schema = json.load(handle)
        attrs = schema["unified_schema"]["sensitive_attributes"]
        assert "ethnicity" not in attrs


def test_domain_vocabulary_still_describes_ethnicity() -> None:
    """The removal is scoped to run axes, not to the schema vocabulary.

    ``configs/domain/cardiac.yaml`` documents what the column *means* if a
    dataset ever supplies it. Dropping it there would lose that description
    and silently change how such a dataset is interpreted.
    """
    domain = _yaml("domain/cardiac.yaml")
    assert "ethnicity" in yaml.dump(domain)


class TestMitigationModelFamilies:
    """All four families are mitigated, in both experiment configs.

    The exclusion these assertions replace was a cardio70k hardware budget,
    not a correctness limit: SVM's RBF kernel is O(n^2) in rows and the
    resampling techniques add rows. At 303 and 918 rows that cost is nil.
    """

    _ALL = {"logistic_regression", "random_forest", "svm", "xgboost"}

    def test_combinatorial_supported_families(self) -> None:
        config = _yaml("experiments/combinatorial.yaml")
        supported = {str(m).strip().lower() for m in config["mitigation_supported_model_types"]}
        assert supported == self._ALL

    def test_combinatorial_combo_families(self) -> None:
        config = _yaml("experiments/combinatorial.yaml")
        combos = {str(m).strip().lower() for m in config["mitigation_combo_model_types"]}
        assert combos == self._ALL

    def test_mitigation_experiment_families(self) -> None:
        config = _yaml("experiments/mitigation.yaml")
        families = {str(m).strip().lower() for m in config["model_types"]}
        assert families == self._ALL


def test_both_svm_variants_are_enabled() -> None:
    """Linear SVM and logistic regression are near-duplicates on this data.

    The RBF arm is the one that can separate them, so it has to be live for
    the SVM family to contribute anything the LR family does not.
    """
    variants = _yaml("experiments/combinatorial.yaml")["model_variants"]["svm"]
    names = {str(v["name"]) for v in variants}
    assert names == {"svm_linear", "svm_rbf"}

    kernels = {str(v["name"]): v["params"]["kernel"] for v in variants}
    assert kernels == {"svm_linear": "linear", "svm_rbf": "rbf"}
