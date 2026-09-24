"""Stages 7, 10 and 11 must resolve the same hyperparameters.

Before the shared resolver, stage 7 merged the HPO best params, stage 10 ignored
HPO entirely, and stage 11 merged HPO and then let the ``model_variants`` block
overwrite it. So the tuned baseline of chapter section 6.2 was not the model the
mitigation arms were built from, and no arm could be compared to another. These
tests pin the layering and the agreement between the three stages.
"""

import json
import sys
from pathlib import Path

import pytest
import yaml

_FAIRXAI_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_FAIRXAI_ROOT / "src"))
sys.path.insert(0, str(_FAIRXAI_ROOT / "scripts" / "common"))
sys.path.insert(0, str(_FAIRXAI_ROOT / "scripts" / "experiments"))

from fairxai.training.grid_search import (  # noqa: E402
    hpo_params_dir,
    load_base_params,
    resolve_model_params,
)

BASE_PARAMS = {"C": 0.1, "penalty": "l2", "solver": "lbfgs", "max_iter": 1000, "random_state": 42}
HPO_BEST = {"C": 1.0, "max_iter": 2000}


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """A minimal project root with one model config and one HPO study."""
    models = tmp_path / "configs" / "models"
    models.mkdir(parents=True)
    (models / "logistic_regression.yaml").write_text(
        yaml.safe_dump({"hyperparameters": BASE_PARAMS}), encoding="utf-8"
    )

    hpo = tmp_path / "output" / "cardiac" / "studies" / "hpo" / "study_1"
    hpo.mkdir(parents=True)
    (hpo / "best_params_cleveland_logistic_regression.json").write_text(
        json.dumps({"best_params": HPO_BEST}), encoding="utf-8"
    )
    (hpo.parent / "latest.txt").write_text("study_1", encoding="utf-8")
    return tmp_path


def test_hpo_overrides_only_the_searched_keys(project: Path) -> None:
    resolved = resolve_model_params(
        project,
        "logistic_regression",
        dataset="cleveland",
        hpo_dir=hpo_params_dir(project, "cardiac"),
    )

    assert resolved.hpo_used
    assert resolved.hpo_keys == ("C", "max_iter")
    assert resolved.params == {**BASE_PARAMS, **HPO_BEST}
    assert resolved.variant_name == "tuned"


def test_without_an_hpo_study_the_config_defaults_stand(tmp_path: Path, project: Path) -> None:
    resolved = resolve_model_params(
        project,
        "logistic_regression",
        dataset="cleveland",
        hpo_dir=hpo_params_dir(tmp_path / "empty", "cardiac"),
    )

    assert not resolved.hpo_used
    assert resolved.params == BASE_PARAMS
    assert resolved.variant_name == "default"


def test_overrides_and_hardware_are_applied_after_hpo(project: Path) -> None:
    """A variant refines the tuned model; hardware has the final word.

    The old sweep applied variant params over untuned defaults and then let HPO
    overwrite nothing, which is how hand-written values reached the results
    tables as tuned ones.
    """
    resolved = resolve_model_params(
        project,
        "logistic_regression",
        dataset="cleveland",
        hpo_dir=hpo_params_dir(project, "cardiac"),
        overrides={"C": 0.5, "max_iter": 500},
        hardware={"max_iter": 123},
    )

    assert resolved.params["C"] == 0.5
    assert resolved.params["max_iter"] == 123
    assert resolved.hpo_used  # still recorded as tuned-plus-override


def test_a_missing_model_config_is_not_fatal(project: Path) -> None:
    resolved = resolve_model_params(project, "no_such_family", random_state=7)

    assert resolved.params == {"random_state": 7}
    assert not resolved.hpo_used


def test_stage_7_and_stage_10_resolve_identical_params(project: Path) -> None:
    from run_mitigation_comparison import _load_model_params
    from train_baseline import _build_model_params

    hpo_dir = hpo_params_dir(project, "cardiac")

    stage_7 = _build_model_params(
        "logistic_regression",
        42,
        project,
        hpo_dir=hpo_dir,
        dataset_name="cleveland",
    )
    stage_10 = _load_model_params(
        project, "logistic_regression", dataset="cleveland", hpo_dir=hpo_dir
    )

    assert stage_7 == stage_10 == {**BASE_PARAMS, **HPO_BEST}


def test_stage_11_cell_matches_the_other_two_stages(project: Path) -> None:
    from run_combinatorial_experiments import _resolve_model_variants
    from train_baseline import _build_model_params

    hpo_dir = hpo_params_dir(project, "cardiac")
    variants = _resolve_model_variants(
        {},  # no model_variants configured
        "logistic_regression",
        project,
        dataset="cleveland",
        hpo_dir=hpo_dir,
    )

    assert len(variants) == 1
    assert variants[0]["name"] == "tuned"
    assert variants[0]["params"] == _build_model_params(
        "logistic_regression", 42, project, hpo_dir=hpo_dir, dataset_name="cleveland"
    )


def test_stage_11_variant_refines_the_tuned_params(project: Path) -> None:
    from run_combinatorial_experiments import _resolve_model_variants

    variants = _resolve_model_variants(
        {"model_variants": {"logistic_regression": [{"name": "untuned_c", "params": {"C": 0.01}}]}},
        "logistic_regression",
        project,
        dataset="cleveland",
        hpo_dir=hpo_params_dir(project, "cardiac"),
    )

    assert [v["name"] for v in variants] == ["untuned_c"]
    assert variants[0]["params"]["C"] == 0.01
    # The HPO value the variant did not name survives.
    assert variants[0]["params"]["max_iter"] == HPO_BEST["max_iter"]


def test_shipped_configs_declare_no_variants_so_every_family_is_one_cell() -> None:
    """Guards the compute saving A3 bought: one cell per family, not two or three."""
    config = yaml.safe_load(
        (_FAIRXAI_ROOT / "configs" / "experiments" / "combinatorial.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert config["model_variants"] == {}


def test_load_base_params_reads_the_shipped_model_configs() -> None:
    for family in ("logistic_regression", "random_forest", "svm", "xgboost"):
        params = load_base_params(_FAIRXAI_ROOT, family)
        assert params, f"no hyperparameters block for {family}"
