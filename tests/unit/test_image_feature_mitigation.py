"""Unit tests for feature-space dermatology mitigation (stage 11, part 2).

Synthetic feature matrices only — no torch, no image decode, no model load.
The frozen-backbone extraction is injected as a callable so the whole
pre/in-processing matrix can be exercised without a checkpoint.

What these pin:
  * the technique catalog mirrors cardiac (ros/rus stay excluded on purpose),
  * the delta reference is the *unmitigated head over the same features*, not
    the CNN — otherwise a head swap would be reported as a mitigation effect,
  * features are standardised before mitigation (SMOTE/ADASYN are distance
    based; unscaled CNN activations would let one channel own the neighbourhood),
  * one failing technique never kills the rest of the matrix,
  * the report says out loud that the intervention is on the head, not the
    representation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fairxai.fairness.image_feature_mitigation import (
    DEFAULT_FEATURE_TECHNIQUES,
    mitigate_features_frame,
    mitigate_run_features,
    technique_stage,
)
from fairxai.fairness.mitigation import MitigationEngine

ROOT = Path(__file__).resolve().parents[2]


def _synthetic_split(n: int, n_features: int, seed: int) -> tuple[np.ndarray, pd.DataFrame]:
    """Features whose label coupling differs by sex, so the head is unfair."""
    rng = np.random.default_rng(seed)
    sex = np.array([0] * (n // 2) + [1] * (n - n // 2))
    signal = rng.normal(size=n)
    # Group 1's label depends on the signal much more weakly -> the head that
    # fits group 0 well transfers badly, which is the unfairness we mitigate.
    logit = np.where(sex == 0, 2.5 * signal, 0.4 * signal) + 0.3
    y = (logit + rng.normal(scale=0.5, size=n) > 0).astype(int)
    features = rng.normal(size=(n, n_features))
    features[:, 0] = signal
    features[:, 1] = sex + rng.normal(scale=0.2, size=n)
    meta = pd.DataFrame(
        {
            "sex": sex,
            "fitzpatrick_group": np.where(sex == 0, "I-II", "III-IV"),
            "y_true": y,
        }
    )
    return features, meta


def _run_frame(techniques=None, n=240, n_features=6, **kwargs):
    train_x, train_meta = _synthetic_split(n, n_features, seed=1)
    test_x, test_meta = _synthetic_split(n, n_features, seed=2)
    return mitigate_features_frame(
        train_x,
        train_meta,
        test_x,
        test_meta,
        ["sex"],
        techniques=techniques,
        min_group_samples=20,
        **kwargs,
    )


class TestTechniqueCatalog:
    def test_defaults_mirror_the_cardiac_selection(self) -> None:
        assert set(DEFAULT_FEATURE_TECHNIQUES) == {
            "reweighting",
            "smote",
            "adasyn",
            "exponentiated_gradient",
            "grid_search",
        }

    def test_random_resampling_stays_excluded(self) -> None:
        # Dropped for cardiac in configs/experiments/combinatorial.yaml (no
        # fairness guarantee, discards information). Dermatology must not
        # quietly reintroduce a wider selection than the domain it mirrors.
        assert "ros" not in DEFAULT_FEATURE_TECHNIQUES
        assert "rus" not in DEFAULT_FEATURE_TECHNIQUES

    def test_every_default_is_a_technique_the_shared_engine_implements(self) -> None:
        known = set(MitigationEngine.VALID_PREPROCESSING) | set(MitigationEngine.VALID_INPROCESSING)
        assert set(DEFAULT_FEATURE_TECHNIQUES).issubset(known)

    def test_stage_lookup_matches_the_shared_engine(self) -> None:
        for name in DEFAULT_FEATURE_TECHNIQUES:
            stage = technique_stage(name)
            if name in MitigationEngine.VALID_PREPROCESSING:
                assert stage == "pre-processing"
            else:
                assert stage == "in-processing"

    def test_unknown_technique_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            technique_stage("threshold_optimizer")


class TestFeatureMitigationReport:
    def test_every_requested_technique_is_reported(self) -> None:
        report = _run_frame(techniques=["reweighting", "smote"])
        techs = report["sensitive_attributes"]["sex"]["techniques"]
        assert set(techs) == {"reweighting", "smote"}

    def test_delta_reference_is_the_head_baseline_not_the_cnn(self) -> None:
        report = _run_frame(techniques=["reweighting"])
        attr = report["sensitive_attributes"]["sex"]
        head_summary = attr["summary_baseline"]
        cell = attr["techniques"]["reweighting"]
        assert cell["summary_before"] == head_summary
        for key, delta in cell["summary_deltas"].items():
            if delta is None:
                continue
            assert delta == pytest.approx(cell["summary_after"][key] - cell["summary_before"][key])

    def test_head_baseline_is_trained_on_the_same_features(self) -> None:
        report = _run_frame(techniques=["reweighting"])
        assert report["head_baseline"]["n_test"] == report["n_test"]
        assert report["overall_baseline"]["accuracy"] is not None

    def test_features_are_standardised_before_mitigation(self) -> None:
        """A column rescaled x1000 must not change the result.

        StandardScaler makes the head scale-invariant; without it the widest
        activation channel dominates the SMOTE/ADASYN neighbourhood and the
        arms stop being comparable to cardiac's scaled-feature runs.
        """
        train_x, train_meta = _synthetic_split(240, 6, seed=1)
        test_x, test_meta = _synthetic_split(240, 6, seed=2)
        blown_train = train_x.copy()
        blown_test = test_x.copy()
        blown_train[:, 2] *= 1000.0
        blown_test[:, 2] *= 1000.0

        def _acc(a, b):
            return mitigate_features_frame(
                a, train_meta, b, test_meta, ["sex"], techniques=["smote"], min_group_samples=20
            )["overall_baseline"]["accuracy"]

        assert _acc(train_x, test_x) == pytest.approx(_acc(blown_train, blown_test), abs=1e-9)

    def test_report_states_the_intervention_is_on_the_head(self) -> None:
        report = _run_frame(techniques=["reweighting"])
        assert "head" in report["scope"].lower()
        assert report["standardized"] is True

    def test_train_group_support_is_recorded(self) -> None:
        report = _run_frame(techniques=["reweighting"])
        support = report["sensitive_attributes"]["sex"]["group_support_train"]
        # Keyed on the *decoded* labels, so support lines up with the group names
        # the fairness report uses rather than the raw 0/1 encoding.
        assert set(support) == {"Female", "Male"}
        assert sum(support.values()) == report["n_train"]

    def test_one_failing_technique_does_not_kill_the_matrix(self, monkeypatch) -> None:
        from fairxai.fairness import image_feature_mitigation as mod

        real = MitigationEngine.apply_technique

        def _explode(self, technique_name, *args, **kwargs):
            if technique_name == "smote":
                raise RuntimeError("boom")
            return real(self, technique_name, *args, **kwargs)

        monkeypatch.setattr(mod.MitigationEngine, "apply_technique", _explode)
        report = _run_frame(techniques=["reweighting", "smote"])
        techs = report["sensitive_attributes"]["sex"]["techniques"]
        assert "boom" in techs["smote"]["error"]
        assert "summary_deltas" in techs["reweighting"]

    def test_missing_attribute_is_skipped_not_fatal(self) -> None:
        train_x, train_meta = _synthetic_split(200, 5, seed=3)
        test_x, test_meta = _synthetic_split(200, 5, seed=4)
        report = mitigate_features_frame(
            train_x,
            train_meta,
            test_x,
            test_meta,
            ["sex", "race_group"],
            techniques=["reweighting"],
            min_group_samples=20,
        )
        assert "sex" in report["sensitive_attributes"]
        assert "race_group" not in report["sensitive_attributes"]


class TestRunLevelOrchestration:
    def _make_run(self, tmp_path: Path) -> Path:
        run_root = tmp_path / "runs" / "r1"
        results = run_root / "baseline" / "results"
        models = run_root / "baseline" / "models"
        results.mkdir(parents=True)
        models.mkdir(parents=True)
        (models / "pad_ufes_20_resnet18.pt").write_bytes(b"stub")
        (results / "pad_ufes_20_resnet18_metrics.json").write_text(
            json.dumps(
                {
                    "status": "success",
                    "model_type": "resnet18",
                    "model_file": str(models / "pad_ufes_20_resnet18.pt"),
                    "feature_cache": True,
                    "test_metrics": {"accuracy": 0.71, "auc_roc": 0.78},
                }
            )
        )
        return run_root

    def _fake_extractor(self):
        def _extract(checkpoint_path, csv_path, **kwargs):
            seed = 1 if "train" in str(csv_path) else 2
            return _synthetic_split(200, 5, seed=seed)

        return _extract

    def test_run_writes_report_json_md_and_csv(self, tmp_path: Path) -> None:
        run_root = self._make_run(tmp_path)
        reports = mitigate_run_features(
            run_root,
            ["sex"],
            processed_dir=tmp_path / "processed",
            techniques=["reweighting"],
            min_group_samples=20,
            extractor=self._fake_extractor(),
        )
        out = run_root / "baseline" / "mitigation" / "feature_space"
        assert "pad_ufes_20_resnet18" in reports
        assert (out / "feature_mitigation_report.json").exists()
        assert (out / "feature_mitigation_report.md").exists()
        rows = pd.read_csv(out / "feature_mitigation_summary.csv")
        assert set(rows["technique"]) == {"reweighting"}
        assert set(rows["run_key"]) == {"pad_ufes_20_resnet18"}

    def test_run_carries_the_cnn_metrics_for_context(self, tmp_path: Path) -> None:
        run_root = self._make_run(tmp_path)
        reports = mitigate_run_features(
            run_root,
            ["sex"],
            processed_dir=tmp_path / "processed",
            techniques=["reweighting"],
            min_group_samples=20,
            extractor=self._fake_extractor(),
        )
        report = reports["pad_ufes_20_resnet18"]
        # The CNN softmax head and the mitigated linear head are different
        # classifiers; the CNN number is context, never the delta reference.
        assert report["cnn_test_metrics"]["accuracy"] == pytest.approx(0.71)
        assert report["cnn_test_metrics"] is not report["overall_baseline"]

    def test_model_type_filter_is_honoured(self, tmp_path: Path) -> None:
        run_root = self._make_run(tmp_path)
        reports = mitigate_run_features(
            run_root,
            ["sex"],
            processed_dir=tmp_path / "processed",
            techniques=["reweighting"],
            min_group_samples=20,
            model_types=["densenet121"],
            extractor=self._fake_extractor(),
        )
        assert reports == {}


class TestConfigAndDocs:
    def test_dermatology_config_declares_feature_space_mitigation(self) -> None:
        import yaml

        cfg = yaml.safe_load((ROOT / "configs" / "pipelines" / "dermatology.yaml").read_text())
        block = cfg["mitigation"]["feature_space"]
        assert block["enabled"] is True
        assert set(block["techniques"]) == set(DEFAULT_FEATURE_TECHNIQUES)

    def test_decisions_doc_records_the_feature_space_scope(self) -> None:
        """The doc used to say CNN pre/in-processing was out of scope. It is not."""
        text = (ROOT / "docs" / "architecture" / "decisions.md").read_text().lower()
        assert "feature-space" in text or "feature space" in text
