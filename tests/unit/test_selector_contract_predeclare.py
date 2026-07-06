"""Regression: selector contract must PREDECLARE the primary feature mode and
model families, never let higher test-set scores from the feature-selection study
replace them.

A test-derived mode/model that outscores the predeclared baseline is preserved as
informational ranking only; swapping it into the primary recommendation would leak
the held-out test set into the final model.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_FAIRXAI_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_FAIRXAI_ROOT / "src"))
sys.path.insert(0, str(_FAIRXAI_ROOT / "scripts" / "studies"))

import build_selector_contract as bsc  # noqa: E402


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _metrics(f1: float) -> dict:
    # _composite_score needs accuracy, recall, f1_score, auc_roc all present.
    return {"accuracy": f1, "recall": f1, "f1_score": f1, "auc_roc": f1}


def _seed_study(project_root: Path) -> None:
    """One feature-selection study where include_all_sensitive outscores the
    predeclared exclude_sensitive baseline."""
    fs_root = project_root / "output/cardiac/studies/feature_selection"
    study_dir = fs_root / "run_20260706_000000"
    study_dir.mkdir(parents=True)
    (fs_root / "latest.txt").write_text("run_20260706_000000", encoding="utf-8")

    runs = []
    # (mode, model, f1) — include_all_sensitive scores higher than the baseline,
    # and random_forest outranks logistic_regression on the test set.
    for mode, model, f1 in [
        ("exclude_sensitive", "logistic_regression", 0.70),
        ("include_all_sensitive", "logistic_regression", 0.90),
        ("include_all_sensitive", "random_forest", 0.95),
    ]:
        sub_key = f"fs_{mode}__{model}"
        baseline_root = study_dir / "runs" / sub_key / "baseline"
        _write_json(
            baseline_root / "results" / "training_results.json",
            {"cleveland_uci": {model: {"test_metrics": _metrics(f1)}}},
        )
        runs.append(
            {
                "status": "success",
                "mode": mode,
                "model": model,
                "baseline_root": str(baseline_root.relative_to(project_root)),
            }
        )
    _write_json(study_dir / "study_manifest.json", {"runs": runs, "rfe_top_k": 10})


def _seed_configs(project_root: Path) -> None:
    pipeline_cfg = (
        "runtime:\n"
        "  datasets: [cleveland_uci]\n"
        "training:\n"
        "  target: heart_disease\n"
        "  feature_selection_mode: exclude_sensitive\n"
        "  model_types: [logistic_regression, random_forest, svm]\n"
        "  rfe_top_k: 10\n"
        "fairness:\n"
        "  sensitive_attributes: [age_group, sex]\n"
    )
    cfg_path = project_root / "configs/pipelines/cardiac.yaml"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(pipeline_cfg, encoding="utf-8")


def test_predeclared_baseline_survives_higher_scoring_study_mode(
    tmp_path: Path, monkeypatch
) -> None:
    _seed_configs(tmp_path)
    _seed_study(tmp_path)

    monkeypatch.setattr(bsc, "get_project_root", lambda _current_file: tmp_path)
    monkeypatch.setattr(sys, "argv", ["build_selector_contract.py", "--pipeline", "cardiac"])

    bsc.main()

    contract = json.loads(
        (tmp_path / "output/cardiac/recommendations/selector_contract.latest.json").read_text(
            encoding="utf-8"
        )
    )
    recs = contract["recommendations"]
    fs = contract["studies"]["feature_selection"]

    # The study genuinely surfaced a higher-scoring non-baseline mode/model...
    assert fs["highest_scoring_mode"] == "include_all_sensitive"
    assert fs["recommended_model_types"][0] == "random_forest"

    # ...but the primary recommendation stays predeclared, not the study winner.
    assert recs["feature_selection_mode"] == "exclude_sensitive"
    assert recs["feature_selection_mode_source"] == "predeclared_baseline"
    assert recs["model_types"] == ["logistic_regression", "random_forest", "svm"]
    assert recs["model_types_source"] == "predeclared_requested"
    # Test-derived ordering is kept as informational ranking only.
    assert recs["model_ranking"][0] == "random_forest"
    assert "svm" not in recs["model_ranking"]  # svm study run absent, but not dropped above
