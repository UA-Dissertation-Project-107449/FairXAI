"""Unit tests for the profiling-sensitivity study runner (paired, repeated seeds)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "studies"))

import generate_profiling_sensitivity_plots as plots  # noqa: E402
import run_profiling_sensitivity_study as runner  # noqa: E402

from fairxai.data.synthetic import build_grid  # noqa: E402


def test_condition_keys_are_unique_per_tier_and_keep_the_mechanism():
    grid = build_grid()
    for tier in ("abstract", "healthcare"):
        keys = [runner._condition(cfg) for cfg in grid if cfg.tier == tier]
        assert len(keys) == len(set(keys)) == 12
        assert "missingness_mcar_0.2" in keys and "missingness_mar_0.2" in keys
        assert runner.BASELINE_CONDITION in keys


def _row(replicate, condition, ebm, tier="abstract"):
    return {
        "replicate": replicate,
        "seed": 100 + replicate,
        "tier": tier,
        "label": condition.split("_")[0],
        "condition": condition,
        "knob_value": 0.1,
        "ebmDifficulty": ebm,
    }


def test_paired_deltas_subtract_the_same_replicates_baseline():
    rows = [
        _row(0, "base", 0.30),
        _row(0, "imbalance_0.1", 0.50),
        _row(1, "base", 0.10),
        _row(1, "imbalance_0.1", 0.40),
    ]
    deltas = pd.DataFrame(runner._build_paired_deltas(rows))
    ebm = deltas[(deltas["metric"] == "ebmDifficulty") & (deltas["condition"] != "base")]
    assert ebm["delta"].round(6).tolist() == [0.2, 0.3]

    summary = pd.DataFrame(runner._build_knob_response(runner._build_paired_deltas(rows)))
    row = summary[(summary["condition"] == "imbalance_0.1")].iloc[0]
    assert row["n_replicates"] == 2
    assert row["delta_mean"] == pytest.approx(0.25)
    assert row["delta_ci95_low"] < 0.25 < row["delta_ci95_high"]


def test_smoke_study_writes_replicated_tables_and_environment(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "get_project_root", lambda _path: tmp_path)
    code = runner.main(["--grid-size", "smoke", "--limit", "2", "--replicates", "2"])
    assert code == 0

    study_root = next(
        (tmp_path / "output" / "synthetic" / "studies" / runner.STUDY_TYPE).glob("run_*")
    )
    datasets = pd.read_csv(study_root / "dataset_results.csv")
    assert sorted(datasets["replicate"].unique()) == [0, 1]
    assert datasets.groupby("replicate")["seed"].nunique().eq(1).all()
    assert (study_root / "knob_response_summary.csv").exists()
    # Only replicate 0 keeps its generated artefacts.
    assert len(list((study_root / "datasets").glob("*.csv"))) == 2
    assert not (study_root / ".scratch").exists()

    manifest = json.loads((study_root / "study_manifest.json").read_text())
    assert manifest["seeds"] == [20260625, 20260626]
    assert manifest["environment"]["packages"]["pandas"] == pd.__version__
    assert "commit" in manifest["environment"]["git"]
    confusion = pd.read_csv(study_root / "type_confusion.csv")
    assert set(confusion["tier"]) == {"abstract"}
    by_tier = manifest["type_accuracy_by_tier"]["abstract"]
    assert by_tier["columns"] == len(pd.read_csv(study_root / "column_results.csv"))

    # The plots read the replicated tables, including the paired-delta figures.
    monkeypatch.setattr(plots, "_ROOT", tmp_path)
    assert plots.main(["--study-id", study_root.name]) == 0
    assert (study_root / "figures" / "complexity" / "paired_delta_forest.pdf").exists()
    assert (study_root / "figures" / "complexity" / "paired_delta_heatmap.pdf").exists()
