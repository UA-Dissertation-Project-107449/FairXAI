"""Unit tests for dermatology baseline comparison (stage 9).

No model, no training: each test lays down a synthetic run directory with
metrics + fairness JSON and exercises the collation / rendering logic.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from fairxai.comparison.dermatology import (
    compare_run,
    render_markdown,
)


def _write_metrics(
    results: Path, dataset: str, model: str, *, auc: float, status: str = "success"
) -> None:
    (results / f"{dataset}_{model}_metrics.json").write_text(
        json.dumps(
            {
                "status": status,
                "model_type": model,
                "architecture": model,
                "weights_name": "IMAGENET1K_V1",
                "feature_cache": True,
                "n_train": 1616,
                "n_test": 682,
                "train_time_seconds": 64.6,
                "test_metrics": {
                    "accuracy": 0.72,
                    "precision": 0.75,
                    "recall": 0.64,
                    "f1_score": 0.69,
                    "auc_roc": auc,
                },
            }
        )
    )


def _write_fairness(run_root: Path, run_keys: list[str]) -> None:
    out = run_root / "baseline" / "prediction_fairness"
    out.mkdir(parents=True, exist_ok=True)
    report = {
        key: {
            "sensitive_attributes": {
                "sex": {
                    "group_fairness": {
                        "demographic_parity": {"max_difference": 0.05},
                        "equalized_odds": {
                            "tpr_max_difference": 0.016,
                            "fpr_max_difference": 0.066,
                        },
                        "equal_opportunity": {"max_difference": 0.016},
                    }
                }
            }
        }
        for key in run_keys
    }
    (out / "fairness_report.json").write_text(json.dumps(report))


def _make_run(tmp_path: Path, *, with_fairness: bool = True) -> Path:
    run_root = tmp_path / "runs" / "run_x"
    results = run_root / "baseline" / "results"
    results.mkdir(parents=True)
    _write_metrics(results, "pad_ufes_20", "resnet18", auc=0.81)
    _write_metrics(results, "pad_ufes_20", "efficientnet_b0", auc=0.78)
    if with_fairness:
        _write_fairness(run_root, ["pad_ufes_20_resnet18", "pad_ufes_20_efficientnet_b0"])
    return run_root


def test_compare_run_writes_outputs(tmp_path: Path) -> None:
    run_root = _make_run(tmp_path)
    rows = compare_run(run_root)

    out = run_root / "baseline" / "comparison"
    assert (out / "model_comparison.csv").exists()
    assert (out / "model_comparison.md").exists()
    assert {r["model"] for r in rows} == {"resnet18", "efficientnet_b0"}


def test_rows_carry_performance_and_identity(tmp_path: Path) -> None:
    run_root = _make_run(tmp_path)
    rows = compare_run(run_root)
    resnet = next(r for r in rows if r["model"] == "resnet18")
    assert resnet["dataset"] == "pad_ufes_20"
    assert resnet["auc"] == 0.81
    assert resnet["f1"] == 0.69
    assert resnet["architecture"] == "resnet18"
    assert resnet["n_test"] == 682


def test_rows_carry_fairness_deltas(tmp_path: Path) -> None:
    run_root = _make_run(tmp_path)
    rows = compare_run(run_root)
    resnet = next(r for r in rows if r["model"] == "resnet18")
    assert resnet["sex_dp_delta"] == 0.05
    assert resnet["sex_tpr_delta"] == 0.016
    assert resnet["sex_fpr_delta"] == 0.066
    assert resnet["sex_eo_delta"] == 0.016


def test_fairness_columns_blank_when_report_missing(tmp_path: Path) -> None:
    run_root = _make_run(tmp_path, with_fairness=False)
    rows = compare_run(run_root)
    # No fairness report -> no per-attribute columns at all.
    assert all(not k.startswith("sex_") for r in rows for k in r)


def test_failed_models_excluded(tmp_path: Path) -> None:
    run_root = tmp_path / "runs" / "run_y"
    results = run_root / "baseline" / "results"
    results.mkdir(parents=True)
    _write_metrics(results, "pad_ufes_20", "resnet18", auc=0.81)
    _write_metrics(results, "pad_ufes_20", "densenet121", auc=0.0, status="failed")
    rows = compare_run(run_root)
    assert {r["model"] for r in rows} == {"resnet18"}


def test_model_type_filter(tmp_path: Path) -> None:
    run_root = _make_run(tmp_path)
    rows = compare_run(run_root, model_types=["resnet18"])
    assert {r["model"] for r in rows} == {"resnet18"}


def test_render_markdown_has_perf_and_fairness_tables(tmp_path: Path) -> None:
    run_root = _make_run(tmp_path)
    rows = compare_run(run_root)
    md = render_markdown(rows, ["sex"])
    assert "# Dermatology Model Comparison" in md
    assert "## Test performance" in md
    assert "## Fairness — sex" in md
    assert "resnet18" in md


def test_csv_is_one_row_per_model(tmp_path: Path) -> None:
    run_root = _make_run(tmp_path)
    compare_run(run_root)
    df = pd.read_csv(run_root / "baseline" / "comparison" / "model_comparison.csv")
    assert len(df) == 2
    assert {"dataset", "model", "auc", "sex_tpr_delta"} <= set(df.columns)


def test_empty_run_yields_no_rows(tmp_path: Path) -> None:
    run_root = tmp_path / "runs" / "empty"
    (run_root / "baseline" / "results").mkdir(parents=True)
    rows = compare_run(run_root)
    assert rows == []
    assert "_No baseline models found" in render_markdown(rows, [])
