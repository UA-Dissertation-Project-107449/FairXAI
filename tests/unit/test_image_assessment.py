"""Unit tests for dermatology post-prediction fairness assessment.

No model, no training: every test builds a synthetic predictions DataFrame and
exercises the gating / decode / report logic directly.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from fairxai.fairness.image_assessment import (
    _flatten_for_csv,
    assess_group_views_frame,
    assess_predictions_frame,
    assess_run,
    decode_groups,
    derive_age_coarse,
    derive_group_view_columns,
    render_markdown,
)


def _synthetic_predictions(n_big: int = 60, n_small: int = 5) -> pd.DataFrame:
    """Two well-supported sex groups + a tiny Fitzpatrick V-VI cell."""
    rows = []
    # Female (sex=0): biased toward correct negatives.
    for i in range(n_big):
        y = i % 2
        rows.append(
            {
                "sex": 0,
                "fitzpatrick_group": "I-II",
                "y_true": y,
                "y_pred": y,
                "y_proba": 0.8 if y else 0.2,
            }
        )
    # Male (sex=1): some errors so metrics differ across groups.
    for i in range(n_big):
        y = i % 2
        pred = y if i % 3 else 1 - y
        rows.append(
            {
                "sex": 1,
                "fitzpatrick_group": "III-IV",
                "y_true": y,
                "y_pred": pred,
                "y_proba": 0.6 if pred else 0.4,
            }
        )
    # Tiny cells (sex=-1 Unknown, Fitzpatrick V-VI) — skipped under min=50.
    for i in range(n_small):
        rows.append(
            {"sex": -1, "fitzpatrick_group": "V-VI", "y_true": 1, "y_pred": 1, "y_proba": 0.9}
        )
    return pd.DataFrame(rows)


def test_decode_groups_maps_sex_codes() -> None:
    df = pd.DataFrame({"sex": [0, 1, -1]})
    decoded = decode_groups(df, "sex").tolist()
    assert decoded == ["Female", "Male", "Unknown"]


def test_decode_groups_passthrough_for_plaintext() -> None:
    df = pd.DataFrame({"fitzpatrick_group": ["I-II", "unknown"]})
    assert decode_groups(df, "fitzpatrick_group").tolist() == ["I-II", "unknown"]


def test_age_coarse_mapping_handles_pad_groups_and_unknowns() -> None:
    df = pd.DataFrame({"age_group": ["<20", "20-39", "40-59", "60-79", "80+", "unknown", None]})
    assert derive_age_coarse(df).tolist() == [
        "<40",
        "<40",
        "40-59",
        "60+",
        "60+",
        "Unknown",
        "Unknown",
    ]


def test_sex_x_fitzpatrick_uses_decoded_sex_labels() -> None:
    df = pd.DataFrame(
        {
            "sex": [0, 1, -1],
            "fitzpatrick_group": ["I-II", "III-IV", "unknown"],
            "y_true": [0, 1, 0],
            "y_pred": [0, 1, 0],
            "y_proba": [0.1, 0.9, 0.2],
        }
    )
    work, meta = derive_group_view_columns(df, ["sex_x_fitzpatrick"])
    assert work["sex_x_fitzpatrick"].tolist() == [
        "Female x I-II",
        "Male x III-IV",
        "Unknown x unknown",
    ]
    assert meta["sex_x_fitzpatrick"]["exploratory"] is True


def test_overall_performance_present() -> None:
    df = _synthetic_predictions()
    report = assess_predictions_frame(df, ["sex"], min_group_samples=50)
    op = report["overall_performance"]
    assert set(op) == {"accuracy", "precision", "recall", "f1", "auc"}
    assert 0.0 <= op["accuracy"] <= 1.0
    assert op["auc"] is not None  # both classes present


def test_sex_decoded_into_group_keys() -> None:
    df = _synthetic_predictions()
    report = assess_predictions_frame(df, ["sex"], min_group_samples=50)
    groups = report["sensitive_attributes"]["sex"]["group_performance"]
    assert set(groups) == {"Female", "Male"}
    assert all(g["n"] == 60 for g in groups.values())


def test_tiny_group_is_skipped_and_reported() -> None:
    df = _synthetic_predictions()
    report = assess_predictions_frame(df, ["fitzpatrick_group"], min_group_samples=50)
    fitz = report["sensitive_attributes"]["fitzpatrick_group"]
    assert "V-VI" not in fitz["group_performance"]
    assert any(s["group"] == "V-VI" and s["count"] == 5 for s in fitz["skipped_groups"])


def test_single_valid_group_yields_no_comparison() -> None:
    # One group of 60, one of 10: at min=50 only the large group survives.
    rows = [{"sex": 0, "y_true": i % 2, "y_pred": i % 2, "y_proba": 0.7} for i in range(60)] + [
        {"sex": 1, "y_true": i % 2, "y_pred": i % 2, "y_proba": 0.7} for i in range(10)
    ]
    df = pd.DataFrame(rows)
    report = assess_predictions_frame(df, ["sex"], min_group_samples=50)
    sex = report["sensitive_attributes"]["sex"]
    assert set(sex["group_performance"]) == {"Female"}
    assert sex["group_fairness"] == {}
    assert "note" in sex


def test_group_fairness_computed_for_two_groups() -> None:
    df = _synthetic_predictions()
    report = assess_predictions_frame(df, ["sex"], min_group_samples=50)
    gf = report["sensitive_attributes"]["sex"]["group_fairness"]
    assert "demographic_parity" in gf and "equalized_odds" in gf
    assert gf["equalized_odds"]["tpr_max_difference"] >= 0.0


def test_group_view_report_respects_intersection_min_support() -> None:
    df = _synthetic_predictions(n_big=40, n_small=5)
    report = assess_group_views_frame(
        df,
        ["sex_x_fitzpatrick"],
        min_group_samples=50,
        intersection_min_group_samples=30,
    )
    view = report["group_views"]["sex_x_fitzpatrick"]
    assert view["min_group_samples"] == 30
    assert "Female x I-II" in view["group_performance"]
    assert "Male x III-IV" in view["group_performance"]
    assert any(s["group"] == "Unknown x V-VI" and s["count"] == 5 for s in view["skipped_groups"])


def test_degenerate_group_excluded_from_deltas() -> None:
    # Female/Male both have positives+negatives; a 60-row all-benign group has no
    # positives → must be flagged degenerate and excluded from the delta math.
    rows = []
    for i in range(60):
        rows.append({"sex": 0, "y_true": i % 2, "y_pred": i % 2, "y_proba": 0.7})
        rows.append({"sex": 1, "y_true": i % 2, "y_pred": (i + 1) % 2, "y_proba": 0.4})
    for i in range(60):  # sex=-1 Unknown, all benign
        rows.append({"sex": -1, "y_true": 0, "y_pred": 0, "y_proba": 0.1})
    df = pd.DataFrame(rows)

    report = assess_predictions_frame(df, ["sex"], min_group_samples=50)
    sex = report["sensitive_attributes"]["sex"]

    # Unknown kept in the performance table but flagged degenerate (no positives).
    assert "Unknown" in sex["group_performance"]
    assert sex["group_performance"]["Unknown"]["recall"] is None
    assert any(
        d["group"] == "Unknown" and d["reason"] == "no positives" for d in sex["degenerate_groups"]
    )

    # Deltas computed over Female/Male only (both well-formed) → still produced.
    assert "demographic_parity" in sex["group_fairness"]


def test_missing_attribute_is_skipped() -> None:
    df = _synthetic_predictions()
    report = assess_predictions_frame(df, ["sex", "not_a_column"], min_group_samples=50)
    assert "sex" in report["sensitive_attributes"]
    assert "not_a_column" not in report["sensitive_attributes"]


def test_render_markdown_and_flatten() -> None:
    df = _synthetic_predictions()
    report = assess_predictions_frame(df, ["sex", "fitzpatrick_group"], min_group_samples=50)
    md = render_markdown({"pad_ufes_20_resnet18": report})
    assert "# Dermatology Prediction Fairness Report" in md
    assert "Female" in md and "skipped (under min)" in md

    flat = _flatten_for_csv({"pad_ufes_20_resnet18": report})
    assert {"run_key", "sensitive_attribute", "group", "auc"} <= set(flat.columns)
    assert (flat["group"] == "Female").any()


def test_assess_run_writes_outputs(tmp_path: Path) -> None:
    run_root = tmp_path / "runs" / "run_x"
    results = run_root / "baseline" / "results"
    preds = results / "predictions"
    preds.mkdir(parents=True)

    df = _synthetic_predictions()
    csv_path = preds / "pad_ufes_20_resnet18_test.csv"
    df.to_csv(csv_path, index=False)
    (results / "pad_ufes_20_resnet18_metrics.json").write_text(
        json.dumps({"test_predictions": str(csv_path)})
    )

    reports = assess_run(run_root, ["sex", "fitzpatrick_group"], min_group_samples=50)

    assert "pad_ufes_20_resnet18" in reports
    out = run_root / "baseline" / "prediction_fairness"
    assert (out / "fairness_report.json").exists()
    assert (out / "fairness_report.md").exists()
    assert (out / "fairness_groups.csv").exists()


def test_assess_run_writes_group_view_outputs_without_mutating_base_report(tmp_path: Path) -> None:
    run_root = tmp_path / "runs" / "run_group_views"
    results = run_root / "baseline" / "results"
    preds = results / "predictions"
    preds.mkdir(parents=True)

    df = _synthetic_predictions()
    csv_path = preds / "pad_ufes_20_resnet18_test.csv"
    df.to_csv(csv_path, index=False)
    (results / "pad_ufes_20_resnet18_metrics.json").write_text(
        json.dumps({"test_predictions": str(csv_path)})
    )

    assess_run(
        run_root,
        ["sex"],
        min_group_samples=50,
        write_group_views=True,
        group_views=["age_coarse", "sex_x_fitzpatrick"],
        group_view_min_group_samples=50,
        intersection_min_group_samples=30,
    )

    out = run_root / "baseline" / "prediction_fairness"
    base = json.loads((out / "fairness_report.json").read_text())
    assert "group_views" not in base["pad_ufes_20_resnet18"]

    group_dir = out / "group_views"
    assert (group_dir / "group_view_report.json").exists()
    assert (group_dir / "group_view_report.md").exists()
    assert (group_dir / "group_view_groups.csv").exists()
    group_report = json.loads((group_dir / "group_view_report.json").read_text())
    assert {"age_coarse", "sex_x_fitzpatrick"} <= set(
        group_report["pad_ufes_20_resnet18"]["group_views"]
    )


def test_assess_run_model_type_filter(tmp_path: Path) -> None:
    run_root = tmp_path / "runs" / "run_y"
    results = run_root / "baseline" / "results"
    preds = results / "predictions"
    preds.mkdir(parents=True)
    df = _synthetic_predictions()
    for model in ("resnet18", "efficientnet_b0"):
        csv_path = preds / f"pad_ufes_20_{model}_test.csv"
        df.to_csv(csv_path, index=False)
        (results / f"pad_ufes_20_{model}_metrics.json").write_text(
            json.dumps({"test_predictions": str(csv_path)})
        )

    reports = assess_run(run_root, ["sex"], min_group_samples=50, model_types=["resnet18"])
    assert set(reports) == {"pad_ufes_20_resnet18"}
