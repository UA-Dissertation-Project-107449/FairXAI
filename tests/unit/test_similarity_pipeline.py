"""Unit tests for the post-assess similarity analysis pipeline."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from fairxai.similarity.similarity_pipeline import (
    load_all_model_predictions,
    resolve_feature_cols,
    run_similarity,
    run_similarity_for_predictions,
)


def _pred_df(n_per: int = 25, seed: int = 0) -> pd.DataFrame:
    """Two feature clusters aligned with prediction, plus sensitive cols."""
    rng = np.random.default_rng(seed)
    feat = np.concatenate([rng.normal(0, 0.3, n_per), rng.normal(5, 0.3, n_per)])
    noise = rng.uniform(0, 1000, size=2 * n_per)
    pred = np.array([0] * n_per + [1] * n_per)
    return pd.DataFrame(
        {
            "feat": feat,
            "chol_big": noise,
            "sex_cat": (["M"] * n_per + ["F"] * n_per),
            "group_cluster": ([0] * n_per + [1] * n_per),
            "y_true": pred,
            "y_pred": pred,
        }
    )


def test_resolve_feature_cols_drops_meta_and_sensitive():
    df = _pred_df()
    cols = resolve_feature_cols(df, exclude=["sex_cat", "group_cluster"])
    assert "feat" in cols and "chol_big" in cols
    assert "y_pred" not in cols and "y_true" not in cols
    assert "group_cluster" not in cols and "sex_cat" not in cols


def test_resolve_feature_cols_drops_raw_metadata():
    """`*_raw` columns are analysis metadata (continuous source of a sensitive
    attribute), never k-NN features — the model never saw them."""
    df = _pred_df()
    df["age_raw"] = range(len(df))
    cols = resolve_feature_cols(df, exclude=["sex_cat", "group_cluster"])
    assert "age_raw" not in cols
    assert "feat" in cols


def test_run_for_predictions_writes_scores_and_per_group(tmp_path):
    df = _pred_df()
    out = tmp_path / "logreg"
    summary = run_similarity_for_predictions(
        df,
        feature_cols=["feat", "chol_big"],
        sensitive_attrs=["sex_cat", "group_cluster"],
        k_values=[5, 10],
        out_dir=out,
    )
    assert summary is not None
    assert (out / "similarity_fairness_scores.csv").exists()
    assert (out / "per_group_consistency.json").exists()
    pg = json.loads((out / "per_group_consistency.json").read_text())
    assert "sex_cat" in pg and "group_cluster" in pg
    # Scaled distance ignores chol_big → clusters align with pred → high consistency.
    assert summary["overall_mean_consistency"] > 0.7


def test_run_for_predictions_returns_none_without_pred_col(tmp_path):
    df = _pred_df().drop(columns=["y_pred"])
    result = run_similarity_for_predictions(
        df, feature_cols=["feat"], sensitive_attrs=["sex_cat"], k_values=[5], out_dir=tmp_path / "m"
    )
    assert result is None


def _write_model_preds(run_root, dataset, model, train_df, test_df):
    pdir = run_root / "baseline" / "results" / "predictions"
    pdir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(pdir / f"{dataset}_{model}_train.csv", index=False)
    test_df.to_csv(pdir / f"{dataset}_{model}_test.csv", index=False)


def test_load_all_models_returns_every_model(tmp_path):
    df = _pred_df()
    for model in ("logistic_regression", "random_forest"):
        _write_model_preds(tmp_path, "cleveland", model, df.iloc[:30], df.iloc[30:])
    loaded = load_all_model_predictions(tmp_path, "cleveland")
    assert set(loaded.keys()) == {"logistic_regression", "random_forest"}
    assert all(len(v) == len(df) for v in loaded.values())


def test_run_similarity_produces_per_model_dirs(tmp_path):
    df = _pred_df()
    run_root = tmp_path / "run"
    for model in ("logistic_regression", "random_forest"):
        _write_model_preds(run_root, "cleveland", model, df.iloc[:30], df.iloc[30:])

    summary = run_similarity(
        run_root=run_root,
        dataset="cleveland",
        sensitive_attrs=["sex_cat", "group_cluster"],
        k_values=[5],
        out_base=tmp_path / "out",
        feature_exclude=["sex_cat", "group_cluster", "y_true", "y_pred"],
    )
    assert summary is not None
    for model in ("logistic_regression", "random_forest"):
        assert (tmp_path / "out" / "cleveland" / model / "similarity_fairness_scores.csv").exists()


def test_run_similarity_no_predictions_returns_none(tmp_path):
    result = run_similarity(
        run_root=tmp_path / "empty",
        dataset="cleveland",
        sensitive_attrs=["sex_cat"],
        k_values=[5],
        out_base=tmp_path / "out",
    )
    assert result is None
