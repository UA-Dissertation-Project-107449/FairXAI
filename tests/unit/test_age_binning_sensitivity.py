"""Unit tests for the age-binning fairness sensitivity sweep."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fairxai.experiments.age_binning_sensitivity import (
    before_after_deltas,
    compute_age_binning_sensitivity,
    load_mitigation_predictions,
    run_age_binning,
)


def _pred_df(n: int = 120, seed: int = 0, shift: float = 0.0) -> pd.DataFrame:
    """Predictions with continuous age_raw + two features.

    `shift` nudges the positive rate of older patients, so a "mitigated" regime
    can differ from "baseline".
    """
    rng = np.random.default_rng(seed)
    age = rng.uniform(30, 80, size=n)
    f1 = rng.normal(size=n)
    f2 = rng.normal(size=n)
    logit = 0.04 * (age - 55) + f1 + shift * (age > 55)
    y_true = (logit + rng.normal(0, 0.3, size=n) > 0).astype(int)
    y_pred = (logit > 0).astype(int)
    return pd.DataFrame(
        {
            "age_raw": age,
            "f1": f1,
            "f2": f2,
            "y_true": y_true,
            "y_pred": y_pred,
        }
    )


def test_grid_covers_regimes_and_strategies():
    preds = {"before": _pred_df(seed=1), "after": _pred_df(seed=1, shift=-2.0)}
    grid = compute_age_binning_sensitivity(
        preds,
        strategies=["quantile_2", "quantile_3"],
        feature_cols=["f1", "f2"],
        k=5,
    )
    assert set(grid["regime"]) == {"before", "after"}
    assert set(grid["strategy"]) == {"quantile_2", "quantile_3"}
    for col in ("bin", "n", "positive_rate", "dp_gap", "eo_tpr_gap", "mean_consistency", "low_n"):
        assert col in grid.columns
    # quantile_3 yields more bins than quantile_2 within a regime.
    q2 = grid[(grid.regime == "before") & (grid.strategy == "quantile_2")]
    q3 = grid[(grid.regime == "before") & (grid.strategy == "quantile_3")]
    assert len(q3) > len(q2)


def test_low_n_bins_flagged():
    preds = {"before": _pred_df(n=120, seed=2)}
    grid = compute_age_binning_sensitivity(
        preds, strategies=["quantile_2"], feature_cols=["f1", "f2"], k=5, min_bin_size=80
    )
    # Each quantile_2 bin holds ~60 < 80 → all flagged low_n.
    assert grid["low_n"].all()


def test_skips_regime_without_age_column():
    preds = {"before": _pred_df(seed=3), "after": _pred_df(seed=3).drop(columns=["age_raw"])}
    grid = compute_age_binning_sensitivity(
        preds, strategies=["quantile_2"], feature_cols=["f1", "f2"], k=5
    )
    assert set(grid["regime"]) == {"before"}


def test_before_after_deltas_pivot():
    preds = {"before": _pred_df(seed=4), "after": _pred_df(seed=4, shift=-3.0)}
    grid = compute_age_binning_sensitivity(
        preds, strategies=["quantile_2"], feature_cols=["f1", "f2"], k=5
    )
    deltas = before_after_deltas(grid, metric="dp_gap")
    assert {"strategy", "bin", "before", "after", "delta"} <= set(deltas.columns)
    # delta = after - before, elementwise.
    assert np.allclose(deltas["delta"], deltas["after"] - deltas["before"])


def _write_baseline_preds(run_root, dataset, model, df):
    pdir = run_root / "baseline" / "results" / "predictions"
    pdir.mkdir(parents=True, exist_ok=True)
    n = len(df)
    df.iloc[: n // 2].to_csv(pdir / f"{dataset}_{model}_train.csv", index=False)
    df.iloc[n // 2 :].to_csv(pdir / f"{dataset}_{model}_test.csv", index=False)


def _write_mitigation_preds(run_root, dataset, technique, constraint, df):
    mdir = run_root / "experiments" / "mitigation" / "predictions"
    mdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(mdir / f"{dataset}_{technique}_{constraint}.csv", index=False)


def test_load_mitigation_predictions_keys_by_technique_constraint():
    run_root_df = _pred_df(seed=5)
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        run_root = __import__("pathlib").Path(d)
        _write_mitigation_preds(run_root, "cleveland", "reweighing", "sex", run_root_df)
        loaded = load_mitigation_predictions(run_root, "cleveland")
    assert "reweighing_sex" in loaded
    assert "age_raw" in loaded["reweighing_sex"].columns


def test_run_age_binning_writes_baseline_and_before_after(tmp_path):
    base = _pred_df(seed=6)
    mit = _pred_df(seed=6, shift=-2.0)
    run_root = tmp_path / "run"
    _write_baseline_preds(run_root, "cleveland", "logistic_regression", base)
    _write_mitigation_preds(run_root, "cleveland", "reweighing", "sex", mit)

    out_base = tmp_path / "out"
    summary = run_age_binning(
        run_root=run_root,
        dataset="cleveland",
        strategies=["quantile_2"],
        out_base=out_base,
        k=5,
    )
    assert summary is not None
    # Axis B: per-model baseline sweep.
    assert (out_base / "cleveland" / "logistic_regression" / "age_binning_sensitivity.csv").exists()
    # Axis A: before/after vs the mitigation set.
    ba = out_base / "cleveland" / "before_after" / "reweighing_sex"
    assert (ba / "age_binning_sensitivity.csv").exists()
    assert (ba / "before_after_deltas.csv").exists()
