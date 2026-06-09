"""Age-binning fairness sensitivity sweep (post-hoc, analysis only).

`age_group` is a sensitive attribute measured under **one** binning strategy, yet
many strategies exist and the attribute-binning study is stage 9 — *after*
train/assess. But predictions are **independent** of the age binning (age is not a
model feature), so per-bin fairness can be recomputed under any number of
strategies with **no retraining**.

This module recomputes, on *given* predictions, two orthogonal views:

* **Axis A — mitigation effect**: hold a strategy fixed, compare ``before``
  (baseline preds) vs ``after`` (mitigated preds) per age bin.
* **Axis B — binning sensitivity**: hold a prediction set fixed, vary the
  strategy — is a gap real or a bucketing artifact?

Reuses, does not reinvent: ``create_binning_strategy`` / ``apply_binning``
(binning), :class:`FairnessMetrics` (group gaps),
:class:`SimilarityEngine.per_group_consistency` (individual fairness).

See the root design note ``AGE_BINNING_FAIRNESS_SENSITIVITY.md``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from fairxai.fairness.metrics import FairnessMetrics
from fairxai.similarity.engine import SimilarityEngine

from .attribute_binning import apply_binning, create_binning_strategy

logger = logging.getLogger(__name__)

_BIN_COL = "age_bin_exp"

# Output column order for the tidy sweep grid.
GRID_COLUMNS = [
    "regime",
    "strategy",
    "bin",
    "n",
    "positive_rate",
    "dp_gap",
    "eo_tpr_gap",
    "eo_fpr_gap",
    "mean_consistency",
    "low_n",
]


def compute_age_binning_sensitivity(
    predictions: Dict[str, pd.DataFrame],
    strategies: List[str],
    feature_cols: List[str],
    age_col: str = "age_raw",
    pred_col: str = "y_pred",
    true_col: str = "y_true",
    k: int = 5,
    min_bin_size: int = 20,
) -> pd.DataFrame:
    """Recompute per-age-bin fairness across regimes × strategies.

    Parameters
    ----------
    predictions
        ``{regime: df}`` — e.g. ``{"before": baseline_preds, "after": mitigated}``.
        A regime whose frame lacks *age_col* is skipped (logged).
    strategies
        Age-binning strategy names (``create_binning_strategy`` resolvable, e.g.
        ``quantile_3``, ``fixed_10yr``).
    feature_cols
        Columns for the k-NN individual-fairness distance (already excluding
        ``*_raw``/sensitive/meta — see ``resolve_feature_cols``).
    min_bin_size
        Bins with fewer rows are flagged ``low_n`` (noisy gaps; same min-size
        lens as the clustering validity gate).

    Returns
    -------
    Tidy DataFrame, one row per (regime, strategy, bin), columns
    :data:`GRID_COLUMNS`. Strategy-level gaps (``dp_gap``/``eo_*``) repeat across
    that strategy's bins; bin-level values (``n``/``positive_rate``/
    ``mean_consistency``) vary per row.
    """
    valid_feats = [c for c in feature_cols if c]
    rows: List[Dict] = []

    for regime, df in predictions.items():
        if age_col not in df.columns:
            logger.info("[INFO] age-binning sweep: regime %r has no %r; skipping", regime, age_col)
            continue
        present_feats = [c for c in valid_feats if c in df.columns]

        for strategy in strategies:
            try:
                bins, labels = create_binning_strategy(df, strategy, col=age_col)
            except ValueError as exc:
                logger.warning("[WARN] age-binning sweep: strategy %r failed: %s", strategy, exc)
                continue
            binned = apply_binning(df, bins, labels, col=age_col, output_col=_BIN_COL)

            fm = FairnessMetrics(sensitive_attributes=[_BIN_COL])
            dp = fm.demographic_parity(binned, _BIN_COL, pred_col=pred_col)
            eo = fm.equalized_odds(binned, _BIN_COL, true_col=true_col, pred_col=pred_col)

            bin_cons: Dict[str, Dict] = {}
            if present_feats:
                engine = SimilarityEngine(k_values=[k], pred_col=pred_col)
                bin_cons = engine.per_group_consistency(
                    binned, present_feats, group_cols=[_BIN_COL], k=k
                ).get(_BIN_COL, {})

            for bin_label, grp in dp["group_rates"].items():
                n = grp["count"]
                rows.append(
                    {
                        "regime": regime,
                        "strategy": strategy,
                        "bin": bin_label,
                        "n": n,
                        "positive_rate": grp["positive_rate"],
                        "dp_gap": dp["max_difference"],
                        "eo_tpr_gap": eo["tpr_max_difference"],
                        "eo_fpr_gap": eo["fpr_max_difference"],
                        "mean_consistency": bin_cons.get(bin_label, {}).get("mean_consistency"),
                        "low_n": n < min_bin_size,
                    }
                )

    return pd.DataFrame(rows, columns=GRID_COLUMNS)


def before_after_deltas(
    grid: pd.DataFrame,
    metric: str = "dp_gap",
    before: str = "before",
    after: str = "after",
) -> pd.DataFrame:
    """Axis A pivot: per (strategy, bin), ``delta = after - before`` for *metric*."""
    sub = grid[grid["regime"].isin([before, after])]
    wide = sub.pivot_table(
        index=["strategy", "bin"], columns="regime", values=metric, aggfunc="first"
    ).reset_index()
    for regime in (before, after):
        if regime not in wide.columns:
            wide[regime] = pd.NA
    wide["delta"] = wide[after] - wide[before]
    return wide[["strategy", "bin", before, after, "delta"]]


def load_mitigation_predictions(run_root: Path, dataset: str) -> Dict[str, pd.DataFrame]:
    """Load persisted mitigated per-sample predictions for *dataset*.

    Reads ``experiments/mitigation/predictions/<dataset>_<technique>_<constraint>.csv``
    (written by ``run_mitigation_comparison``). Returns
    ``{"<technique>_<constraint>": df}`` — each an "after" regime.
    """
    mdir = run_root / "experiments" / "mitigation" / "predictions"
    out: Dict[str, pd.DataFrame] = {}
    if not mdir.exists():
        return out
    prefix = f"{dataset}_"
    for path in sorted(mdir.glob(f"{dataset}_*.csv")):
        regime = path.stem[len(prefix) :]
        out[regime] = pd.read_csv(path)
    return out


def run_age_binning(
    run_root: Path,
    dataset: str,
    strategies: List[str],
    out_base: Path,
    feature_exclude: Optional[List[str]] = None,
    baseline_model: str = "logistic_regression",
    k: int = 5,
    min_bin_size: int = 20,
    age_col: str = "age_raw",
) -> Optional[Dict]:
    """Age-binning sweep for *dataset* under *run_root*.

    Two outputs per the design note:

    * **Axis B** (binning sensitivity) — per baseline model, sweep strategies on
      its predictions → ``out_base/<dataset>/<model>/``.
    * **Axis A** (mitigation effect) — pair the *baseline_model* baseline preds
      ("before") against each persisted mitigation set ("after") →
      ``out_base/<dataset>/before_after/<technique>_<constraint>/``.

    Returns a summary dict, or ``None`` when no baseline predictions exist.
    """
    # Local import avoids a hard package-level dependency cycle (similarity ↔ experiments).
    from fairxai.similarity.similarity_pipeline import (
        load_all_model_predictions,
        resolve_feature_cols,
    )

    baseline_models = load_all_model_predictions(run_root, dataset)
    if not baseline_models:
        logger.info("[INFO] age-binning sweep: no baseline predictions for %s", dataset)
        return None

    # Loud guard: without `age_col` in the baseline predictions there is no "before"
    # — Axis B is empty and Axis A degenerates to after-only. Almost always means
    # age_raw was not carried into the baseline prediction CSVs (re-run train).
    if not any(age_col in df.columns for df in baseline_models.values()):
        logger.warning(
            "[WARN] age-binning sweep: %r missing from ALL baseline predictions for %s — "
            "no 'before' regime (Axis B empty, Axis A after-only). Re-run the train stage "
            "so %r is carried into baseline prediction CSVs.",
            age_col,
            dataset,
            age_col,
        )

    summary: Dict[str, object] = {"axis_b_models": [], "axis_a_regimes": []}

    # Axis B — per-model binning sensitivity on baseline predictions.
    for model, df in baseline_models.items():
        feats = resolve_feature_cols(df, exclude=feature_exclude)
        grid = run_age_binning_sensitivity(
            {"baseline": df},
            strategies=strategies,
            feature_cols=feats,
            out_dir=out_base / dataset / model,
            k=k,
            min_bin_size=min_bin_size,
            age_col=age_col,
        )
        if grid is not None:
            summary["axis_b_models"].append(model)

    # Axis A — before (baseline_model) vs each mitigation set.
    mitigation = load_mitigation_predictions(run_root, dataset)
    before_df = baseline_models.get(baseline_model)
    if mitigation and before_df is not None:
        feats = resolve_feature_cols(before_df, exclude=feature_exclude)
        for regime, after_df in mitigation.items():
            grid = run_age_binning_sensitivity(
                {"before": before_df, "after": after_df},
                strategies=strategies,
                feature_cols=feats,
                out_dir=out_base / dataset / "before_after" / regime,
                k=k,
                min_bin_size=min_bin_size,
                age_col=age_col,
            )
            if grid is not None:
                summary["axis_a_regimes"].append(regime)

    return summary


def run_age_binning_sensitivity(
    predictions: Dict[str, pd.DataFrame],
    strategies: List[str],
    feature_cols: List[str],
    out_dir: Path,
    delta_metric: str = "dp_gap",
    **kwargs,
) -> Optional[pd.DataFrame]:
    """Compute the sweep and persist tidy CSV + before/after deltas + summary.

    Returns the grid, or ``None`` if no regime produced rows.
    """
    grid = compute_age_binning_sensitivity(predictions, strategies, feature_cols, **kwargs)
    if grid.empty:
        logger.info("[INFO] age-binning sweep: no rows for %s", out_dir)
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    grid.to_csv(out_dir / "age_binning_sensitivity.csv", index=False)

    if {"before", "after"} <= set(grid["regime"]):
        before_after_deltas(grid, metric=delta_metric).to_csv(
            out_dir / "before_after_deltas.csv", index=False
        )

    summary = {
        "regimes": sorted(grid["regime"].unique().tolist()),
        "strategies": sorted(grid["strategy"].unique().tolist()),
        "n_bins_total": int(len(grid)),
        "n_low_n_bins": int(grid["low_n"].sum()),
        "dp_gap_by_strategy": grid.groupby("strategy")["dp_gap"].first().to_dict(),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return grid
