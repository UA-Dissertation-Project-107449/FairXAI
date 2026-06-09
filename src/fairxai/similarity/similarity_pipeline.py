"""Post-assess similarity analysis (individual fairness), wired per model.

Runs AFTER training/assess — it needs predictions. Unlike clustering (which
becomes a training-affecting sensitive attribute), this is **analysis only**:
k-NN prediction consistency per model, a per-sensitive-group breakdown, and a
violation-density map. Features are z-scored before distance so high-magnitude
columns don't dominate the neighbourhood.

Shared by the standalone study (``run_grouping_analysis.py``) and the gated
pipeline step (``scripts/cardiac/similarity_analysis.py``).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .density import ViolationDensityMapper
from .engine import SimilarityEngine

logger = logging.getLogger(__name__)

# Columns that are never k-NN features (targets, predictions, ids, decoded cats).
_META_COLS = {"y_true", "y_pred", "y_prob", "y_score", "index", "Unnamed: 0"}

DEFAULT_FEATURE_EXCLUDE = [
    "heart_disease",
    "target",
    "age_group",
    "sex",
    "ethnicity",
    "group_cluster",
]


def resolve_feature_cols(df: pd.DataFrame, exclude: Optional[List[str]] = None) -> List[str]:
    """Numeric feature columns for distance, dropping meta/sensitive/decoded cols."""
    exclude_set = set(exclude or []) | _META_COLS
    cols = []
    for c in df.select_dtypes(include="number").columns:
        if c in exclude_set or c.endswith("_cat") or c.endswith("_raw"):
            continue
        cols.append(c)
    return cols


def _model_key(stem: str, dataset: str, split_suffix: str) -> Optional[str]:
    prefix = f"{dataset}_"
    if not stem.startswith(prefix) or not stem.endswith(split_suffix):
        return None
    return stem[len(prefix) : -len(split_suffix)]


def _load_layout(
    results_dir: Path, dataset: str, train_suffix: str, test_suffix: str
) -> Dict[str, pd.DataFrame]:
    """Collect every model's train+test prediction pair under one layout."""
    if not results_dir.exists():
        return {}
    train_by_model = {
        m: p
        for p in sorted(results_dir.glob(f"{dataset}_*{train_suffix}.csv"))
        if (m := _model_key(p.stem, dataset, train_suffix)) is not None
    }
    test_by_model = {
        m: p
        for p in sorted(results_dir.glob(f"{dataset}_*{test_suffix}.csv"))
        if (m := _model_key(p.stem, dataset, test_suffix)) is not None
    }
    out: Dict[str, pd.DataFrame] = {}
    for model in sorted(set(train_by_model) & set(test_by_model)):
        train_df = pd.read_csv(train_by_model[model])
        test_df = pd.read_csv(test_by_model[model])
        out[model] = pd.concat([train_df, test_df], ignore_index=True)
    return out


def load_all_model_predictions(run_root: Path, dataset: str) -> Dict[str, pd.DataFrame]:
    """Load train+test predictions for **every** model of *dataset*.

    Supports the current ``baseline/results/predictions/`` layout and the legacy
    flat ``baseline/results/`` layout. Returns ``{model: concat(train, test)}``.
    """
    current = _load_layout(
        run_root / "baseline" / "results" / "predictions", dataset, "_train", "_test"
    )
    if current:
        return current
    return _load_layout(
        run_root / "baseline" / "results", dataset, "_train_predictions", "_test_predictions"
    )


def run_similarity_for_predictions(
    pred_df: pd.DataFrame,
    feature_cols: List[str],
    sensitive_attrs: List[str],
    k_values: List[int],
    out_dir: Path,
    pred_col: str = "y_pred",
) -> Optional[Dict]:
    """Core per-model analysis: multi-k consistency + per-group + density map.

    Returns a summary dict, or ``None`` when *pred_col* or features are missing.
    """
    valid_cols = [c for c in feature_cols if c in pred_df.columns]
    if pred_col not in pred_df.columns or not valid_cols:
        logger.info("[INFO] similarity: missing %r or no features; skipping %s", pred_col, out_dir)
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    engine = SimilarityEngine(k_values=k_values, pred_col=pred_col)

    sim_result = engine.compute(pred_df, feature_cols=valid_cols)
    engine.save_scores(sim_result, out_dir)

    present_sensitive = [a for a in sensitive_attrs if a in pred_df.columns]
    per_group = engine.per_group_consistency(
        pred_df, valid_cols, group_cols=present_sensitive, k=min(k_values)
    )
    (out_dir / "per_group_consistency.json").write_text(json.dumps(per_group, indent=2))

    mapper = ViolationDensityMapper(k=min(k_values))
    map_result = mapper.compute(
        pred_df,
        feature_cols=valid_cols,
        pred_col=pred_col,
        output_file=out_dir / "violation_density_map.png",
        similarity_engine=engine,
    )

    overall = (
        sum(r.mean_consistency for r in sim_result.rows) / len(sim_result.rows)
        if sim_result.rows
        else 0.0
    )
    return {
        "overall_mean_consistency": overall,
        "k_rows": len(sim_result.rows),
        "per_group": per_group,
        "density_map": str(map_result.output_file) if map_result.output_file else None,
    }


def run_similarity(
    run_root: Path,
    dataset: str,
    sensitive_attrs: List[str],
    k_values: List[int],
    out_base: Path,
    feature_exclude: Optional[List[str]] = None,
    pred_col: str = "y_pred",
) -> Optional[Dict]:
    """Run similarity analysis for **all** models of *dataset* under *run_root*.

    Writes ``out_base/<dataset>/<model>/`` and returns a per-model summary, or
    ``None`` when no compatible predictions are found.
    """
    models = load_all_model_predictions(run_root, dataset)
    if not models:
        logger.info("[INFO] similarity: no predictions for %s under %s", dataset, run_root)
        return None

    feature_exclude = feature_exclude if feature_exclude is not None else DEFAULT_FEATURE_EXCLUDE
    summaries: Dict[str, Dict] = {}
    for model, pred_df in models.items():
        feature_cols = resolve_feature_cols(pred_df, exclude=feature_exclude)
        summary = run_similarity_for_predictions(
            pred_df,
            feature_cols=feature_cols,
            sensitive_attrs=sensitive_attrs,
            k_values=k_values,
            out_dir=out_base / dataset / model,
            pred_col=pred_col,
        )
        if summary is not None:
            summaries[model] = summary
            logger.info(
                "[SUCCESS] similarity model=%s mean_consistency=%.4f",
                model,
                summary["overall_mean_consistency"],
            )

    return summaries or None
