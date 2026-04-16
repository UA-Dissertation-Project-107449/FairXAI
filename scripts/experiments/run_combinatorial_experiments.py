"""Combinatorial experiment orchestration for fairness mitigation analysis."""

import argparse
import copy
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
# Ensure local experiment helpers (e.g., _gates.py) are importable from wrapper entrypoints.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _gates import evaluate_fairness_gate, evaluate_recall_gate, load_gate_thresholds

from fairxai.cli.runner_base import get_project_root, setup_phase_logging
from fairxai.cli.runner_utils import (
    append_run_history,
    get_run_root,
    resolve_run_id,
    update_latest_pointer,
)
from fairxai.data.schemas import available_sensitive, preferred_sensitive
from fairxai.experiments.data_io import (
    build_schema_excludes,
    default_exclude_columns,
)
from fairxai.experiments.data_io import load_schema_config as load_schema_config_shared
from fairxai.experiments.versioning import ExperimentVersioning
from fairxai.explainability.tabular import (
    adaptive_shap_sample_cap,
    build_lime_explainer,
    lime_explain_instance,
    shap_explain_tabular,
)
from fairxai.fairness.metrics import FairnessMetrics
from fairxai.fairness.mitigation import MitigationEngine
from fairxai.models import get_model_class
from fairxai.models.baseline import BaselineLogisticRegression
from fairxai.models.cv_trainer import CVTrainer
from fairxai.utils.config import load_yaml_config
from fairxai.utils.gpu import detect_accelerator

logger = logging.getLogger(__name__)

STAGE_MAP = {
    "reweighting": "pre-processing",
    "smote": "pre-processing",
    "ros": "pre-processing",
    "rus": "pre-processing",
    "adasyn": "pre-processing",
    "exponentiated_gradient": "in-processing",
    "grid_search": "in-processing",
    "threshold_optimizer": "post-processing",
}

# Mitigation engine currently assumes logistic baseline for pre/in/post mitigation.
# This set is now config-driven (mitigation_supported_model_types in combinatorial.yaml).
# The module-level default is kept as a fallback when the config key is absent.
_DEFAULT_MITIGATION_SUPPORTED_MODEL_TYPES = {"logistic_regression"}

# Cache repeated dataset/binning CSV loads across experiment combinations.
_PROCESSED_DATA_CACHE: Dict[tuple, Dict[str, pd.DataFrame]] = {}

SCORE_WEIGHTS = {
    "f1": 0.40,
    "recall": 0.30,
    "accuracy": 0.20,
    "auc": 0.10,
}


def _extract_fairness_gap(fairness_metrics: Dict[str, Any]) -> Optional[float]:
    """Extract a scalar fairness gap from a calculate_all_metrics() result.

    Returns the maximum demographic-parity ``max_difference`` across all
    sensitive attributes, or ``None`` if the metric is unavailable.
    """
    group_fairness = fairness_metrics.get("group_fairness", {})
    gaps = []
    for attr_metrics in group_fairness.values():
        dp = attr_metrics.get("demographic_parity", {})
        val = dp.get("max_difference")
        if val is not None:
            gaps.append(float(val))
    return max(gaps) if gaps else None


def _annotate_gate_fields(result: Dict[str, Any], thresholds: Dict[str, float]) -> None:
    """Evaluate and inject gate fields into an experiment result dict in-place.

    Fields added:
    - ``gate_recall_passed`` (bool)
    - ``gate_recall_tier`` ('full_pass' | 'lower_tier' | 'fail')
    - ``gate_recall_reason`` (str)
    - ``gate_fairness_passed`` (bool)
    - ``gate_fairness_reason`` (str)
    - ``fairness_gap`` (float | None)
    """
    if result.get("execution", {}).get("status") != "success":
        result.update(
            {
                "gate_recall_passed": False,
                "gate_recall_tier": "fail",
                "gate_recall_reason": "experiment failed",
                "gate_fairness_passed": False,
                "gate_fairness_reason": "experiment failed",
                "fairness_gap": None,
            }
        )
        return

    if result.get("training_method") == "kfold_cv":
        recall = result.get("cv_results", {}).get("recall", {}).get("mean")
    else:
        recall = result.get("test_metrics", {}).get("recall")

    recall_gate = evaluate_recall_gate(
        recall,
        thresholds["recall_hard_floor"],
        thresholds["min_recall"],
    )
    result["gate_recall_passed"] = recall_gate.passed
    result["gate_recall_tier"] = recall_gate.tier
    result["gate_recall_reason"] = recall_gate.reason

    fairness_gap = _extract_fairness_gap(result.get("fairness_metrics", {}))
    fairness_gate = evaluate_fairness_gate(fairness_gap, thresholds["max_fairness_violation"])
    result["gate_fairness_passed"] = fairness_gate.passed
    result["gate_fairness_reason"] = fairness_gate.reason
    result["fairness_gap"] = fairness_gap


def _resolve_xgb_device(config: Dict[str, Any]) -> str:
    """Resolve XGBoost compute device from config accelerator setting.

    Returns 'cuda' when an NVIDIA GPU is detected (or explicitly requested),
    otherwise 'cpu'. AMD/ROCm is not supported and falls back to 'cpu'.
    """
    return detect_accelerator(config.get("accelerator", "auto"))


def _resolve_model_n_jobs(outer_n_jobs: int) -> int:
    """Return the n_jobs value to pass to individual model constructors.

    When the outer experiment loop already runs multiple workers in parallel,
    each model should use a single thread to avoid CPU/RAM over-subscription.
    Only allow the model to use all cores when experiments run sequentially.

    Args:
        outer_n_jobs: Number of parallel experiment workers (from config/CLI).

    Returns:
        ``1`` when ``outer_n_jobs > 1``, ``-1`` (all cores) otherwise.
    """
    return 1 if outer_n_jobs > 1 else -1


def _coerce_probability_vector(values: Any) -> np.ndarray:
    """Convert probability outputs into a single positive-class score vector."""
    arr = np.asarray(values)
    if arr.ndim == 0:
        return np.asarray([float(arr)])
    if arr.ndim == 1:
        return arr.reshape(-1)
    if arr.ndim == 2:
        if arr.shape[1] == 0:
            return np.zeros(arr.shape[0], dtype=float)
        if arr.shape[1] == 1:
            return arr[:, 0]
        return arr[:, 1]
    return arr.reshape(arr.shape[0], -1)[:, 0]


def _coerce_label_vector(values: Any) -> np.ndarray:
    """Convert prediction outputs into a 1D label vector."""
    arr = np.asarray(values)
    if arr.ndim <= 1:
        return arr.reshape(-1)
    return np.argmax(arr, axis=1)


def _is_shap_enabled_for_model(model_type: str, xai_cfg: Optional[Dict[str, Any]]) -> bool:
    cfg = xai_cfg or {}
    skip_models = {str(m).strip().lower() for m in cfg.get("skip_shap_model_types", ["svm"])}
    return (model_type or "").strip().lower() not in skip_models


def _resolve_xai_mode(xai_cfg: Optional[Dict[str, Any]]) -> str:
    cfg = xai_cfg or {}
    mode = str(cfg.get("mode", "all")).strip().lower()
    if mode not in {"all", "top_configs", "disabled"}:
        return "all"
    return mode


def _is_xai_enabled_for_phase(xai_cfg: Optional[Dict[str, Any]]) -> bool:
    cfg = xai_cfg or {}
    if not cfg.get("enabled", True):
        return False
    return _resolve_xai_mode(cfg) == "all"


def _result_score(result: Dict[str, Any]) -> Optional[float]:
    """Compute composite score compatible with comparison phase ranking.

    Returns ``None`` (hard exclusion) for failed experiments or configs that
    did not pass the recall hard floor gate (``gate_recall_tier == 'fail'``).
    """
    if result.get("execution", {}).get("status") != "success":
        return None

    # Hard gate: drop configs that fell below recall_hard_floor.
    # gate_recall_tier is set by _annotate_gate_fields(); if absent (legacy
    # results without annotation), fall through to the score computation.
    if result.get("gate_recall_tier") == "fail":
        return None

    if result.get("training_method") == "kfold_cv":
        cv = result.get("cv_results", {})
        f1 = cv.get("f1_score", {}).get("mean")
        recall = cv.get("recall", {}).get("mean")
        accuracy = cv.get("accuracy", {}).get("mean")
        auc = cv.get("auc_roc", {}).get("mean")
    else:
        tm = result.get("test_metrics", {})
        f1 = tm.get("f1_score")
        recall = tm.get("recall")
        accuracy = tm.get("accuracy")
        auc = tm.get("auc_roc")

    values = [f1, recall, accuracy, auc]
    if any(v is None for v in values):
        return None

    return (
        SCORE_WEIGHTS["f1"] * float(f1)
        + SCORE_WEIGHTS["recall"] * float(recall)
        + SCORE_WEIGHTS["accuracy"] * float(accuracy)
        + SCORE_WEIGHTS["auc"] * float(auc)
    )


def _select_top_experiments_for_xai(
    results: List[Dict[str, Any]],
    top_k: int,
    per_dataset: bool,
) -> List[Tuple[Dict[str, Any], float]]:
    """Select top-k experiments for deferred XAI, using fairness-first ordering.

    Ranking tiers (applied per-dataset when ``per_dataset=True``):
    1. Tier 1 (strict): recall full_pass AND fairness gate passed; sorted by score desc
    2. Tier 2 (strict): recall lower_tier AND fairness gate passed; sorted by score desc
    3. Fallback (when strict set is empty): all scored configs; sorted by
       fairness_gap asc then score desc; tagged ``selection_mode='fallback'``

    Experiments that failed the recall hard floor gate (``gate_recall_tier='fail'``)
    are already excluded by ``_result_score()`` returning ``None``.
    """
    scored: List[Tuple[Dict[str, Any], float]] = []
    for result in results:
        score = _result_score(result)
        if score is not None:
            scored.append((result, score))

    if not scored:
        return []

    def _tier_rank(
        items: List[Tuple[Dict[str, Any], float]], top_k: int
    ) -> List[Tuple[Dict[str, Any], float]]:
        tier1 = [
            (r, s)
            for r, s in items
            if r.get("gate_recall_tier") == "full_pass" and r.get("gate_fairness_passed")
        ]
        tier2 = [
            (r, s)
            for r, s in items
            if r.get("gate_recall_tier") == "lower_tier" and r.get("gate_fairness_passed")
        ]
        strict = sorted(tier1, key=lambda x: x[1], reverse=True) + sorted(
            tier2, key=lambda x: x[1], reverse=True
        )
        if strict:
            ranked = strict[:top_k]
            for r, _ in ranked:
                r["selection_mode"] = "strict"
            return ranked
        # Fallback: rank by smallest fairness_gap, then score desc
        fallback = sorted(
            items,
            key=lambda x: (x[0].get("fairness_gap") or 1.0, -x[1]),
        )[:top_k]
        for r, _ in fallback:
            r["selection_mode"] = "fallback"
        return fallback

    if not per_dataset:
        return _tier_rank(scored, top_k)

    selected: List[Tuple[Dict[str, Any], float]] = []
    datasets = sorted({item[0].get("configuration", {}).get("dataset", "") for item in scored})
    for dataset in datasets:
        dataset_scored = [
            item for item in scored if item[0].get("configuration", {}).get("dataset") == dataset
        ]
        selected.extend(_tier_rank(dataset_scored, top_k))
    return selected


def _load_model_config(project_root: Path, model_type: str) -> Dict[str, Any]:
    """Load base hyperparameters from configs/models/<model_type>.yaml."""
    path = project_root / "configs" / "models" / f"{model_type}.yaml"
    cfg = load_yaml_config(str(path))
    return dict(cfg.get("hyperparameters", {}))


def _resolve_model_variants(
    config: Dict[str, Any],
    model_type: str,
    project_root: Path,
    xgb_device: Optional[str] = None,
    outer_n_jobs: int = 1,
    dataset: Optional[str] = None,
    hpo_dir: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Resolve model variants: base params from model file, overrides from config."""
    from fairxai.training.grid_search import load_hpo_params

    base_params = _load_model_config(project_root, model_type)
    if model_type == "xgboost" and xgb_device is not None:
        base_params["device"] = xgb_device
    if model_type == "random_forest" and xgb_device == "cuda":
        # Enable RAPIDS cuML GPU backend when a CUDA device is available.
        base_params["use_gpu"] = True
    # Prevent CPU/RAM over-subscription: when outer experiment workers > 1,
    # each model must use a single thread (not all cores).
    if model_type in {"random_forest", "xgboost"} and "n_jobs" in base_params:
        base_params["n_jobs"] = _resolve_model_n_jobs(outer_n_jobs)
    # Merge HPO best params when available (override defaults, keep GPU/n_jobs overrides after).
    if hpo_dir is not None and dataset is not None:
        hpo_best = load_hpo_params(hpo_dir, dataset, model_type)
        if hpo_best:
            logger.debug(f"[HPO] Loaded params for {model_type}/{dataset}: {hpo_best}")
            base_params.update(hpo_best)
            # Re-apply hardware overrides that must not be clobbered by HPO.
            if model_type == "xgboost" and xgb_device is not None:
                base_params["device"] = xgb_device
            if model_type in {"random_forest", "xgboost"} and "n_jobs" in base_params:
                base_params["n_jobs"] = _resolve_model_n_jobs(outer_n_jobs)

    variants = config.get("model_variants", {}).get(model_type, [])
    if not variants:
        return [{"name": "default", "params": base_params}]

    resolved = []
    for variant in variants:
        variant_name = str(variant.get("name", "variant")).strip() or "variant"
        merged_params = dict(base_params)
        merged_params.update(variant.get("params", {}))
        resolved.append({"name": variant_name, "params": merged_params})
    return resolved


def load_processed_data(
    dataset_name: str, binning_strategy: str, processed_dir: Path
) -> Dict[str, pd.DataFrame]:
    """
    Load preprocessed data for specific dataset and binning strategy.

    Args:
        dataset_name: Dataset name (cleveland, kaggle_heart)
        binning_strategy: Binning strategy name
        processed_dir: Path to processed data directory

    Returns:
        Dictionary with train/test data
    """
    cache_key = (str(processed_dir.resolve()), dataset_name, binning_strategy)
    if cache_key in _PROCESSED_DATA_CACHE:
        cached = _PROCESSED_DATA_CACHE[cache_key]
        return {
            "train_df": cached["train_df"].copy(),
            "test_df": cached["test_df"].copy(),
        }

    data_dir = processed_dir / f"{dataset_name}_{binning_strategy}"

    if not data_dir.exists():
        raise FileNotFoundError(
            f"Processed data not found: {data_dir}\n"
            f"Run preprocessing with --binning-strategy {binning_strategy}"
        )

    # Load scaled data with sensitive attributes
    train_df = pd.read_csv(data_dir / f"{dataset_name}_train_scaled.csv")
    test_df = pd.read_csv(data_dir / f"{dataset_name}_test_scaled.csv")

    payload = {"train_df": train_df, "test_df": test_df}
    _PROCESSED_DATA_CACHE[cache_key] = payload
    return {
        "train_df": payload["train_df"].copy(),
        "test_df": payload["test_df"].copy(),
    }


def load_schema_config(project_root: Path, pipeline: str) -> Dict[str, Any]:
    return load_schema_config_shared(project_root, pipeline)


def schema_excludes(schema_cfg: Dict[str, Any], dataset_name: str) -> List[str]:
    return build_schema_excludes(schema_cfg, dataset_name)


def prepare_data_splits(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    exclude_cols: List[str],
    sensitive_attrs: List[str],
    target_col: str,
) -> Dict[str, Any]:
    """
    Prepare train/test splits with feature/target separation.

    Args:
        train_df: Training dataframe
        test_df: Test dataframe
        exclude_cols: Columns to exclude from features

    Returns:
        Dictionary with X, y, sensitive attributes for train and test
    """
    # Filter exclude columns that actually exist
    exclude_cols = [col for col in exclude_cols if col in train_df.columns]

    # Separate features, target, and sensitive/group attributes
    X_train = train_df.drop(columns=exclude_cols)
    y_train = train_df[target_col]
    X_test = test_df.drop(columns=exclude_cols)
    y_test = test_df[target_col]

    sens_cols_train = available_sensitive(train_df, sensitive_attrs)
    sens_cols_test = available_sensitive(test_df, sensitive_attrs)

    sensitive_train = (
        train_df[sens_cols_train].copy() if sens_cols_train else pd.DataFrame(index=train_df.index)
    )
    sensitive_test = (
        test_df[sens_cols_test].copy() if sens_cols_test else pd.DataFrame(index=test_df.index)
    )

    return {
        "X_train": X_train,
        "y_train": y_train,
        "sensitive_train": sensitive_train,
        "X_test": X_test,
        "y_test": y_test,
        "sensitive_test": sensitive_test,
        "sensitive_cols": list(dict.fromkeys(sens_cols_train + sens_cols_test)),
    }


def _unwrap_for_xai(raw_model: Any) -> Optional[Any]:
    """Extract a sklearn-compatible estimator from fairlearn/wrapper models.

    ``_run_xai_for_fold`` only does ``getattr(model, 'model', model)`` which
    is insufficient for fairlearn in-processing models.  This helper digs
    into ``predictors_`` / ``.model`` to find a real sklearn estimator that
    exposes ``predict_proba`` (or at least ``predict``).

    Returns:
        A sklearn-like estimator, or *None* if nothing usable is found.
    """
    candidate = raw_model
    # Fairlearn in-processing (ExponentiatedGradient, GridSearch)
    # Prefer fitted predictors_ first; estimator_/estimator can be unfitted
    # base templates in some fairlearn objects.
    if hasattr(candidate, "predictors_"):
        predictor = next(
            (p for p in candidate.predictors_ if hasattr(p, "predict_proba")),
            next(
                (p for p in candidate.predictors_ if hasattr(p, "predict")),
                None,
            ),
        )
        if predictor is not None:
            candidate = predictor
    # Fairlearn post-processing wrappers (e.g. ThresholdOptimizer)
    elif hasattr(candidate, "estimator_"):
        candidate = candidate.estimator_
    elif hasattr(candidate, "estimator"):
        candidate = candidate.estimator
    # Custom wrappers (e.g. BaselineLogisticRegression)
    if hasattr(candidate, "model"):
        candidate = candidate.model
    # Final check: usable by SHAP (callable or sklearn) and LIME (predict_proba)
    if (
        hasattr(candidate, "predict_proba")
        or hasattr(candidate, "predict")
        or hasattr(candidate, "decision_function")
    ):
        return candidate
    return None


def save_experiment_xai(
    exp_id: str,
    model: Any,
    X_ref: pd.DataFrame,
    X_lime: pd.DataFrame,
    versioning: ExperimentVersioning,
    dataset_name: str,
    model_type: str,
    base_model: Optional[Any] = None,
    xai_cfg: Optional[Dict[str, Any]] = None,
) -> None:
    if xai_cfg is None:
        xai_cfg = {}
    if not xai_cfg.get("enabled", True):
        return

    dataset_xai_dir = versioning.latest_dir / "xai" / dataset_name / "holdout"
    shap_dir = dataset_xai_dir / "shap"
    lime_dir = dataset_xai_dir / "lime"
    shap_dir.mkdir(parents=True, exist_ok=True)
    lime_dir.mkdir(parents=True, exist_ok=True)
    max_samples = adaptive_shap_sample_cap(
        n_rows=len(X_ref),
        base_cap=int(xai_cfg.get("max_samples", 200)),
        large_cap=int(xai_cfg.get("large_dataset_max_samples", 80)),
    )
    lime_instances = int(xai_cfg.get("lime_instances", 2))
    allow_svm_shap = bool(xai_cfg.get("allow_svm_shap", False))
    shap_enabled = _is_shap_enabled_for_model(model_type, xai_cfg)

    def _mean_abs_shap_values(shap_values: Any) -> np.ndarray:
        def _reduce(arr: np.ndarray) -> np.ndarray:
            if arr.ndim == 3:
                return np.mean(np.abs(arr), axis=(0, 2))
            if arr.ndim == 2:
                return np.mean(np.abs(arr), axis=0)
            if arr.ndim == 1:
                return np.abs(arr)
            return np.mean(np.abs(arr), axis=0)

        if isinstance(shap_values, list):
            values = [_reduce(np.asarray(v)) for v in shap_values]
            return np.mean(values, axis=0)

        return _reduce(np.asarray(shap_values))

    def _resolve_shap_model(raw_model: Any) -> Any:
        candidate = _unwrap_for_xai(raw_model) or raw_model
        if hasattr(candidate, "predictors_"):
            predictor = next(
                (
                    p
                    for p in candidate.predictors_
                    if hasattr(p, "predict_proba") or hasattr(p, "predict")
                ),
                None,
            )
            if predictor is not None:
                candidate = predictor
        if hasattr(candidate, "model"):
            candidate = candidate.model

        if hasattr(candidate, "predict_proba"):
            return lambda X: candidate.predict_proba(X)
        if hasattr(candidate, "decision_function"):
            wrapped = _wrap_decision_function(candidate)
            return lambda X: wrapped.predict_proba(X)
        if hasattr(candidate, "predict"):
            return lambda X: candidate.predict(X)
        return candidate

    def _wrap_decision_function(df_model: Any):
        class _DecisionFunctionWrapper:
            def __init__(self, base_model: Any):
                self.base_model = base_model

            def predict_proba(self, X):
                scores = self.base_model.decision_function(X)
                scores = np.asarray(scores)
                if scores.ndim == 1:
                    prob_pos = 1.0 / (1.0 + np.exp(-scores))
                    return np.vstack([1 - prob_pos, prob_pos]).T
                exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
                return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        return _DecisionFunctionWrapper(df_model)

    def _resolve_lime_model(raw_model: Any) -> Optional[Any]:
        candidate = _unwrap_for_xai(raw_model) or raw_model
        if hasattr(candidate, "predict_proba"):
            return candidate
        if hasattr(candidate, "decision_function"):
            return _wrap_decision_function(candidate)
        if hasattr(candidate, "predictors_"):
            predictor = next(
                (
                    p
                    for p in candidate.predictors_
                    if hasattr(p, "predict_proba") or hasattr(p, "decision_function")
                ),
                None,
            )
            if predictor is not None:
                return predictor
        if hasattr(candidate, "model") and hasattr(candidate.model, "predict_proba"):
            return candidate.model
        if hasattr(candidate, "model") and hasattr(candidate.model, "decision_function"):
            return _wrap_decision_function(candidate.model)
        return None

    xai_model_raw = base_model if base_model is not None else model
    xai_model = _unwrap_for_xai(xai_model_raw) or xai_model_raw
    if shap_enabled:
        try:
            shap_model = _resolve_shap_model(xai_model)
            # Global SHAP (train reference)
            shap_global = shap_explain_tabular(
                shap_model,
                X_ref,
                max_samples=max_samples,
                allow_svm=allow_svm_shap,
            )
            mean_abs_global = _mean_abs_shap_values(shap_global.shap_values)
            shap_global_df = pd.DataFrame(
                {"feature": shap_global.feature_names, "mean_abs_shap": mean_abs_global}
            ).sort_values("mean_abs_shap", ascending=False)
            shap_global_file = shap_dir / f"{exp_id}_global.csv"
            shap_global_df.to_csv(shap_global_file, index=False)

            # Local SHAP (test/reference for local context)
            shap_local = shap_explain_tabular(
                shap_model,
                X_lime,
                max_samples=max_samples,
                allow_svm=allow_svm_shap,
            )
            mean_abs_local = _mean_abs_shap_values(shap_local.shap_values)
            shap_local_df = pd.DataFrame(
                {"feature": shap_local.feature_names, "mean_abs_shap": mean_abs_local}
            ).sort_values("mean_abs_shap", ascending=False)
            shap_local_file = shap_dir / f"{exp_id}_local.csv"
            shap_local_df.to_csv(shap_local_file, index=False)
        except Exception as exc:
            logging.getLogger(__name__).warning(f"SHAP failed for {exp_id}: {exc}")
    else:
        logging.getLogger(__name__).info(
            f"SHAP skipped for exp_id={exp_id} model_type={model_type}"
        )

    # LIME examples (requires predict_proba)
    try:
        lime_model = _resolve_lime_model(xai_model)
        if lime_instances > 0 and lime_model is not None:
            lime_rows = X_lime.sample(n=min(lime_instances, len(X_lime)), random_state=42)
            lime_results = []
            # Build explainer once and reuse across all LIME instances.
            lime_expl = build_lime_explainer(
                X_ref,
                feature_names=list(X_ref.columns),
                class_names=["no_disease", "disease"],
            )
            for idx, row in lime_rows.iterrows():
                exp = lime_explain_instance(
                    model=lime_model,
                    data_row=row,
                    training_data=X_ref,
                    feature_names=list(X_ref.columns),
                    class_names=["no_disease", "disease"],
                    num_features=10,
                    explainer=lime_expl,
                )
                for feat, weight in exp.weights:
                    lime_results.append(
                        {
                            "instance_id": int(idx),
                            "feature": feat,
                            "weight": weight,
                            "intercept": exp.intercept,
                            "score": exp.score,
                            "local_pred": exp.local_pred,
                        }
                    )
            lime_df = pd.DataFrame(lime_results)
            lime_file = lime_dir / f"{exp_id}_examples.csv"
            lime_df.to_csv(lime_file, index=False)
        elif lime_instances > 0:
            logging.getLogger(__name__).warning(
                f"LIME skipped for {exp_id}: no predict_proba/decision_function"
            )
    except Exception as exc:
        logging.getLogger(__name__).warning(f"LIME failed for {exp_id}: {exc}")


def save_cv_experiment_xai(
    exp_id: str,
    fold_results: List[Dict[str, Any]],
    versioning: ExperimentVersioning,
    dataset_name: str,
    xai_cfg: Optional[Dict[str, Any]] = None,
) -> None:
    """Aggregate and save cross-validated XAI outputs (SHAP + LIME).

    Uses :meth:`CVTrainer.aggregate_cv_shap` and
    :meth:`CVTrainer.aggregate_cv_lime` to combine per-fold XAI data,
    then writes the results to ``xai/{dataset}/cv/shap/`` and
    ``xai/{dataset}/cv/lime/``.

    Args:
        exp_id: Experiment identifier.
        fold_results: Per-fold result dicts (each must contain an ``xai`` key).
        versioning: Versioning system instance.
        dataset_name: Dataset name for directory structure.
        xai_cfg: XAI configuration dict.
    """
    if xai_cfg is None:
        xai_cfg = {}
    if not xai_cfg.get("enabled", True):
        return

    logger = logging.getLogger(__name__)

    cv_xai_dir = versioning.latest_dir / "xai" / dataset_name / "cv"
    shap_dir = cv_xai_dir / "shap"
    lime_dir = cv_xai_dir / "lime"
    shap_dir.mkdir(parents=True, exist_ok=True)
    lime_dir.mkdir(parents=True, exist_ok=True)

    # Aggregate SHAP across folds (global = train data, local = val data)
    cv_shap_global = CVTrainer.aggregate_cv_shap(fold_results, scope="global")
    if cv_shap_global is not None:
        shap_file = shap_dir / f"{exp_id}_global.csv"
        cv_shap_global.to_csv(shap_file, index=False)
        logger.info(f"  CV SHAP global saved: {shap_file.name}")

    cv_shap_local = CVTrainer.aggregate_cv_shap(fold_results, scope="local")
    if cv_shap_local is not None:
        shap_file = shap_dir / f"{exp_id}_local.csv"
        cv_shap_local.to_csv(shap_file, index=False)
        logger.info(f"  CV SHAP local saved: {shap_file.name}")

    # Aggregate LIME across folds
    cv_lime_df = CVTrainer.aggregate_cv_lime(fold_results)
    if cv_lime_df is not None:
        lime_file = lime_dir / f"{exp_id}_tracked.csv"
        cv_lime_df.to_csv(lime_file, index=False)
        logger.info(f"  CV LIME tracked saved: {lime_file.name}")


def aggregate_dataset_shap(xai_dir: Path, suffix: str) -> None:
    files = [
        p for p in xai_dir.glob(f"*_{suffix}.csv") if not p.name.endswith(f"{suffix}_summary.csv")
    ]
    if not files:
        return

    rows = []
    for file_path in files:
        df = pd.read_csv(file_path)
        if "feature" not in df.columns or "mean_abs_shap" not in df.columns:
            continue
        df = df[["feature", "mean_abs_shap"]].copy()
        df["source_file"] = file_path.name
        rows.append(df)

    if not rows:
        return

    combined = pd.concat(rows, ignore_index=True)
    grouped = combined.groupby("feature")["mean_abs_shap"]
    summary = grouped.agg(
        count="count",
        mean="mean",
        std="std",
        min="min",
        max="max",
    ).reset_index()
    summary["p25"] = grouped.quantile(0.25).values
    summary["p50"] = grouped.quantile(0.50).values
    summary["p75"] = grouped.quantile(0.75).values
    summary = summary.sort_values("mean", ascending=False)

    out_file = xai_dir / f"{suffix}_summary.csv"
    summary.to_csv(out_file, index=False)


def run_single_experiment(
    exp_id: str,
    config: Dict[str, Any],
    versioning: ExperimentVersioning,
    processed_dir: Path,
    schema_cfg: Dict[str, Any],
    logger: logging.Logger,
    target_col: str,
) -> Dict[str, Any]:
    """
    Run a single experiment with given configuration.

    Args:
        exp_id: Experiment ID
        config: Experiment configuration
        versioning: Versioning system instance
        processed_dir: Path to processed data
        logger: Logger instance

    Returns:
        Dictionary with experiment results
    """
    start_time = datetime.now()

    try:
        logger.info(
            f"[EXPERIMENT] id={exp_id} dataset={config['dataset']} "
            f"binning={config['binning_strategy']} mitigation={config['mitigation_technique']} "
            f"training={config['training_method']} "
            f"model={config.get('model_type', 'logistic_regression')}[{config.get('model_variant', 'default')}]"
        )

        # Load data
        data = load_processed_data(config["dataset"], config["binning_strategy"], processed_dir)

        # Prepare splits
        exclude_cols = default_exclude_columns(
            schema_cfg,
            config["dataset"],
            target=target_col,
            sensitive_attrs=config.get("sensitive_attributes", []),
        )
        splits = prepare_data_splits(
            data["train_df"],
            data["test_df"],
            exclude_cols,
            config["sensitive_attributes"],
            target_col,
        )

        if config["mitigation_technique"] != "baseline" and not splits["sensitive_cols"]:
            raise ValueError(
                "No sensitive/group columns available for mitigation; check preprocessing and config"
            )

        logger.info(
            f"[DATA] train_rows={len(splits['X_train'])} test_rows={len(splits['X_test'])} "
            f"n_features={splits['X_train'].shape[1]}"
        )

        # Train model based on training method
        if config["training_method"] == "kfold_cv":
            results = run_cv_experiment(exp_id, config, splits, versioning, logger)
        else:
            results = run_single_split_experiment(exp_id, config, splits, versioning, logger)

        # Add execution metadata
        duration = (datetime.now() - start_time).total_seconds()
        results["execution"] = {
            "duration_seconds": duration,
            "timestamp": datetime.now().isoformat(),
            "status": "success",
        }

        logger.info(
            f"[SUCCESS] Experiment complete: exp_id={exp_id} duration_seconds={duration:.1f}"
        )
        return results

    except Exception as e:
        logger.error(f"Experiment {exp_id} failed: {str(e)}")
        duration = (datetime.now() - start_time).total_seconds()

        return {
            "experiment_id": exp_id,
            "configuration": config,
            "execution": {
                "duration_seconds": duration,
                "timestamp": datetime.now().isoformat(),
                "status": "failed",
                "error": str(e),
            },
            "results": None,
        }


def run_single_split_experiment(
    exp_id: str,
    config: Dict[str, Any],
    splits: Dict[str, Any],
    versioning: ExperimentVersioning,
    logger: logging.Logger,
) -> Dict[str, Any]:
    """
    Run experiment with single train/test split.
    """
    logger.info("[TRAINING] method=single_split")
    xai_cfg = config.get("xai", {})
    xai_enabled = _is_xai_enabled_for_phase(xai_cfg)

    # Initialize mitigation engine
    engine = MitigationEngine()

    # Apply mitigation technique
    mitigation = config["mitigation_technique"]
    stage = STAGE_MAP.get(mitigation)
    if mitigation == "baseline":
        stage = "baseline"
    base_model = None
    sensitive_attr = next(
        (
            c
            for c in config["sensitive_attributes"]
            if c in splits["sensitive_train"].columns and c != "age_group"
        ),
        next((c for c in splits["sensitive_train"].columns), None),
    )

    if mitigation != "baseline" and sensitive_attr is None:
        raise ValueError(
            "Mitigation requires at least one sensitive/group column; none found in splits"
        )

    combo_chain = config.get("mitigation_combo")  # set for combo experiments; None otherwise

    if mitigation == "baseline":
        # Train baseline model
        model_type = config.get("model_type", "logistic_regression")
        model_class = get_model_class(model_type)
        model = model_class(**config.get("model_params", {}))
        model.train(splits["X_train"], splits["y_train"])
        test_metrics = model.evaluate(splits["X_test"], splits["y_test"])

        # Get predictions
        y_pred = _coerce_label_vector(model.predict(splits["X_test"]))
        y_proba = _coerce_probability_vector(model.predict_proba(splits["X_test"]))

        result = {
            "test_metrics": test_metrics,
            "predictions": {"y_pred": y_pred, "y_proba": y_proba},
        }
    elif combo_chain:
        # Sequential combo: pre to in to post chain
        fairness_base_params = config.get("fairness_base_model_params")
        result = engine.apply_combo(
            techniques=combo_chain,
            X_train=splits["X_train"],
            y_train=splits["y_train"],
            X_test=splits["X_test"],
            y_test=splits["y_test"],
            sensitive_train=splits["sensitive_train"],
            sensitive_test=splits["sensitive_test"],
            sensitive_attr=sensitive_attr,
            base_model_params=fairness_base_params,
        )
    else:
        if stage is None:
            raise ValueError(f"Unknown mitigation technique: {mitigation}")

        if stage == "post-processing":
            base_model = BaselineLogisticRegression(**config.get("model_params", {}))
            base_model.train(splits["X_train"], splits["y_train"])

        # Apply single mitigation technique
        fairness_base_params = config.get("fairness_base_model_params")
        result = engine.apply_technique(
            technique_name=mitigation,
            stage=stage,
            X_train=splits["X_train"],
            y_train=splits["y_train"],
            X_test=splits["X_test"],
            y_test=splits["y_test"],
            sensitive_train=splits["sensitive_train"],
            sensitive_test=splits["sensitive_test"],
            sensitive_attr=sensitive_attr,
            base_model=base_model,
            base_model_params=fairness_base_params,
        )

    # Calculate fairness metrics
    y_pred_series = _coerce_label_vector(result["predictions"]["y_pred"])
    _raw_proba = result["predictions"]["y_proba"]
    if _raw_proba is None:
        y_proba_series = np.full(len(splits["y_test"]), np.nan)
    else:
        y_proba_series = _coerce_probability_vector(_raw_proba)
    predictions_df = pd.DataFrame(
        {
            "y_true": splits["y_test"].values,
            "y_pred": y_pred_series,
            "y_proba": y_proba_series,
        }
    )

    for col in splits["sensitive_cols"]:
        predictions_df[col] = splits["sensitive_test"][col].values

    # Add features for individual fairness
    for col in splits["X_test"].columns:
        predictions_df[col] = splits["X_test"][col].values

    fairness_calc = FairnessMetrics(
        available_sensitive(predictions_df, config["sensitive_attributes"])
    )
    fairness_results = fairness_calc.calculate_all_metrics(
        predictions_df, feature_cols=list(splits["X_test"].columns)
    )

    # Save predictions
    versioning.save_predictions(
        exp_id, predictions_df, dataset=config["dataset"], split_method="holdout"
    )

    # XAI outputs (single-split only)
    model_for_xai = result.get("model") if isinstance(result, dict) else None
    if mitigation == "baseline":
        model_for_xai = model
    if stage == "post-processing" and base_model is not None:
        model_for_xai = base_model
    if model_for_xai is not None and xai_enabled:
        model_for_xai = _unwrap_for_xai(model_for_xai) or model_for_xai
        save_experiment_xai(
            exp_id,
            model_for_xai,
            splits["X_train"],
            splits["X_test"],
            versioning,
            config["dataset"],
            model_type=config.get("model_type", "logistic_regression"),
            base_model=base_model,
            xai_cfg=xai_cfg,
        )

    # Save model to temp dir (comparison script will promote top-N and prune _temp/)
    if model_for_xai is not None:
        versioning.save_temp_model(exp_id, model_for_xai)

    logger.info(f"  Accuracy: {result['test_metrics']['accuracy']:.3f}")
    logger.info(f"  Recall: {result['test_metrics']['recall']:.3f}")
    logger.info(f"  F1: {result['test_metrics']['f1_score']:.3f}")

    return {
        "experiment_id": exp_id,
        "configuration": config,
        "test_metrics": result["test_metrics"],
        "fairness_metrics": fairness_results,
        "training_method": "single_split",
        "n_folds": 1,
    }


def run_cv_experiment(
    exp_id: str,
    config: Dict[str, Any],
    splits: Dict[str, Any],
    versioning: ExperimentVersioning,
    logger: logging.Logger,
) -> Dict[str, Any]:
    """
    Run experiment with k-fold cross-validation.
    """
    n_folds = config.get("cv_folds", 5)
    logger.info(f"[TRAINING] method=kfold_cv folds={n_folds}")

    # Combine train and test for CV (we'll use full dataset for CV)
    X_full = pd.concat([splits["X_train"], splits["X_test"]], ignore_index=True)
    y_full = pd.concat([splits["y_train"], splits["y_test"]], ignore_index=True)
    sensitive_full = pd.concat(
        [splits["sensitive_train"], splits["sensitive_test"]], ignore_index=True
    )

    # Initialize CV trainer
    cv_trainer = CVTrainer(n_folds=n_folds, random_state=config.get("random_seed", 42))

    mitigation = config.get("mitigation_technique", "baseline")
    stage = STAGE_MAP.get(mitigation)

    def _aggregate_fold_metrics(fold_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        metrics_names = ["accuracy", "precision", "recall", "f1_score", "auc_roc"]
        aggregated = {}
        for metric_name in metrics_names:
            values = [fold["val_metrics"][metric_name] for fold in fold_results]
            aggregated[metric_name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "folds": values,
            }
        return aggregated

    # XAI setup for CV
    xai_cfg = config.get("xai", {})
    xai_enabled = _is_xai_enabled_for_phase(xai_cfg)
    model_type = config.get("model_type", "logistic_regression")
    shap_enabled = _is_shap_enabled_for_model(model_type, xai_cfg)
    allow_svm_shap = bool(xai_cfg.get("allow_svm_shap", False))
    tracked_indices = None
    feature_names = list(X_full.columns)
    shap_max_samples = adaptive_shap_sample_cap(
        n_rows=len(X_full),
        base_cap=int(xai_cfg.get("max_samples", 200)),
        large_cap=int(xai_cfg.get("large_dataset_max_samples", 80)),
    )
    if xai_enabled:
        lime_n = int(xai_cfg.get("lime_instances", 2))
        rng = np.random.RandomState(config.get("random_seed", 42))
        tracked_indices = rng.choice(
            len(X_full), size=min(lime_n, len(X_full)), replace=False
        ).tolist()

    if mitigation == "baseline":
        model_type = config.get("model_type", "logistic_regression")
        model_class = get_model_class(model_type)
        # Run CV experiment (baseline only)
        cv_results = cv_trainer.run_cv_experiment(
            model_class=model_class,
            X=X_full,
            y=y_full,
            sensitive_attrs=sensitive_full,
            model_params=config.get("model_params", {}),
            xai_enabled=xai_enabled,
            shap_enabled=shap_enabled,
            allow_svm_shap=allow_svm_shap,
            tracked_indices=tracked_indices,
            feature_names=feature_names,
            shap_max_samples=shap_max_samples,
        )

        # Get fold predictions for fairness calculation
        model = model_class(**config.get("model_params", {}))
        fold_predictions = cv_trainer.get_fold_predictions(model, X_full, y_full, sensitive_full)
    else:
        combo_chain = config.get("mitigation_combo")
        if combo_chain is None and stage is None:
            raise ValueError(f"Unknown mitigation technique for CV: {mitigation}")

        engine = MitigationEngine()
        folds = cv_trainer.create_stratified_folds(X_full, y_full, sensitive_full)
        fold_results = []
        all_predictions = []

        sensitive_attr = next(
            (
                c
                for c in config["sensitive_attributes"]
                if c in sensitive_full.columns and c != "age_group"
            ),
            next((c for c in sensitive_full.columns), None),
        )

        if sensitive_attr is None:
            raise ValueError(
                "Mitigation requires at least one sensitive/group column; none found in CV splits"
            )

        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            logger.info(f"[FOLD] method=mitigation_cv fold={fold_idx + 1}/{n_folds}")

            X_train = X_full.iloc[train_idx]
            y_train = y_full.iloc[train_idx]
            X_val = X_full.iloc[val_idx]
            y_val = y_full.iloc[val_idx]
            sensitive_train = sensitive_full.iloc[train_idx]
            sensitive_val = sensitive_full.iloc[val_idx]

            if combo_chain:
                fairness_base_params = config.get("fairness_base_model_params")
                result = engine.apply_combo(
                    techniques=combo_chain,
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_val,
                    y_test=y_val,
                    sensitive_train=sensitive_train,
                    sensitive_test=sensitive_val,
                    sensitive_attr=sensitive_attr,
                    base_model_params=fairness_base_params,
                )
            else:
                base_model = None
                if stage == "post-processing":
                    base_model = BaselineLogisticRegression(**config.get("model_params", {}))
                    base_model.train(X_train, y_train)

                result = engine.apply_technique(
                    technique_name=mitigation,
                    stage=stage,
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_val,
                    y_test=y_val,
                    sensitive_train=sensitive_train,
                    sensitive_test=sensitive_val,
                    sensitive_attr=sensitive_attr,
                    base_model=base_model,
                )

            fold_result_entry = {
                "fold_idx": fold_idx,
                "train_indices": train_idx,
                "val_indices": val_idx,
                "train_metrics": result.get("train_metrics", None),
                "val_metrics": result["test_metrics"],
            }

            # Per-fold XAI (while fold model is still available)
            if xai_enabled:
                raw_fold_model = None
                if stage == "post-processing" and base_model is not None:
                    raw_fold_model = base_model
                else:
                    raw_fold_model = result.get("model")
                # Unwrap fairlearn / wrapper models to sklearn estimator
                fold_model = _unwrap_for_xai(raw_fold_model) if raw_fold_model is not None else None
                if fold_model is not None:
                    try:
                        fold_xai = cv_trainer._run_xai_for_fold(
                            model=fold_model,
                            X_train=X_train,
                            X_val=X_val,
                            feature_names=feature_names,
                            tracked_indices=tracked_indices,
                            val_indices=val_idx,
                            fold_idx=fold_idx,
                            max_samples=shap_max_samples,
                            shap_enabled=shap_enabled,
                            allow_svm_shap=allow_svm_shap,
                        )
                        fold_result_entry["xai"] = fold_xai
                    except Exception as exc:
                        logger.warning(f"  Fold {fold_idx} XAI failed: {exc}")
                elif raw_fold_model is not None:
                    logger.info(
                        f"[XAI] fold={fold_idx + 1}/{n_folds} skipped=model_type_not_xai_compatible"
                    )

            fold_results.append(fold_result_entry)

            y_pred = _coerce_label_vector(result["predictions"]["y_pred"])
            y_proba_raw = result["predictions"]["y_proba"]
            if y_proba_raw is None:
                y_proba = y_pred.astype(float)
            else:
                y_proba = _coerce_probability_vector(y_proba_raw)

            fold_preds = pd.DataFrame(
                {
                    "fold": fold_idx,
                    "sample_idx": val_idx,
                    "y_true": y_val.values,
                    "y_pred": y_pred,
                    "y_proba": y_proba,
                }
            )

            for col in ["age_group", "sex", "ethnicity", "group_cluster"]:
                if col in sensitive_full.columns:
                    fold_preds[col] = sensitive_full.iloc[val_idx][col].values

            all_predictions.append(fold_preds)

        fold_predictions = pd.concat(all_predictions, ignore_index=True)
        fold_predictions = fold_predictions.sort_values("sample_idx").reset_index(drop=True)

        cv_results = {
            "fold_results": fold_results,
            "aggregated_metrics": _aggregate_fold_metrics(fold_results),
            "n_folds": n_folds,
            "random_state": config.get("random_seed", 42),
        }

    # Save fold predictions (both baseline and mitigated paths)
    versioning.save_predictions(
        exp_id, fold_predictions, dataset=config["dataset"], split_method="cv"
    )

    # CV XAI outputs
    if xai_enabled:
        save_cv_experiment_xai(
            exp_id, cv_results["fold_results"], versioning, config["dataset"], xai_cfg=xai_cfg
        )

    # Calculate fairness metrics on full CV predictions
    predictions_df = fold_predictions.copy()
    for col in X_full.columns:
        predictions_df[col] = X_full[col].values

    fairness_calc = FairnessMetrics(
        available_sensitive(predictions_df, config["sensitive_attributes"])
    )
    fairness_results = fairness_calc.calculate_all_metrics(
        predictions_df, feature_cols=list(X_full.columns)
    )

    agg = cv_results["aggregated_metrics"]
    logger.info(f"  Accuracy: {agg['accuracy']['mean']:.3f} +/- {agg['accuracy']['std']:.3f}")
    logger.info(f"  Recall: {agg['recall']['mean']:.3f} +/- {agg['recall']['std']:.3f}")
    logger.info(f"  F1: {agg['f1_score']['mean']:.3f} +/- {agg['f1_score']['std']:.3f}")

    return {
        "experiment_id": exp_id,
        "configuration": config,
        "cv_results": cv_results["aggregated_metrics"],
        "fold_results": cv_results["fold_results"],
        "fairness_metrics": fairness_results,
        "training_method": "kfold_cv",
        "n_folds": n_folds,
    }


def run_combinatorial_analysis(
    config_path: str,
    pipeline: str = "cardiac",
    n_jobs: int = 1,
    verbose: int = 0,
    archive_previous: bool = True,
    run_id: Optional[str] = None,
    output_root: Optional[str] = None,
    datasets: Optional[list[str]] = None,
    model_types_override: Optional[list[str]] = None,
):
    """Main orchestration for combinatorial experiments."""
    project_root = get_project_root(Path(__file__))
    use_run_id = bool(run_id or os.getenv("RUN_ID") or os.getenv("PREFECT__RUNTIME__FLOW_RUN_ID"))
    run_id = resolve_run_id(run_id) if use_run_id else None
    setup_phase_logging(
        project_root,
        "combinatorial_experiments.log",
        verbose=verbose,
        run_id=run_id,
        stage_name="combinatorial",
    )
    logger = logging.getLogger(__name__)

    logger.info("[PHASE] Combinatorial experiments started")
    logger.info(
        f"[RUN_CONTEXT] pipeline={pipeline} run_id={run_id or 'none'} config_path={config_path}"
    )

    # Load configuration
    config_path = Path(config_path)
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    selected_datasets = [
        str(d).strip() for d in (datasets or config.get("datasets", [])) if str(d).strip()
    ]
    if not selected_datasets:
        logger.error("No datasets selected. Provide --datasets or define datasets in config.")
        sys.exit(1)

    selected_model_types = [
        str(m).strip().lower()
        for m in (model_types_override or config.get("model_types", ["logistic_regression"]))
        if str(m).strip()
    ]
    if not selected_model_types:
        logger.error(
            "No model types selected. Provide --model-types or define model_types in config."
        )
        sys.exit(1)

    # Load gate thresholds: experiment config overrides with thresholds.yaml fallback.
    # Must happen before any experiment result is annotated or ranked.
    gate_thresholds = load_gate_thresholds(config, project_root)
    logger.info(
        f"Gate thresholds: recall_hard_floor={gate_thresholds['recall_hard_floor']:.2f}, "
        f"min_recall={gate_thresholds['min_recall']:.2f}, "
        f"max_fairness_violation={gate_thresholds['max_fairness_violation']:.2f}"
    )

    xgb_device = _resolve_xgb_device(config)
    logger.info(f"XGBoost device: {xgb_device}")

    xai_cfg_global = config.get("xai", {})
    xai_mode = _resolve_xai_mode(xai_cfg_global)
    logger.info(f"XAI execution mode: {xai_mode}")

    if n_jobs is None:
        n_jobs = int(config.get("n_jobs", 1))

    logger.info(f"Loaded configuration from: {config_path}")

    sensitive_attrs = preferred_sensitive(config.get("sensitive_attributes"))

    # Load pipeline config
    pipeline_cfg = load_yaml_config(str(project_root / f"configs/pipelines/{pipeline}.yaml"))
    target_col = pipeline_cfg.get("training", {}).get("target", "heart_disease")

    # Initialize versioning
    base_output_dir = Path(output_root) if output_root else Path(config["paths"]["results_dir"])
    if run_id:
        base_output_dir = (
            Path(output_root) if output_root else (project_root / f"output/{pipeline}")
        )
        run_dir = get_run_root(base_output_dir, run_id) / "experiments"
        run_dir.mkdir(parents=True, exist_ok=True)
        versioning = ExperimentVersioning(base_output_dir, run_dir=run_dir)
    else:
        versioning = ExperimentVersioning(base_output_dir)
        # Archive previous run if requested
        if archive_previous:
            versioning.archive_previous_run()

    if run_id:
        append_run_history(
            base_output_dir,
            {
                "run_id": run_id,
                "pipeline": pipeline,
                "mode": "full",
                "phase": "combinatorial",
                "datasets": selected_datasets,
                "output_dir": str(versioning.latest_dir),
                "status": "started",
            },
        )

    # Generate all experiment combinations
    experiments = []
    # HPO: load best params directory (auto-detected; silently skipped when absent).
    hpo_output_dir = project_root / f"output/{pipeline}/studies/hpo"
    hpo_dir: Optional[Path] = hpo_output_dir if hpo_output_dir.exists() else None
    if hpo_dir:
        logger.info(f"[HPO] Using pre-computed HPO params from: {hpo_dir}")
    fairness_base_params_cfg = config.get("fairness_base_model_params")
    model_types = selected_model_types
    mitigation_supported_model_types = {
        str(m).strip().lower()
        for m in config.get(
            "mitigation_supported_model_types",
            list(_DEFAULT_MITIGATION_SUPPORTED_MODEL_TYPES),
        )
    }
    logger.info(f"Mitigation supported model types: {sorted(mitigation_supported_model_types)}")
    for dataset in selected_datasets:
        if isinstance(fairness_base_params_cfg, dict) and dataset in fairness_base_params_cfg:
            fairness_base_params = fairness_base_params_cfg.get(dataset)
        else:
            fairness_base_params = fairness_base_params_cfg
        if not fairness_base_params:
            fairness_base_params = _load_model_config(project_root, "logistic_regression")
        for binning in config["binning_strategies"]:
            for mitigation in config["mitigation_techniques"]:
                for training_method in config["training_methods"]:
                    for model_type in model_types:
                        # Skip mitigation for models not in the supported set (baseline always runs).
                        if (
                            mitigation != "baseline"
                            and model_type not in mitigation_supported_model_types
                        ):
                            logger.debug(
                                f"Skipping mitigation={mitigation} for model_type={model_type} "
                                f"(not in mitigation_supported_model_types)"
                            )
                            continue

                        for variant in _resolve_model_variants(
                            config,
                            model_type,
                            project_root,
                            xgb_device,
                            outer_n_jobs=n_jobs,
                            dataset=dataset,
                            hpo_dir=hpo_dir,
                        ):
                            exp_id = versioning.generate_experiment_id()
                            exp_config = {
                                "dataset": dataset,
                                "binning_strategy": binning,
                                "mitigation_technique": mitigation,
                                "training_method": training_method,
                                "cv_folds": config.get("cv_folds", 5),
                                "random_seed": config.get("random_seed", 42),
                                "model_type": model_type,
                                "model_variant": variant["name"],
                                "model_params": variant["params"],
                                "fairness_base_model_params": fairness_base_params or None,
                                "sensitive_attributes": sensitive_attrs,
                                "xai": config.get("xai", {}),
                            }

                            experiments.append((exp_id, exp_config))

    # Combo experiments: pre to in to post chains, logistic_regression only.
    for combo in config.get("mitigation_combos", []):
        for dataset in selected_datasets:
            if isinstance(fairness_base_params_cfg, dict) and dataset in fairness_base_params_cfg:
                fairness_base_params = fairness_base_params_cfg.get(dataset)
            else:
                fairness_base_params = fairness_base_params_cfg
            if not fairness_base_params:
                fairness_base_params = _load_model_config(project_root, "logistic_regression")
            for binning in config["binning_strategies"]:
                for training_method in config["training_methods"]:
                    for variant in _resolve_model_variants(
                        config,
                        "logistic_regression",
                        project_root,
                        xgb_device,
                        outer_n_jobs=n_jobs,
                        dataset=dataset,
                        hpo_dir=hpo_dir,
                    ):
                        exp_id = versioning.generate_experiment_id()
                        exp_config = {
                            "dataset": dataset,
                            "binning_strategy": binning,
                            "mitigation_technique": "+".join(combo),
                            "mitigation_combo": combo,
                            "training_method": training_method,
                            "cv_folds": config.get("cv_folds", 5),
                            "random_seed": config.get("random_seed", 42),
                            "model_type": "logistic_regression",
                            "model_variant": variant["name"],
                            "model_params": variant["params"],
                            "fairness_base_model_params": fairness_base_params or None,
                            "sensitive_attributes": sensitive_attrs,
                            "xai": config.get("xai", {}),
                        }
                        experiments.append((exp_id, exp_config))

    total_experiments = len(experiments)
    logger.info(f"[PLAN] Total experiments: {total_experiments}")
    logger.info(f"  Datasets ({len(selected_datasets)}): {selected_datasets}")
    logger.info(f"  Binning strategies: {len(config['binning_strategies'])}")
    logger.info(f"  Mitigation techniques: {len(config['mitigation_techniques'])}")
    logger.info(f"  Training methods: {len(config['training_methods'])}")
    logger.info(f"  Model types ({len(model_types)}): {model_types}")
    logger.info(f"  Parallel jobs: {n_jobs}")

    # Save manifests first
    logger.info("Saving experiment manifests...")
    for exp_id, exp_config in experiments:
        _sm = "holdout" if exp_config["training_method"] == "single_split" else "cv"
        versioning.save_manifest(exp_id, exp_config, split_method=_sm)

    # Run experiments
    logger.info("Starting experiments...")
    project_root = Path(__file__).parent.parent.parent
    processed_dir = Path(config["paths"]["processed_dir"])
    if not processed_dir.is_absolute():
        processed_dir = project_root / processed_dir
    schema_cfg = load_schema_config(project_root, pipeline)

    if n_jobs == 1:
        # Sequential execution
        results = []
        for i, (exp_id, exp_config) in enumerate(experiments, 1):
            logger.info(f"[RUN] Experiment {i}/{total_experiments}: exp_id={exp_id}")
            result = run_single_experiment(
                exp_id, exp_config, versioning, processed_dir, schema_cfg, logger, target_col
            )
            _annotate_gate_fields(result, gate_thresholds)
            results.append(result)
            _sm = "holdout" if result.get("training_method") == "single_split" else "cv"
            versioning.save_results(exp_id, result, split_method=_sm)
    else:
        # Parallel execution
        logger.info(f"Running experiments in parallel with {n_jobs} jobs...")
        parallel_verbose = int(config.get("parallel_verbose", 0))
        results = Parallel(n_jobs=n_jobs, verbose=parallel_verbose)(
            delayed(run_single_experiment)(
                exp_id, exp_config, versioning, processed_dir, schema_cfg, logger, target_col
            )
            for exp_id, exp_config in experiments
        )

        # Annotate gate fields and save results
        for result in results:
            _annotate_gate_fields(result, gate_thresholds)
            _sm = "holdout" if result.get("training_method") == "single_split" else "cv"
            versioning.save_results(result["experiment_id"], result, split_method=_sm)

    # Deferred XAI pass: run only for top-ranked configurations.
    if xai_cfg_global.get("enabled", True) and xai_mode == "top_configs":
        top_k = int(xai_cfg_global.get("top_k", 5))
        per_dataset = bool(xai_cfg_global.get("top_k_per_dataset", True))
        selected = _select_top_experiments_for_xai(results, top_k=top_k, per_dataset=per_dataset)

        logger.info("[XAI] Starting deferred replay for top configurations")
        logger.info(f"  Selected configs: {len(selected)}")

        if selected:
            replay_xai_cfg = copy.deepcopy(xai_cfg_global)
            replay_xai_cfg["enabled"] = True
            replay_xai_cfg["mode"] = "all"

            if "top_max_samples" in replay_xai_cfg:
                replay_xai_cfg["max_samples"] = int(replay_xai_cfg["top_max_samples"])

            if replay_xai_cfg.get("allow_svm_shap_on_top_configs", True):
                replay_xai_cfg["allow_svm_shap"] = True
                skip_models = [
                    str(m).strip().lower() for m in replay_xai_cfg.get("skip_shap_model_types", [])
                ]
                replay_xai_cfg["skip_shap_model_types"] = [m for m in skip_models if m != "svm"]

            for rank, (result_item, score_value) in enumerate(selected, start=1):
                exp_id = result_item.get("experiment_id")
                exp_cfg = copy.deepcopy(result_item.get("configuration", {}))
                if not exp_id or not exp_cfg:
                    continue

                exp_cfg["xai"] = copy.deepcopy(replay_xai_cfg)
                logger.info(
                    f"  [XAI {rank}/{len(selected)}] exp_id={exp_id} score={score_value:.4f} "
                    f"dataset={exp_cfg.get('dataset')} model={exp_cfg.get('model_type')}"
                )

                replay_result = run_single_experiment(
                    exp_id,
                    exp_cfg,
                    versioning,
                    processed_dir,
                    schema_cfg,
                    logger,
                    target_col,
                )

                if replay_result.get("execution", {}).get("status") != "success":
                    logger.warning(
                        f"  Deferred XAI replay failed for exp_id={exp_id}: "
                        f"{replay_result.get('execution', {}).get('error')}"
                    )

    # Create summary
    versioning.create_summary()
    logger.info("[PHASE] Combinatorial experiments complete")

    # Count successes and failures
    n_success = sum(1 for r in results if r["execution"]["status"] == "success")
    n_failed = total_experiments - n_success

    logger.info("[SUMMARY] Results")
    logger.info(f"  Total experiments: {total_experiments}")
    logger.info(f"  Successful: {n_success}")
    logger.info(f"  Failed: {n_failed}")
    logger.info(f"Results saved to: {versioning.latest_dir}")

    if run_id:
        update_latest_pointer(base_output_dir, versioning.latest_dir, logger)
        append_run_history(
            base_output_dir,
            {
                "run_id": run_id,
                "pipeline": pipeline,
                "mode": "full",
                "phase": "combinatorial",
                "datasets": selected_datasets,
                "output_dir": str(versioning.latest_dir),
                "status": "completed",
            },
        )

    # Aggregate dataset-level SHAP summaries (global/local for both holdout and CV)
    xai_root = versioning.latest_dir / "xai"
    for dataset in selected_datasets:
        shap_dir = xai_root / dataset / "holdout" / "shap"
        if shap_dir.exists():
            aggregate_dataset_shap(shap_dir, "global")
            aggregate_dataset_shap(shap_dir, "local")
        cv_shap_dir = xai_root / dataset / "cv" / "shap"
        if cv_shap_dir.exists():
            aggregate_dataset_shap(cv_shap_dir, "global")
            aggregate_dataset_shap(cv_shap_dir, "local")


def main():
    parser = argparse.ArgumentParser(
        description="Run combinatorial fairness mitigation experiments"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiments/combinatorial.yaml",
        help="Path to experiment configuration file",
    )
    parser.add_argument(
        "--pipeline", type=str, default="cardiac", help="Pipeline name (e.g., cardiac, dermatology)"
    )
    parser.add_argument(
        "--n-jobs", type=int, default=None, help="Number of parallel jobs (-1 for all cores)"
    )
    parser.add_argument(
        "-v", "--verbose", action="count", default=0, help="Verbosity: -v=info, -vv=debug"
    )
    parser.add_argument(
        "--archive-previous",
        action="store_true",
        default=os.getenv("ARCHIVE_PREVIOUS", "true").lower() == "true",
        help="Archive previous run before starting",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=os.getenv("RUN_ID"),
        help="Run identifier (optional, enables run-scoped outputs)",
    )
    parser.add_argument(
        "--output-root", type=str, default=None, help="Base output directory for run outputs"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Optional dataset override (CLI > config > defaults).",
    )
    parser.add_argument(
        "--model-types",
        nargs="+",
        default=None,
        help="Optional model types override (CLI > config > defaults).",
    )

    args = parser.parse_args()

    run_combinatorial_analysis(
        config_path=args.config,
        pipeline=args.pipeline,
        n_jobs=args.n_jobs,
        verbose=args.verbose,
        archive_previous=args.archive_previous,
        run_id=args.run_id,
        output_root=args.output_root,
        datasets=args.datasets,
        model_types_override=args.model_types,
    )


if __name__ == "__main__":
    main()
