"""Schema harmonization and sensitive-column helpers for data workflows.

Canonical domain metadata (dataset names, sensitive attributes, sex-mapping
constants, age-clipping bounds) is loaded from
``configs/domain/cardiac.yaml`` at first access and cached for the process
lifetime.  Built-in fallbacks ensure the module works even when the YAML is
unavailable (e.g. in unit-test contexts).
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from pathlib import Path

import pandas as pd

from ..utils.config import load_yaml_config

logger = logging.getLogger(__name__)

# ── Domain config cache ───────────────────────────────────────────────
_domain_cfg: dict | None = None

# Built-in fallbacks (used when YAML is not found).
_FALLBACK_DATASETS = {"cleveland", "kaggle_heart", "cardio70k"}
_FALLBACK_SENSITIVE_ATTRS: Sequence[str] = (
    "age_group", "sex", "ethnicity", "group_cluster",
)
_FALLBACK_SEX_NUMERIC = {0: "Female", 1: "Male", 2: "Male"}
_FALLBACK_SEX_STRING = {
    "F": "Female", "Female": "Female", "0": "Female",
    "M": "Male", "Male": "Male", "1": "Male", "2": "Male",
}
_FALLBACK_AGE_CLIP_YEARS = 120
_FALLBACK_AGE_CLIP_DAYS = 43830


def _resolve_project_root() -> Path | None:
    """Walk upward from this file to find the repo root (has ``configs/``)."""
    candidate = Path(__file__).resolve().parent
    for _ in range(8):
        if (candidate / "configs").is_dir():
            return candidate
        candidate = candidate.parent
    return None


def _load_domain_config() -> dict:
    """Load ``configs/domain/cardiac.yaml`` once and cache it."""
    global _domain_cfg
    if _domain_cfg is not None:
        return _domain_cfg

    root = _resolve_project_root()
    if root is not None:
        cfg_path = root / "configs" / "domain" / "cardiac.yaml"
        if cfg_path.exists():
            try:
                _domain_cfg = load_yaml_config(str(cfg_path))
                return _domain_cfg
            except Exception:
                logger.debug("Failed to load %s; using built-in fallbacks.", cfg_path)

    _domain_cfg = {}
    return _domain_cfg


# ── Public accessors (replace former module-level constants) ──────────

def get_cardiac_datasets() -> set[str]:
    """Return the set of known cardiac dataset names from domain config."""
    cfg = _load_domain_config()
    datasets_section = cfg.get("datasets", {})
    if datasets_section:
        return set(datasets_section.keys())
    return set(_FALLBACK_DATASETS)


def get_default_sensitive_attrs() -> list[str]:
    """Return canonical sensitive/group columns in priority order."""
    cfg = _load_domain_config()
    order = cfg.get("sensitive_attribute_order")
    if order:
        return list(order)
    return list(_FALLBACK_SENSITIVE_ATTRS)


def get_sex_mapping(kind: str = "numeric") -> dict:
    """Return canonical sex-value mapping from domain config.

    Parameters
    ----------
    kind : ``"numeric"`` | ``"string"``
    """
    cfg = _load_domain_config()
    mappings = cfg.get("sex_mapping", {})
    raw = mappings.get(kind, {})
    if raw:
        if kind == "numeric":
            return {int(k): v for k, v in raw.items()}
        return {str(k): v for k, v in raw.items()}
    return dict(_FALLBACK_SEX_NUMERIC if kind == "numeric" else _FALLBACK_SEX_STRING)


def get_age_clip_bounds(dataset: str) -> tuple[int, int]:
    """Return ``(lower, upper)`` age-clipping bounds for *dataset*."""
    cfg = _load_domain_config()
    clip = cfg.get("age_clipping", {})
    if dataset == "cardio70k":
        upper = clip.get("cardio70k_days_upper", _FALLBACK_AGE_CLIP_DAYS)
    else:
        upper = clip.get("years_upper", _FALLBACK_AGE_CLIP_YEARS)
    return 0, int(upper)


def get_age_unit(dataset: str) -> str:
    """Return the age unit (``'years'`` or ``'days'``) declared for *dataset*.

    Reads ``age_unit`` from ``configs/domain/cardiac.yaml``.
    Falls back to ``'years'`` if not declared — safe default for all datasets
    that store age in years (cleveland, kaggle_heart, and future datasets).
    """
    cfg = _load_domain_config()
    return str(cfg.get("datasets", {}).get(dataset, {}).get("age_unit", "years"))


# ── Schema harmonization ─────────────────────────────────────────────

def harmonize_cardiac_schema(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Harmonize cardiac datasets to a unified schema.

    - Unify age to ``age_raw`` and derive ``age_group`` (if bins later, we keep existing).
    - Normalize sex to ``sex`` (Female/Male), add ``sex_extended`` duplicate for
      plotting, and ``sex_bin`` numeric encoding (Female=0, Male=1).
    - Map target to ``heart_disease`` if present under dataset-specific names.
    - Preserve existing columns; avoid overwriting if already standardized.
    """
    df = df.copy()

    # Age raw: support both 'age' and 'Age'
    if "age_raw" not in df.columns:
        if "age" in df.columns:
            df["age_raw"] = pd.to_numeric(df["age"], errors="coerce")
        elif "Age" in df.columns:
            df["age_raw"] = pd.to_numeric(df["Age"], errors="coerce")

    # Clip to reasonable human range to avoid outliers corrupting scaling.
    if "age_raw" in df.columns:
        lower, upper = get_age_clip_bounds(dataset)
        if dataset == "cardio70k" and df["age_raw"].max() > 130:
            df["age_raw"] = df["age_raw"].clip(lower=lower, upper=upper)
        else:
            _, years_upper = get_age_clip_bounds("years")  # always years for non-cardio70k
            df["age_raw"] = df["age_raw"].clip(lower=0, upper=years_upper)

    # Sex: harmonize to string labels where possible
    sex_num = get_sex_mapping("numeric")
    sex_str = get_sex_mapping("string")
    if "sex" not in df.columns and "Sex" in df.columns:
        df["sex"] = df["Sex"].map(sex_str).fillna(df["Sex"])
    elif "sex" in df.columns and pd.api.types.is_numeric_dtype(df["sex"]):
        df["sex"] = df["sex"].map(sex_num).astype("object")

    # Add sex_extended (same as sex) and sex_bin (Female=0, Male=1)
    if "sex" in df.columns:
        df["sex_extended"] = df["sex"].astype("object")
        df["sex_bin"] = df["sex"].map({"Female": 0, "Male": 1})

    # Target: unify to heart_disease
    if "heart_disease" not in df.columns:
        if "HeartDisease" in df.columns:
            df["heart_disease"] = pd.to_numeric(df["HeartDisease"], errors="coerce")
        elif "condition" in df.columns:
            df["heart_disease"] = pd.to_numeric(df["condition"], errors="coerce")

    return df


# ── Sensitive-column helpers ──────────────────────────────────────────

def preferred_sensitive(preferred: Iterable[str] | None = None) -> list[str]:
    """Return the preferred sensitive/group columns in priority order."""
    if preferred is None:
        return get_default_sensitive_attrs()
    return list(dict.fromkeys(preferred))  # de-duplicate while preserving order


def available_sensitive(df: pd.DataFrame, preferred: Iterable[str] | None = None) -> list[str]:
    """Return sensitive/group columns that exist in the dataframe."""
    return [col for col in preferred_sensitive(preferred) if col in df.columns]
