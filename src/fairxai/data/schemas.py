from typing import Iterable, List, Sequence, Tuple
import pandas as pd


CARDIAC_DATASETS = {"cleveland", "kaggle_heart"}

# Canonical sensitive/group columns to support across areas.
# Order matters for stratification and reporting.
DEFAULT_SENSITIVE_ATTRS: Sequence[str] = (
    "age_group",  # canonical age bins
    "sex",        # canonical sex
    "ethnicity",  # optional (not present in current cardiac datasets)
    "group_cluster",  # optional derived cluster/similarity group
)


def harmonize_cardiac_schema(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """
    Harmonize cardiac datasets to a unified schema.

    - Unify age to `age_raw` and derive `age_group` (if bins later, we keep existing).
    - Normalize sex to `sex` (Female/Male), add `sex_extended` duplicate for plotting,
      and `sex_bin` numeric encoding (Female=0, Male=1).
    - Map target to `heart_disease` if present under dataset-specific names.
    - Preserve existing columns; avoid overwriting if already standardized.
    """
    df = df.copy()

    # Age raw: support both 'age' and 'Age'
    if "age_raw" not in df.columns:
        if "age" in df.columns:
            df["age_raw"] = pd.to_numeric(df["age"], errors="coerce")
        elif "Age" in df.columns:
            df["age_raw"] = pd.to_numeric(df["Age"], errors="coerce")

    # Clip to reasonable human range to avoid outliers corrupting scaling
    if "age_raw" in df.columns:
        df["age_raw"] = df["age_raw"].clip(lower=0, upper=120)

    # Sex: harmonize to string labels where possible
    if "sex" not in df.columns and "Sex" in df.columns:
        df["sex"] = df["Sex"].map({"F": "Female", "M": "Male"}).fillna(df["Sex"])
    elif "sex" in df.columns and pd.api.types.is_numeric_dtype(df["sex"]):
        df["sex"] = df["sex"].map({0: "Female", 1: "Male"}).astype("object")

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


def get_sensitive_columns() -> Tuple[str, str]:
    """Return canonical sensitive columns for cardiac workflows."""
    return "age_group", "sex"


def preferred_sensitive(preferred: Iterable[str] = None) -> List[str]:
    """Return the preferred sensitive/group columns in priority order."""
    if preferred is None:
        return list(DEFAULT_SENSITIVE_ATTRS)
    return list(dict.fromkeys(preferred))  # de-duplicate while preserving order


def available_sensitive(df: pd.DataFrame, preferred: Iterable[str] = None) -> List[str]:
    """Return sensitive/group columns that exist in the dataframe."""
    return [col for col in preferred_sensitive(preferred) if col in df.columns]
