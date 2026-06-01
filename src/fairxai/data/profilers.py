"""Data profiling utilities for fairness assessment."""

from itertools import combinations
from typing import Any, Callable, Dict, List

import pandas as pd

from ..profiling import compute_complexity_metrics
from .schemas import available_sensitive, preferred_sensitive

# Columns present in raw/standardized files but absent from the model feature
# set.  They are redundant aliases (sex_extended == sex, sex_bin == numeric sex,
# age_raw == continuous age before binning) and inflate feature-count-sensitive
# complexity metrics (N3, L1, L2) if left in.
_COMPLEXITY_EXCLUDE_COLS: list[str] = [
    "sex_extended",
    "sex_bin",
    "age_raw",
    # pipeline metadata columns
    "_dataset_source",
    "_dataset_file",
    # raw-name aliases that may survive standardization
    "age",
    "Age",
    "Sex",
    "gender",
    "id",
    # image-pipeline metadata
    "image_path",
    "patient_id",
    "lesion_id",
    "diagnostic",
    "diagnostic_label",
]


class DataProfiler:
    """Profile datasets for fairness assessment before modeling."""

    def __init__(
        self,
        sensitive_attrs: List[str] = None,
        min_group_samples: int = 50,
        subgroup_generators: List[Callable[[pd.DataFrame], pd.DataFrame]] | None = None,
    ):
        """
        Initialize profiler.

        Args:
            sensitive_attrs: List of sensitive attribute column names
        """
        self.sensitive_attrs = preferred_sensitive(sensitive_attrs)
        self.min_group_samples = min_group_samples
        self.subgroup_generators = subgroup_generators or []

    @staticmethod
    def _to_builtin(value: Any) -> Any:
        if hasattr(value, "item"):
            return value.item()
        return value

    @staticmethod
    def _stringify(obj) -> Dict:
        """Return a dict with stringified keys for JSON safety."""
        items = obj.items() if isinstance(obj, dict) else obj.to_dict().items()
        return {str(k): v for k, v in items}

    @staticmethod
    def _complexity_ready_frame(df: pd.DataFrame, target: str) -> pd.DataFrame:
        """Fill numeric feature NaNs for sklearn-based complexity metrics only."""
        prepared = df.copy()
        numeric_cols = prepared.select_dtypes(include="number").columns
        for col in numeric_cols:
            if col == target or not prepared[col].isna().any():
                continue
            fill_value = prepared[col].median()
            if pd.isna(fill_value):
                fill_value = 0
            prepared[col] = prepared[col].fillna(fill_value)
        return prepared

    def profile_dataset(
        self, df: pd.DataFrame, target: str = "heart_disease", dataset_name: str = "unknown"
    ) -> Dict:
        """
        Generate comprehensive data profile including fairness metrics.

        Args:
            df: DataFrame to profile
            target: Target variable column name
            dataset_name: Dataset identifier

        Returns:
            Dictionary with profiling results
        """
        available_attrs = available_sensitive(df, self.sensitive_attrs)
        subgroup_df = self._build_subgroup_frame(df, available_attrs)
        complexity_df = self._complexity_ready_frame(df, target)
        subgroup_complexity_df = self._complexity_ready_frame(subgroup_df, target)

        profile = {
            "dataset_name": dataset_name,
            "basic_stats": self._basic_statistics(df, target),
            "sensitive_attr_distribution": self._sensitive_distribution(df),
            "target_distribution": self._target_distribution(df, target),
            "group_statistics": self._group_statistics(df, target),
            "representation_balance": self._representation_balance(df),
            "label_imbalance_by_group": self._label_imbalance_by_group(df, target),
            "missing_value_analysis": self._missing_value_analysis(df),
            "complexity_metrics": compute_complexity_metrics(
                complexity_df, target=target, exclude_cols=_COMPLEXITY_EXCLUDE_COLS
            ),
            "group_complexity_metrics": self._group_complexity_metrics(
                subgroup_complexity_df,
                target=target,
                sensitive_attrs=available_attrs,
            ),
            "intersection_complexity_metrics": self._intersection_complexity_metrics(
                subgroup_complexity_df,
                target=target,
                sensitive_attrs=available_attrs,
            ),
        }

        return profile

    def _basic_statistics(self, df: pd.DataFrame, target: str) -> Dict:
        """Basic dataset statistics."""
        return {
            "n_samples": len(df),
            "n_features": len(df.columns),
            "target_name": target,
            "target_prevalence": float(df[target].mean()) if target in df.columns else None,
        }

    def _sensitive_distribution(self, df: pd.DataFrame) -> Dict:
        """Distribution of sensitive attributes."""
        distributions = {}

        for attr in self.sensitive_attrs:
            if attr in df.columns:
                counts = df[attr].value_counts()
                distributions[attr] = {
                    "counts": self._stringify(counts),
                    "proportions": self._stringify(counts / len(df)),
                }

        return distributions

    def _target_distribution(self, df: pd.DataFrame, target: str) -> Dict:
        """Overall target variable distribution."""
        if target not in df.columns:
            return {}

        counts = df[target].value_counts()
        return {
            "counts": self._stringify(counts),
            "proportions": self._stringify(counts / len(df)),
            "imbalance_ratio": float(counts.max() / counts.min()) if len(counts) > 1 else 1.0,
        }

    def _group_statistics(self, df: pd.DataFrame, target: str) -> Dict:
        """Statistics stratified by sensitive attributes."""
        group_stats = {}

        for attr in self.sensitive_attrs:
            if attr not in df.columns or target not in df.columns:
                continue

            groups = {}
            for group_value in df[attr].dropna().unique():
                group_df = df[df[attr] == group_value]

                groups[str(group_value)] = {
                    "n_samples": len(group_df),
                    "proportion_of_total": float(len(group_df) / len(df)),
                    "target_prevalence": float(group_df[target].mean()),
                    "target_counts": group_df[target].value_counts().to_dict(),
                }

            group_stats[attr] = groups

        return group_stats

    def _representation_balance(self, df: pd.DataFrame) -> Dict:
        """
        Calculate representation balance metrics.

        Uses coefficient of variation (CV) to measure disparity in group sizes.
        CV = std / mean, lower is more balanced.
        """
        balance = {}

        for attr in self.sensitive_attrs:
            if attr not in df.columns:
                continue

            counts = df[attr].value_counts()
            mean_count = counts.mean()
            std_count = counts.std()

            balance[attr] = {
                "coefficient_of_variation": (
                    float(std_count / mean_count) if mean_count > 0 else None
                ),
                "min_group_size": int(counts.min()),
                "max_group_size": int(counts.max()),
                "size_ratio": float(counts.max() / counts.min()) if counts.min() > 0 else None,
                "counts": self._stringify(counts),
            }

        return balance

    def _label_imbalance_by_group(self, df: pd.DataFrame, target: str) -> Dict:
        """
        Calculate label imbalance within each sensitive group.

        Statistical parity difference: P(Y=1 | A=a) - P(Y=1 | A=b)
        """
        if target not in df.columns:
            return {}

        imbalances = {}

        for attr in self.sensitive_attrs:
            if attr not in df.columns:
                continue

            # Calculate positive rate for each group
            group_rates = df.groupby(attr, observed=True)[target].mean()

            imbalances[attr] = {
                "positive_rates": self._stringify(group_rates),
                "statistical_parity_difference": {
                    "max_difference": float(group_rates.max() - group_rates.min()),
                    "max_ratio": (
                        float(group_rates.max() / group_rates.min())
                        if group_rates.min() > 0
                        else None
                    ),
                },
            }

        return imbalances

    def _missing_value_analysis(self, df: pd.DataFrame) -> Dict:
        """Analyze missing values overall and by sensitive groups."""
        missing_overall = df.isnull().sum()
        missing_overall = missing_overall[missing_overall > 0]

        analysis = {
            "total_missing": int(df.isnull().sum().sum()),
            "columns_with_missing": missing_overall.to_dict(),
            "missing_by_group": {},
        }

        # Check if missing values vary by sensitive groups
        for attr in self.sensitive_attrs:
            if attr not in df.columns:
                continue

            group_missing = {}
            for col in df.columns:
                if df[col].isnull().sum() > 0:
                    missing_by_group = df.groupby(attr)[col].apply(lambda x: x.isnull().sum())
                    group_missing[col] = self._stringify(missing_by_group)

            if group_missing:
                analysis["missing_by_group"][attr] = group_missing

        return analysis

    def _build_subgroup_frame(self, df: pd.DataFrame, sensitive_attrs: List[str]) -> pd.DataFrame:
        subgroup_df = df.copy()
        for generator in self.subgroup_generators:
            try:
                generated = generator(subgroup_df)
                if isinstance(generated, pd.DataFrame):
                    subgroup_df = generated
            except Exception:
                continue

        if len(sensitive_attrs) >= 2:
            for left, right in combinations(sensitive_attrs, 2):
                if left in subgroup_df.columns and right in subgroup_df.columns:
                    inter_col = f"{left}__{right}"
                    subgroup_df[inter_col] = (
                        subgroup_df[left].astype("object").astype(str)
                        + "|"
                        + subgroup_df[right].astype("object").astype(str)
                    )
        return subgroup_df

    def _group_complexity_metrics(
        self,
        df: pd.DataFrame,
        target: str,
        sensitive_attrs: List[str],
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        result: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for attr in sensitive_attrs:
            if attr not in df.columns:
                continue
            group_result: Dict[str, Dict[str, Any]] = {}
            for group_value, group_df in df.groupby(attr, observed=True):
                key = str(group_value)
                n_samples = len(group_df)
                if n_samples < self.min_group_samples:
                    group_result[key] = {
                        "n_samples": int(n_samples),
                        "status": f"skipped (n < {self.min_group_samples})",
                        "complexity_metrics": {},
                    }
                    continue

                metrics = compute_complexity_metrics(
                    group_df, target=target, exclude_cols=_COMPLEXITY_EXCLUDE_COLS
                )
                group_result[key] = {
                    "n_samples": int(n_samples),
                    "status": "ok" if metrics else "unavailable",
                    "complexity_metrics": metrics,
                }
            result[attr] = group_result
        return result

    def _intersection_complexity_metrics(
        self,
        df: pd.DataFrame,
        target: str,
        sensitive_attrs: List[str],
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        intersections: Dict[str, Dict[str, Dict[str, Any]]] = {}
        if len(sensitive_attrs) < 2:
            return intersections

        for left, right in combinations(sensitive_attrs, 2):
            if left not in df.columns or right not in df.columns:
                continue
            pair_key = f"{left}__{right}"
            pair_groups: Dict[str, Dict[str, Any]] = {}
            grouped = df.groupby([left, right], observed=True)
            for (left_value, right_value), group_df in grouped:
                subgroup_key = f"{left}={left_value}|{right}={right_value}"
                n_samples = len(group_df)
                if n_samples < self.min_group_samples:
                    pair_groups[subgroup_key] = {
                        "n_samples": int(n_samples),
                        "status": f"skipped (n < {self.min_group_samples})",
                        "complexity_metrics": {},
                    }
                    continue
                metrics = compute_complexity_metrics(
                    group_df, target=target, exclude_cols=_COMPLEXITY_EXCLUDE_COLS
                )
                pair_groups[subgroup_key] = {
                    "n_samples": int(n_samples),
                    "status": "ok" if metrics else "unavailable",
                    "complexity_metrics": metrics,
                }

            intersections[pair_key] = pair_groups

        return intersections


def compare_datasets(profiles: List[Dict]) -> Dict:
    """
    Compare multiple dataset profiles.

    Args:
        profiles: List of profile dictionaries from DataProfiler

    Returns:
        Comparison summary
    """
    comparison = {
        "n_datasets": len(profiles),
        "dataset_names": [p["dataset_name"] for p in profiles],
        "total_samples": sum(p["basic_stats"]["n_samples"] for p in profiles),
        "sample_sizes": {p["dataset_name"]: p["basic_stats"]["n_samples"] for p in profiles},
        "target_prevalence": {
            p["dataset_name"]: p["basic_stats"]["target_prevalence"] for p in profiles
        },
    }

    return comparison
