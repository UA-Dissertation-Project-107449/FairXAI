"""Data preprocessing utilities for cardiac datasets."""

from __future__ import annotations

import json
import logging
import math
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .schemas import available_sensitive, preferred_sensitive


class CardiacPreprocessor:
    """Preprocess cardiac datasets for modeling."""

    def __init__(self, sensitive_attrs: list[str] | None = None):
        """
        Initialize preprocessor.

        Args:
            sensitive_attrs: List of sensitive attribute column names
        """
        self.sensitive_attrs = preferred_sensitive(sensitive_attrs)
        self.scalers = {}
        self.encoders = {}
        self.impute_values: dict[str, object] = {}
        self.metadata = {}

    def analyze_missing_values(self, df: pd.DataFrame) -> dict[str, object]:
        """
        Analyze missing values in dataset.

        Args:
            df: DataFrame to analyze

        Returns:
            Dictionary with missing value analysis
        """
        missing = df.isnull().sum()
        missing = missing[missing > 0]

        analysis = {
            "total_missing": int(df.isnull().sum().sum()),
            "missing_by_column": {},
            "rows_with_missing": int(df.isnull().any(axis=1).sum()),
            "complete_rows": int((~df.isnull().any(axis=1)).sum()),
        }

        for col in missing.index:
            n_missing = int(missing[col])
            pct_missing = float(missing[col] / len(df) * 100)

            analysis["missing_by_column"][col] = {
                "count": n_missing,
                "percentage": pct_missing,
                "action": self._determine_missing_action(pct_missing),
            }

        return analysis

    def _determine_missing_action(self, pct_missing: float) -> str:
        """Determine action for missing values based on percentage."""
        if pct_missing == 0:
            return "none"
        elif pct_missing < 5:
            return "drop_rows"
        elif pct_missing < 50:
            return "impute_or_flag"
        else:
            return "consider_dropping_column"

    def handle_missing_values(
        self, df: pd.DataFrame, strategy: str = "analyze_only"
    ) -> tuple[pd.DataFrame, dict[str, object]]:
        """
        Handle missing values according to strategy.

        Args:
            df: DataFrame with potential missing values
            strategy: 'analyze_only' (log only), 'keep' (retain rows for later
                holdout-safe imputation), or 'drop_rows' (complete-case). Any other
                value raises ValueError.

        Returns:
            Tuple of (processed DataFrame, actions taken)
        """
        allowed = {"analyze_only", "keep", "drop_rows"}
        if strategy not in allowed:
            raise ValueError(
                f"Unknown missing-value strategy {strategy!r}; "
                f"expected one of {sorted(allowed)}."
            )

        df_processed = df.copy()
        actions = {"strategy": strategy, "actions_taken": []}

        missing_analysis = self.analyze_missing_values(df)

        if missing_analysis["total_missing"] == 0:
            logging.info("No missing values found")
            return df_processed, actions

        if strategy == "analyze_only":
            logging.warning(f"Found {missing_analysis['total_missing']} missing values")
            for col, info in missing_analysis["missing_by_column"].items():
                logging.warning(
                    f"  {col}: {info['count']} ({info['percentage']:.1f}%) - Suggested: {info['action']}"
                )
            return df_processed, actions

        elif strategy == "keep":
            # Retain every row; missing values are imputed later in prepare_features
            # (holdout-safe: fitted on train, reused on test). Required for pooled
            # cohorts with structural, site-dependent missingness where drop_rows
            # would bias.
            actions["actions_taken"].append(
                "Kept all rows; missing values deferred to holdout-safe imputation"
            )
            logging.info("Missing-value strategy=keep: %d rows retained", len(df_processed))
            return df_processed, actions

        elif strategy == "drop_rows":
            initial_len = len(df_processed)
            df_processed = df_processed.dropna()
            dropped = initial_len - len(df_processed)
            actions["actions_taken"].append(f"Dropped {dropped} rows with missing values")
            logging.info(f"Dropped {dropped} rows with missing values")

        return df_processed, actions

    def _impute_missing(self, X: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Fill missing values: median for numeric columns, mode for categoricals.

        Holdout-safe: with ``fit=True`` the fill value per column is computed from
        *X* (the training holdout) and cached in ``self.impute_values``; with
        ``fit=False`` the cached training values are reused, so the held-out test
        set never sees its own statistics.

        Cross-validation is handled separately and is also leak-free: the CV path
        feeds the raw (pre-imputation, pre-scaling) matrix to
        :class:`FoldPreprocessor` via ``CVTrainer.fold_preprocessor_factory``,
        which refits imputation+scaling inside every fold on that fold's training
        rows only. This method governs the single holdout split.

        Args:
            X: Feature matrix (modified in-place and returned).
            fit: Learn and store fill values (train) vs. reuse stored ones (test).

        Returns:
            The same DataFrame with NaNs filled.
        """
        # is_numeric_dtype catches pandas nullable integers (Int64) that
        # select_dtypes(include=[np.number]) can miss, so nullable columns are
        # imputed with a numeric fallback rather than a string.
        numeric_cols = {col for col in X.columns if pd.api.types.is_numeric_dtype(X[col])}

        if fit:
            self.impute_values = {}
            for col in numeric_cols:
                self.impute_values[col] = X[col].median()
            for col in X.columns:
                if col in numeric_cols:
                    continue
                mode = X[col].mode(dropna=True)
                self.impute_values[col] = mode.iloc[0] if not mode.empty else "unknown"

        for col in X.columns:
            if not X[col].isnull().any():
                continue
            fill_value = self.impute_values.get(col)
            # pd.isna handles every all-missing case: a float NaN median, a
            # pd.NA from an all-missing nullable Int64 column, or a None for an
            # unseen column — all fall back to a dtype-appropriate constant.
            if fill_value is None or pd.isna(fill_value):
                fill_value = 0 if col in numeric_cols else "unknown"
            X[col] = X[col].fillna(fill_value)

        return X

    def _encode_categoricals(self, X: pd.DataFrame) -> pd.DataFrame:
        """Label-encode all object/category columns.

        Encoders are cached in ``self.encoders`` so that test data can be
        transformed with the same mapping.

        Args:
            X: Feature matrix (modified in-place and returned).

        Returns:
            The same DataFrame with categorical columns integer-encoded.
        """
        categorical_cols = X.select_dtypes(include=["object", "category"]).columns

        for col in categorical_cols:
            if col not in self.encoders:
                self.encoders[col] = LabelEncoder()
                X[col] = self.encoders[col].fit_transform(X[col].astype(str))
            else:
                X[col] = self.encoders[col].transform(X[col].astype(str))

        return X

    def prepare_features(
        self,
        df: pd.DataFrame,
        target: str = "heart_disease",
        exclude_cols: list[str] | None = None,
        extra_exclude: list[str] | None = None,
        fit: bool = True,
    ) -> tuple[pd.DataFrame, pd.Series, list[str]]:
        """Prepare feature matrix and target vector.

        Orchestrates column exclusion, missing-value imputation, and
        categorical encoding via :meth:`_impute_missing` and
        :meth:`_encode_categoricals`.

        Args:
            df: Input DataFrame
            target: Target column name
            exclude_cols: Columns to exclude from features
            extra_exclude: Per-dataset model-only exclusions (kept in the canonical
                data for provenance/profiling, dropped from the model matrix — e.g.
                source_site, or the sparse ca/thal/slope/chol panel columns).
            fit: Fit imputation on this frame (train) vs. reuse fitted values (test).

        Returns:
            Tuple of (X, y, feature_names)
        """
        if exclude_cols is None:
            # Pipeline metadata and the raw age column (age_group is kept)
            exclude_cols = [
                target,
                "_dataset_source",
                "_dataset_file",
                "age_raw",
            ]

        # Raw / original-name aliases that duplicate harmonized columns
        exclude_cols = list(
            dict.fromkeys(
                exclude_cols
                + [
                    "age",
                    "Age",  # raw age aliases
                    "Sex",
                    "gender",  # raw sex aliases
                    "condition",
                    "HeartDisease",
                    "cardio",  # raw target aliases
                    "id",  # row identifier
                ]
            )
        )

        # Sensitive / demographic columns — excluded from the scaled feature
        # matrix. They are re-attached unscaled in the save step so downstream
        # stages pick them up explicitly. sex_bin / age_group_idx are the
        # numeric encodings of the demographic attributes.
        exclude_cols = list(
            dict.fromkeys(
                exclude_cols
                + self.sensitive_attrs
                + [
                    "sex_extended",
                    "sex_bin",
                    "age_group_idx",
                ]
            )
        )

        if extra_exclude:
            exclude_cols = list(dict.fromkeys(exclude_cols + list(extra_exclude)))

        feature_cols = [col for col in df.columns if col not in exclude_cols]

        X = df[feature_cols].copy()
        y = df[target].copy()

        X = self._impute_missing(X, fit=fit)
        X = self._encode_categoricals(X)

        return X, y, list(feature_cols)

    def scale_features(
        self, X_train: pd.DataFrame, X_test: pd.DataFrame, method: str = "standard"
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Scale numerical features.

        Args:
            X_train: Training features
            X_test: Test features
            method: 'standard' or 'none'

        Returns:
            Tuple of (X_train_scaled, X_test_scaled)
        """
        if method == "none":
            return X_train, X_test

        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()

        # Identify numerical columns (exclude already encoded categoricals)
        numerical_cols = X_train.select_dtypes(include=[np.number]).columns

        if method == "standard":
            scaler = StandardScaler()
            X_train_scaled[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
            X_test_scaled[numerical_cols] = scaler.transform(X_test[numerical_cols])
            self.scalers["standard"] = scaler
            logging.info(f"Standardized {len(numerical_cols)} numerical features")

        return X_train_scaled, X_test_scaled

    def stratified_split(
        self,
        df: pd.DataFrame,
        target: str = "heart_disease",
        test_size: float = 0.3,
        random_state: int = 42,
        context_label: str | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Perform stratified train/test split.

        Stratifies by target AND sensitive attributes to maintain group distributions.

        Args:
            df: Input DataFrame
            target: Target column name
            test_size: Fraction for test set
            random_state: Random seed
            context_label: Optional context shown in warnings (e.g., dataset/binning)

        Returns:
            Tuple of (train_df, test_df)
        """
        # Prefer the most detailed target + sensitive-attribute key that sklearn
        # can stratify without discarding rare rows. If a detailed key contains
        # singleton groups, progressively remove the least-preferred sensitive
        # attribute; target-only and finally random splitting are safe fallbacks.
        strat_cols = [target] + [attr for attr in self.sensitive_attrs if attr in df.columns]
        strat_cols = list(dict.fromkeys(strat_cols))
        context = f"[{context_label}] " if context_label else ""
        working = df.copy()
        n_test = (
            math.ceil(len(working) * test_size) if isinstance(test_size, float) else int(test_size)
        )
        n_train = len(working) - n_test
        max_strata = min(n_train, n_test)
        strat_key = None
        selected_cols = None

        for width in range(len(strat_cols), 0, -1):
            candidate_cols = strat_cols[:width]
            candidate_frame = working[candidate_cols]
            if candidate_frame.columns.duplicated().any():
                candidate_frame = candidate_frame.loc[:, ~candidate_frame.columns.duplicated()]
            candidate_key = candidate_frame.astype(str).agg("_".join, axis=1)
            counts = candidate_key.value_counts()
            valid = (
                candidate_key.nunique() >= 2
                and counts.min() >= 2
                and candidate_key.nunique() <= max_strata
            )
            if valid:
                strat_key = candidate_key
                selected_cols = candidate_cols
                break
            logging.warning(
                "%sCannot stratify on %s without invalid rare strata; trying a coarser key",
                context,
                candidate_cols,
            )

        # Perform stratified split; a random split is the final non-dropping
        # fallback when even target-only stratification is impossible.
        if strat_key is None:
            logging.warning(
                "%sStratified split unavailable; using a non-dropping random split",
                context,
            )
            train_df, test_df = train_test_split(
                working, test_size=test_size, random_state=random_state, shuffle=True
            )
        else:
            if selected_cols != strat_cols:
                logging.warning(
                    "%sUsing coarser stratification key %s; all %d rows retained",
                    context,
                    selected_cols,
                    len(working),
                )
            train_df, test_df = train_test_split(
                working,
                test_size=test_size,
                stratify=strat_key,
                random_state=random_state,
            )

        logging.info(f"[SUCCESS] Split: {len(train_df)} train, {len(test_df)} test")
        logging.info(f"  Test size: {test_size:.1%}")

        return train_df, test_df

    def verify_split_fairness(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame, target: str = "heart_disease"
    ) -> dict[str, object]:
        """
        Verify that train/test split maintains group distributions.

        Args:
            train_df: Training set
            test_df: Test set
            target: Target column name

        Returns:
            Dictionary with verification metrics
        """
        verification = {
            "split_sizes": {
                "train": len(train_df),
                "test": len(test_df),
                "train_pct": len(train_df) / (len(train_df) + len(test_df)),
            },
            "target_distribution": {},
            "sensitive_distribution": {},
        }

        # Check target distribution
        train_target_dist = train_df[target].value_counts(normalize=True).to_dict()
        test_target_dist = test_df[target].value_counts(normalize=True).to_dict()

        verification["target_distribution"] = {"train": train_target_dist, "test": test_target_dist}

        # Check sensitive attribute distributions
        for attr in available_sensitive(train_df, self.sensitive_attrs):
            logging.info(f"[INFO] Checking distribution for attribute: {attr}")
            train_dist = train_df[attr].value_counts(normalize=True).to_dict()
            test_dist = test_df[attr].value_counts(normalize=True).to_dict()

            verification["sensitive_distribution"][attr] = {"train": train_dist, "test": test_dist}

        return verification

    def save_metadata(self, filepath: str) -> None:
        """Save preprocessing metadata to JSON."""
        scaler_params = {}
        for name, scaler in self.scalers.items():
            params = {}
            if hasattr(scaler, "mean_"):
                params["mean"] = scaler.mean_.tolist()
            if hasattr(scaler, "scale_"):
                params["scale"] = scaler.scale_.tolist()
            scaler_params[name] = params if params else "fitted"

        metadata = {
            "encoders": {k: list(v.classes_) for k, v in self.encoders.items()},
            "scalers": scaler_params,
            "sensitive_attrs": self.sensitive_attrs,
        }

        with open(filepath, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        logging.info(f"[SUCCESS] Saved preprocessing metadata to: {filepath}")


class DermatologyPreprocessor(CardiacPreprocessor):
    """Preprocess dermatology image metadata without dropping unknown groups."""

    def handle_missing_values(
        self, df: pd.DataFrame, strategy: str = "image_metadata"
    ) -> tuple[pd.DataFrame, dict[str, object]]:
        """Keep image rows and make metadata safe for downstream profiling."""
        df_processed = df.copy()
        target = "skin_cancer"
        required = [
            col for col in [target, "image_path", "patient_id"] if col in df_processed.columns
        ]
        initial_len = len(df_processed)
        if required:
            df_processed = df_processed.dropna(subset=required).copy()

        for col in self.sensitive_attrs + ["sex_extended", "fitzpatrick", "fitzpatrick_group"]:
            if col in df_processed.columns:
                df_processed[col] = df_processed[col].fillna("unknown")

        remaining_missing = int(df_processed.isna().sum().sum())
        if remaining_missing:
            for col in df_processed.columns:
                if not df_processed[col].isna().any():
                    continue
                if pd.api.types.is_numeric_dtype(df_processed[col]):
                    fill_value = df_processed[col].median()
                    if pd.isna(fill_value):
                        fill_value = 0
                    df_processed[col] = df_processed[col].fillna(fill_value)
                else:
                    df_processed[col] = df_processed[col].fillna("unknown")

        actions = {
            "strategy": strategy,
            "actions_taken": [
                f"Dropped {initial_len - len(df_processed)} rows missing required image/target/id fields",
                "Filled missing sensitive attributes with 'unknown'",
                f"Filled {remaining_missing} remaining metadata missing values for profiling",
            ],
        }
        for action in actions["actions_taken"]:
            logging.info(action)
        return df_processed, actions

    def patient_stratified_split(
        self,
        df: pd.DataFrame,
        target: str = "skin_cancer",
        patient_col: str = "patient_id",
        test_size: float = 0.3,
        random_state: int = 42,
        context_label: str | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split by patient to avoid lesion/image leakage across train/test."""
        if patient_col not in df.columns:
            logging.warning("%s missing; falling back to row-level split", patient_col)
            return self.stratified_split(df, target, test_size, random_state, context_label)

        patient_targets = (
            df.groupby(patient_col)[target]
            .agg(lambda values: int(pd.Series(values).mode(dropna=True).iloc[0]))
            .reset_index()
        )
        stratify = patient_targets[target]
        if stratify.value_counts().min() < 2:
            stratify = None
            logging.warning("Patient-level stratification unavailable; using random patient split")

        train_patients, test_patients = train_test_split(
            patient_targets[patient_col],
            test_size=test_size,
            random_state=random_state,
            shuffle=True,
            stratify=stratify,
        )
        train_ids = set(train_patients.astype(str))
        test_ids = set(test_patients.astype(str))
        train_df = df[df[patient_col].astype(str).isin(train_ids)].copy()
        test_df = df[df[patient_col].astype(str).isin(test_ids)].copy()
        leakage = train_ids & test_ids
        if leakage:
            raise RuntimeError(f"Patient leakage detected after split: {sorted(leakage)[:10]}")

        logging.info("[SUCCESS] Patient split: %d train, %d test", len(train_df), len(test_df))
        logging.info("  Patients: %d train, %d test", len(train_ids), len(test_ids))
        logging.info("  Test size: %.1f%%", test_size * 100)
        return train_df, test_df

    def prepare_features(
        self, df: pd.DataFrame, target: str = "skin_cancer", exclude_cols: list[str] | None = None
    ) -> tuple[pd.DataFrame, pd.Series, list[str]]:
        """Metadata-only fallback feature prep; image training uses CSVs directly."""
        if exclude_cols is None:
            exclude_cols = [
                target,
                "image_path",
                "patient_id",
                "lesion_id",
                "diagnostic_label",
                "_dataset_source",
                "_dataset_file",
                "age_raw",
            ]
        return super().prepare_features(df, target=target, exclude_cols=exclude_cols)


class FoldPreprocessor:
    """Leak-free per-fold imputation + standardization for cross-validation.

    A single ``StandardScaler`` fit on a train/test holdout, then re-pooled and
    re-split for k-fold CV, leaks out-of-fold statistics into every fold. This
    transformer restores fold isolation: fit statistics (per-numeric-column
    median for imputation, ``StandardScaler`` mean/std) are learned **only** from
    the fold's training rows via :meth:`fit_transform`, then applied unchanged to
    the validation rows via :meth:`transform`. A fresh instance is created per
    fold, so no state ever crosses fold boundaries.

    It mirrors the single-split preprocessing (median impute + standardize all
    numeric feature columns) so CV and holdout use the same transform family,
    just fit on the correct rows. Row index and column order are preserved so
    downstream index-based lookups (e.g. LIME tracked instances) keep working.
    """

    def __init__(self) -> None:
        self.numeric_cols: list[str] = []
        self.medians: dict[str, float] = {}
        self.object_fills: dict[str, object] = {}
        self.scaler: StandardScaler | None = None

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Learn impute/scale statistics from *X* (fold-train) and apply them."""
        self.numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
        self.medians = {c: X[c].median() for c in self.numeric_cols}
        self.object_fills = {}
        for col in X.columns:
            if col in self.numeric_cols:
                continue
            mode = X[col].mode(dropna=True)
            self.object_fills[col] = mode.iloc[0] if not mode.empty else "unknown"

        out = self._impute(X.copy())
        if self.numeric_cols:
            self.scaler = StandardScaler()
            out[self.numeric_cols] = self.scaler.fit_transform(out[self.numeric_cols])
        return out

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the fold-train statistics to *X* (fold-validation)."""
        out = self._impute(X.copy())
        if self.scaler is not None and self.numeric_cols:
            out[self.numeric_cols] = self.scaler.transform(out[self.numeric_cols])
        return out

    def _impute(self, X: pd.DataFrame) -> pd.DataFrame:
        for col in self.numeric_cols:
            if not X[col].isnull().any():
                continue
            fill = self.medians.get(col)
            if fill is None or pd.isna(fill):
                # Column all-missing in the fold-train rows: safe numeric fallback.
                fill = 0
            X[col] = X[col].fillna(fill)
        for col, fill in self.object_fills.items():
            if col in X.columns and X[col].isnull().any():
                X[col] = X[col].fillna(fill)
        return X


def apply_fold_preprocessing(
    factory: Optional[Callable[[], Any]],
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit a fresh per-fold transformer on *X_train* and apply it to both splits.

    Single choke point shared by every CV surface (sequential/parallel CVTrainer
    folds, prediction folds, and the combinatorial mitigation loop) so leak-free
    fold preprocessing behaves identically everywhere. Returns the splits
    unchanged when *factory* is ``None``.
    """
    if factory is None:
        return X_train, X_val
    transformer = factory()
    return transformer.fit_transform(X_train), transformer.transform(X_val)
