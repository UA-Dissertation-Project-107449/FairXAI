"""Data preprocessing utilities for cardiac datasets."""

from __future__ import annotations

import json
import logging

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
            strategy: 'analyze_only', 'drop_rows', 'drop_columns', or 'median'

        Returns:
            Tuple of (processed DataFrame, actions taken)
        """
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

        elif strategy == "drop_rows":
            initial_len = len(df_processed)
            df_processed = df_processed.dropna()
            dropped = initial_len - len(df_processed)
            actions["actions_taken"].append(f"Dropped {dropped} rows with missing values")
            logging.info(f"Dropped {dropped} rows with missing values")

        return df_processed, actions

    def _impute_missing(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values: median for numeric columns, mode for categoricals.

        Args:
            X: Feature matrix (modified in-place and returned).

        Returns:
            The same DataFrame with NaNs filled.
        """
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        categorical_cols = X.select_dtypes(include=["object", "category"]).columns

        for col in numeric_cols:
            if X[col].isnull().any():
                X[col] = X[col].fillna(X[col].median())

        for col in categorical_cols:
            if X[col].isnull().any():
                mode = X[col].mode(dropna=True)
                fill_value = mode.iloc[0] if not mode.empty else "unknown"
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
        self, df: pd.DataFrame, target: str = "heart_disease", exclude_cols: list[str] | None = None
    ) -> tuple[pd.DataFrame, pd.Series, list[str]]:
        """Prepare feature matrix and target vector.

        Orchestrates column exclusion, missing-value imputation, and
        categorical encoding via :meth:`_impute_missing` and
        :meth:`_encode_categoricals`.

        Args:
            df: Input DataFrame
            target: Target column name
            exclude_cols: Columns to exclude from features

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

        # Sensitive / demographic columns — excluded to prevent leakage
        exclude_cols = list(
            dict.fromkeys(
                exclude_cols
                + self.sensitive_attrs
                + [
                    "sex_extended",
                    "sex_bin",
                ]
            )
        )

        feature_cols = [col for col in df.columns if col not in exclude_cols]

        X = df[feature_cols].copy()
        y = df[target].copy()

        X = self._impute_missing(X)
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
        # Create stratification variable combining target and sensitive attributes
        strat_cols = [target] + [attr for attr in self.sensitive_attrs if attr in df.columns]
        strat_cols = list(dict.fromkeys(strat_cols))

        # Create combined stratification key (dedupe duplicate columns if present)
        strat_df = df[strat_cols]
        if strat_df.columns.duplicated().any():
            strat_df = strat_df.loc[:, ~strat_df.columns.duplicated()]
        strat_values = strat_df.astype(str).to_numpy()
        strat_key = ["_".join(row) for row in strat_values]
        df["_strat_key"] = pd.Series(strat_key, index=df.index)

        # Remove rare combinations (< 2 samples) to avoid split errors
        strat_counts = df["_strat_key"].value_counts()
        valid_strats = strat_counts[strat_counts >= 2].index
        df_valid = df[df["_strat_key"].isin(valid_strats)].copy()
        context = f"[{context_label}] " if context_label else ""

        def _format_index_preview(indices: list[int], limit: int = 10) -> str:
            head = indices[:limit]
            suffix = "..." if len(indices) > limit else ""
            return f"{head}{suffix}"

        if len(df_valid) < len(df):
            dropped = len(df) - len(df_valid)
            dropped_indices = sorted(df.index.difference(df_valid.index).tolist())
            logging.warning(
                "%sDropped %d samples with rare group combinations; dropped_row_indices=%s",
                context,
                dropped,
                _format_index_preview(dropped_indices),
            )

            # Fallback: if too many dropped, retry with fewer stratification columns
            if len(df_valid) < 0.9 * len(df):
                fallback_cols = [target]
                primary_sensitive = next(
                    (attr for attr in self.sensitive_attrs if attr in df.columns), None
                )
                if primary_sensitive:
                    fallback_cols.append(primary_sensitive)
                fallback_df = df[fallback_cols]
                if fallback_df.columns.duplicated().any():
                    fallback_df = fallback_df.loc[:, ~fallback_df.columns.duplicated()]
                fallback_values = fallback_df.astype(str).to_numpy()
                fallback_key = ["_".join(row) for row in fallback_values]
                df["_strat_key"] = pd.Series(fallback_key, index=df.index)
                strat_counts = df["_strat_key"].value_counts()
                valid_strats = strat_counts[strat_counts >= 2].index
                df_valid = df[df["_strat_key"].isin(valid_strats)].copy()
                dropped = len(df) - len(df_valid)
                dropped_indices = sorted(df.index.difference(df_valid.index).tolist())
                logging.warning(
                    "%sFallback stratification dropped %d samples; dropped_row_indices=%s",
                    context,
                    dropped,
                    _format_index_preview(dropped_indices),
                )

        # Perform stratified split (fallback to unstratified if no valid groups)
        if df_valid.empty or df_valid["_strat_key"].nunique() < 2:
            logging.warning(
                "%sStratified split unavailable; falling back to random split",
                context,
            )
            train_df, test_df = train_test_split(
                df, test_size=test_size, random_state=random_state, shuffle=True
            )
        else:
            train_df, test_df = train_test_split(
                df_valid,
                test_size=test_size,
                stratify=df_valid["_strat_key"],
                random_state=random_state,
            )

        # Remove stratification key
        train_df = train_df.drop(columns=["_strat_key"])
        test_df = test_df.drop(columns=["_strat_key"])

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
