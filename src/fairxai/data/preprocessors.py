"""Data preprocessing utilities for cardiac datasets."""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple, Optional
import json
import logging

from .schemas import available_sensitive, preferred_sensitive


class CardiacPreprocessor:
    """Preprocess cardiac datasets for modeling."""
    
    def __init__(self, sensitive_attrs: Optional[List[str]] = None):
        """
        Initialize preprocessor.
        
        Args:
            sensitive_attrs: List of sensitive attribute column names
        """
        self.sensitive_attrs = preferred_sensitive(sensitive_attrs)
        self.scalers = {}
        self.encoders = {}
        self.metadata = {}
        
    def analyze_missing_values(self, df: pd.DataFrame) -> Dict:
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
            'total_missing': int(df.isnull().sum().sum()),
            'missing_by_column': {},
            'rows_with_missing': int(df.isnull().any(axis=1).sum()),
            'complete_rows': int((~df.isnull().any(axis=1)).sum())
        }
        
        for col in missing.index:
            n_missing = int(missing[col])
            pct_missing = float(missing[col] / len(df) * 100)
            
            analysis['missing_by_column'][col] = {
                'count': n_missing,
                'percentage': pct_missing,
                'action': self._determine_missing_action(pct_missing)
            }
        
        return analysis
    
    def _determine_missing_action(self, pct_missing: float) -> str:
        """Determine action for missing values based on percentage."""
        if pct_missing == 0:
            return 'none'
        elif pct_missing < 5:
            return 'drop_rows'
        elif pct_missing < 50:
            return 'impute_or_flag'
        else:
            return 'consider_dropping_column'
    
    def handle_missing_values(
        self, 
        df: pd.DataFrame, 
        strategy: str = 'analyze_only'
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Handle missing values according to strategy.
        
        Args:
            df: DataFrame with potential missing values
            strategy: 'analyze_only', 'drop_rows', 'drop_columns', or 'median'
            
        Returns:
            Tuple of (processed DataFrame, actions taken)
        """
        df_processed = df.copy()
        actions = {'strategy': strategy, 'actions_taken': []}
        
        missing_analysis = self.analyze_missing_values(df)
        
        if missing_analysis['total_missing'] == 0:
            logging.info("✓ No missing values found")
            return df_processed, actions
        
        if strategy == 'analyze_only':
            logging.warning(f"⚠️  Found {missing_analysis['total_missing']} missing values")
            for col, info in missing_analysis['missing_by_column'].items():
                logging.warning(f"  {col}: {info['count']} ({info['percentage']:.1f}%) - Suggested: {info['action']}")
            return df_processed, actions
        
        elif strategy == 'drop_rows':
            initial_len = len(df_processed)
            df_processed = df_processed.dropna()
            dropped = initial_len - len(df_processed)
            actions['actions_taken'].append(f"Dropped {dropped} rows with missing values")
            logging.info(f"Dropped {dropped} rows with missing values")
        
        return df_processed, actions
    
    def prepare_features(
        self,
        df: pd.DataFrame,
        target: str = 'heart_disease',
        exclude_cols: List[str] = None
    ) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """
        Prepare feature matrix and target vector.
        
        Args:
            df: Input DataFrame
            target: Target column name
            exclude_cols: Columns to exclude from features
            
        Returns:
            Tuple of (X, y, feature_names)
        """
        if exclude_cols is None:
            exclude_cols = [
                target,
                '_dataset_source',
                '_dataset_file',
                'age_raw',  # Keep age_group instead
            ]

        # Always exclude sensitive/group columns from model features to avoid leakage
        exclude_cols = list(dict.fromkeys(exclude_cols + self.sensitive_attrs + [
            'sex_extended', 'sex_bin'
        ]))
        
        # Exclude target and metadata columns
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols].copy()
        y = df[target].copy()
        
        # Encode categorical variables
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_cols:
            if col not in self.encoders:
                self.encoders[col] = LabelEncoder()
                X[col] = self.encoders[col].fit_transform(X[col].astype(str))
            else:
                X[col] = self.encoders[col].transform(X[col].astype(str))
        
        return X, y, list(feature_cols)
    
    def scale_features(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        method: str = 'standard'
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Scale numerical features.
        
        Args:
            X_train: Training features
            X_test: Test features
            method: 'standard' or 'none'
            
        Returns:
            Tuple of (X_train_scaled, X_test_scaled)
        """
        if method == 'none':
            return X_train, X_test
        
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        
        # Identify numerical columns (exclude already encoded categoricals)
        numerical_cols = X_train.select_dtypes(include=[np.number]).columns
        
        if method == 'standard':
            scaler = StandardScaler()
            X_train_scaled[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
            X_test_scaled[numerical_cols] = scaler.transform(X_test[numerical_cols])
            self.scalers['standard'] = scaler
            logging.info(f"✓ Standardized {len(numerical_cols)} numerical features")
        
        return X_train_scaled, X_test_scaled
    
    def stratified_split(
        self,
        df: pd.DataFrame,
        target: str = 'heart_disease',
        test_size: float = 0.3,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Perform stratified train/test split.
        
        Stratifies by target AND sensitive attributes to maintain group distributions.
        
        Args:
            df: Input DataFrame
            target: Target column name
            test_size: Fraction for test set
            random_state: Random seed
            
        Returns:
            Tuple of (train_df, test_df)
        """
        # Create stratification variable combining target and sensitive attributes
        strat_cols = [target] + [attr for attr in self.sensitive_attrs if attr in df.columns]
        
        # Create combined stratification key
        df['_strat_key'] = df[strat_cols].astype(str).agg('_'.join, axis=1)
        
        # Remove rare combinations (< 2 samples) to avoid split errors
        strat_counts = df['_strat_key'].value_counts()
        valid_strats = strat_counts[strat_counts >= 2].index
        df_valid = df[df['_strat_key'].isin(valid_strats)].copy()
        
        if len(df_valid) < len(df):
            dropped = len(df) - len(df_valid)
            logging.warning(f"⚠️  Dropped {dropped} samples with rare group combinations")
        
        # Perform stratified split
        train_df, test_df = train_test_split(
            df_valid,
            test_size=test_size,
            stratify=df_valid['_strat_key'],
            random_state=random_state
        )
        
        # Remove stratification key
        train_df = train_df.drop(columns=['_strat_key'])
        test_df = test_df.drop(columns=['_strat_key'])
        
        logging.info(f"✓ Split: {len(train_df)} train, {len(test_df)} test")
        logging.info(f"  Test size: {test_size:.1%}")
        
        return train_df, test_df
    
    def verify_split_fairness(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        target: str = 'heart_disease'
    ) -> Dict:
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
            'split_sizes': {
                'train': len(train_df),
                'test': len(test_df),
                'train_pct': len(train_df) / (len(train_df) + len(test_df))
            },
            'target_distribution': {},
            'sensitive_distribution': {}
        }
        
        # Check target distribution
        train_target_dist = train_df[target].value_counts(normalize=True).to_dict()
        test_target_dist = test_df[target].value_counts(normalize=True).to_dict()
        
        verification['target_distribution'] = {
            'train': train_target_dist,
            'test': test_target_dist
        }
        
        # Check sensitive attribute distributions
        for attr in available_sensitive(train_df, self.sensitive_attrs):
            
            train_dist = train_df[attr].value_counts(normalize=True).to_dict()
            test_dist = test_df[attr].value_counts(normalize=True).to_dict()
            
            verification['sensitive_distribution'][attr] = {
                'train': train_dist,
                'test': test_dist
            }
        
        return verification
    
    def save_metadata(self, filepath: str):
        """Save preprocessing metadata to JSON."""
        metadata = {
            'encoders': {k: list(v.classes_) for k, v in self.encoders.items()},
            'scalers': {k: 'fitted' for k in self.scalers.keys()},
            'sensitive_attrs': self.sensitive_attrs
        }
        
        with open(filepath, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logging.info(f"✓ Saved preprocessing metadata to: {filepath}")
