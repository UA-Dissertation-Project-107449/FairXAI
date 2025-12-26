"""Data profiling utilities for fairness assessment."""

import pandas as pd
from typing import Dict, List

from .schemas import available_sensitive, preferred_sensitive


class DataProfiler:
    """Profile datasets for fairness assessment before modeling."""
    
    def __init__(self, sensitive_attrs: List[str] = None):
        """
        Initialize profiler.
        
        Args:
            sensitive_attrs: List of sensitive attribute column names
        """
        self.sensitive_attrs = preferred_sensitive(sensitive_attrs)

    @staticmethod
    def _stringify(obj) -> Dict:
        """Return a dict with stringified keys for JSON safety."""
        items = obj.items() if isinstance(obj, dict) else obj.to_dict().items()
        return {str(k): v for k, v in items}
        
    def profile_dataset(
        self, 
        df: pd.DataFrame, 
        target: str = 'heart_disease',
        dataset_name: str = 'unknown'
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
        profile = {
            'dataset_name': dataset_name,
            'basic_stats': self._basic_statistics(df, target),
            'sensitive_attr_distribution': self._sensitive_distribution(df),
            'target_distribution': self._target_distribution(df, target),
            'group_statistics': self._group_statistics(df, target),
            'representation_balance': self._representation_balance(df),
            'label_imbalance_by_group': self._label_imbalance_by_group(df, target),
            'missing_value_analysis': self._missing_value_analysis(df)
        }
        
        return profile
    
    def _basic_statistics(self, df: pd.DataFrame, target: str) -> Dict:
        """Basic dataset statistics."""
        return {
            'n_samples': len(df),
            'n_features': len(df.columns),
            'target_name': target,
            'target_prevalence': float(df[target].mean()) if target in df.columns else None
        }
    
    def _sensitive_distribution(self, df: pd.DataFrame) -> Dict:
        """Distribution of sensitive attributes."""
        distributions = {}
        
        for attr in self.sensitive_attrs:
            if attr in df.columns:
                counts = df[attr].value_counts()
                distributions[attr] = {
                    'counts': self._stringify(counts),
                    'proportions': self._stringify(counts / len(df))
                }
        
        return distributions
    
    def _target_distribution(self, df: pd.DataFrame, target: str) -> Dict:
        """Overall target variable distribution."""
        if target not in df.columns:
            return {}
        
        counts = df[target].value_counts()
        return {
            'counts': self._stringify(counts),
            'proportions': self._stringify(counts / len(df)),
            'imbalance_ratio': float(counts.max() / counts.min()) if len(counts) > 1 else 1.0
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
                    'n_samples': len(group_df),
                    'proportion_of_total': float(len(group_df) / len(df)),
                    'target_prevalence': float(group_df[target].mean()),
                    'target_counts': group_df[target].value_counts().to_dict()
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
                'coefficient_of_variation': float(std_count / mean_count) if mean_count > 0 else None,
                'min_group_size': int(counts.min()),
                'max_group_size': int(counts.max()),
                'size_ratio': float(counts.max() / counts.min()) if counts.min() > 0 else None,
                'counts': self._stringify(counts)
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
                'positive_rates': self._stringify(group_rates),
                'statistical_parity_difference': {
                    'max_difference': float(group_rates.max() - group_rates.min()),
                    'max_ratio': float(group_rates.max() / group_rates.min()) if group_rates.min() > 0 else None
                }
            }
        
        return imbalances
    
    def _missing_value_analysis(self, df: pd.DataFrame) -> Dict:
        """Analyze missing values overall and by sensitive groups."""
        missing_overall = df.isnull().sum()
        missing_overall = missing_overall[missing_overall > 0]
        
        analysis = {
            'total_missing': int(df.isnull().sum().sum()),
            'columns_with_missing': missing_overall.to_dict(),
            'missing_by_group': {}
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
                analysis['missing_by_group'][attr] = group_missing
        
        return analysis


def compare_datasets(profiles: List[Dict]) -> Dict:
    """
    Compare multiple dataset profiles.
    
    Args:
        profiles: List of profile dictionaries from DataProfiler
        
    Returns:
        Comparison summary
    """
    comparison = {
        'n_datasets': len(profiles),
        'dataset_names': [p['dataset_name'] for p in profiles],
        'total_samples': sum(p['basic_stats']['n_samples'] for p in profiles),
        'sample_sizes': {p['dataset_name']: p['basic_stats']['n_samples'] for p in profiles},
        'target_prevalence': {p['dataset_name']: p['basic_stats']['target_prevalence'] for p in profiles}
    }
    
    return comparison
