"""Data loading utilities for cardiac datasets."""

import pandas as pd
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class CardiacDataLoader:
    """Loader for cardiac disease datasets with schema mapping."""
    
    def __init__(self, config_path: str):
        """
        Initialize loader with schema mapping configuration.
        
        Args:
            config_path: Path to schema_mapping.json
        """
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.datasets = self.config['datasets']
        self.cardiac_datasets = self.config['cardiac_relevant_datasets']
        
    def load_dataset(self, dataset_name: str, data_dir: str) -> pd.DataFrame:
        """
        Load a single dataset by name.
        
        Args:
            dataset_name: Name from schema_mapping.json (e.g., 'cleveland')
            data_dir: Directory containing raw CSV files
            
        Returns:
            Raw DataFrame
        """
        if dataset_name not in self.datasets:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        dataset_config = self.datasets[dataset_name]
        filepath = Path(data_dir) / dataset_config['filename']
        
        if not filepath.exists():
            raise FileNotFoundError(f"Dataset file not found: {filepath}")
        
        logging.info(f"Loading {dataset_name} from {filepath}")
        df = pd.read_csv(filepath)
        
        # Add metadata columns
        df['_dataset_source'] = dataset_name
        df['_dataset_file'] = dataset_config['filename']
        
        return df
    
    def load_all_cardiac_datasets(self, data_dir: str) -> Dict[str, pd.DataFrame]:
        """
        Load all cardiac-relevant datasets.
        
        Args:
            data_dir: Directory containing raw CSV files
            
        Returns:
            Dictionary mapping dataset names to DataFrames
        """
        datasets = {}
        for name in self.cardiac_datasets:
            try:
                datasets[name] = self.load_dataset(name, data_dir)
                logging.info(f"✓ Loaded {name}: {len(datasets[name])} rows")
            except Exception as e:
                logging.error(f"✗ Failed to load {name}: {e}")
        
        return datasets
    
    def get_sensitive_attributes(self, dataset_name: str) -> Dict[str, dict]:
        """Get sensitive attribute configuration for a dataset."""
        return self.datasets[dataset_name]['sensitive_attributes']
    
    def get_target_column(self, dataset_name: str) -> str:
        """Get target column name for a dataset."""
        return self.datasets[dataset_name]['target']
    
    def get_clinical_features(self, dataset_name: str) -> List[str]:
        """Get list of clinical feature columns for a dataset."""
        return self.datasets[dataset_name]['clinical_features']
    
    def standardize_sensitive_attributes(
        self, 
        df: pd.DataFrame, 
        dataset_name: str
    ) -> pd.DataFrame:
        """
        Standardize sensitive attributes to unified schema.
        
        Creates new columns: 'age_group', 'sex'
        
        Args:
            df: Raw DataFrame
            dataset_name: Dataset identifier
            
        Returns:
            DataFrame with standardized sensitive attributes
        """
        df = df.copy()
        sens_attrs = self.get_sensitive_attributes(dataset_name)
        
        # Standardize age
        if 'age' in sens_attrs:
            age_col = 'age'
        elif 'Age' in sens_attrs:
            age_col = 'Age'
        else:
            raise ValueError(f"No age column found for {dataset_name}")
        
        age_config = sens_attrs[age_col]
        df['age_group'] = pd.cut(
            df[age_col],
            bins=age_config['bins'],
            labels=age_config['labels'],
            include_lowest=True
        )
        df['age_raw'] = df[age_col]  # Keep original
        
        # Standardize sex
        if 'sex' in sens_attrs:
            sex_col = 'sex'
        elif 'Sex' in sens_attrs:
            sex_col = 'Sex'
        elif 'Gender' in sens_attrs:
            sex_col = 'Gender'
        else:
            raise ValueError(f"No sex/gender column found for {dataset_name}")
        
        sex_mapping = sens_attrs[sex_col]['mapping']
        # Convert keys to match data types (handle both int and string keys)
        if df[sex_col].dtype in ['int64', 'int32', 'int16', 'int8']:
            # Convert string keys to integers for integer columns
            sex_mapping = {int(k): v for k, v in sex_mapping.items()}
        df['sex'] = df[sex_col].map(sex_mapping)
        
        return df
    
    def standardize_target(
        self, 
        df: pd.DataFrame, 
        dataset_name: str
    ) -> pd.DataFrame:
        """
        Standardize target variable to 'heart_disease' (binary: 0/1).
        
        Args:
            df: DataFrame with dataset-specific target
            dataset_name: Dataset identifier
            
        Returns:
            DataFrame with 'heart_disease' column
        """
        df = df.copy()
        target_col = self.get_target_column(dataset_name)
        
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in {dataset_name}")
        
        # Binary mapping (assume 0=no disease, 1=disease)
        df['heart_disease'] = df[target_col].astype(int)
        
        return df


def get_dataset_summary(df: pd.DataFrame, dataset_name: str) -> Dict:
    """
    Generate summary statistics for a dataset.
    
    Args:
        df: DataFrame to summarize
        dataset_name: Name for identification
        
    Returns:
        Dictionary with summary statistics
    """
    summary = {
        'dataset': dataset_name,
        'n_rows': len(df),
        'n_cols': len(df.columns),
        'columns': list(df.columns),
        'missing_values': df.isnull().sum().to_dict(),
        'dtypes': df.dtypes.astype(str).to_dict()
    }
    
    return summary
