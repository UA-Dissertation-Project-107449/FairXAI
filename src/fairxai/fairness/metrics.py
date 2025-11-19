"""
Fairness metrics for post-prediction assessment.

This module implements group fairness metrics and individual fairness analysis
for evaluating ML model predictions across sensitive attributes.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy.spatial.distance import pdist, squareform
import logging


class FairnessMetrics:
    """Calculate fairness metrics on predictions."""
    
    def __init__(self, sensitive_attributes: List[str] = ['age_group', 'sex']):
        """
        Initialize fairness metrics calculator.
        
        Args:
            sensitive_attributes: List of sensitive attribute column names
        """
        self.sensitive_attributes = sensitive_attributes
    
    def demographic_parity(
        self,
        df: pd.DataFrame,
        sensitive_attr: str,
        pred_col: str = 'y_pred'
    ) -> Dict:
        """
        Calculate demographic parity (statistical parity).
        
        Measures: P(Y_hat=1 | A=a) for each group a
        Fairness violation: max|P(Y_hat=1 | A=a) - P(Y_hat=1 | A=b)|
        
        Args:
            df: DataFrame with predictions and sensitive attributes
            sensitive_attr: Name of sensitive attribute column
            pred_col: Name of prediction column
            
        Returns:
            Dictionary with per-group rates and max difference
        """
        results = {
            'metric': 'demographic_parity',
            'sensitive_attribute': sensitive_attr,
            'group_rates': {},
            'overall_rate': df[pred_col].mean()
        }
        
        for group in df[sensitive_attr].unique():
            group_df = df[df[sensitive_attr] == group]
            positive_rate = group_df[pred_col].mean()
            results['group_rates'][str(group)] = {
                'positive_rate': positive_rate,
                'count': len(group_df)
            }
        
        # Calculate max difference
        rates = [v['positive_rate'] for v in results['group_rates'].values()]
        results['max_difference'] = max(rates) - min(rates)
        results['is_fair'] = results['max_difference'] < 0.1  # 10% threshold
        
        return results
    
    def equalized_odds(
        self,
        df: pd.DataFrame,
        sensitive_attr: str,
        true_col: str = 'y_true',
        pred_col: str = 'y_pred'
    ) -> Dict:
        """
        Calculate equalized odds.
        
        Measures: 
        - TPR (recall): P(Y_hat=1 | Y=1, A=a) for each group
        - FPR: P(Y_hat=1 | Y=0, A=a) for each group
        
        Fairness: Equal TPR and FPR across groups
        
        Args:
            df: DataFrame with predictions, labels, and sensitive attributes
            sensitive_attr: Name of sensitive attribute column
            true_col: Name of true label column
            pred_col: Name of prediction column
            
        Returns:
            Dictionary with TPR/FPR per group and max differences
        """
        results = {
            'metric': 'equalized_odds',
            'sensitive_attribute': sensitive_attr,
            'group_metrics': {}
        }
        
        for group in df[sensitive_attr].unique():
            group_df = df[df[sensitive_attr] == group]
            
            # True positives and false negatives
            positives = group_df[group_df[true_col] == 1]
            tp = ((positives[pred_col] == 1).sum())
            fn = ((positives[pred_col] == 0).sum())
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            
            # True negatives and false positives
            negatives = group_df[group_df[true_col] == 0]
            tn = ((negatives[pred_col] == 0).sum())
            fp = ((negatives[pred_col] == 1).sum())
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            
            results['group_metrics'][str(group)] = {
                'tpr': tpr,
                'fpr': fpr,
                'count': len(group_df),
                'positive_count': len(positives),
                'negative_count': len(negatives)
            }
        
        # Calculate max differences
        tprs = [v['tpr'] for v in results['group_metrics'].values()]
        fprs = [v['fpr'] for v in results['group_metrics'].values()]
        
        results['tpr_max_difference'] = max(tprs) - min(tprs)
        results['fpr_max_difference'] = max(fprs) - min(fprs)
        results['is_fair'] = (
            results['tpr_max_difference'] < 0.1 and 
            results['fpr_max_difference'] < 0.1
        )
        
        return results
    
    def equal_opportunity(
        self,
        df: pd.DataFrame,
        sensitive_attr: str,
        true_col: str = 'y_true',
        pred_col: str = 'y_pred'
    ) -> Dict:
        """
        Calculate equal opportunity (equalized TPR only).
        
        Measures: P(Y_hat=1 | Y=1, A=a) for each group
        Fairness: Equal TPR across groups
        
        Args:
            df: DataFrame with predictions, labels, and sensitive attributes
            sensitive_attr: Name of sensitive attribute column
            true_col: Name of true label column
            pred_col: Name of prediction column
            
        Returns:
            Dictionary with TPR per group and max difference
        """
        results = {
            'metric': 'equal_opportunity',
            'sensitive_attribute': sensitive_attr,
            'group_tpr': {}
        }
        
        for group in df[sensitive_attr].unique():
            group_df = df[df[sensitive_attr] == group]
            positives = group_df[group_df[true_col] == 1]
            
            tp = ((positives[pred_col] == 1).sum())
            fn = ((positives[pred_col] == 0).sum())
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            
            results['group_tpr'][str(group)] = {
                'tpr': tpr,
                'true_positive_count': len(positives)
            }
        
        tprs = [v['tpr'] for v in results['group_tpr'].values()]
        results['max_difference'] = max(tprs) - min(tprs)
        results['is_fair'] = results['max_difference'] < 0.1
        
        return results
    
    def predictive_parity(
        self,
        df: pd.DataFrame,
        sensitive_attr: str,
        true_col: str = 'y_true',
        pred_col: str = 'y_pred'
    ) -> Dict:
        """
        Calculate predictive parity (precision parity).
        
        Measures: P(Y=1 | Y_hat=1, A=a) for each group (precision)
        Fairness: Equal precision across groups
        
        Args:
            df: DataFrame with predictions, labels, and sensitive attributes
            sensitive_attr: Name of sensitive attribute column
            true_col: Name of true label column
            pred_col: Name of prediction column
            
        Returns:
            Dictionary with precision per group and max difference
        """
        results = {
            'metric': 'predictive_parity',
            'sensitive_attribute': sensitive_attr,
            'group_precision': {}
        }
        
        for group in df[sensitive_attr].unique():
            group_df = df[df[sensitive_attr] == group]
            predicted_positive = group_df[group_df[pred_col] == 1]
            
            tp = ((predicted_positive[true_col] == 1).sum())
            fp = ((predicted_positive[true_col] == 0).sum())
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            
            results['group_precision'][str(group)] = {
                'precision': precision,
                'predicted_positive_count': len(predicted_positive)
            }
        
        precisions = [v['precision'] for v in results['group_precision'].values()]
        results['max_difference'] = max(precisions) - min(precisions)
        results['is_fair'] = results['max_difference'] < 0.1
        
        return results
    
    def calibration_by_group(
        self,
        df: pd.DataFrame,
        sensitive_attr: str,
        true_col: str = 'y_true',
        proba_col: str = 'y_proba',
        n_bins: int = 10
    ) -> Dict:
        """
        Calculate calibration metrics by group.
        
        Calibration: Does P(Y=1 | score=s) match the predicted score s?
        
        Args:
            df: DataFrame with predictions, labels, probabilities, and sensitive attributes
            sensitive_attr: Name of sensitive attribute column
            true_col: Name of true label column
            proba_col: Name of probability column
            n_bins: Number of bins for calibration curve
            
        Returns:
            Dictionary with calibration metrics per group
        """
        results = {
            'metric': 'calibration',
            'sensitive_attribute': sensitive_attr,
            'n_bins': n_bins,
            'group_calibration': {}
        }
        
        for group in df[sensitive_attr].unique():
            group_df = df[df[sensitive_attr] == group]
            
            # Create bins
            bins = np.linspace(0, 1, n_bins + 1)
            bin_indices = np.digitize(group_df[proba_col], bins[:-1]) - 1
            bin_indices = np.clip(bin_indices, 0, n_bins - 1)
            
            bin_metrics = []
            for bin_idx in range(n_bins):
                bin_mask = bin_indices == bin_idx
                bin_data = group_df[bin_mask]
                
                if len(bin_data) > 0:
                    mean_predicted = bin_data[proba_col].mean()
                    mean_true = bin_data[true_col].mean()
                    count = len(bin_data)
                    
                    bin_metrics.append({
                        'bin': bin_idx,
                        'bin_range': (bins[bin_idx], bins[bin_idx + 1]),
                        'mean_predicted': mean_predicted,
                        'mean_true': mean_true,
                        'calibration_error': abs(mean_predicted - mean_true),
                        'count': count
                    })
            
            # Expected Calibration Error (ECE)
            total_count = len(group_df)
            ece = sum(
                (m['count'] / total_count) * m['calibration_error']
                for m in bin_metrics
            )
            
            results['group_calibration'][str(group)] = {
                'ece': ece,
                'bins': bin_metrics,
                'count': total_count
            }
        
        # Max ECE difference
        eces = [v['ece'] for v in results['group_calibration'].values()]
        results['max_ece_difference'] = max(eces) - min(eces)
        results['is_fair'] = results['max_ece_difference'] < 0.05
        
        return results
    
    def individual_fairness_consistency(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        pred_col: str = 'y_pred',
        k: int = 5
    ) -> Dict:
        """
        Calculate individual fairness via k-nearest neighbor consistency.
        
        Individual fairness: Similar individuals should receive similar predictions.
        Consistency score: How often do k-nearest neighbors have the same prediction?
        
        Args:
            df: DataFrame with features and predictions
            feature_cols: List of feature column names
            pred_col: Name of prediction column
            k: Number of nearest neighbors to consider
            
        Returns:
            Dictionary with consistency metrics
        """
        # Extract features
        X = df[feature_cols].values
        y_pred = df[pred_col].values
        
        # Compute pairwise distances
        distances = squareform(pdist(X, metric='euclidean'))
        
        # For each sample, find k nearest neighbors (excluding itself)
        consistencies = []
        for i in range(len(df)):
            # Get k+1 nearest (including itself), then exclude itself
            nearest_indices = np.argsort(distances[i])[:k+1]
            nearest_indices = nearest_indices[nearest_indices != i][:k]
            
            # Check prediction consistency
            same_prediction = (y_pred[nearest_indices] == y_pred[i]).sum()
            consistency = same_prediction / k
            consistencies.append(consistency)
        
        results = {
            'metric': 'individual_fairness_consistency',
            'k': k,
            'mean_consistency': np.mean(consistencies),
            'std_consistency': np.std(consistencies),
            'min_consistency': np.min(consistencies),
            'median_consistency': np.median(consistencies)
        }
        
        return results
    
    def calculate_all_metrics(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None
    ) -> Dict:
        """
        Calculate all fairness metrics for all sensitive attributes.
        
        Args:
            df: DataFrame with predictions, labels, and sensitive attributes
            feature_cols: Optional list of feature columns for individual fairness
            
        Returns:
            Dictionary with all metrics organized by type
        """
        results = {
            'group_fairness': {},
            'calibration': {},
            'individual_fairness': {}
        }
        
        for attr in self.sensitive_attributes:
            if attr not in df.columns:
                logging.warning(f"Sensitive attribute '{attr}' not found in dataframe")
                continue
            
            # Group fairness metrics
            results['group_fairness'][attr] = {
                'demographic_parity': self.demographic_parity(df, attr),
                'equalized_odds': self.equalized_odds(df, attr),
                'equal_opportunity': self.equal_opportunity(df, attr),
                'predictive_parity': self.predictive_parity(df, attr)
            }
            
            # Calibration
            if 'y_proba' in df.columns:
                results['calibration'][attr] = self.calibration_by_group(df, attr)
        
        # Individual fairness
        if feature_cols and all(col in df.columns for col in feature_cols):
            results['individual_fairness'] = self.individual_fairness_consistency(
                df, feature_cols
            )
        
        return results


def summarize_fairness_results(results: Dict) -> pd.DataFrame:
    """
    Create a summary table of fairness metrics.
    
    Args:
        results: Dictionary from calculate_all_metrics
        
    Returns:
        DataFrame with fairness summary
    """
    rows = []
    
    for attr, metrics in results['group_fairness'].items():
        for metric_name, metric_data in metrics.items():
            row = {
                'sensitive_attribute': attr,
                'metric': metric_name,
                'is_fair': metric_data.get('is_fair', False)
            }
            
            # Extract key difference value
            if 'max_difference' in metric_data:
                row['max_difference'] = metric_data['max_difference']
            elif 'tpr_max_difference' in metric_data:
                row['tpr_max_difference'] = metric_data['tpr_max_difference']
                row['fpr_max_difference'] = metric_data['fpr_max_difference']
            
            rows.append(row)
    
    return pd.DataFrame(rows)
