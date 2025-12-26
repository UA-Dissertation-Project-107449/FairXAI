"""
Baseline model training script.
"""
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def train_baseline_model(X_train, y_train, X_test, y_test, dataset_name):
    """
    Train baseline logistic regression model.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        dataset_name: Name of dataset for tracking
        
    Returns:
        tuple: (model, y_pred, y_proba, metrics_dict)
    """
    model = LogisticRegression(
        C=1.0,
        class_weight='balanced',
        max_iter=1000,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'dataset': dataset_name,
        'technique': 'baseline',
        'stage': 'none',
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, zero_division=0)),
        'f1': float(f1_score(y_test, y_pred, zero_division=0)),
        'auc_roc': float(roc_auc_score(y_test, y_proba))
    }
    
    return model, y_pred, y_proba, metrics
