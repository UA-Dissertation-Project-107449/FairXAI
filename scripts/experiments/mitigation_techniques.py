"""
Fairness mitigation techniques implementation.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Pre-processing
from imblearn.over_sampling import SMOTE, RandomOverSampler, ADASYN
from imblearn.under_sampling import RandomUnderSampler

# In-processing
from fairlearn.reductions import ExponentiatedGradient, GridSearch, DemographicParity, EqualizedOdds

# Post-processing
from fairlearn.postprocessing import ThresholdOptimizer


def apply_preprocessing(X_train, y_train, X_test, y_test, sensitive_train, sensitive_test, 
                       technique_name, dataset_name):
    """
    Apply pre-processing mitigation technique.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        sensitive_train: Sensitive attributes for training
        sensitive_test: Sensitive attributes for test
        technique_name: Name of technique ('smote', 'ros', 'rus', 'adasyn')
        dataset_name: Name of dataset
        
    Returns:
        dict: Metrics dictionary
    """
    # Select sampler
    if technique_name == 'smote':
        sampler = SMOTE(sampling_strategy='auto', k_neighbors=5, random_state=42)
    elif technique_name == 'ros':
        sampler = RandomOverSampler(sampling_strategy='auto', random_state=42)
    elif technique_name == 'rus':
        sampler = RandomUnderSampler(sampling_strategy='auto', random_state=42)
    elif technique_name == 'adasyn':
        sampler = ADASYN(sampling_strategy='auto', n_neighbors=5, random_state=42)
    else:
        raise ValueError(f"Unknown preprocessing technique: {technique_name}")
    
    # Resample training data
    X_train_resampled, y_train_resampled = sampler.fit_resample(X_train, y_train)
    
    # Train model on resampled data
    model = LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, random_state=42)
    model.fit(X_train_resampled, y_train_resampled)
    
    # Predict on test set
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'dataset': dataset_name,
        'technique': technique_name,
        'stage': 'pre-processing',
        'samples_before': len(X_train),
        'samples_after': len(X_train_resampled),
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, zero_division=0)),
        'f1': float(f1_score(y_test, y_pred, zero_division=0)),
        'auc_roc': float(roc_auc_score(y_test, y_proba))
    }
    
    return metrics


def apply_inprocessing(X_train, y_train, X_test, y_test, sensitive_train, sensitive_test,
                      technique_name, dataset_name):
    """
    Apply in-processing mitigation technique.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        sensitive_train: Sensitive attributes for training
        sensitive_test: Sensitive attributes for test
        technique_name: Name of technique ('exponentiated_gradient', 'grid_search')
        dataset_name: Name of dataset
        
    Returns:
        dict: Metrics dictionary
    """
    base_model = LogisticRegression(max_iter=1000, random_state=42)
    
    if technique_name == 'exponentiated_gradient':
        constraint = DemographicParity()
        model = ExponentiatedGradient(
            base_model,
            constraints=constraint,
            eps=0.05,
            max_iter=50
        )
        model.fit(X_train, y_train, sensitive_features=sensitive_train['sex'])
        
    elif technique_name == 'grid_search':
        constraint = EqualizedOdds()
        model = GridSearch(
            base_model,
            constraints=constraint,
            grid_size=20
        )
        model.fit(X_train, y_train, sensitive_features=sensitive_train['sex'])
    else:
        raise ValueError(f"Unknown in-processing technique: {technique_name}")
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Get probabilities
    if hasattr(model, 'predictors_') and len(model.predictors_) > 0:
        y_proba = model.predictors_[0].predict_proba(X_test)[:, 1]
    elif hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = y_pred
    
    # Calculate metrics
    metrics = {
        'dataset': dataset_name,
        'technique': technique_name,
        'stage': 'in-processing',
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, zero_division=0)),
        'f1': float(f1_score(y_test, y_pred, zero_division=0)),
        'auc_roc': float(roc_auc_score(y_test, y_proba)) if len(np.unique(y_proba)) > 1 else 0.0
    }
    
    return metrics


def apply_postprocessing(base_model, X_train, y_train, X_test, y_test, 
                        sensitive_train, sensitive_test, technique_name, dataset_name):
    """
    Apply post-processing mitigation technique.
    
    Args:
        base_model: Pre-trained baseline model
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        sensitive_train: Sensitive attributes for training
        sensitive_test: Sensitive attributes for test
        technique_name: Name of technique ('threshold_optimizer')
        dataset_name: Name of dataset
        
    Returns:
        dict: Metrics dictionary
    """
    if technique_name == 'threshold_optimizer':
        postprocessor = ThresholdOptimizer(
            estimator=base_model,
            constraints='equalized_odds',
            objective='balanced_accuracy_score',
            prefit=True
        )
        
        # Fit on training data
        postprocessor.fit(X_train, y_train, sensitive_features=sensitive_train['sex'])
        
        # Predict on test set
        y_pred = postprocessor.predict(X_test, sensitive_features=sensitive_test['sex'])
        y_proba = base_model.predict_proba(X_test)[:, 1]
    else:
        raise ValueError(f"Unknown post-processing technique: {technique_name}")
    
    # Calculate metrics
    metrics = {
        'dataset': dataset_name,
        'technique': technique_name,
        'stage': 'post-processing',
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, zero_division=0)),
        'f1': float(f1_score(y_test, y_pred, zero_division=0)),
        'auc_roc': float(roc_auc_score(y_test, y_proba))
    }
    
    return metrics
