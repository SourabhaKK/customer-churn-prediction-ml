"""
Model evaluation module for customer churn prediction.

This module provides functions to evaluate trained models
and compute performance metrics.
"""

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    confusion_matrix
)
from sklearn.base import BaseEstimator
from typing import Union, Dict, Any


def evaluate_model(
    model: BaseEstimator,
    X_test: Union[np.ndarray, 'pd.DataFrame'],
    y_test: Union[np.ndarray, 'pd.Series']
) -> Dict[str, Any]:
    """
    Evaluate a trained model on test data.
    
    Computes multiple classification metrics to assess model performance:
    - ROC-AUC: Area under the ROC curve (threshold-independent)
    - Precision: Of predicted positives, how many are actually positive?
    - Recall: Of actual positives, how many did we predict?
    - Confusion Matrix: Breakdown of predictions vs. actuals
    
    Args:
        model: Trained sklearn model with predict and predict_proba methods
        X_test: Test feature matrix (numpy array or pandas DataFrame)
        y_test: True labels for test set (numpy array or pandas Series)
        
    Returns:
        dict: Dictionary containing evaluation metrics:
            - 'roc_auc': float - ROC-AUC score (0 to 1)
            - 'precision': float - Precision score (0 to 1)
            - 'recall': float - Recall score (0 to 1)
            - 'confusion_matrix': np.ndarray - 2x2 confusion matrix
            
    Raises:
        ValueError: If model or inputs are None
        AttributeError: If model doesn't have required methods
        
    Example:
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> import numpy as np
        >>> 
        >>> # Train a model
        >>> X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        >>> y_train = np.array([0, 0, 1, 1])
        >>> model = RandomForestClassifier(random_state=42)
        >>> model.fit(X_train, y_train)
        >>> 
        >>> # Evaluate on test data
        >>> X_test = np.array([[2, 3], [6, 7]])
        >>> y_test = np.array([0, 1])
        >>> metrics = evaluate_model(model, X_test, y_test)
        >>> 
        >>> print(f"ROC-AUC: {metrics['roc_auc']:.3f}")
        >>> print(f"Precision: {metrics['precision']:.3f}")
        >>> print(f"Recall: {metrics['recall']:.3f}")
    """
    # Validate inputs
    if model is None:
        raise ValueError("Model cannot be None")
    
    if X_test is None:
        raise ValueError("X_test cannot be None")
    
    if y_test is None:
        raise ValueError("y_test cannot be None")
    
    # Validate model has required methods
    if not hasattr(model, 'predict'):
        raise AttributeError("Model must have 'predict' method")
    
    if not hasattr(model, 'predict_proba'):
        raise AttributeError("Model must have 'predict_proba' method")
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of positive class
    
    # Compute metrics
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    # Return metrics dictionary
    return {
        'roc_auc': float(roc_auc),
        'precision': float(precision),
        'recall': float(recall),
        'confusion_matrix': cm
    }
