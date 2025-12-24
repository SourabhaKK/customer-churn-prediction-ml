"""
Prediction interface module for customer churn prediction.

This module provides functions to make predictions using trained models
and return both class predictions and probability scores.
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from typing import Union, Dict


def predict(model: BaseEstimator, X: Union[pd.DataFrame, np.ndarray]) -> Dict:
    """
    Make predictions using a trained model.
    
    This function takes a trained model and input features, returning
    both class predictions and probability scores. It handles both single
    and batch inputs, and supports both DataFrame and NumPy array formats.
    
    Args:
        model: Trained sklearn model with predict and predict_proba methods
        X: Feature matrix (pandas DataFrame or numpy array)
        
    Returns:
        dict: Dictionary containing:
            - 'predictions': Array of predicted class labels
            - 'probabilities': Array of probability scores for each class
            
    Raises:
        ValueError: If model or input is None
        AttributeError: If model doesn't have required methods
        
    Example:
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> import pandas as pd
        >>> 
        >>> # Train a model
        >>> X_train = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
        >>> y_train = pd.Series([0, 1, 0])
        >>> model = RandomForestClassifier(random_state=42)
        >>> model.fit(X_train, y_train)
        >>> 
        >>> # Make predictions
        >>> X_test = pd.DataFrame({'feature1': [2], 'feature2': [5]})
        >>> result = predict(model, X_test)
        >>> print(result['predictions'])
        [1]
        >>> print(result['probabilities'])
        [[0.3 0.7]]
    """
    # Validate inputs
    if model is None:
        raise ValueError("Model cannot be None")
    
    if X is None:
        raise ValueError("Input data cannot be None")
    
    # Validate model has required methods
    if not hasattr(model, 'predict'):
        raise AttributeError("Model must have 'predict' method")
    
    if not hasattr(model, 'predict_proba'):
        raise AttributeError("Model must have 'predict_proba' method")
    
    # Make predictions
    try:
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)
    except Exception as e:
        # Re-raise with more context
        raise ValueError(f"Error making predictions: {str(e)}") from e
    
    # Return results as dictionary
    return {
        'predictions': predictions,
        'probabilities': probabilities
    }
