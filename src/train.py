"""
Model training module for customer churn prediction.

This module provides functions to train machine learning models
for predicting customer churn.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator
from typing import Union, Optional


def train_model(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    random_state: Optional[int] = 42
) -> BaseEstimator:
    """
    Train a machine learning model for customer churn prediction.
    
    This function trains a Random Forest classifier on the provided
    training data. The model is configured with a fixed random state
    for reproducibility.
    
    Args:
        X: Feature matrix (pandas DataFrame or numpy array)
        y: Target variable (pandas Series or numpy array)
        random_state: Random seed for reproducibility (default: 42)
        
    Returns:
        BaseEstimator: Trained sklearn model (fitted Random Forest classifier)
        
    Example:
        >>> X = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
        >>> y = pd.Series([0, 1, 0])
        >>> model = train_model(X, y)
        >>> predictions = model.predict(X)
    """
    # Initialize Random Forest classifier with reproducible random state
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1  # Use all available cores
    )
    
    # Train the model
    model.fit(X, y)
    
    return model
