"""
Preprocessing module for customer churn prediction.

This module provides preprocessing functions to transform raw customer data
into features suitable for machine learning models.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from typing import Union


def preprocess_data(df: pd.DataFrame) -> np.ndarray:
    """
    Preprocess customer churn data for machine learning.
    
    This function performs the following transformations:
    - Separates features from target column ('Churn')
    - Identifies numerical and categorical features
    - One-hot encodes categorical features
    - Scales numerical features using StandardScaler
    - Returns a NumPy array ready for model training
    
    Args:
        df: pandas DataFrame containing customer data with 'Churn' column
        
    Returns:
        np.ndarray: Preprocessed feature matrix with all transformations applied
        
    Example:
        >>> df = pd.DataFrame({
        ...     'tenure': [12, 24],
        ...     'MonthlyCharges': [50.5, 75.2],
        ...     'Contract': ['Month-to-month', 'One year'],
        ...     'Churn': ['No', 'Yes']
        ... })
        >>> result = preprocess_data(df)
        >>> result.shape
        (2, 5)  # 2 numerical + 3 one-hot encoded features
    """
    # Separate features from target
    if 'Churn' in df.columns:
        X = df.drop('Churn', axis=1)
    else:
        X = df.copy()
    
    # Identify numerical and categorical columns
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    
    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )
    
    # Fit and transform the data
    X_processed = preprocessor.fit_transform(X)
    
    return X_processed
