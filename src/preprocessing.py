"""
Preprocessing module for customer churn prediction.

This module provides preprocessing functions to transform raw customer data
into features suitable for machine learning models.
"""

import logging

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from typing import Tuple

logger = logging.getLogger(__name__)

# Columns that must never reach the encoder: identifiers produce cardinality
# explosions; TotalCharges is a numeric field stored as object in the raw CSV.
_COLUMNS_TO_DROP = ['customerID']
_NUMERIC_COERCE_COLUMNS = ['TotalCharges']


def _prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a clean feature DataFrame ready for dtype-based column selection.

    Performs two transformations that are invisible to callers but are required
    for correctness on the real Telco CSV:

    1. Drops ``customerID`` (and any column in ``_COLUMNS_TO_DROP``) so that
       the OHE step does not produce ~5 685 spurious binary columns.
    2. Coerces ``TotalCharges`` (and any column in ``_NUMERIC_COERCE_COLUMNS``)
       from ``object`` dtype to ``float64``, filling blank strings / NaN with
       0.  This moves the column from the categorical pipeline to the numeric
       (StandardScaler) pipeline where it belongs.

    Args:
        df: Feature DataFrame (target column already removed by the caller).

    Returns:
        Cleaned DataFrame with correct dtypes.
    """
    X = df.copy()

    # Drop identifier columns
    cols_to_drop = [c for c in _COLUMNS_TO_DROP if c in X.columns]
    if cols_to_drop:
        logger.info("Dropping identifier column(s): %s", cols_to_drop)
        X = X.drop(columns=cols_to_drop)

    # Coerce known numeric-but-object columns
    for col in _NUMERIC_COERCE_COLUMNS:
        if col in X.columns and X[col].dtype == object:
            before_nulls = X[col].isnull().sum()
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
            after_nulls = X[col].isnull().sum()
            imputed = int(X[col].eq(0).sum()) - int(before_nulls)
            if imputed > 0:
                logger.warning(
                    "Column '%s': %d blank/non-numeric value(s) coerced to 0.",
                    col, imputed,
                )

    return X


def preprocess_data(df: pd.DataFrame) -> np.ndarray:
    """
    Preprocess customer churn data for machine learning.
    
    ⚠️ WARNING: This function uses fit_transform internally.
    For production use with train/test splits, use fit_preprocess() 
    and transform_preprocess() instead to prevent data leakage.
    
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

    # Drop identifier columns and coerce known numeric-but-object columns
    X = _prepare_features(X)

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


def fit_preprocess(df: pd.DataFrame) -> Tuple[ColumnTransformer, np.ndarray]:
    """
    Fit preprocessing transformers on training data and return both preprocessor and transformed data.
    
    This function should be used on TRAINING data only. It fits the encoders and scalers
    on the training data and returns both the fitted preprocessor (for later use on test data)
    and the transformed training data.
    
    This prevents data leakage by ensuring test data statistics do not influence training.
    
    Args:
        df: pandas DataFrame containing training data with 'Churn' column
        
    Returns:
        Tuple containing:
            - preprocessor: Fitted ColumnTransformer that can be reused
            - X_processed: Transformed training data as NumPy array
            
    Example:
        >>> train_df = pd.DataFrame({
        ...     'tenure': [12, 24, 36],
        ...     'MonthlyCharges': [50.5, 75.2, 90.0],
        ...     'Contract': ['Month-to-month', 'One year', 'Two year'],
        ...     'Churn': ['No', 'Yes', 'No']
        ... })
        >>> preprocessor, X_train = fit_preprocess(train_df)
        >>> # Later use preprocessor on test data
        >>> X_test = transform_preprocess(test_df, preprocessor)
    """
    # Separate features from target
    if 'Churn' in df.columns:
        X = df.drop('Churn', axis=1)
    else:
        X = df.copy()

    # Drop identifier columns and coerce known numeric-but-object columns
    X = _prepare_features(X)

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

    # Fit and transform the training data
    X_processed = preprocessor.fit_transform(X)

    return preprocessor, X_processed


def transform_preprocess(df: pd.DataFrame, preprocessor: ColumnTransformer) -> np.ndarray:
    """
    Transform data using an already-fitted preprocessor.
    
    This function should be used on TEST data. It uses the preprocessor that was
    fitted on training data to transform test data, ensuring that test data statistics
    do not leak into the model.
    
    ⚠️ IMPORTANT: This function does NOT refit the preprocessor. It only transforms
    the data using the statistics learned from the training data.
    
    Args:
        df: pandas DataFrame containing test data with 'Churn' column
        preprocessor: Already-fitted ColumnTransformer from fit_preprocess()
        
    Returns:
        np.ndarray: Transformed test data using training statistics
        
    Example:
        >>> # First, fit on training data
        >>> preprocessor, X_train = fit_preprocess(train_df)
        >>> 
        >>> # Then, transform test data using the same preprocessor
        >>> X_test = transform_preprocess(test_df, preprocessor)
        >>> 
        >>> # X_test is scaled using training data statistics (no leakage!)
    """
    # Separate features from target
    if 'Churn' in df.columns:
        X = df.drop('Churn', axis=1)
    else:
        X = df.copy()

    # Drop identifier columns and coerce known numeric-but-object columns
    # (must mirror the same preparation done inside fit_preprocess)
    X = _prepare_features(X)

    # Transform using the already-fitted preprocessor
    # This does NOT refit - it only transforms using learned statistics
    X_processed = preprocessor.transform(X)

    return X_processed
