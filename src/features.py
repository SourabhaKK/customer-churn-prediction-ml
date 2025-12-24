"""
Feature engineering module for customer churn prediction.

This module provides functions to create derived features from
preprocessed customer data to improve model performance.
"""

import pandas as pd
import numpy as np
from typing import Optional


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer features for customer churn prediction.
    
    This function creates derived features including:
    - Tenure groups (categorical buckets based on customer tenure)
    - Charge-based ratios and metrics
    - Other domain-specific features
    
    All original columns are preserved, and new features are added.
    
    Args:
        df: pandas DataFrame containing preprocessed customer data
        
    Returns:
        pd.DataFrame: DataFrame with original columns plus engineered features
        
    Example:
        >>> df = pd.DataFrame({
        ...     'tenure': [1, 12, 36],
        ...     'MonthlyCharges': [20.0, 50.0, 75.0],
        ...     'TotalCharges': [20.0, 600.0, 2700.0],
        ...     'Contract': ['Month-to-month', 'One year', 'Two year']
        ... })
        >>> result = engineer_features(df)
        >>> 'tenure_group' in result.columns
        True
    """
    # Create a copy to avoid modifying the original DataFrame
    df_engineered = df.copy()
    
    # Feature 1: Tenure Groups (categorical buckets)
    df_engineered['tenure_group'] = _create_tenure_groups(df_engineered['tenure'])
    
    # Feature 2: Average Monthly Charge (TotalCharges / tenure)
    # Handle division by zero for tenure = 0
    df_engineered['avg_monthly_charge'] = np.where(
        df_engineered['tenure'] > 0,
        df_engineered['TotalCharges'] / df_engineered['tenure'],
        df_engineered['MonthlyCharges']  # Use MonthlyCharges for tenure = 0
    )
    
    # Feature 3: Charge Ratio (MonthlyCharges / TotalCharges)
    # Handle division by zero for TotalCharges = 0
    df_engineered['charge_ratio'] = np.where(
        df_engineered['TotalCharges'] > 0,
        df_engineered['MonthlyCharges'] / df_engineered['TotalCharges'],
        1.0  # Default ratio for zero TotalCharges
    )
    
    # Feature 4: Total Charges per Month (normalized)
    df_engineered['charges_per_month'] = np.where(
        df_engineered['tenure'] > 0,
        df_engineered['TotalCharges'] / df_engineered['tenure'],
        0.0
    )
    
    return df_engineered


def _create_tenure_groups(tenure: pd.Series) -> pd.Series:
    """
    Create categorical tenure groups based on customer tenure.
    
    Groups:
    - 'new': 0-12 months
    - 'short': 13-24 months
    - 'medium': 25-48 months
    - 'long': 49-72 months
    - 'very_long': 73+ months
    
    Args:
        tenure: pandas Series containing tenure values in months
        
    Returns:
        pd.Series: Categorical tenure groups
    """
    def categorize_tenure(months):
        if months <= 12:
            return 'new'
        elif months <= 24:
            return 'short'
        elif months <= 48:
            return 'medium'
        elif months <= 72:
            return 'long'
        else:
            return 'very_long'
    
    return tenure.apply(categorize_tenure)
