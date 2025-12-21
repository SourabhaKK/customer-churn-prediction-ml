"""
Data validation module for customer churn prediction.

This module provides validation functions to ensure the customer churn dataset
meets required schema and data quality constraints before processing.
"""

import pandas as pd
from typing import List


def validate_dataframe(df: pd.DataFrame) -> None:
    """
    Validate that a DataFrame meets the required schema and quality constraints.
    
    This function checks:
    - DataFrame is not empty
    - Target column 'Churn' exists
    - Required columns are present
    - Critical columns do not contain null values
    
    Args:
        df: pandas DataFrame to validate
        
    Raises:
        ValueError: If validation fails, with a descriptive error message
        
    Returns:
        None: If validation passes, returns nothing
    """
    # Check if DataFrame is empty
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    # Define required columns
    required_columns = ['tenure', 'MonthlyCharges', 'TotalCharges']
    
    # Check if target column exists
    if 'Churn' not in df.columns:
        raise ValueError("Target column 'Churn' is missing from the DataFrame")
    
    # Check if required columns exist
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Missing required columns: {', '.join(missing_columns)}"
        )
    
    # Check for null values in critical columns
    critical_columns = ['tenure', 'MonthlyCharges', 'TotalCharges']
    
    for column in critical_columns:
        if df[column].isnull().any():
            raise ValueError(
                f"Column '{column}' contains null values. "
                f"Critical columns must not have missing data."
            )
