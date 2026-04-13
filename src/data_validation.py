"""
Data validation module for customer churn prediction.

This module provides validation functions to ensure the customer churn dataset
meets required schema and data quality constraints before processing.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def validate_dataframe(df: pd.DataFrame) -> None:
    """
    Validate that a DataFrame meets the required schema and quality constraints.

    This function checks:
    - DataFrame is not empty
    - Target column 'Churn' exists
    - Required columns are present
    - TotalCharges is coercible to numeric (blank strings in the Telco CSV
      cause silent NaN propagation if not caught here).
    - Critical numeric columns do not contain null values after coercion

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

    # CRITICAL: TotalCharges arrives as dtype=object from the Telco CSV because
    # 11 rows contain blank strings instead of numbers.  Coerce to numeric and
    # count how many rows become NaN — any non-zero count is a data quality
    # failure that must be surfaced before preprocessing.
    total_charges_numeric = pd.to_numeric(df['TotalCharges'], errors='coerce')
    blank_count = total_charges_numeric.isna().sum()
    if blank_count > 0:
        logger.warning(
            "TotalCharges contains %d non-numeric value(s) (blank strings). "
            "These rows will be treated as errors.",
            blank_count,
        )
        raise ValueError(
            f"Column 'TotalCharges' contains {blank_count} non-numeric "
            f"value(s) (e.g. blank strings). "
            f"Run pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0) "
            f"before calling validate_dataframe()."
        )

    # Check for null values in critical numeric columns
    critical_columns = ['tenure', 'MonthlyCharges', 'TotalCharges']

    for column in critical_columns:
        null_count = df[column].isnull().sum()
        if null_count > 0:
            raise ValueError(
                f"Column '{column}' contains {null_count} null values. "
                f"Critical columns must not have missing data."
            )
