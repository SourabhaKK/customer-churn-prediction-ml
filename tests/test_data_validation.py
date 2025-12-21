"""
Unit tests for data validation module.

Tests validate the behavior of validate_dataframe() function
which ensures the customer churn dataset meets required schema
and data quality constraints.
"""

import pytest
import pandas as pd
from src.data_validation import validate_dataframe


class TestDataValidation:
    """Test suite for data validation functionality."""

    def test_validate_dataframe_with_valid_data(self):
        """Test that validation passes for a valid DataFrame with all required columns."""
        # Arrange: Create a valid DataFrame with all required columns
        valid_df = pd.DataFrame({
            'customerID': ['C001', 'C002', 'C003'],
            'gender': ['Male', 'Female', 'Male'],
            'SeniorCitizen': [0, 1, 0],
            'Partner': ['Yes', 'No', 'Yes'],
            'Dependents': ['No', 'Yes', 'No'],
            'tenure': [12, 24, 6],
            'PhoneService': ['Yes', 'Yes', 'No'],
            'MonthlyCharges': [50.5, 75.2, 30.0],
            'TotalCharges': [606.0, 1804.8, 180.0],
            'Churn': ['No', 'Yes', 'No']
        })
        
        # Act & Assert: Should not raise any exception
        validate_dataframe(valid_df)

    def test_validate_dataframe_missing_churn_column(self):
        """Test that validation fails when target column 'Churn' is missing."""
        # Arrange: Create DataFrame without Churn column
        df_no_churn = pd.DataFrame({
            'customerID': ['C001', 'C002'],
            'tenure': [12, 24],
            'MonthlyCharges': [50.5, 75.2],
            'TotalCharges': [606.0, 1804.8]
        })
        
        # Act & Assert: Should raise ValueError with descriptive message
        with pytest.raises(ValueError, match="Target column 'Churn' is missing"):
            validate_dataframe(df_no_churn)

    def test_validate_dataframe_missing_required_columns(self):
        """Test that validation fails when required columns are missing."""
        # Arrange: Create DataFrame missing critical columns
        df_missing_cols = pd.DataFrame({
            'customerID': ['C001', 'C002'],
            'Churn': ['No', 'Yes']
            # Missing: tenure, MonthlyCharges, TotalCharges
        })
        
        # Act & Assert: Should raise ValueError mentioning missing columns
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_dataframe(df_missing_cols)

    def test_validate_dataframe_null_in_tenure(self):
        """Test that validation fails when tenure column contains null values."""
        # Arrange: Create DataFrame with null in tenure
        df_null_tenure = pd.DataFrame({
            'customerID': ['C001', 'C002', 'C003'],
            'tenure': [12, None, 6],
            'MonthlyCharges': [50.5, 75.2, 30.0],
            'TotalCharges': [606.0, 1804.8, 180.0],
            'Churn': ['No', 'Yes', 'No']
        })
        
        # Act & Assert: Should raise ValueError about null values in tenure
        with pytest.raises(ValueError, match="tenure.*null values"):
            validate_dataframe(df_null_tenure)

    def test_validate_dataframe_null_in_monthly_charges(self):
        """Test that validation fails when MonthlyCharges column contains null values."""
        # Arrange: Create DataFrame with null in MonthlyCharges
        df_null_monthly = pd.DataFrame({
            'customerID': ['C001', 'C002', 'C003'],
            'tenure': [12, 24, 6],
            'MonthlyCharges': [50.5, None, 30.0],
            'TotalCharges': [606.0, 1804.8, 180.0],
            'Churn': ['No', 'Yes', 'No']
        })
        
        # Act & Assert: Should raise ValueError about null values in MonthlyCharges
        with pytest.raises(ValueError, match="MonthlyCharges.*null values"):
            validate_dataframe(df_null_monthly)

    def test_validate_dataframe_null_in_total_charges(self):
        """Test that validation fails when TotalCharges column contains null values."""
        # Arrange: Create DataFrame with null in TotalCharges
        df_null_total = pd.DataFrame({
            'customerID': ['C001', 'C002', 'C003'],
            'tenure': [12, 24, 6],
            'MonthlyCharges': [50.5, 75.2, 30.0],
            'TotalCharges': [606.0, None, 180.0],
            'Churn': ['No', 'Yes', 'No']
        })
        
        # Act & Assert: Should raise ValueError about null values in TotalCharges
        with pytest.raises(ValueError, match="TotalCharges.*null values"):
            validate_dataframe(df_null_total)

    def test_validate_dataframe_empty_dataframe(self):
        """Test that validation fails for an empty DataFrame."""
        # Arrange: Create empty DataFrame
        empty_df = pd.DataFrame()
        
        # Act & Assert: Should raise ValueError
        with pytest.raises(ValueError):
            validate_dataframe(empty_df)

    def test_validate_dataframe_with_extra_columns(self):
        """Test that validation passes even when DataFrame has extra columns."""
        # Arrange: Create DataFrame with all required columns plus extras
        df_with_extras = pd.DataFrame({
            'customerID': ['C001', 'C002'],
            'tenure': [12, 24],
            'MonthlyCharges': [50.5, 75.2],
            'TotalCharges': [606.0, 1804.8],
            'Churn': ['No', 'Yes'],
            'ExtraColumn1': ['A', 'B'],
            'ExtraColumn2': [1, 2]
        })
        
        # Act & Assert: Should not raise any exception
        validate_dataframe(df_with_extras)
