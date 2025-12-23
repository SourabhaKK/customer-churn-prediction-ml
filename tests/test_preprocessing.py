"""
Unit tests for preprocessing module.

Tests validate the behavior of preprocess_data() function
which handles feature separation, encoding, and scaling for
the customer churn prediction pipeline.
"""

import pytest
import pandas as pd
import numpy as np
from src.preprocessing import preprocess_data


class TestPreprocessing:
    """Test suite for data preprocessing functionality."""

    def test_preprocess_data_returns_numpy_array(self):
        """Test that preprocessing returns a NumPy array."""
        # Arrange: Create a simple DataFrame with mixed types
        df = pd.DataFrame({
            'tenure': [12, 24, 6],
            'MonthlyCharges': [50.5, 75.2, 30.0],
            'Contract': ['Month-to-month', 'One year', 'Two year'],
            'Churn': ['No', 'Yes', 'No']
        })
        
        # Act
        result = preprocess_data(df)
        
        # Assert: Should return NumPy array
        assert isinstance(result, np.ndarray), "Output should be a NumPy array"

    def test_preprocess_data_separates_features_from_target(self):
        """Test that preprocessing excludes the target column 'Churn'."""
        # Arrange: Create DataFrame with target column
        df = pd.DataFrame({
            'tenure': [12, 24, 6],
            'MonthlyCharges': [50.5, 75.2, 30.0],
            'Contract': ['Month-to-month', 'One year', 'Two year'],
            'Churn': ['No', 'Yes', 'No']
        })
        
        # Act
        result = preprocess_data(df)
        
        # Assert: Number of columns should be features only (not including Churn)
        # After one-hot encoding 'Contract' (3 categories) and 2 numerical features
        # Expected: 2 numerical + 3 one-hot encoded = 5 features
        assert result.shape[1] == 5, "Should have 5 features after preprocessing"

    def test_preprocess_data_handles_categorical_encoding(self):
        """Test that categorical features are one-hot encoded."""
        # Arrange: Create DataFrame with categorical feature
        df = pd.DataFrame({
            'tenure': [12, 24, 6],
            'MonthlyCharges': [50.5, 75.2, 30.0],
            'Contract': ['Month-to-month', 'One year', 'Month-to-month'],
            'Churn': ['No', 'Yes', 'No']
        })
        
        # Act
        result = preprocess_data(df)
        
        # Assert: Should have more columns than original due to one-hot encoding
        # Original: 2 numerical + 1 categorical = 3 features (excluding Churn)
        # After encoding: 2 numerical + encoded categorical > 3
        assert result.shape[1] > 3, "One-hot encoding should increase feature count"

    def test_preprocess_data_scales_numerical_features(self):
        """Test that numerical features are scaled (standardized)."""
        # Arrange: Create DataFrame with numerical features
        df = pd.DataFrame({
            'tenure': [1, 50, 100],  # Wide range to test scaling
            'MonthlyCharges': [20.0, 50.0, 100.0],
            'Contract': ['Month-to-month', 'One year', 'Two year'],
            'Churn': ['No', 'Yes', 'No']
        })
        
        # Act
        result = preprocess_data(df)
        
        # Assert: Scaled features should have values roughly in range [-3, 3] for standard scaling
        # Check that values are not in original range
        numerical_cols = result[:, :2]  # First 2 columns are numerical
        assert np.max(numerical_cols) < 10, "Numerical features should be scaled"
        assert np.min(numerical_cols) > -10, "Numerical features should be scaled"

    def test_preprocess_data_consistent_output_shape(self):
        """Test that same input schema produces same output shape."""
        # Arrange: Create two DataFrames with same schema but different values
        df1 = pd.DataFrame({
            'tenure': [12, 24, 6],
            'MonthlyCharges': [50.5, 75.2, 30.0],
            'Contract': ['Month-to-month', 'One year', 'Two year'],
            'Churn': ['No', 'Yes', 'No']
        })
        
        df2 = pd.DataFrame({
            'tenure': [5, 15, 25],
            'MonthlyCharges': [40.0, 60.0, 80.0],
            'Contract': ['One year', 'Two year', 'Month-to-month'],
            'Churn': ['Yes', 'No', 'Yes']
        })
        
        # Act
        result1 = preprocess_data(df1)
        result2 = preprocess_data(df2)
        
        # Assert: Both should have same shape
        assert result1.shape == result2.shape, "Same schema should produce same output shape"

    def test_preprocess_data_preserves_row_count(self):
        """Test that preprocessing preserves the number of rows."""
        # Arrange: Create DataFrame with known number of rows
        df = pd.DataFrame({
            'tenure': [12, 24, 6, 36, 48],
            'MonthlyCharges': [50.5, 75.2, 30.0, 90.0, 100.0],
            'Contract': ['Month-to-month', 'One year', 'Two year', 'One year', 'Two year'],
            'Churn': ['No', 'Yes', 'No', 'Yes', 'No']
        })
        
        # Act
        result = preprocess_data(df)
        
        # Assert: Number of rows should be preserved
        assert result.shape[0] == 5, "Should preserve number of rows"

    def test_preprocess_data_handles_multiple_categorical_features(self):
        """Test that preprocessing handles multiple categorical columns."""
        # Arrange: Create DataFrame with multiple categorical features
        df = pd.DataFrame({
            'tenure': [12, 24, 6],
            'MonthlyCharges': [50.5, 75.2, 30.0],
            'Contract': ['Month-to-month', 'One year', 'Two year'],
            'PaymentMethod': ['Electronic check', 'Mailed check', 'Bank transfer'],
            'Churn': ['No', 'Yes', 'No']
        })
        
        # Act
        result = preprocess_data(df)
        
        # Assert: Should handle multiple categorical features
        # 2 numerical + 3 (Contract) + 3 (PaymentMethod) = 8 features
        assert result.shape[1] == 8, "Should encode all categorical features"

    def test_preprocess_data_handles_binary_categorical(self):
        """Test that preprocessing handles binary categorical features."""
        # Arrange: Create DataFrame with binary categorical feature
        df = pd.DataFrame({
            'tenure': [12, 24, 6],
            'MonthlyCharges': [50.5, 75.2, 30.0],
            'gender': ['Male', 'Female', 'Male'],
            'Churn': ['No', 'Yes', 'No']
        })
        
        # Act
        result = preprocess_data(df)
        
        # Assert: Binary categorical should be encoded
        # 2 numerical + 2 (gender one-hot) = 4 features
        assert result.shape[1] == 4, "Should encode binary categorical features"

    def test_preprocess_data_no_missing_values_in_output(self):
        """Test that preprocessing output contains no NaN values."""
        # Arrange: Create DataFrame
        df = pd.DataFrame({
            'tenure': [12, 24, 6],
            'MonthlyCharges': [50.5, 75.2, 30.0],
            'Contract': ['Month-to-month', 'One year', 'Two year'],
            'Churn': ['No', 'Yes', 'No']
        })
        
        # Act
        result = preprocess_data(df)
        
        # Assert: No NaN values in output
        assert not np.isnan(result).any(), "Output should not contain NaN values"

    def test_preprocess_data_output_is_numeric(self):
        """Test that all output values are numeric."""
        # Arrange: Create DataFrame with mixed types
        df = pd.DataFrame({
            'tenure': [12, 24, 6],
            'MonthlyCharges': [50.5, 75.2, 30.0],
            'Contract': ['Month-to-month', 'One year', 'Two year'],
            'Churn': ['No', 'Yes', 'No']
        })
        
        # Act
        result = preprocess_data(df)
        
        # Assert: All values should be numeric (float or int)
        assert np.issubdtype(result.dtype, np.number), "All output values should be numeric"
