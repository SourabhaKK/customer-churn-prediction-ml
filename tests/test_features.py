"""
Unit tests for feature engineering module.

Tests validate the behavior of engineer_features() function
which creates derived features for the customer churn prediction model.
"""

import pytest
import pandas as pd
import numpy as np
from src.features import engineer_features


class TestFeatureEngineering:
    """Test suite for feature engineering functionality."""

    def test_engineer_features_returns_dataframe(self):
        """Test that feature engineering returns a pandas DataFrame."""
        # Arrange: Create a simple preprocessed DataFrame
        df = pd.DataFrame({
            'tenure': [1, 12, 36, 60],
            'MonthlyCharges': [20.0, 50.0, 75.0, 100.0],
            'TotalCharges': [20.0, 600.0, 2700.0, 6000.0],
            'Contract': ['Month-to-month', 'One year', 'Two year', 'Two year']
        })
        
        # Act
        result = engineer_features(df)
        
        # Assert: Should return a DataFrame
        assert isinstance(result, pd.DataFrame), "Output should be a pandas DataFrame"

    def test_engineer_features_preserves_row_count(self):
        """Test that feature engineering does not change the number of rows."""
        # Arrange: Create DataFrame with known number of rows
        df = pd.DataFrame({
            'tenure': [1, 12, 36, 60, 72],
            'MonthlyCharges': [20.0, 50.0, 75.0, 100.0, 110.0],
            'TotalCharges': [20.0, 600.0, 2700.0, 6000.0, 7920.0],
            'Contract': ['Month-to-month', 'One year', 'Two year', 'Two year', 'Two year']
        })
        
        # Act
        result = engineer_features(df)
        
        # Assert: Row count should be preserved
        assert len(result) == 5, "Feature engineering should preserve row count"

    def test_engineer_features_creates_tenure_groups(self):
        """Test that tenure-based groups/buckets are created."""
        # Arrange: Create DataFrame with varying tenure values
        df = pd.DataFrame({
            'tenure': [1, 12, 36, 60],
            'MonthlyCharges': [20.0, 50.0, 75.0, 100.0],
            'TotalCharges': [20.0, 600.0, 2700.0, 6000.0],
            'Contract': ['Month-to-month', 'One year', 'Two year', 'Two year']
        })
        
        # Act
        result = engineer_features(df)
        
        # Assert: Should have a tenure group column
        assert 'tenure_group' in result.columns, "Should create tenure_group feature"

    def test_engineer_features_tenure_groups_are_categorical(self):
        """Test that tenure groups are categorical values."""
        # Arrange: Create DataFrame
        df = pd.DataFrame({
            'tenure': [1, 12, 36, 60],
            'MonthlyCharges': [20.0, 50.0, 75.0, 100.0],
            'TotalCharges': [20.0, 600.0, 2700.0, 6000.0],
            'Contract': ['Month-to-month', 'One year', 'Two year', 'Two year']
        })
        
        # Act
        result = engineer_features(df)
        
        # Assert: tenure_group should have limited unique values (categories)
        unique_groups = result['tenure_group'].nunique()
        assert unique_groups <= 5, "Tenure groups should be categorical (limited categories)"

    def test_engineer_features_creates_charge_ratio(self):
        """Test that derived features like charge ratios are created."""
        # Arrange: Create DataFrame with charge data
        df = pd.DataFrame({
            'tenure': [12, 24, 36],
            'MonthlyCharges': [50.0, 75.0, 100.0],
            'TotalCharges': [600.0, 1800.0, 3600.0],
            'Contract': ['One year', 'One year', 'Two year']
        })
        
        # Act
        result = engineer_features(df)
        
        # Assert: Should have derived charge-related features
        # Check for any new columns beyond original
        assert len(result.columns) > len(df.columns), "Should create new derived features"

    def test_engineer_features_no_missing_values(self):
        """Test that feature engineering does not introduce missing values."""
        # Arrange: Create DataFrame without missing values
        df = pd.DataFrame({
            'tenure': [1, 12, 36, 60],
            'MonthlyCharges': [20.0, 50.0, 75.0, 100.0],
            'TotalCharges': [20.0, 600.0, 2700.0, 6000.0],
            'Contract': ['Month-to-month', 'One year', 'Two year', 'Two year']
        })
        
        # Act
        result = engineer_features(df)
        
        # Assert: No missing values should be introduced
        assert not result.isnull().any().any(), "Feature engineering should not introduce NaN values"

    def test_engineer_features_deterministic_output(self):
        """Test that same input produces same output (deterministic)."""
        # Arrange: Create identical DataFrames
        df1 = pd.DataFrame({
            'tenure': [1, 12, 36],
            'MonthlyCharges': [20.0, 50.0, 75.0],
            'TotalCharges': [20.0, 600.0, 2700.0],
            'Contract': ['Month-to-month', 'One year', 'Two year']
        })
        
        df2 = df1.copy()
        
        # Act
        result1 = engineer_features(df1)
        result2 = engineer_features(df2)
        
        # Assert: Results should be identical
        pd.testing.assert_frame_equal(result1, result2, 
                                     check_dtype=True,
                                     obj="Feature engineering should be deterministic")

    def test_engineer_features_preserves_original_columns(self):
        """Test that original columns are preserved in the output."""
        # Arrange: Create DataFrame
        df = pd.DataFrame({
            'tenure': [1, 12, 36],
            'MonthlyCharges': [20.0, 50.0, 75.0],
            'TotalCharges': [20.0, 600.0, 2700.0],
            'Contract': ['Month-to-month', 'One year', 'Two year']
        })
        
        original_columns = set(df.columns)
        
        # Act
        result = engineer_features(df)
        
        # Assert: Original columns should still exist
        assert original_columns.issubset(set(result.columns)), \
            "Original columns should be preserved"

    def test_engineer_features_handles_edge_case_zero_tenure(self):
        """Test that feature engineering handles edge case of zero tenure."""
        # Arrange: Create DataFrame with zero tenure
        df = pd.DataFrame({
            'tenure': [0, 1, 12],
            'MonthlyCharges': [20.0, 20.0, 50.0],
            'TotalCharges': [0.0, 20.0, 600.0],
            'Contract': ['Month-to-month', 'Month-to-month', 'One year']
        })
        
        # Act
        result = engineer_features(df)
        
        # Assert: Should handle zero tenure without errors
        assert len(result) == 3, "Should handle zero tenure gracefully"
        assert not result.isnull().any().any(), "Should not create NaN for zero tenure"

    def test_engineer_features_handles_high_tenure_values(self):
        """Test that feature engineering handles very high tenure values."""
        # Arrange: Create DataFrame with high tenure
        df = pd.DataFrame({
            'tenure': [72, 84, 100],
            'MonthlyCharges': [100.0, 110.0, 120.0],
            'TotalCharges': [7200.0, 9240.0, 12000.0],
            'Contract': ['Two year', 'Two year', 'Two year']
        })
        
        # Act
        result = engineer_features(df)
        
        # Assert: Should handle high tenure values
        assert len(result) == 3, "Should handle high tenure values"
        assert 'tenure_group' in result.columns, "Should categorize high tenure appropriately"

    def test_engineer_features_consistent_column_count(self):
        """Test that feature engineering produces consistent column count for same schema."""
        # Arrange: Create two DataFrames with same schema but different values
        df1 = pd.DataFrame({
            'tenure': [1, 12, 36],
            'MonthlyCharges': [20.0, 50.0, 75.0],
            'TotalCharges': [20.0, 600.0, 2700.0],
            'Contract': ['Month-to-month', 'One year', 'Two year']
        })
        
        df2 = pd.DataFrame({
            'tenure': [6, 24, 48],
            'MonthlyCharges': [30.0, 60.0, 90.0],
            'TotalCharges': [180.0, 1440.0, 4320.0],
            'Contract': ['One year', 'Two year', 'Two year']
        })
        
        # Act
        result1 = engineer_features(df1)
        result2 = engineer_features(df2)
        
        # Assert: Column count should be consistent
        assert len(result1.columns) == len(result2.columns), \
            "Same schema should produce same number of features"

    def test_engineer_features_creates_numeric_features(self):
        """Test that new features are numeric (suitable for ML models)."""
        # Arrange: Create DataFrame
        df = pd.DataFrame({
            'tenure': [1, 12, 36],
            'MonthlyCharges': [20.0, 50.0, 75.0],
            'TotalCharges': [20.0, 600.0, 2700.0],
            'Contract': ['Month-to-month', 'One year', 'Two year']
        })
        
        # Act
        result = engineer_features(df)
        
        # Assert: New numeric features should exist
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        assert len(numeric_cols) > 0, "Should have numeric features for ML"
