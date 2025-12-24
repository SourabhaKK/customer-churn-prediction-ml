"""
Unit tests for model training module.

Tests validate the behavior of train_model() function
which trains a machine learning model for customer churn prediction.
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from src.train import train_model


class TestModelTraining:
    """Test suite for model training functionality."""

    @pytest.fixture
    def sample_data(self):
        """Create sample training data for tests."""
        # Create simple binary classification data
        X = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'feature2': [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
            'feature3': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        })
        y = pd.Series([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        return X, y

    def test_train_model_returns_fitted_model(self, sample_data):
        """Test that train_model returns a fitted model object."""
        # Arrange
        X, y = sample_data
        
        # Act
        model = train_model(X, y)
        
        # Assert: Should return a fitted sklearn estimator
        assert isinstance(model, BaseEstimator), "Should return a sklearn estimator"
        assert hasattr(model, 'predict'), "Model should have predict method"

    def test_train_model_completes_without_errors(self, sample_data):
        """Test that training completes without raising exceptions."""
        # Arrange
        X, y = sample_data
        
        # Act & Assert: Should not raise any exception
        try:
            model = train_model(X, y)
            assert model is not None
        except Exception as e:
            pytest.fail(f"Training raised an exception: {e}")

    def test_train_model_can_make_predictions(self, sample_data):
        """Test that trained model can make predictions."""
        # Arrange
        X, y = sample_data
        
        # Act
        model = train_model(X, y)
        predictions = model.predict(X)
        
        # Assert: Predictions should have correct shape
        assert len(predictions) == len(y), "Predictions should match input length"
        assert all(pred in [0, 1] for pred in predictions), "Predictions should be binary"

    def test_train_model_reproducible_with_random_state(self, sample_data):
        """Test that model training is reproducible with fixed random state."""
        # Arrange
        X, y = sample_data
        
        # Act: Train model twice with same data
        model1 = train_model(X, y, random_state=42)
        model2 = train_model(X, y, random_state=42)
        
        predictions1 = model1.predict(X)
        predictions2 = model2.predict(X)
        
        # Assert: Predictions should be identical
        np.testing.assert_array_equal(predictions1, predictions2,
                                     err_msg="Model should be reproducible with fixed random_state")

    def test_train_model_different_random_states_may_differ(self, sample_data):
        """Test that different random states can produce different results."""
        # Arrange
        X, y = sample_data
        
        # Act: Train with different random states
        model1 = train_model(X, y, random_state=42)
        model2 = train_model(X, y, random_state=123)
        
        # Assert: Models should exist (may or may not have different predictions)
        assert model1 is not None
        assert model2 is not None

    def test_train_model_handles_dataframe_input(self, sample_data):
        """Test that model accepts pandas DataFrame as input."""
        # Arrange
        X, y = sample_data
        
        # Act
        model = train_model(X, y)
        
        # Assert: Should handle DataFrame input
        assert model is not None
        predictions = model.predict(X)
        assert len(predictions) == len(X)

    def test_train_model_handles_numpy_array_input(self, sample_data):
        """Test that model accepts numpy arrays as input."""
        # Arrange
        X, y = sample_data
        X_array = X.values
        y_array = y.values
        
        # Act
        model = train_model(X_array, y_array)
        
        # Assert: Should handle numpy array input
        assert model is not None
        predictions = model.predict(X_array)
        assert len(predictions) == len(y_array)

    def test_train_model_handles_small_dataset(self):
        """Test that model can train on small datasets."""
        # Arrange: Very small dataset
        X = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [5, 4, 3, 2, 1]
        })
        y = pd.Series([0, 0, 1, 1, 1])
        
        # Act
        model = train_model(X, y)
        
        # Assert: Should handle small dataset
        assert model is not None
        predictions = model.predict(X)
        assert len(predictions) == 5

    def test_train_model_handles_multiple_features(self):
        """Test that model handles datasets with many features."""
        # Arrange: Dataset with multiple features
        np.random.seed(42)
        X = pd.DataFrame(np.random.rand(20, 10))  # 20 samples, 10 features
        y = pd.Series(np.random.randint(0, 2, 20))  # Binary target
        
        # Act
        model = train_model(X, y)
        
        # Assert: Should handle multiple features
        assert model is not None
        predictions = model.predict(X)
        assert len(predictions) == 20

    def test_train_model_returns_model_with_classes(self, sample_data):
        """Test that trained model has classes_ attribute."""
        # Arrange
        X, y = sample_data
        
        # Act
        model = train_model(X, y)
        
        # Assert: Classifier should have classes_ attribute
        assert hasattr(model, 'classes_'), "Model should have classes_ attribute"
        assert len(model.classes_) == 2, "Should recognize 2 classes for binary classification"

    def test_train_model_predictions_are_valid_classes(self, sample_data):
        """Test that predictions are from valid class labels."""
        # Arrange
        X, y = sample_data
        
        # Act
        model = train_model(X, y)
        predictions = model.predict(X)
        
        # Assert: All predictions should be valid class labels
        valid_classes = set(y.unique())
        assert all(pred in valid_classes for pred in predictions), \
            "All predictions should be from training classes"

    def test_train_model_can_predict_probabilities(self, sample_data):
        """Test that model can predict class probabilities."""
        # Arrange
        X, y = sample_data
        
        # Act
        model = train_model(X, y)
        
        # Assert: Should have predict_proba method
        assert hasattr(model, 'predict_proba'), "Model should support probability predictions"
        probabilities = model.predict_proba(X)
        assert probabilities.shape == (len(X), 2), "Should return probabilities for both classes"
        assert np.allclose(probabilities.sum(axis=1), 1.0), "Probabilities should sum to 1"
