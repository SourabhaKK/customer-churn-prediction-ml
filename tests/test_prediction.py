"""
Unit tests for prediction interface module.

Tests validate the behavior of predict() function
which makes predictions using a trained model.
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from src.predict import predict


class TestPredictionInterface:
    """Test suite for prediction interface functionality."""

    @pytest.fixture
    def trained_model(self):
        """Create a simple trained model for testing."""
        # Create simple training data
        X_train = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [5, 4, 3, 2, 1]
        })
        y_train = pd.Series([0, 0, 1, 1, 1])
        
        # Train a simple model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        return model

    @pytest.fixture
    def single_input(self):
        """Create single row input for prediction."""
        return pd.DataFrame({
            'feature1': [3],
            'feature2': [3]
        })

    @pytest.fixture
    def batch_input(self):
        """Create batch input for prediction."""
        return pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [5, 4, 3, 2, 1]
        })

    def test_predict_returns_predictions(self, trained_model, single_input):
        """Test that predict returns prediction results."""
        # Act
        result = predict(trained_model, single_input)
        
        # Assert: Should return predictions
        assert result is not None, "Predict should return a result"

    def test_predict_single_input_returns_correct_shape(self, trained_model, single_input):
        """Test that single input returns single prediction."""
        # Act
        result = predict(trained_model, single_input)
        
        # Assert: Should return single prediction
        assert 'predictions' in result, "Result should contain predictions"
        assert len(result['predictions']) == 1, "Single input should return single prediction"

    def test_predict_batch_input_returns_correct_shape(self, trained_model, batch_input):
        """Test that batch input returns multiple predictions."""
        # Act
        result = predict(trained_model, batch_input)
        
        # Assert: Should return multiple predictions
        assert 'predictions' in result, "Result should contain predictions"
        assert len(result['predictions']) == 5, "Batch input should return multiple predictions"

    def test_predict_returns_probability_scores(self, trained_model, single_input):
        """Test that predict returns probability scores."""
        # Act
        result = predict(trained_model, single_input)
        
        # Assert: Should include probabilities
        assert 'probabilities' in result, "Result should contain probabilities"
        assert result['probabilities'] is not None, "Probabilities should not be None"

    def test_predict_probabilities_sum_to_one(self, trained_model, single_input):
        """Test that probability scores sum to 1 for each prediction."""
        # Act
        result = predict(trained_model, single_input)
        
        # Assert: Probabilities should sum to 1
        probs = result['probabilities']
        assert np.allclose(probs.sum(axis=1), 1.0), "Probabilities should sum to 1"

    def test_predict_probabilities_correct_shape(self, trained_model, batch_input):
        """Test that probabilities have correct shape (n_samples, n_classes)."""
        # Act
        result = predict(trained_model, batch_input)
        
        # Assert: Probabilities should have shape (5, 2) for binary classification
        probs = result['probabilities']
        assert probs.shape == (5, 2), "Probabilities should have shape (n_samples, n_classes)"

    def test_predict_returns_valid_class_labels(self, trained_model, single_input):
        """Test that predictions are valid class labels."""
        # Act
        result = predict(trained_model, single_input)
        
        # Assert: Predictions should be valid classes (0 or 1 for binary)
        predictions = result['predictions']
        assert all(pred in [0, 1] for pred in predictions), "Predictions should be valid class labels"

    def test_predict_handles_dataframe_input(self, trained_model):
        """Test that predict handles pandas DataFrame input."""
        # Arrange
        df_input = pd.DataFrame({
            'feature1': [2, 3],
            'feature2': [4, 3]
        })
        
        # Act
        result = predict(trained_model, df_input)
        
        # Assert: Should handle DataFrame
        assert result is not None
        assert len(result['predictions']) == 2

    def test_predict_handles_numpy_array_input(self, trained_model):
        """Test that predict handles numpy array input."""
        # Arrange
        array_input = np.array([[2, 4], [3, 3]])
        
        # Act
        result = predict(trained_model, array_input)
        
        # Assert: Should handle numpy array
        assert result is not None
        assert len(result['predictions']) == 2

    def test_predict_raises_error_on_invalid_input(self, trained_model):
        """Test that predict raises error on invalid input."""
        # Arrange: Invalid input (wrong number of features)
        invalid_input = pd.DataFrame({
            'feature1': [1, 2, 3]
            # Missing feature2
        })
        
        # Act & Assert: Should raise an error
        with pytest.raises((ValueError, KeyError, Exception)):
            predict(trained_model, invalid_input)

    def test_predict_raises_error_on_none_model(self, single_input):
        """Test that predict raises error when model is None."""
        # Act & Assert: Should raise an error
        with pytest.raises((ValueError, AttributeError, TypeError)):
            predict(None, single_input)

    def test_predict_raises_error_on_none_input(self, trained_model):
        """Test that predict raises error when input is None."""
        # Act & Assert: Should raise an error
        with pytest.raises((ValueError, AttributeError, TypeError)):
            predict(trained_model, None)

    def test_predict_returns_dict_structure(self, trained_model, single_input):
        """Test that predict returns a dictionary with expected keys."""
        # Act
        result = predict(trained_model, single_input)
        
        # Assert: Should be a dictionary with expected keys
        assert isinstance(result, dict), "Result should be a dictionary"
        assert 'predictions' in result, "Result should have 'predictions' key"
        assert 'probabilities' in result, "Result should have 'probabilities' key"

    def test_predict_consistent_results(self, trained_model, single_input):
        """Test that predict returns consistent results for same input."""
        # Act: Predict twice with same input
        result1 = predict(trained_model, single_input)
        result2 = predict(trained_model, single_input)
        
        # Assert: Results should be identical
        np.testing.assert_array_equal(result1['predictions'], result2['predictions'],
                                     err_msg="Predictions should be consistent")
        np.testing.assert_array_almost_equal(result1['probabilities'], result2['probabilities'],
                                            err_msg="Probabilities should be consistent")
