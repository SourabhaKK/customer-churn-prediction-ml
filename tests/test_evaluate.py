"""
Unit tests for model evaluation module.

Tests validate the behavior of evaluate_model() function
which computes evaluation metrics for trained models.
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from src.evaluate import evaluate_model


class TestModelEvaluation:
    """Test suite for model evaluation functionality."""

    @pytest.fixture
    def trained_model_and_data(self):
        """Create a trained model and test data for evaluation."""
        # Create simple training data
        X_train = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5, 6, 7, 8],
            'feature2': [8, 7, 6, 5, 4, 3, 2, 1]
        })
        y_train = pd.Series([0, 0, 0, 0, 1, 1, 1, 1])
        
        # Train a simple model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        # Create test data
        X_test = pd.DataFrame({
            'feature1': [2, 3, 6, 7],
            'feature2': [7, 6, 3, 2]
        })
        y_test = pd.Series([0, 0, 1, 1])
        
        return model, X_test, y_test

    def test_evaluate_model_returns_dictionary(self, trained_model_and_data):
        """Test that evaluate_model returns a dictionary."""
        # Arrange
        model, X_test, y_test = trained_model_and_data
        
        # Act
        result = evaluate_model(model, X_test, y_test)
        
        # Assert
        assert isinstance(result, dict), "evaluate_model should return a dictionary"

    def test_evaluate_model_contains_required_keys(self, trained_model_and_data):
        """Test that result contains all required metric keys."""
        # Arrange
        model, X_test, y_test = trained_model_and_data
        
        # Act
        result = evaluate_model(model, X_test, y_test)
        
        # Assert: Check for required keys
        required_keys = ['roc_auc', 'precision', 'recall', 'confusion_matrix']
        for key in required_keys:
            assert key in result, f"Result should contain '{key}' key"

    def test_evaluate_model_roc_auc_is_float(self, trained_model_and_data):
        """Test that roc_auc is a float value."""
        # Arrange
        model, X_test, y_test = trained_model_and_data
        
        # Act
        result = evaluate_model(model, X_test, y_test)
        
        # Assert
        assert isinstance(result['roc_auc'], (float, np.floating)), \
            "roc_auc should be a float"

    def test_evaluate_model_precision_is_float(self, trained_model_and_data):
        """Test that precision is a float value."""
        # Arrange
        model, X_test, y_test = trained_model_and_data
        
        # Act
        result = evaluate_model(model, X_test, y_test)
        
        # Assert
        assert isinstance(result['precision'], (float, np.floating)), \
            "precision should be a float"

    def test_evaluate_model_recall_is_float(self, trained_model_and_data):
        """Test that recall is a float value."""
        # Arrange
        model, X_test, y_test = trained_model_and_data
        
        # Act
        result = evaluate_model(model, X_test, y_test)
        
        # Assert
        assert isinstance(result['recall'], (float, np.floating)), \
            "recall should be a float"

    def test_evaluate_model_confusion_matrix_is_array(self, trained_model_and_data):
        """Test that confusion_matrix is a numpy array."""
        # Arrange
        model, X_test, y_test = trained_model_and_data
        
        # Act
        result = evaluate_model(model, X_test, y_test)
        
        # Assert
        assert isinstance(result['confusion_matrix'], np.ndarray), \
            "confusion_matrix should be a numpy array"

    def test_evaluate_model_confusion_matrix_shape(self, trained_model_and_data):
        """Test that confusion_matrix has correct shape for binary classification."""
        # Arrange
        model, X_test, y_test = trained_model_and_data
        
        # Act
        result = evaluate_model(model, X_test, y_test)
        
        # Assert
        assert result['confusion_matrix'].shape == (2, 2), \
            "confusion_matrix should be 2x2 for binary classification"

    def test_evaluate_model_metrics_in_valid_range(self, trained_model_and_data):
        """Test that metric values are in valid ranges."""
        # Arrange
        model, X_test, y_test = trained_model_and_data
        
        # Act
        result = evaluate_model(model, X_test, y_test)
        
        # Assert: Metrics should be between 0 and 1
        assert 0 <= result['roc_auc'] <= 1, "roc_auc should be between 0 and 1"
        assert 0 <= result['precision'] <= 1, "precision should be between 0 and 1"
        assert 0 <= result['recall'] <= 1, "recall should be between 0 and 1"

    def test_evaluate_model_completes_without_errors(self, trained_model_and_data):
        """Test that evaluation completes without raising exceptions."""
        # Arrange
        model, X_test, y_test = trained_model_and_data
        
        # Act & Assert: Should not raise any exception
        try:
            result = evaluate_model(model, X_test, y_test)
            assert result is not None
        except Exception as e:
            pytest.fail(f"evaluate_model raised an exception: {e}")

    def test_evaluate_model_handles_numpy_arrays(self):
        """Test that evaluate_model handles numpy array inputs."""
        # Arrange: Create model and numpy arrays
        X_train = np.array([[1, 8], [2, 7], [3, 6], [4, 5], 
                           [5, 4], [6, 3], [7, 2], [8, 1]])
        y_train = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        X_test = np.array([[2, 7], [3, 6], [6, 3], [7, 2]])
        y_test = np.array([0, 0, 1, 1])
        
        # Act
        result = evaluate_model(model, X_test, y_test)
        
        # Assert
        assert result is not None
        assert 'roc_auc' in result

    def test_evaluate_model_handles_perfect_predictions(self):
        """Test evaluation with perfect predictions."""
        # Arrange: Create simple data where model will predict perfectly
        X_train = np.array([[1], [2], [3], [4]])
        y_train = np.array([0, 0, 1, 1])
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        # Test on same data (will predict perfectly)
        X_test = X_train
        y_test = y_train
        
        # Act
        result = evaluate_model(model, X_test, y_test)
        
        # Assert: Perfect predictions should give high scores
        assert result['precision'] >= 0.9, "Perfect predictions should have high precision"
        assert result['recall'] >= 0.9, "Perfect predictions should have high recall"

    def test_evaluate_model_confusion_matrix_values_are_integers(self, trained_model_and_data):
        """Test that confusion matrix contains integer counts."""
        # Arrange
        model, X_test, y_test = trained_model_and_data
        
        # Act
        result = evaluate_model(model, X_test, y_test)
        
        # Assert: Confusion matrix should contain integer counts
        cm = result['confusion_matrix']
        assert np.issubdtype(cm.dtype, np.integer), \
            "confusion_matrix should contain integer values"

    def test_evaluate_model_confusion_matrix_sum_equals_test_size(self, trained_model_and_data):
        """Test that confusion matrix sum equals number of test samples."""
        # Arrange
        model, X_test, y_test = trained_model_and_data
        
        # Act
        result = evaluate_model(model, X_test, y_test)
        
        # Assert: Sum of confusion matrix should equal test set size
        cm_sum = result['confusion_matrix'].sum()
        assert cm_sum == len(y_test), \
            "confusion_matrix sum should equal number of test samples"

    def test_evaluate_model_raises_error_on_none_model(self):
        """Test that evaluate_model raises error when model is None."""
        # Arrange
        X_test = np.array([[1, 2], [3, 4]])
        y_test = np.array([0, 1])
        
        # Act & Assert
        with pytest.raises((ValueError, AttributeError, TypeError)):
            evaluate_model(None, X_test, y_test)

    def test_evaluate_model_raises_error_on_none_inputs(self, trained_model_and_data):
        """Test that evaluate_model raises error when inputs are None."""
        # Arrange
        model, _, _ = trained_model_and_data
        
        # Act & Assert: None X_test
        with pytest.raises((ValueError, AttributeError, TypeError)):
            evaluate_model(model, None, np.array([0, 1]))
        
        # Act & Assert: None y_test
        with pytest.raises((ValueError, AttributeError, TypeError)):
            evaluate_model(model, np.array([[1, 2]]), None)
