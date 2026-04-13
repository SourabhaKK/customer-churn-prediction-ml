"""
Pipeline orchestration module for customer churn prediction.

This module provides end-to-end pipeline execution from raw data
to trained model and evaluation metrics.
"""

import logging

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from typing import Tuple, Dict, Any
from sklearn.base import BaseEstimator

from src.data_validation import validate_dataframe
from src.features import engineer_features
from src.preprocessing import fit_preprocess, transform_preprocess
from src.train import train_model
from src.predict import predict

logger = logging.getLogger(__name__)


def run_pipeline(
    data_path: str,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[BaseEstimator, Dict[str, Any]]:
    """
    Execute the complete ML pipeline from raw data to trained model.
    
    Pipeline Steps:
    1. Load raw CSV data
    2. Validate data schema and quality
    3. Split into train/test sets (BEFORE any preprocessing)
    4. Engineer features on both sets independently
    5. Preprocess train set (fit + transform)
    6. Preprocess test set (transform only, using train statistics)
    7. Train model on preprocessed training data
    8. Evaluate model on test set
    9. Return trained model and evaluation metrics
    
    Args:
        data_path: Path to raw CSV file
        test_size: Proportion of data for test set (default: 0.2)
        random_state: Random seed for reproducibility (default: 42)
        
    Returns:
        Tuple containing:
            - model: Trained sklearn model
            - metrics: Dictionary with evaluation metrics including:
                - accuracy, precision, recall, f1, roc_auc
                - confusion_matrix, classification_report
                
    Example:
        >>> model, metrics = run_pipeline('data/customer_churn_dataset.csv')
        >>> print(f"Accuracy: {metrics['accuracy']:.3f}")
        >>> print(f"ROC-AUC: {metrics['roc_auc']:.3f}")
    """
    # Step 1: Load raw data
    logger.info("Step 1: Loading raw data from %s", data_path)
    df_raw = pd.read_csv(data_path)
    logger.info("Loaded %d rows, %d columns", len(df_raw), len(df_raw.columns))

    # Step 1b: Coerce TotalCharges to numeric BEFORE validation.
    # The Telco CSV stores TotalCharges as object with 11 blank strings.
    # pd.to_numeric + fillna(0) converts them to 0.0 so that validate_dataframe
    # does not raise on legitimately fixable data quality issues.
    if 'TotalCharges' in df_raw.columns:
        n_blank = df_raw['TotalCharges'].eq('').sum() + pd.to_numeric(
            df_raw['TotalCharges'], errors='coerce'
        ).isna().sum()
        if n_blank > 0:
            logger.warning(
                "TotalCharges: coercing %d blank/non-numeric value(s) to 0.0",
                n_blank,
            )
        df_raw['TotalCharges'] = pd.to_numeric(
            df_raw['TotalCharges'], errors='coerce'
        ).fillna(0.0)

    # Step 2: Validate raw data
    logger.info("Step 2: Validating data schema and quality")
    validate_dataframe(df_raw)
    logger.info("Data validation passed")

    # Step 3: Split into train/test BEFORE any preprocessing
    # This prevents data leakage
    logger.info("Step 3: Splitting data (test_size=%s)", test_size)
    train_df, test_df = train_test_split(
        df_raw,
        test_size=test_size,
        random_state=random_state,
        stratify=df_raw['Churn'] if 'Churn' in df_raw.columns else None
    )
    logger.info("Train set: %d rows | Test set: %d rows", len(train_df), len(test_df))

    # Step 4: Engineer features on both sets
    logger.info("Step 4: Engineering features")
    train_df_features = engineer_features(train_df)
    test_df_features = engineer_features(test_df)
    logger.info("Features after engineering: %d columns", len(train_df_features.columns))

    # Step 5: Separate features and target
    logger.info("Step 5: Separating features and target")
    X_train_raw = train_df_features.drop('Churn', axis=1)
    y_train = train_df_features['Churn'].map({'No': 0, 'Yes': 1})

    X_test_raw = test_df_features.drop('Churn', axis=1)
    y_test = test_df_features['Churn'].map({'No': 0, 'Yes': 1})
    logger.info("Target distribution (train): %s", y_train.value_counts().to_dict())

    # Step 6: Preprocess data using leakage-safe functions
    logger.info("Step 6: Preprocessing features (fit on train only)")

    # Add 'Churn' column temporarily for preprocessing compatibility
    X_train_with_target = X_train_raw.copy()
    X_train_with_target['Churn'] = y_train

    X_test_with_target = X_test_raw.copy()
    X_test_with_target['Churn'] = y_test

    # Fit preprocessor on training data only
    preprocessor, X_train_processed = fit_preprocess(X_train_with_target)
    logger.info("Preprocessor fitted on training data")

    # Transform test data using training statistics (no refitting)
    X_test_processed = transform_preprocess(X_test_with_target, preprocessor)
    logger.info("Test data transformed using training statistics")

    logger.info("Processed shape: %s", X_train_processed.shape)

    # Step 7: Train model
    logger.info("Step 7: Training model")
    model = train_model(X_train_processed, y_train, random_state=random_state)
    logger.info("Model trained: %s", type(model).__name__)

    # Step 8: Evaluate model on test set
    logger.info("Step 8: Evaluating model on test set")
    predictions = predict(model, X_test_processed)
    y_pred = predictions['predictions']
    y_pred_proba = predictions['probabilities'][:, 1]  # Probability of class 1

    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred)
    }

    logger.info(
        "EVALUATION — accuracy=%.4f precision=%.4f recall=%.4f "
        "f1=%.4f roc_auc=%.4f",
        metrics['accuracy'], metrics['precision'], metrics['recall'],
        metrics['f1'], metrics['roc_auc'],
    )

    return model, metrics


if __name__ == "__main__":
    # Example usage
    model, metrics = run_pipeline('data/customer_churn_dataset.csv')
    print("\n✓ Pipeline completed successfully!")
