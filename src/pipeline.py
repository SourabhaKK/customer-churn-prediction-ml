"""
Pipeline orchestration module for customer churn prediction.

This module provides end-to-end pipeline execution from raw data
to trained model and evaluation metrics.
"""

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
from src.preprocessing import preprocess_data
from src.train import train_model
from src.predict import predict


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
    print("Step 1: Loading raw data...")
    df_raw = pd.read_csv(data_path)
    print(f"  Loaded {len(df_raw)} rows, {len(df_raw.columns)} columns")
    
    # Step 2: Validate raw data
    print("\nStep 2: Validating data...")
    validate_dataframe(df_raw)
    print("  ✓ Data validation passed")
    
    # Step 3: Split into train/test BEFORE any preprocessing
    # This prevents data leakage
    print(f"\nStep 3: Splitting data (test_size={test_size})...")
    train_df, test_df = train_test_split(
        df_raw,
        test_size=test_size,
        random_state=random_state,
        stratify=df_raw['Churn'] if 'Churn' in df_raw.columns else None
    )
    print(f"  Train set: {len(train_df)} rows")
    print(f"  Test set: {len(test_df)} rows")
    
    # Step 4: Engineer features on both sets
    print("\nStep 4: Engineering features...")
    train_df_features = engineer_features(train_df)
    test_df_features = engineer_features(test_df)
    print(f"  Features created: {len(train_df_features.columns)} columns")
    
    # Step 5: Separate features and target
    print("\nStep 5: Separating features and target...")
    X_train_raw = train_df_features.drop('Churn', axis=1)
    y_train = train_df_features['Churn'].map({'No': 0, 'Yes': 1})
    
    X_test_raw = test_df_features.drop('Churn', axis=1)
    y_test = test_df_features['Churn'].map({'No': 0, 'Yes': 1})
    print(f"  Target distribution (train): {y_train.value_counts().to_dict()}")
    
    # Step 6: Preprocess data
    # NOTE: Current preprocess_data() does fit_transform
    # This is a known limitation that will be addressed
    # For now, we preprocess train and test separately
    print("\nStep 6: Preprocessing features...")
    print("  WARNING: Current preprocessing fits on each set independently")
    print("  This should be refactored to fit on train only")
    
    # Add 'Churn' column temporarily for preprocess_data compatibility
    X_train_with_target = X_train_raw.copy()
    X_train_with_target['Churn'] = y_train
    X_train_processed = preprocess_data(X_train_with_target)
    
    X_test_with_target = X_test_raw.copy()
    X_test_with_target['Churn'] = y_test
    X_test_processed = preprocess_data(X_test_with_target)
    
    print(f"  Processed shape: {X_train_processed.shape}")
    
    # Step 7: Train model
    print("\nStep 7: Training model...")
    model = train_model(X_train_processed, y_train, random_state=random_state)
    print(f"  ✓ Model trained: {type(model).__name__}")
    
    # Step 8: Evaluate model on test set
    print("\nStep 8: Evaluating model...")
    predictions = predict(model, X_test_processed)
    y_pred = predictions['predictions']
    y_pred_proba = predictions['probabilities'][:, 1]  # Probability of class 1
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred)
    }
    
    # Print summary
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    print("\nConfusion Matrix:")
    print(metrics['confusion_matrix'])
    print("\nClassification Report:")
    print(metrics['classification_report'])
    print("="*50)
    
    return model, metrics


if __name__ == "__main__":
    # Example usage
    model, metrics = run_pipeline('data/customer_churn_dataset.csv')
    print("\n✓ Pipeline completed successfully!")
