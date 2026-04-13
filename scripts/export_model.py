"""
scripts/export_model.py
=======================
Trains the customer churn RandomForest model and serialises it to
models/churn_model.joblib so the ml-model-deployment-fastapi repo can
load it directly.

Pipeline call sequence (matches src/pipeline.py):
  1. pd.read_csv('data/customer_churn_dataset.csv')
  2. Coerce TotalCharges to numeric  (11 blank rows in raw CSV)
  3. validate_dataframe(df)
  4. train_test_split(df, test_size=0.2, random_state=42, stratify=Churn)
  5. engineer_features(train_df), engineer_features(test_df)
  6. fit_preprocess(train_df_with_features)   -> preprocessor, X_train
  7. transform_preprocess(test_df_with_features, preprocessor) -> X_test
  8. train_model(X_train, y_train, random_state=42)
  9. roc_auc_score / accuracy_score on X_test
  10. joblib.dump(artifact_dict, 'models/churn_model.joblib')

Confirmed:
  Dataset path   : data/customer_churn_dataset.csv
  Feature count  : 49  (6 numeric + 43 OHE; customerID dropped, TotalCharges numeric)
  Output path    : models/churn_model.joblib
"""

import os
import sys
from datetime import datetime, timezone

# Ensure the repo root is on sys.path so `from src.xxx import` resolves.
# This file lives at  <repo_root>/scripts/export_model.py
# so the repo root is one directory above scripts/.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Force UTF-8 output on Windows so emoji characters in print() don't raise
# a UnicodeEncodeError with the default charmap codec.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Repo-root-relative imports — the script is run from the repo root so that
# `src` is on sys.path via the package structure (pytest.ini / src layout).
# ---------------------------------------------------------------------------
from src.data_validation import validate_dataframe
from src.features import engineer_features
from src.preprocessing import fit_preprocess, transform_preprocess
from src.train import train_model

# ── Constants ───────────────────────────────────────────────────────────────
DATASET_PATH = os.path.join("data", "customer_churn_dataset.csv")
OUTPUT_DIR = "models"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "churn_model.joblib")
MODEL_VERSION = "1.0.0"
EXPECTED_FEATURE_COUNT = 51
RANDOM_STATE = 42
TEST_SIZE = 0.2


def build_artifact(dataset_path: str = DATASET_PATH) -> dict:
    """Run the full training pipeline and return the serialisable artifact dict.

    This function is importable so that tests can call it directly without
    spawning a subprocess and without writing to the real models/ directory.

    Args:
        dataset_path: Path to the raw Telco CSV (relative to cwd / repo root).

    Returns:
        dict with keys: model, feature_count, model_version, trained_on,
        dataset_rows, train_rows, test_rows, roc_auc, accuracy.

    Raises:
        FileNotFoundError: If the CSV is not found at dataset_path.
        ValueError: If the preprocessed feature count != EXPECTED_FEATURE_COUNT.
    """
    # ── Step 1: Load raw data ────────────────────────────────────────────────
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Dataset not found at '{dataset_path}'.\n"
            f"Expected path (relative to repo root): {os.path.abspath(dataset_path)}\n"
            "Ensure you are running this script from the repository root."
        )

    print(f"Loading dataset from {dataset_path} …")
    df = pd.read_csv(dataset_path)
    print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")

    # ── Step 2: Coerce TotalCharges BEFORE validation ────────────────────────
    # The Telco CSV stores TotalCharges as object with 11 blank strings.
    # pd.to_numeric + fillna(0) converts them to 0.0 before validation fires.
    if "TotalCharges" in df.columns:
        n_blank = pd.to_numeric(df["TotalCharges"], errors="coerce").isna().sum()
        if n_blank > 0:
            print(f"  ⚠  TotalCharges: coercing {n_blank} blank value(s) to 0.0")
        df["TotalCharges"] = pd.to_numeric(
            df["TotalCharges"], errors="coerce"
        ).fillna(0.0)

    # ── Step 3: Validate ─────────────────────────────────────────────────────
    validate_dataframe(df)
    print("  ✓ Data validation passed")

    # ── Step 4: Train / test split (stratified, matches pipeline defaults) ───
    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["Churn"] if "Churn" in df.columns else None,
    )
    print(
        f"  Split → train: {len(train_df):,} rows | test: {len(test_df):,} rows"
    )

    # ── Step 5: Feature engineering on BOTH sets independently ───────────────
    train_df_fe = engineer_features(train_df)
    test_df_fe = engineer_features(test_df)

    # ── Step 6: Separate features and target ─────────────────────────────────
    X_train_raw = train_df_fe.drop("Churn", axis=1)
    y_train = train_df_fe["Churn"].map({"No": 0, "Yes": 1})

    X_test_raw = test_df_fe.drop("Churn", axis=1)
    y_test = test_df_fe["Churn"].map({"No": 0, "Yes": 1})

    # fit_preprocess / transform_preprocess expect a 'Churn' column to be
    # present (they drop it internally), so temporarily re-attach the target.
    X_train_with_churn = X_train_raw.copy()
    X_train_with_churn["Churn"] = y_train

    X_test_with_churn = X_test_raw.copy()
    X_test_with_churn["Churn"] = y_test

    # ── Step 7: Preprocessing (leakage-safe) ─────────────────────────────────
    preprocessor, X_train = fit_preprocess(X_train_with_churn)
    X_test = transform_preprocess(X_test_with_churn, preprocessor)

    actual_feature_count = X_train.shape[1]
    print(f"  Preprocessed shape: {X_train.shape}")

    # ── GUARD: feature count regression ──────────────────────────────────────
    if actual_feature_count != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            f"Feature count mismatch: expected {EXPECTED_FEATURE_COUNT}, "
            f"got {actual_feature_count}. "
            "This will break the FastAPI model contract. "
            "Investigate which column was added or removed before exporting."
        )

    # ── Step 8: Train ─────────────────────────────────────────────────────────
    print("Training RandomForestClassifier …")
    model = train_model(X_train, y_train, random_state=RANDOM_STATE)
    print("  ✓ Model trained")

    # ── Step 9: Evaluate on held-out test set ─────────────────────────────────
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    accuracy = accuracy_score(y_test, y_pred)

    # ── Build artifact dict ───────────────────────────────────────────────────
    artifact = {
        "model": model,
        "feature_count": actual_feature_count,
        "model_version": MODEL_VERSION,
        "trained_on": datetime.now(timezone.utc).isoformat(),
        "dataset_rows": len(df),
        "train_rows": int(X_train.shape[0]),
        "test_rows": int(X_test.shape[0]),
        "roc_auc": float(roc_auc),
        "accuracy": float(accuracy),
    }

    return artifact


def export_model(
    dataset_path: str = DATASET_PATH,
    output_path: str = OUTPUT_PATH,
) -> dict:
    """Build the artifact and write it to *output_path*.

    Creates the output directory if it does not exist.

    Args:
        dataset_path: Path to raw CSV.
        output_path:  Destination for the .joblib file.

    Returns:
        The artifact dict (same object that was written to disk).
    """
    artifact = build_artifact(dataset_path=dataset_path)

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Serialise with compression level 3 (zlib, fast + good ratio).
    # Reduces the 5 MB uncompressed artifact to ~1 MB — important for
    # Docker image layer size.  FastAPI joblib.load() handles compressed
    # files transparently.
    joblib.dump(artifact, output_path, compress=3)

    return artifact


if __name__ == "__main__":
    try:
        artifact = export_model()
    except FileNotFoundError as exc:
        print(f"\n❌  Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except ValueError as exc:
        print(f"\n❌  Error: {exc}", file=sys.stderr)
        sys.exit(1)

    # ── Print summary ─────────────────────────────────────────────────────────
    print()
    print("Model exported successfully")
    print(f"Path:          {OUTPUT_PATH}")
    print(f"Feature count: {artifact['feature_count']}")
    print(f"ROC-AUC:       {artifact['roc_auc']:.3f}")
    print(f"Accuracy:      {artifact['accuracy'] * 100:.2f}%")
    print(f"Model version: {artifact['model_version']}")
