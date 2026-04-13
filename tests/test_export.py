"""
Tests for model export script.

Validates that the export produces a correct artifact structure without
writing to the real models/ directory. All tests use pytest's tmp_path
fixture so no disk state leaks between runs.

These tests call build_artifact() / export_model() directly — no subprocess
spawning, no dependency on the real CSV being present at a fixed path.
"""

import os
import pytest
import joblib
import numpy as np


# ---------------------------------------------------------------------------
# Shared fixture — build and export the artifact once per test session to
# avoid re-running the 10-second training pipeline for every test.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def exported_artifact(tmp_path_factory):
    """Build and export the artifact to a temp directory exactly once.

    Returns both the artifact dict and the path where it was written.
    """
    # Use a session-scoped tmp dir so all tests in this module share the
    # same trained model without re-training for each test function.
    tmp = tmp_path_factory.mktemp("model_export")
    output_path = str(tmp / "churn_model.joblib")

    # Import here so that a missing dataset causes a clean skip, not a
    # collection-time import error.
    from scripts.export_model import export_model, DATASET_PATH

    if not os.path.exists(DATASET_PATH):
        pytest.skip(
            f"Real dataset not found at '{DATASET_PATH}'. "
            "Skipping export tests that require the CSV."
        )

    artifact = export_model(output_path=output_path)
    return artifact, output_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestExportedArtifactStructure:
    """Verify the artifact dict contains all required keys with correct types."""

    def test_exported_artifact_has_required_keys(self, exported_artifact):
        """Exported joblib file must contain all required keys."""
        artifact, path = exported_artifact
        loaded = joblib.load(path)

        required_keys = {
            "model",
            "feature_count",
            "model_version",
            "trained_on",
            "dataset_rows",
            "train_rows",
            "test_rows",
            "roc_auc",
            "accuracy",
        }

        missing = required_keys - set(loaded.keys())
        assert not missing, f"Artifact missing required keys: {missing}"

    def test_exported_feature_count_is_51(self, exported_artifact):
        """Feature count in artifact must be 51 (6 numeric + 45 OHE post-encoding on real data)."""
        artifact, path = exported_artifact
        loaded = joblib.load(path)
        assert loaded["feature_count"] == 51, (
            f"Expected feature_count=51, got {loaded['feature_count']}. "
            "This will break the FastAPI model contract."
        )

    def test_exported_model_can_predict(self, exported_artifact):
        """Loaded model must accept a 51-feature vector and return a prediction."""
        artifact, path = exported_artifact
        loaded = joblib.load(path)

        n_features = loaded["feature_count"]
        X = np.zeros((1, n_features))

        pred = loaded["model"].predict(X)
        assert pred.shape == (1,), (
            f"Expected prediction shape (1,), got {pred.shape}"
        )
        assert pred[0] in (0, 1), f"Expected binary prediction, got {pred[0]}"

    def test_exported_roc_auc_above_threshold(self, exported_artifact):
        """ROC-AUC must be above 0.80 — regression guard against data bugs."""
        artifact, path = exported_artifact
        loaded = joblib.load(path)

        roc_auc = loaded["roc_auc"]
        assert roc_auc > 0.80, (
            f"ROC-AUC {roc_auc:.4f} is below the 0.80 threshold. "
            "This indicates a data or preprocessing regression."
        )

    def test_models_directory_created_if_missing(self, tmp_path):
        """export_model() must create the output directory if it does not exist."""
        from scripts.export_model import export_model, DATASET_PATH

        if not os.path.exists(DATASET_PATH):
            pytest.skip(
                f"Real dataset not found at '{DATASET_PATH}'. "
                "Skipping directory-creation test."
            )

        # Deliberately target a nested subdirectory that does not yet exist
        new_dir = tmp_path / "nested" / "models"
        output_path = str(new_dir / "churn_model.joblib")

        assert not new_dir.exists(), "Directory should not exist before export"

        export_model(output_path=output_path)

        assert new_dir.exists(), "export_model() must create the output directory"
        assert os.path.isfile(output_path), "Joblib file must exist after export"
