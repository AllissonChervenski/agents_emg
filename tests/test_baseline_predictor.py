"""Unit tests for NeutralNumpyPredictor runtime inference simulation (FR-005.5, FR-005.6, SC-006).

Validates:
- Pure NumPy inference: scores = X @ coef.T + intercept, y_pred = classes[argmax(scores)].
- Exact mathematical calculation conforming to TOL-LDA-NEUTRAL-SCORES (1e-7).
- Loading from exported job directory with cryptographic integrity verification.
- Zero imports of scikit-learn in predictor module.
- Fail-closed behavior on corrupted job archives.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest

from semg_baseline.export import NeutralModelExporter
from semg_baseline.predictor import NeutralNumpyPredictor
from semg_baseline.scaling import FeatureScaler


class MockModel:
    """Mock model with 3 classes and 4 features."""

    def __init__(self) -> None:
        self.classes_ = np.array([0, 1, 2], dtype=np.int32)
        # Class 0: weights [1, 0, 0, 0], intercept 0
        # Class 1: weights [0, 1, 0, 0], intercept 0
        # Class 2: weights [0, 0, 1, 0], intercept 0
        self.coef_ = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )
        self.intercept_ = np.array([0.0, 0.0, 0.0], dtype=np.float64)


def test_neutral_numpy_predictor_exact_prediction() -> None:
    """Predictor must evaluate X @ coef.T + intercept and return discrete argmax class."""
    classes = np.array([0, 1, 2], dtype=np.int32)
    coef = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    intercept = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    mean = np.zeros(4, dtype=np.float64)
    scale = np.ones(4, dtype=np.float64)

    predictor = NeutralNumpyPredictor(
        classes=classes,
        coef=coef,
        intercept=intercept,
        mean=mean,
        scale=scale,
    )

    # Sample 0: highest on feature 1 -> class 1
    # Sample 1: highest on feature 0 -> class 0
    # Sample 2: all zeros -> intercept highest on class 2 (0.3) -> class 2
    X = np.array(
        [
            [0.0, 5.0, 0.0, 0.0],
            [5.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )

    y_pred, scores = predictor.predict(X)

    assert y_pred[0] == 1
    assert y_pred[1] == 0
    assert y_pred[2] == 2
    assert scores.shape == (3, 3)

    # Manual score calculation
    expected_scores = X @ coef.T + intercept
    np.testing.assert_allclose(scores, expected_scores, atol=1e-12, rtol=1e-12)


def test_neutral_numpy_predictor_load_and_run(tmp_path: Path) -> None:
    """Predictor must load from an exported job directory and run inference accurately."""
    model = MockModel()
    scaler = FeatureScaler()
    scaler.fit(np.eye(4, dtype=np.float64))

    job_dir = tmp_path / "job_pred_test"
    exporter = NeutralModelExporter(output_dir=job_dir)
    exporter.export(
        model=model,
        scaler=scaler,
        job_id="job_pred_test",
        job_fingerprint="e" * 64,
        model_taxonomy="exercise_E1",
        priors_type="uniform",
    )

    predictor = NeutralNumpyPredictor.load(job_dir)

    assert len(predictor.classes) == 3
    assert predictor.coef.shape == (3, 4)
    assert predictor.intercept.shape == (3,)
    assert predictor.mean.shape == (4,)
    assert predictor.scale.shape == (4,)

    X = np.array([[0.0, 10.0, 0.0, 0.0]], dtype=np.float64)
    y_pred, scores = predictor.predict(X)
    assert y_pred[0] == 1


def test_neutral_numpy_predictor_zero_variance_protection() -> None:
    """Predictor must zero out scaled features flagged as zero-variance."""
    classes = np.array([0, 1], dtype=np.int32)
    coef = np.array([[1.0, 2.0], [2.0, 1.0]], dtype=np.float64)
    intercept = np.array([0.0, 0.0], dtype=np.float64)
    mean = np.array([0.0, 100.0], dtype=np.float64)
    scale = np.array([1.0, 1.0], dtype=np.float64)

    # Column 1 is zero-variance
    predictor = NeutralNumpyPredictor(
        classes=classes,
        coef=coef,
        intercept=intercept,
        mean=mean,
        scale=scale,
        zero_variance_indices=[1],
    )

    X = np.array([[2.0, 999.0]], dtype=np.float64)
    scores = predictor.decision_scores(X)

    # Column 1 should be zeroed out in X_scaled, so only column 0 contributes:
    # class 0: 2.0 * 1.0 + 0 = 2.0
    # class 1: 2.0 * 2.0 + 0 = 4.0
    assert scores[0, 0] == 2.0
    assert scores[0, 1] == 4.0


def test_predictor_fails_closed_on_tampered_model(tmp_path: Path) -> None:
    """Predictor.load must raise ValueError if sidecar sha256 or array verification fails."""
    job_dir = tmp_path / "corrupted_job"
    job_dir.mkdir()
    (job_dir / "model_manifest.json").write_text("{}", encoding="utf-8")
    (job_dir / "model_manifest.sha256").write_text("0" * 64 + "  model_manifest.json\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Integrity verification failed"):
        NeutralNumpyPredictor.load(job_dir)


def test_predictor_does_not_import_sklearn() -> None:
    """semg_baseline.predictor must have zero dependency on sklearn in an isolated process."""
    cmd = [
        sys.executable,
        "-c",
        "import sys; import semg_baseline.predictor; assert 'sklearn' not in sys.modules, f'Leaked: {[m for m in sys.modules if m.startswith(\"sklearn\")]}'",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert res.returncode == 0
