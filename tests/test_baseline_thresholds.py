"""Unit tests for data-driven deadzone threshold calibration (FR-012.4, SC-002, Decision-2).

Validates:
- ThresholdCalibration contract and schema validation against training_job_schema.json.
- Exact per-channel RMS rest calculation on training rest windows.
- Deadzone epsilon computation: epsilon[ch] = R * RMS_rest[ch].
- Candidate grid enforcement R in {0.5, 1.0, 1.5, 2.0, 3.0}.
- Zero-variance channel guard (epsilon >= 1e-8).
- Selection via training-pool evaluation callback or default candidate.
"""

from __future__ import annotations

import json
from pathlib import Path
import jsonschema
import numpy as np
import pytest

from semg_baseline.thresholds import ThresholdCalibration, ThresholdCalibrator


REPO_ROOT = Path(__file__).resolve().parent.parent
TRAINING_JOB_SCHEMA_PATH = (
    REPO_ROOT
    / "specs"
    / "004-baseline-training-and-evaluation"
    / "contracts"
    / "training_job_schema.json"
)


def _load_threshold_schema() -> dict:
    with open(TRAINING_JOB_SCHEMA_PATH, "r", encoding="utf-8") as f:
        schema = json.load(f)
    return schema["$defs"]["ThresholdCalibration"]


def test_threshold_calibration_schema_validation() -> None:
    """ThresholdCalibration.to_dict() must strictly comply with JSON schema contract."""
    calib = ThresholdCalibration(
        r_multiplier=1.5,
        rms_rest_train=np.full(12, 0.02, dtype=np.float64),
        epsilon_channels=np.full(12, 0.03, dtype=np.float64),
        selection_method="training_pool_validation",
    )

    d = calib.to_dict()
    assert d["r_multiplier"] == 1.5
    assert d["selection_method"] == "training_pool_validation"
    assert len(d["rms_rest_train"]) == 12
    assert len(d["epsilon_channels"]) == 12

    schema = _load_threshold_schema()
    jsonschema.validate(instance=d, schema=schema)


def test_rest_rms_and_epsilon_exact_calculation() -> None:
    """compute_rest_rms and epsilon_channels must be mathematically exact."""
    # Create 5 rest windows with constant amplitude 2.0 on all 12 channels
    # and 5 active windows with amplitude 10.0
    rest_windows = np.full((5, 400, 12), 2.0, dtype=np.float32)
    active_windows = np.full((5, 400, 12), 10.0, dtype=np.float32)
    train_windows = np.concatenate([rest_windows, active_windows], axis=0)

    train_labels = np.array([0] * 5 + [1] * 5, dtype=np.int16)

    calibrator = ThresholdCalibrator(candidate_grid=(0.5, 1.0, 1.5, 2.0, 3.0))
    calib = calibrator.calibrate(
        train_emg=train_windows,
        train_labels=train_labels,
        rest_label=0,
        r_multiplier=1.5,
    )

    np.testing.assert_allclose(calib.rms_rest_train, 2.0, atol=1e-12)
    np.testing.assert_allclose(calib.epsilon_channels, 3.0, atol=1e-12)
    assert calib.r_multiplier == 1.5


def test_silent_channel_zero_variance_protection() -> None:
    """A completely silent channel (all zeros) must be guarded against epsilon=0."""
    rest_windows = np.zeros((3, 400, 12), dtype=np.float32)
    # Channel 0 has amplitude 1.0, channel 1..11 are exactly 0.0
    rest_windows[:, :, 0] = 1.0
    train_labels = np.zeros(3, dtype=np.int16)

    calibrator = ThresholdCalibrator()
    calib = calibrator.calibrate(
        train_emg=rest_windows,
        train_labels=train_labels,
        rest_label=0,
        r_multiplier=1.0,
    )

    assert calib.rms_rest_train[0] == 1.0
    assert calib.epsilon_channels[0] == 1.0

    # Silent channels must have minimum guard 1e-8
    for ch in range(1, 12):
        assert calib.rms_rest_train[ch] >= 1e-8
        assert calib.epsilon_channels[ch] >= 1e-8


def test_candidate_grid_validation() -> None:
    """Passing an invalid R multiplier not in the candidate grid must raise ValueError."""
    calibrator = ThresholdCalibrator(candidate_grid=(0.5, 1.0, 1.5, 2.0, 3.0))
    rest_windows = np.ones((2, 400, 12), dtype=np.float32)
    train_labels = np.zeros(2, dtype=np.int16)

    with pytest.raises(ValueError, match="not in candidate grid"):
        calibrator.calibrate(
            train_emg=rest_windows,
            train_labels=train_labels,
            rest_label=0,
            r_multiplier=2.5,  # 2.5 is not in candidate grid
        )


def test_missing_rest_windows_raises_value_error() -> None:
    """If the training pool contains no REST windows, calibration must fail closed."""
    windows = np.ones((5, 400, 12), dtype=np.float32)
    labels = np.array([1, 2, 3, 4, 5], dtype=np.int16)  # No label 0

    calibrator = ThresholdCalibrator()
    with pytest.raises(ValueError, match="No REST windows found"):
        calibrator.calibrate(train_emg=windows, train_labels=labels, rest_label=0)


def test_selection_via_evaluation_callback() -> None:
    """Calibrator must evaluate candidate grid and select optimal R from callback."""
    rest_windows = np.full((4, 400, 12), 1.0, dtype=np.float32)
    train_labels = np.zeros(4, dtype=np.int16)

    # Mock score function that favors R = 2.0
    def mock_evaluator(r: float) -> float:
        return 10.0 - abs(r - 2.0)

    calibrator = ThresholdCalibrator(candidate_grid=(0.5, 1.0, 1.5, 2.0, 3.0))
    calib = calibrator.calibrate(
        train_emg=rest_windows,
        train_labels=train_labels,
        rest_label=0,
        evaluator_fn=mock_evaluator,
    )

    assert calib.r_multiplier == 2.0
    np.testing.assert_allclose(calib.epsilon_channels, 2.0, atol=1e-12)
