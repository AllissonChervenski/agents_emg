"""Unit and integration tests for BaselineFeaturePipeline (FR-001, FR-004, FR-012, FR-013, SC-001, SC-002).

Validates:
- End-to-end integration of CausalBandpassFilter, PureWindowSegmenter,
  ThresholdCalibrator, TimeDomainFeatureExtractor, and FeatureScaler.
- Extraction of (N, 60) feature matrices from continuous RecordingData.
- Zero lookahead, zero NaN/Inf, and strict train-only calibration isolation.
- Support for both refined and stimulus dataset views.
"""

from __future__ import annotations

import numpy as np
import pytest

from semg_baseline.pipeline import BaselineFeaturePipeline
from tests.fixtures.synthetic_semg import make_synthetic_recording


def test_baseline_feature_pipeline_fit_transform() -> None:
    """Pipeline must fit on training recordings and produce 60-feature normalized matrix."""
    rec = make_synthetic_recording(seed=42, repetitions=(1, 3, 4, 6))

    pipeline = BaselineFeaturePipeline(
        filter_low_hz=20.0,
        filter_high_hz=450.0,
        fs=2000.0,
        window_len=400,
        stride=100,
        candidate_grid=(0.5, 1.0, 1.5, 2.0, 3.0),
        default_r=1.5,
    )

    X_train, y_train, audits = pipeline.fit_transform([rec], view_name="refined")

    assert X_train.ndim == 2
    assert X_train.shape[1] == 60
    assert len(y_train) == X_train.shape[0]
    assert not np.isnan(X_train).any()
    assert not np.isinf(X_train).any()

    # Verify audit invariants
    assert len(audits) == 1
    assert audits[0].repetition_crossing_windows_admitted == 0
    assert audits[0].admitted_window_count == X_train.shape[0]

    # Verify scaler statistics
    assert pipeline.feature_scaler is not None
    assert pipeline.threshold_calibration is not None
    assert pipeline.threshold_calibration.r_multiplier == 1.5


def test_baseline_feature_pipeline_train_test_isolation() -> None:
    """Transforming test recordings must not mutate training calibration or scaling parameters."""
    rec_train = make_synthetic_recording(seed=101, repetitions=(1, 3, 4, 6))
    rec_test = make_synthetic_recording(seed=102, repetitions=(2, 5))

    pipeline = BaselineFeaturePipeline()
    X_train, y_train, _ = pipeline.fit_transform([rec_train], view_name="refined")

    assert pipeline.feature_scaler is not None
    assert pipeline.threshold_calibration is not None
    orig_mean = np.copy(pipeline.feature_scaler.mean_train)
    orig_scale = np.copy(pipeline.feature_scaler.scale_train)
    orig_eps = np.copy(pipeline.threshold_calibration.epsilon_channels)

    # Transform test recording statelessly
    X_test, y_test, test_audits = pipeline.transform([rec_test], view_name="refined")

    assert X_test.shape[1] == 60
    assert len(y_test) == X_test.shape[0]
    assert not np.isnan(X_test).any()
    assert test_audits[0].repetition_crossing_windows_admitted == 0

    # Invariant: parameters must remain bit-exact
    np.testing.assert_array_equal(pipeline.feature_scaler.mean_train, orig_mean)
    np.testing.assert_array_equal(pipeline.feature_scaler.scale_train, orig_scale)
    np.testing.assert_array_equal(pipeline.threshold_calibration.epsilon_channels, orig_eps)


def test_pipeline_unfitted_transform_raises_runtime_error() -> None:
    """Calling transform on an unfitted pipeline must fail closed with RuntimeError."""
    rec = make_synthetic_recording(seed=1)
    pipeline = BaselineFeaturePipeline()

    with pytest.raises(RuntimeError, match="not fitted"):
        pipeline.transform([rec])


def test_pipeline_supports_both_refined_and_stimulus_views() -> None:
    """Pipeline must cleanly process both refined and stimulus views without contamination."""
    rec = make_synthetic_recording(seed=55)

    pipeline_ref = BaselineFeaturePipeline()
    X_ref, y_ref, _ = pipeline_ref.fit_transform([rec], view_name="refined")

    pipeline_stim = BaselineFeaturePipeline()
    X_stim, y_stim, _ = pipeline_stim.fit_transform([rec], view_name="stimulus")

    assert X_ref.shape[1] == 60
    assert X_stim.shape[1] == 60
    assert len(y_ref) == X_ref.shape[0]
    assert len(y_stim) == X_stim.shape[0]
