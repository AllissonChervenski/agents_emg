"""Unit tests for NinaPro DB2 quantization and channel reduction benchmark."""
from __future__ import annotations

import numpy as np
import pytest

from scripts.benchmark_quantization_and_channels import (
    compute_evaluation_metrics,
    get_channel_feature_indices,
    quantize_signal,
)
from semg_baseline.features import TimeDomainFeatureExtractor
from semg_baseline.lda import CanonicalLdaTrainer, PriorType
from semg_baseline.scaling import FeatureScaler


def test_quantize_signal_float32_identity() -> None:
    arr = np.array([-0.005, 0.0, 0.002, 0.0055], dtype=np.float32)
    q = quantize_signal(arr, "float32")
    assert q.dtype == np.float32
    np.testing.assert_array_equal(arr, q)


def test_quantize_signal_float16_precision() -> None:
    arr = np.array([-0.00512345, 0.00123456], dtype=np.float32)
    q = quantize_signal(arr, "float16")
    assert q.dtype == np.float32
    assert not np.array_equal(arr, q)
    np.testing.assert_allclose(arr, q, rtol=1e-3, atol=1e-6)


def test_quantize_signal_int16_and_int8_scaling() -> None:
    v_max = 0.0055
    arr = np.array([-v_max, 0.0, v_max], dtype=np.float32)

    # int16 should reproduce exact extremes
    q16 = quantize_signal(arr, "int16", v_max=v_max)
    assert q16.dtype == np.float32
    np.testing.assert_allclose(arr, q16, atol=1e-6)

    # int8 should clamp to [-v_max, v_max]
    q8 = quantize_signal(arr, "int8", v_max=v_max)
    assert q8.dtype == np.float32
    np.testing.assert_allclose(arr, q8, atol=1e-4)

    # Test extreme saturation clipping
    over = np.array([-0.010, 0.010], dtype=np.float32)
    q_over = quantize_signal(over, "int8", v_max=v_max)
    assert np.isclose(q_over[0], -v_max, atol=1e-4)
    assert np.isclose(q_over[1], v_max, atol=1e-4)


def test_quantize_signal_invalid_precision() -> None:
    with pytest.raises(ValueError, match="Unsupported precision"):
        quantize_signal(np.zeros(5, dtype=np.float32), "int4")


def test_get_channel_feature_indices() -> None:
    # 12 channels -> 60 features
    idx12 = get_channel_feature_indices(12, total_channels=12)
    assert len(idx12) == 60
    assert idx12 == list(range(60))

    # 1 channel -> 5 features (MAV_0, RMS_0, WL_0, ZC_0, SSC_0)
    idx1 = get_channel_feature_indices(1, total_channels=12)
    assert idx1 == [0, 12, 24, 36, 48]

    # 4 channels -> 20 features
    idx4 = get_channel_feature_indices(4, total_channels=12)
    assert len(idx4) == 20
    assert idx4[:4] == [0, 1, 2, 3]
    assert idx4[4:8] == [12, 13, 14, 15]

    # Invalid channel counts
    with pytest.raises(ValueError):
        get_channel_feature_indices(0)
    with pytest.raises(ValueError):
        get_channel_feature_indices(13)


def test_compute_evaluation_metrics_pure_numpy() -> None:
    y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 3])
    y_pred = np.array([0, 1, 1, 0, 2, 2, 0, 1, 3])
    acc, bal_acc, macro_f1 = compute_evaluation_metrics(y_true, y_pred)
    assert 0.0 <= acc <= 1.0
    assert 0.0 <= bal_acc <= 1.0
    assert 0.0 <= macro_f1 <= 1.0
    assert np.isclose(acc, 7.0 / 9.0)


def test_end_to_end_synthetic_subset_lda_invariance() -> None:
    """Ensure sub-channel extraction from 12-ch features is bitwise equivalent to K-channel extraction."""
    rng = np.random.default_rng(42)
    windows_12 = rng.standard_normal((50, 400, 12), dtype=np.float32) * 1e-4
    labels = np.repeat(np.arange(5), 10)

    # 12-channel extraction
    ext12 = TimeDomainFeatureExtractor(epsilon=1e-5, num_channels=12)
    feats12 = ext12.extract(windows_12)

    # 6-channel extraction
    ext6 = TimeDomainFeatureExtractor(epsilon=1e-5, num_channels=6)
    feats6 = ext6.extract(windows_12[:, :, :6])

    indices6 = get_channel_feature_indices(6, total_channels=12)
    sub_feats = feats12[:, indices6]

    np.testing.assert_allclose(sub_feats, feats6, atol=1e-12)

    # Train and evaluate LDA
    scaler = FeatureScaler()
    scaled_feats = scaler.fit_transform(sub_feats)
    trainer = CanonicalLdaTrainer(prior_type=PriorType.UNIFORM, num_classes=5)
    trainer.fit(scaled_feats, labels)
    preds = trainer.predict(scaled_feats)
    assert len(preds) == 50


def teardown_module() -> None:
    """Evict sklearn from sys.modules to prevent polluting subsequent test state."""
    import sys

    for mod in list(sys.modules):
        if mod == "sklearn" or mod.startswith("sklearn."):
            sys.modules.pop(mod, None)

