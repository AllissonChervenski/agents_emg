"""Unit tests for Ozdemir et al. sEMG quantization and channel reduction benchmark."""
from __future__ import annotations

import sys
import numpy as np
import pytest

from scripts.benchmark_ozdemir_quantization_and_channels import (
    compute_evaluation_metrics,
    get_channel_feature_indices,
    quantize_signal,
)
from semg_baseline.features import TimeDomainFeatureExtractor
from semg_baseline.lda import CanonicalLdaTrainer, PriorType
from semg_baseline.scaling import FeatureScaler


def teardown_module() -> None:
    """Guarantee that scikit-learn is not left resident in sys.modules."""
    sys.modules.pop("sklearn", None)
    for mod in list(sys.modules.keys()):
        if mod.startswith("sklearn."):
            sys.modules.pop(mod, None)


def test_ozdemir_quantize_signal_float32_identity() -> None:
    arr = np.array([-9.5, 0.0, 4.2, 10.0], dtype=np.float32)
    q = quantize_signal(arr, "float32", v_max=10.0)
    assert q.dtype == np.float32
    np.testing.assert_array_equal(arr, q)


def test_ozdemir_quantize_signal_float16_precision() -> None:
    arr = np.array([-9.123456, 3.456789], dtype=np.float32)
    q = quantize_signal(arr, "float16", v_max=10.0)
    assert q.dtype == np.float32
    assert not np.array_equal(arr, q)
    np.testing.assert_allclose(arr, q, rtol=1e-3, atol=1e-5)


def test_ozdemir_quantize_signal_int16_and_int8() -> None:
    v_max = 10.0
    arr = np.array([-v_max, 0.0, v_max], dtype=np.float32)

    q16 = quantize_signal(arr, "int16", v_max=v_max)
    assert q16.dtype == np.float32
    np.testing.assert_allclose(arr, q16, atol=1e-5)

    q8 = quantize_signal(arr, "int8", v_max=v_max)
    assert q8.dtype == np.float32
    np.testing.assert_allclose(arr, q8, atol=1e-3)

    # Saturation clipping test (int8 asymmetric range: [-128, 127])
    over = np.array([-15.0, 15.0], dtype=np.float32)
    q_over = quantize_signal(over, "int8", v_max=v_max)
    assert np.isclose(q_over[0], -v_max * (128 / 127), atol=1e-4)
    assert np.isclose(q_over[1], v_max, atol=1e-4)


def test_ozdemir_channel_feature_indices() -> None:
    # 4 channels -> 20 features
    idx4 = get_channel_feature_indices(4, total_channels=4)
    assert len(idx4) == 20
    assert idx4 == list(range(20))

    # 2 channels -> 10 features
    idx2 = get_channel_feature_indices(2, total_channels=4)
    assert len(idx2) == 10
    assert idx2 == [0, 1, 4, 5, 8, 9, 12, 13, 16, 17]

    # 1 channel -> 5 features
    idx1 = get_channel_feature_indices(1, total_channels=4)
    assert len(idx1) == 5
    assert idx1 == [0, 4, 8, 12, 16]

    with pytest.raises(ValueError):
        get_channel_feature_indices(0, total_channels=4)
    with pytest.raises(ValueError):
        get_channel_feature_indices(5, total_channels=4)


def test_ozdemir_compute_evaluation_metrics_pure_numpy() -> None:
    y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 3])
    y_pred = np.array([0, 1, 1, 0, 2, 2, 0, 1, 3])
    acc, bal_acc, macro_f1 = compute_evaluation_metrics(y_true, y_pred)
    assert 0.0 <= acc <= 1.0
    assert 0.0 <= bal_acc <= 1.0
    assert 0.0 <= macro_f1 <= 1.0
    assert np.isclose(acc, 7.0 / 9.0)


def test_ozdemir_synthetic_subset_lda_invariance() -> None:
    """Ensure sub-channel extraction from 4-ch features matches direct sub-channel extraction."""
    rng = np.random.default_rng(123)
    windows_4 = rng.standard_normal((40, 400, 4), dtype=np.float32) * 0.5
    labels = np.repeat(np.arange(4), 10)

    ext4 = TimeDomainFeatureExtractor(epsilon=1e-4, num_channels=4)
    feats4 = ext4.extract(windows_4)

    ext2 = TimeDomainFeatureExtractor(epsilon=1e-4, num_channels=2)
    feats2_direct = ext2.extract(windows_4[:, :, :2])

    idx2 = get_channel_feature_indices(2, total_channels=4)
    feats2_sliced = feats4[:, idx2]

    np.testing.assert_allclose(feats2_direct, feats2_sliced, rtol=1e-5, atol=1e-7)

    scaler = FeatureScaler()
    X_s = scaler.fit_transform(feats2_sliced)
    trainer = CanonicalLdaTrainer(prior_type=PriorType.UNIFORM, num_classes=4)
    trainer.fit(X_s, labels)
    preds = trainer.predict(X_s)
    acc = np.mean(preds == labels)
    assert acc > 0.0
