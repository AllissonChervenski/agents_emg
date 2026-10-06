"""Analytical tests for classical time-domain feature extractor (FR-004, SC-002, Decision-2).

Validates:
- Analytical mathematical oracles for MAV, RMS, WL, ZC, SSC against TOL-TIME-DOMAIN-FEATURES (1e-12).
- Deadzone noise suppression for ZC and SSC.
- Multi-channel feature matrix extraction shape (N_windows, 60).
- Zero NaN, zero Inf, and vectorized performance.
"""

from __future__ import annotations

import numpy as np

from semg_baseline.features import (
    TimeDomainFeatureExtractor,
    compute_mav,
    compute_rms,
    compute_wl,
    compute_zc,
    compute_ssc,
)


def test_analytical_mav_oracle() -> None:
    """compute_mav must match exact closed-form analytical formula within 1e-12."""
    # Constant signal x = 2.5
    w_const = np.full((1, 400, 12), 2.5, dtype=np.float64)
    mav_const = compute_mav(w_const)
    np.testing.assert_allclose(mav_const, 2.5, atol=1e-12, rtol=1e-12)

    # Alternating signal +1.0, -1.0
    alt_pattern = np.array([1.0 if i % 2 == 0 else -1.0 for i in range(400)], dtype=np.float64)
    w_alt = np.tile(alt_pattern[:, None], (1, 12))[None, :, :]
    mav_alt = compute_mav(w_alt)
    np.testing.assert_allclose(mav_alt, 1.0, atol=1e-12, rtol=1e-12)


def test_analytical_rms_oracle() -> None:
    """compute_rms must match exact mathematical sqrt(mean(x^2)) within 1e-12."""
    w_const = np.full((1, 400, 12), -3.0, dtype=np.float64)
    rms_const = compute_rms(w_const)
    np.testing.assert_allclose(rms_const, 3.0, atol=1e-12, rtol=1e-12)

    # Ramp 0..399
    ramp = np.arange(400, dtype=np.float64)
    expected_rms = np.sqrt(np.mean(ramp**2))
    w_ramp = np.tile(ramp[:, None], (1, 12))[None, :, :]
    rms_ramp = compute_rms(w_ramp)
    np.testing.assert_allclose(rms_ramp, expected_rms, atol=1e-12, rtol=1e-12)


def test_analytical_wl_oracle() -> None:
    """compute_wl must match sum(|x_{i+1} - x_i|) within 1e-12."""
    # Ramp signal: delta = 1.0 at each of 399 steps -> WL = 399.0
    ramp = np.arange(400, dtype=np.float64)
    w_ramp = np.tile(ramp[:, None], (1, 12))[None, :, :]
    wl_ramp = compute_wl(w_ramp)
    np.testing.assert_allclose(wl_ramp, 399.0, atol=1e-12, rtol=1e-12)

    # Alternating +1, -1: delta = 2.0 at each step -> WL = 399 * 2 = 798.0
    alt_pattern = np.array([1.0 if i % 2 == 0 else -1.0 for i in range(400)], dtype=np.float64)
    w_alt = np.tile(alt_pattern[:, None], (1, 12))[None, :, :]
    wl_alt = compute_wl(w_alt)
    np.testing.assert_allclose(wl_alt, 798.0, atol=1e-12, rtol=1e-12)


def test_analytical_zc_oracle_and_deadzone() -> None:
    """compute_zc must count sign crossings exceeding threshold epsilon."""
    # Signal crossing zero 4 times with amplitude 1.0
    # [1, -1, 1, -1, 1] padded to 400 with 1.0
    sig = np.ones(400, dtype=np.float64)
    sig[1] = -1.0
    sig[2] = 1.0
    sig[3] = -1.0
    sig[4] = 1.0
    w_sig = np.tile(sig[:, None], (1, 12))[None, :, :]

    # Threshold 0.1: all 4 crossings have |delta| = 2.0 >= 0.1 -> count = 4
    zc_large = compute_zc(w_sig, epsilon=0.1)
    np.testing.assert_array_equal(zc_large, 4)

    # Threshold 5.0: |delta| = 2.0 < 5.0 -> count = 0
    zc_strict = compute_zc(w_sig, epsilon=5.0)
    np.testing.assert_array_equal(zc_strict, 0)


def test_analytical_ssc_oracle_and_deadzone() -> None:
    """compute_ssc must count slope extrema exceeding threshold epsilon."""
    # Peak at sample 2 (1 -> 3 -> 1), trough at sample 4 (1 -> -1 -> 1)
    sig = np.ones(400, dtype=np.float64)
    sig[2] = 3.0   # peak: 1 < 3 > 1 (|delta|=2)
    sig[4] = -1.0  # trough: 1 > -1 < 1 (|delta|=2)
    w_sig = np.tile(sig[:, None], (1, 12))[None, :, :]

    # Threshold 0.5: both extrema have |delta| = 2.0 >= 0.5 -> count = 2
    ssc_count = compute_ssc(w_sig, epsilon=0.5)
    np.testing.assert_array_equal(ssc_count, 2)

    # Threshold 5.0: |delta| = 2.0 < 5.0 -> count = 0
    ssc_strict = compute_ssc(w_sig, epsilon=5.0)
    np.testing.assert_array_equal(ssc_strict, 0)


def test_feature_extractor_output_shape_and_no_nan() -> None:
    """TimeDomainFeatureExtractor must produce shape (N, 60) with no NaN or Inf."""
    rng = np.random.default_rng(42)
    windows = rng.normal(0.0, 1.0, size=(15, 400, 12)).astype(np.float32)

    extractor = TimeDomainFeatureExtractor(epsilon=0.01)
    feats = extractor.extract(windows)

    assert feats.shape == (15, 60)
    assert feats.dtype == np.float64
    assert not np.isnan(feats).any()
    assert not np.isinf(feats).any()


def test_feature_extractor_per_channel_epsilon() -> None:
    """TimeDomainFeatureExtractor must respect per-channel epsilon array."""
    # Channel 0 has small epsilon (counts crossings), channel 1 has huge epsilon (counts 0)
    sig = np.ones(400, dtype=np.float64)
    sig[1] = -1.0
    sig[2] = 1.0
    w_sig = np.zeros((1, 400, 12), dtype=np.float64)
    w_sig[0, :, 0] = sig
    w_sig[0, :, 1] = sig

    eps = np.full(12, 100.0, dtype=np.float64)
    eps[0] = 0.05  # only channel 0 detects

    extractor = TimeDomainFeatureExtractor(epsilon=eps)
    feats = extractor.extract(w_sig)

    # Columns 36..47 are ZC
    zc_ch0 = feats[0, 36]
    zc_ch1 = feats[0, 37]
    assert zc_ch0 == 2
    assert zc_ch1 == 0
