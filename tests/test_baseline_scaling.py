"""Unit tests for train-only custom Z-score feature scaler (FR-013, SC-002, Decision-2).

Validates:
- Mathematically exact Z-score scaling complying with TOL-FEATURE-SCALING (1e-12).
- Stateless transformation of validation and test splits with zero data leakage.
- Automatic zero-variance feature protection (scale=1.0, z=0.0, zero NaN/Inf).
- Absence of sklearn.preprocessing.StandardScaler imports or couplings.
"""

from __future__ import annotations

import subprocess
import sys
import numpy as np
import pytest

from semg_baseline.scaling import FeatureScaler


def test_feature_scaler_exact_zscore() -> None:
    """FeatureScaler must compute exact Z-score normalization within 1e-12."""
    rng = np.random.default_rng(42)
    X = rng.normal(loc=15.0, scale=4.0, size=(100, 60)).astype(np.float64)

    scaler = FeatureScaler()
    scaler.fit(X)

    expected_mean = np.mean(X, axis=0, dtype=np.float64)
    expected_std = np.std(X, axis=0, ddof=0, dtype=np.float64)

    np.testing.assert_allclose(scaler.mean_train, expected_mean, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(scaler.scale_train, expected_std, atol=1e-12, rtol=1e-12)

    X_scaled = scaler.transform(X)
    np.testing.assert_allclose(np.mean(X_scaled, axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(np.std(X_scaled, axis=0, ddof=0), 1.0, atol=1e-12)


def test_feature_scaler_stateless_test_transform() -> None:
    """Transforming test data must be strictly stateless without mutating training statistics."""
    rng = np.random.default_rng(101)
    X_train = rng.normal(loc=5.0, scale=2.0, size=(50, 60)).astype(np.float64)
    X_test = rng.normal(loc=20.0, scale=8.0, size=(25, 60)).astype(np.float64)

    scaler = FeatureScaler()
    scaler.fit(X_train)

    train_mean_orig = np.copy(scaler.mean_train)
    train_scale_orig = np.copy(scaler.scale_train)

    X_test_scaled = scaler.transform(X_test)

    # Invariant: parameters must remain bit-exact
    np.testing.assert_array_equal(scaler.mean_train, train_mean_orig)
    np.testing.assert_array_equal(scaler.scale_train, train_scale_orig)

    # Test transform matches manual calculation
    expected_test_scaled = (X_test - train_mean_orig) / train_scale_orig
    np.testing.assert_allclose(X_test_scaled, expected_test_scaled, atol=1e-12, rtol=1e-12)


def test_feature_scaler_zero_variance_protection() -> None:
    """Features with zero variance must be guarded (scale=1.0, Z=0.0, zero NaN/Inf)."""
    X = np.ones((40, 60), dtype=np.float64)
    # Give column 0 variance, leave column 1 completely constant (zero variance)
    X[:, 0] = np.linspace(1.0, 10.0, 40)
    X[:, 1] = 42.0  # Constant column

    scaler = FeatureScaler()
    scaler.fit(X)

    assert 1 in scaler.zero_variance_indices
    assert scaler.scale_train[1] == 1.0
    assert not np.isnan(scaler.scale_train).any()

    X_scaled = scaler.transform(X)
    assert not np.isnan(X_scaled).any()
    assert not np.isinf(X_scaled).any()
    assert np.all(X_scaled[:, 1] == 0.0)


def test_feature_scaler_fit_transform_equivalence() -> None:
    """fit_transform(X) must yield identical results to fit(X).transform(X)."""
    rng = np.random.default_rng(202)
    X = rng.normal(0.0, 1.0, size=(30, 60)).astype(np.float64)

    s1 = FeatureScaler()
    z1 = s1.fit(X).transform(X)

    s2 = FeatureScaler()
    z2 = s2.fit_transform(X)

    np.testing.assert_array_equal(z1, z2)


def test_feature_scaler_not_fitted_raises_error() -> None:
    """Calling transform on an unfitted FeatureScaler must raise RuntimeError."""
    scaler = FeatureScaler()
    X = np.zeros((10, 60), dtype=np.float64)

    with pytest.raises(RuntimeError, match="not fitted"):
        scaler.transform(X)


def test_feature_scaler_shape_mismatch_raises_error() -> None:
    """Transforming an array with mismatched feature dimension must raise ValueError."""
    scaler = FeatureScaler()
    scaler.fit(np.zeros((10, 60), dtype=np.float64))

    with pytest.raises(ValueError, match="Feature dimension mismatch"):
        scaler.transform(np.zeros((10, 40), dtype=np.float64))


def test_feature_scaler_does_not_import_sklearn_standard_scaler() -> None:
    """FeatureScaler must be a neutral custom implementation with zero sklearn coupling in an isolated process."""
    cmd = [
        sys.executable,
        "-c",
        "import sys; import semg_baseline.scaling; assert 'sklearn' not in sys.modules, f'Leaked: {[m for m in sys.modules if m.startswith(\"sklearn\")]}'",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert res.returncode == 0
