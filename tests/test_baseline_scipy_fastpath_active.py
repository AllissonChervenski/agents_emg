"""Regression guard proving the host SciPy SOS fast path is actually used.

Original Python DF2T implementation should FAIL this test. The optimized
semg_baseline/filter.py should PASS without modifying semg_dsp/filter.py.
"""
from unittest.mock import patch

import numpy as np
import scipy.signal as signal

from semg_baseline.filter import CausalBandpassFilter
from semg_dsp.filter import CausalSosFilter


def test_host_baseline_invokes_scipy_sosfilt_and_preserves_reference_values():
    rng = np.random.default_rng(20261009)
    x = rng.normal(size=(256, 12)).astype(np.float32)
    filt = CausalBandpassFilter(num_channels=12)
    ref = CausalSosFilter(filt.get_sos_coefficients(), num_channels=12)

    with patch("semg_baseline.filter.signal.sosfilt", wraps=signal.sosfilt) as spy:
        got = filt.process(x)
        assert spy.call_count == 1, "Fast path missing: process() did not invoke sosfilt"

    np.testing.assert_array_equal(got, ref.process_chunk(x))
    np.testing.assert_array_equal(filt._filter.state, ref.state)


def test_host_baseline_empty_input_does_not_call_scipy_sosfilt():
    filt = CausalBandpassFilter(num_channels=12)
    with patch("semg_baseline.filter.signal.sosfilt", wraps=signal.sosfilt) as spy:
        got = filt.process(np.empty((0, 12), dtype=np.float32))
        assert spy.call_count == 0
    assert got.shape == (0, 12)
