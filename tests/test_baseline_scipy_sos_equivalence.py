"""Host-only SciPy SOS fast path versus original NumPy DF2T reference.

Run after applying the patch to semg_baseline/filter.py. No protected tests
or the semg_dsp reference implementation need to change.
"""

import numpy as np
import pytest

from semg_baseline.filter import CausalBandpassFilter
from semg_dsp.filter import CausalSosFilter


@pytest.mark.parametrize("num_channels", [1, 12])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_scipy_matches_reference_bitwise(num_channels, dtype):
    rng = np.random.default_rng(20261009)
    optimized = CausalBandpassFilter(num_channels=num_channels)
    reference = CausalSosFilter(
        optimized.get_sos_coefficients(), num_channels=num_channels
    )
    for n in (0, 1, 7, 259, 4096):
        x = rng.normal(size=(n, num_channels)).astype(dtype)
        float32_x = x.astype(np.float32, copy=False)
        actual = optimized.process(x)
        expected = reference.process_chunk(float32_x)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(optimized._filter.state, reference.state)


@pytest.mark.parametrize("num_channels", [1, 12])
def test_streaming_chunk_invariance_and_reset(num_channels):
    rng = np.random.default_rng(44)
    x = rng.normal(size=(4096, num_channels)).astype(np.float32)
    one_shot = CausalBandpassFilter(num_channels=num_channels)
    streaming = CausalBandpassFilter(num_channels=num_channels)
    reference = CausalSosFilter(
        one_shot.get_sos_coefficients(), num_channels=num_channels
    )

    complete = one_shot.process(x)
    pieces = []
    for start, end in ((0, 1), (1, 37), (37, 37), (37, 525), (525, 4096)):
        piece = streaming.process(x[start:end])
        pieces.append(piece)
        expected = reference.process_chunk(x[start:end])
        np.testing.assert_array_equal(piece, expected)
        np.testing.assert_array_equal(streaming._filter.state, reference.state)

    np.testing.assert_array_equal(complete, np.concatenate(pieces, axis=0))
    np.testing.assert_array_equal(one_shot._filter.state, streaming._filter.state)

    state_alias = streaming._filter.state
    streaming.reset()
    assert state_alias is streaming._filter.state
    np.testing.assert_array_equal(streaming._filter.state, np.zeros_like(state_alias))
    np.testing.assert_array_equal(streaming.process(x), complete)


def test_canonical_coefficients_and_config_remain_identical():
    fast = CausalBandpassFilter()
    other = CausalBandpassFilter()
    np.testing.assert_array_equal(fast.get_sos_coefficients(), other.get_sos_coefficients())
    assert fast.get_config() == other.get_config()


def test_invalid_and_empty_inputs():
    fast = CausalBandpassFilter()
    empty = fast.process(np.empty((0, 12), dtype=np.float32))
    assert empty.shape == (0, 12) and empty.dtype == np.float32
    with pytest.raises(ValueError):
        fast.process(np.zeros((5, 11), dtype=np.float32))
    with pytest.raises(ValueError):
        x = np.zeros((5, 12), dtype=np.float32)
        x[0, 0] = np.nan
        fast.process(x)
