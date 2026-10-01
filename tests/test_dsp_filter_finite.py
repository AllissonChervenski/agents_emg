"""Tests for CausalSosFilter rejection of non-finite input data (T009 per AC-006)."""
import numpy as np
import pytest
from semg_dsp.source import ChunkData
from semg_dsp.filter import CausalSosFilter


def test_causal_sos_filter_rejects_nan_in_numpy_array():
    coeffs = [[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]]
    flt = CausalSosFilter(coeffs, num_channels=1)

    bad_data = np.array([[1.0], [np.nan], [3.0]], dtype=np.float32)
    with pytest.raises(ValueError, match="non-finite"):
        flt.process_chunk(bad_data)


def test_causal_sos_filter_rejects_inf_in_numpy_array():
    coeffs = [[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]]
    flt = CausalSosFilter(coeffs, num_channels=1)

    bad_data = np.array([[1.0], [np.inf], [3.0]], dtype=np.float32)
    with pytest.raises(ValueError, match="non-finite"):
        flt.process_chunk(bad_data)


def test_causal_sos_filter_rejects_nan_in_chunk_data():
    coeffs = [[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]]
    flt = CausalSosFilter(coeffs, num_channels=2)

    bad_data = np.array([[1.0, 0.0], [np.nan, 2.0]], dtype=np.float32)
    chunk = ChunkData(data=bad_data, start_sample_idx=0)
    with pytest.raises(ValueError, match="non-finite"):
        flt.process_chunk(chunk)


def test_causal_sos_filter_rejects_nan_in_sos_coefficients():
    bad_sos = np.array([[np.nan, 0.0, 0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
    with pytest.raises(ValueError, match="finite"):
        CausalSosFilter(bad_sos, num_channels=1)


def test_causal_sos_filter_rejects_inf_in_sos_coefficients():
    bad_sos = np.array([[1.0, np.inf, 0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
    with pytest.raises(ValueError, match="finite"):
        CausalSosFilter(bad_sos, num_channels=1)


def test_causal_sos_filter_rejects_near_zero_a0_overflow():
    bad_sos = np.array([[1.0, 0.0, 0.0, 1e-40, 0.0, 0.0]], dtype=np.float32)
    with pytest.raises(ValueError, match="reciprocal overflow|non-finite"):
        CausalSosFilter(bad_sos, num_channels=1)


