"""Tests for semg_dsp.filter module contracts (T004, numeric_sensitive: true).

Requirements: FR-003, FR-004, FR-005, FR-008, FR-010, FR-011.
Acceptance criteria: AC-002, AC-003, AC-006, AC-011.
Plan decisions: D-002, D-003, D-005, D-006.
"""

from pathlib import Path
import numpy as np
import pytest

from semg_dsp.source import ChunkData

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "dsp" / "sos_test_filter.npz"
TOL_SOS_FILTER_L1_RTOL = 1e-5
TOL_SOS_FILTER_L1_ATOL = 1e-5
TOL_CHUNK_INVARIANCE_RTOL = 1e-6
TOL_CHUNK_INVARIANCE_ATOL = 1e-6


@pytest.fixture(scope="module")
def sos_fixture():
    assert FIXTURE_PATH.is_file(), f"Missing golden fixture: {FIXTURE_PATH}"
    with np.load(FIXTURE_PATH, allow_pickle=False) as data:
        yield {k: data[k] for k in data.files}


def test_causal_sos_filter_init_and_state_shape(sos_fixture):
    from semg_dsp.filter import CausalSosFilter
    sos = sos_fixture["sos"]
    n_sections = sos.shape[0]
    filt = CausalSosFilter(sos_coefficients=sos, num_channels=2)

    assert filt.num_channels == 2
    assert filt.n_sections == n_sections
    assert filt.state.shape == (n_sections, 2, 2)
    assert filt.state.dtype == np.float32
    assert filt.sos_coefficients.dtype == np.float32
    np.testing.assert_array_equal(filt.state, np.zeros((n_sections, 2, 2), dtype=np.float32))


def test_causal_sos_filter_strict_float32_preservation(sos_fixture):
    from semg_dsp.filter import CausalSosFilter
    sos = sos_fixture["sos"]
    filt = CausalSosFilter(sos_coefficients=sos, num_channels=2)

    inp = sos_fixture["sine_in"]
    chunk = ChunkData(data=inp, start_sample_idx=0)
    out = filt.process_chunk(chunk)

    assert isinstance(out, ChunkData)
    assert out.data.dtype == np.float32
    assert filt.state.dtype == np.float32
    assert filt.sos_coefficients.dtype == np.float32
    assert np.all(np.isfinite(out.data))


def test_causal_sos_filter_oracle_l1_cases(sos_fixture):
    from semg_dsp.filter import CausalSosFilter
    sos = sos_fixture["sos"]
    n_sections = sos.shape[0]

    cases = ["zeros", "impulse", "dc", "sine", "multi_tone"]
    for case_name in cases:
        filt = CausalSosFilter(sos_coefficients=sos, num_channels=2)
        inp = sos_fixture[f"{case_name}_in"]
        expected_out = sos_fixture[f"{case_name}_out"]
        expected_zf = sos_fixture[f"{case_name}_zf"]

        chunk = ChunkData(data=inp, start_sample_idx=0)
        out_chunk = filt.process_chunk(chunk)

        assert out_chunk.start_sample_idx == 0
        assert out_chunk.num_samples == len(inp)
        assert out_chunk.num_channels == 2
        assert filt.n_sections == n_sections

        # Check output against SciPy L1 reference within TOL-SOS-FILTER-L1
        np.testing.assert_allclose(
            out_chunk.data,
            expected_out,
            rtol=TOL_SOS_FILTER_L1_RTOL,
            atol=TOL_SOS_FILTER_L1_ATOL,
            err_msg=f"L1 oracle failure for {case_name} output",
        )

        # Check final filter state against SciPy zf within TOL-SOS-FILTER-L1
        np.testing.assert_allclose(
            filt.state,
            expected_zf,
            rtol=TOL_SOS_FILTER_L1_RTOL,
            atol=TOL_SOS_FILTER_L1_ATOL,
            err_msg=f"L1 oracle failure for {case_name} final state",
        )


def test_causal_sos_filter_multichannel_no_crosstalk(sos_fixture):
    """Validate strict channel isolation: impulse on ch0 must NOT leak into ch1."""
    from semg_dsp.filter import CausalSosFilter
    sos = sos_fixture["sos"]
    filt = CausalSosFilter(sos_coefficients=sos, num_channels=2)

    impulse_in = sos_fixture["impulse_in"]  # ch0 has impulse, ch1 is all zeros
    assert np.all(impulse_in[:, 1] == 0.0)

    out_chunk = filt.process_chunk(ChunkData(data=impulse_in, start_sample_idx=0))

    # Channel 1 output must remain strictly bitwise 0.0
    np.testing.assert_array_equal(out_chunk.data[:, 1], np.zeros(len(impulse_in), dtype=np.float32))
    # Channel 1 state must remain strictly bitwise 0.0
    np.testing.assert_array_equal(filt.state[:, :, 1], np.zeros((filt.n_sections, 2), dtype=np.float32))

    # Channel 0 must contain nonzero filtered response
    assert np.any(out_chunk.data[:, 0] != 0.0)
    assert np.any(filt.state[:, :, 0] != 0.0)


def test_causal_sos_filter_chunk_invariance(sos_fixture):
    """Validate chunk invariance: single-shot vs fragmented chunk streaming."""
    from semg_dsp.filter import CausalSosFilter
    sos = sos_fixture["sos"]
    multi_in = sos_fixture["multi_in"]

    # 1. Full batch execution
    filt_full = CausalSosFilter(sos_coefficients=sos, num_channels=2)
    full_out = filt_full.process_chunk(ChunkData(data=multi_in, start_sample_idx=0))

    # 2. Fragmented chunk execution across variable lengths [1, 9, 30, 60, 100]
    filt_chunked = CausalSosFilter(sos_coefficients=sos, num_channels=2)
    lengths = [1, 9, 30, 60, 100]
    assert sum(lengths) == len(multi_in)

    chunk_outputs = []
    cursor = 0
    for length in lengths:
        segment = multi_in[cursor : cursor + length]
        c_in = ChunkData(data=segment, start_sample_idx=cursor)
        c_out = filt_chunked.process_chunk(c_in)
        assert c_out.start_sample_idx == cursor
        assert c_out.num_samples == length
        chunk_outputs.append(c_out.data)
        cursor += length

    concat_out = np.concatenate(chunk_outputs, axis=0)

    # Sample-by-sample DF2T executes identical sequential operations: verify bitwise equivalence
    np.testing.assert_array_equal(concat_out, full_out.data)
    np.testing.assert_array_equal(filt_chunked.state, filt_full.state)

    # Also assert within formal TOL-CHUNK-INVARIANCE contract
    np.testing.assert_allclose(
        concat_out,
        full_out.data,
        rtol=TOL_CHUNK_INVARIANCE_RTOL,
        atol=TOL_CHUNK_INVARIANCE_ATOL,
    )


def test_causal_sos_filter_reset(sos_fixture):
    from semg_dsp.filter import CausalSosFilter
    sos = sos_fixture["sos"]
    filt = CausalSosFilter(sos_coefficients=sos, num_channels=2)

    sine_in = sos_fixture["sine_in"]
    c1 = filt.process_chunk(ChunkData(data=sine_in[:50], start_sample_idx=0))
    assert np.any(filt.state != 0.0)

    filt.reset()
    np.testing.assert_array_equal(filt.state, np.zeros_like(filt.state))

    c2 = filt.process_chunk(ChunkData(data=sine_in[:50], start_sample_idx=0))
    np.testing.assert_array_equal(c1.data, c2.data)


def test_causal_sos_filter_coefficient_normalization():
    """Validate that unnormalized coefficients (a0 != 1.0) are correctly normalized."""
    from semg_dsp.filter import CausalSosFilter
    raw_section = np.array([[0.2, 0.4, 0.2, 2.0, 0.6, 0.2]], dtype=np.float32)
    filt = CausalSosFilter(sos_coefficients=raw_section, num_channels=1)

    # Normalized coefficients: divided by a0 = 2.0
    expected_norm = np.array([[0.1, 0.2, 0.1, 1.0, 0.3, 0.1]], dtype=np.float32)
    np.testing.assert_allclose(filt.sos_coefficients, expected_norm, rtol=1e-6, atol=1e-6)


def test_causal_sos_filter_validation_errors(sos_fixture):
    from semg_dsp.filter import CausalSosFilter
    sos = sos_fixture["sos"]

    # Invalid sos shape
    with pytest.raises(ValueError, match="shape"):
        CausalSosFilter(sos_coefficients=np.zeros((2, 5), dtype=np.float32), num_channels=2)

    # Invalid num_channels
    with pytest.raises(ValueError, match="num_channels"):
        CausalSosFilter(sos_coefficients=sos, num_channels=0)

    # a0 == 0.0
    bad_sos = sos.copy()
    bad_sos[0, 3] = 0.0
    with pytest.raises(ValueError, match="a0"):
        CausalSosFilter(sos_coefficients=bad_sos, num_channels=2)

    filt = CausalSosFilter(sos_coefficients=sos, num_channels=2)

    # Non-2D array
    with pytest.raises(ValueError, match="2D"):
        filt.process_chunk(np.zeros(10, dtype=np.float32))

    # Channel mismatch
    with pytest.raises(ValueError, match="channels"):
        filt.process_chunk(ChunkData(data=np.zeros((10, 3), dtype=np.float32), start_sample_idx=0))

    # Incompatible dtype
    with pytest.raises(TypeError, match="float32"):
        filt.process_chunk(np.zeros((10, 2), dtype=np.float64))
