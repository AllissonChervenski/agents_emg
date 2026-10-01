"""Tests for ChunkData sampling_rate_hz metadata contract (T008 per FR-001)."""
import numpy as np
import pytest


def test_chunk_data_has_sampling_rate_hz_field():
    from semg_dsp.source import ChunkData

    data = np.zeros((10, 2), dtype=np.float32)
    chunk = ChunkData(data=data, start_sample_idx=0, sampling_rate_hz=2000.0)

    assert hasattr(chunk, "sampling_rate_hz")
    assert chunk.sampling_rate_hz == 2000.0


def test_chunk_data_sampling_rate_hz_default_value():
    from semg_dsp.source import ChunkData

    data = np.zeros((10, 2), dtype=np.float32)
    chunk = ChunkData(data=data, start_sample_idx=0)

    assert chunk.sampling_rate_hz == 1000.0


def test_chunk_data_rejects_invalid_sampling_rate_hz():
    from semg_dsp.source import ChunkData

    data = np.zeros((10, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="sampling_rate_hz"):
        ChunkData(data=data, start_sample_idx=0, sampling_rate_hz=0.0)

    with pytest.raises(ValueError, match="sampling_rate_hz"):
        ChunkData(data=data, start_sample_idx=0, sampling_rate_hz=-100.0)

    with pytest.raises(TypeError, match="sampling_rate_hz"):
        ChunkData(data=data, start_sample_idx=0, sampling_rate_hz=False)


def test_synthetic_sample_source_propagates_sampling_rate_hz():
    from semg_dsp.source import SyntheticSampleSource

    src = SyntheticSampleSource(waveform="sine", sampling_rate_hz=2000.0, num_channels=2)
    chunk = src.read_chunk(16)

    assert chunk.sampling_rate_hz == 2000.0


def test_filter_preserves_sampling_rate_hz_when_processing_chunk_data():
    from semg_dsp.source import ChunkData
    from semg_dsp.filter import CausalSosFilter

    coeffs = [[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]]
    flt = CausalSosFilter(coeffs, num_channels=1)

    data = np.zeros((8, 1), dtype=np.float32)
    chunk = ChunkData(data=data, start_sample_idx=10, sampling_rate_hz=500.0)
    out = flt.process_chunk(chunk)

    assert isinstance(out, ChunkData)
    assert out.sampling_rate_hz == 500.0
    assert out.start_sample_idx == 10


def test_synthetic_sample_source_preserves_resolution_beyond_2_to_24_samples():
    from semg_dsp.source import SyntheticSampleSource

    src = SyntheticSampleSource(waveform="sine", frequency_hz=10.0, sampling_rate_hz=1000.0)
    src.total_samples_emitted = 20_000_000  # Strictly > 2^24 = 16,777,216
    chunk = src.read_chunk(100)

    diffs = np.diff(chunk.data[:, 0])
    # Adjacent samples must never be identical (which would indicate precision collapse from float32 integer conversion)
    assert not np.any(diffs == 0.0)
    assert chunk.data.dtype == np.float32


def test_synthetic_sample_source_preserves_sub_microhertz_frequency():
    from semg_dsp.source import SyntheticSampleSource

    src = SyntheticSampleSource(waveform="sine", frequency_hz=1e-7, sampling_rate_hz=1.0)
    chunk = src.read_chunk(100)

    # Must not collapse to constant zero or DC
    diffs = np.diff(chunk.data[:, 0])
    assert not np.all(diffs == 0.0)
    assert np.all(np.isfinite(chunk.data))
    assert chunk.data.dtype == np.float32


def test_synthetic_sample_source_preserves_accuracy_at_10_billion_samples_high_frequency():
    from semg_dsp.source import SyntheticSampleSource

    src = SyntheticSampleSource(waveform="sine", frequency_hz=999.999999, sampling_rate_hz=1000.0)
    src.total_samples_emitted = 10_000_000_000
    chunk = src.read_chunk(100)

    # For 999.999999 Hz at 1000 Hz, num=999999999 and denom=1000000000.
    # At index 10,000,000,000: (10_000_000_000 * 999999999) % 1_000_000_000 == 0.
    # sin(0) == 0.0, avoiding any signed int64 product overflow.
    assert np.isclose(chunk.data[0, 0], 0.0, atol=1e-6)
    assert not np.any(np.isnan(chunk.data))
    assert chunk.data.dtype == np.float32


def test_synthetic_sample_source_preserves_sub_microhertz_frequency_at_1000hz_sampling_rate():
    from semg_dsp.source import SyntheticSampleSource

    src = SyntheticSampleSource(waveform="sine", frequency_hz=1e-7, sampling_rate_hz=1000.0)
    chunk = src.read_chunk(100)

    # Ratio is 1e-7 / 1000 = 1 / 10^10.
    # Must preserve exact non-zero fraction and not collapse to constant zero or DC
    diffs = np.diff(chunk.data[:, 0])
    assert not np.all(diffs == 0.0)
    assert np.all(np.isfinite(chunk.data))
    assert chunk.data.dtype == np.float32


def test_synthetic_sample_source_preserves_1e_minus_38_hz_frequency():
    from semg_dsp.source import SyntheticSampleSource

    src = SyntheticSampleSource(waveform="sine", frequency_hz=1e-38, sampling_rate_hz=1.0)
    chunk = src.read_chunk(100)

    # Denom >= 10^38 exceeds float32 max; phase cycles must not evaluate to float32 Inf or zero
    diffs = np.diff(chunk.data[:, 0])
    assert not np.all(chunk.data == 0.0)
    assert not np.all(diffs == 0.0)
    assert np.all(np.isfinite(chunk.data))
    assert chunk.data.dtype == np.float32


def test_chunk_data_equality_with_nan_compares_equal_to_itself():
    from semg_dsp.source import ChunkData

    data = np.array([[1.0, np.nan], [2.0, 3.0]], dtype=np.float32)
    c1 = ChunkData(data=data, start_sample_idx=0)
    c2 = ChunkData(data=data.copy(), start_sample_idx=0)
    assert c1 == c1
    assert c1 == c2


def test_chunk_data_nan_payloads_have_identical_hash_and_equality():
    from semg_dsp.source import ChunkData

    nan_payload = np.frombuffer(b"\x01\x00\xc0\x7f", dtype=np.float32)[0]
    nan_negative = np.frombuffer(b"\x00\x00\xc0\xff", dtype=np.float32)[0]

    c1 = ChunkData(data=np.array([[nan_payload]], dtype=np.float32), start_sample_idx=0)
    c2 = ChunkData(data=np.array([[nan_negative]], dtype=np.float32), start_sample_idx=0)
    c3 = ChunkData(data=np.array([[np.nan]], dtype=np.float32), start_sample_idx=0)

    assert c1 == c2 == c3
    assert hash(c1) == hash(c2) == hash(c3)


def test_synthetic_sample_source_parameters_stored_as_float32():
    from semg_dsp.source import SyntheticSampleSource

    s = SyntheticSampleSource(waveform="sine", amplitude=1.5, frequency_hz=10.0, phase_rad=0.5)
    assert isinstance(s.amplitude, np.float32)
    assert isinstance(s.frequency_hz, np.float32)
    assert isinstance(s.phase_rad, np.float32)
    assert isinstance(s.dc_offset, np.float32)
    assert isinstance(s.clip_limits[0], np.float32)

    s_m = SyntheticSampleSource(
        waveform="multi_tone",
        amplitude=[1.0, 0.5],
        frequency_hz=[5.0, 10.0],
        phase_rad=[0.1, 0.2],
    )
    assert all(isinstance(a, np.float32) for a in s_m.amplitude)
    assert all(isinstance(f, np.float32) for f in s_m.frequency_hz)
    assert all(isinstance(p, np.float32) for p in s_m.phase_rad)


def test_synthetic_sample_source_preserves_1e_minus_20_hz_at_1_billion_samples():
    from semg_dsp.source import SyntheticSampleSource

    src = SyntheticSampleSource(waveform="sine", frequency_hz=1e-20, sampling_rate_hz=1.0)
    src.total_samples_emitted = 1_000_000_000
    chunk = src.read_chunk(100)

    # Remainder is 10^9 * 1 = 10^9, denom is 10^20.
    # At large indices with large denominators, no int64 OverflowError is raised.
    assert not np.all(chunk.data == 0.0)
    assert np.all(np.isfinite(chunk.data))
    assert chunk.data.dtype == np.float32


def test_synthetic_sample_source_preserves_high_precision_sampling_rate_at_100m_samples():
    from semg_dsp.source import SyntheticSampleSource

    src = SyntheticSampleSource(
        waveform="sine",
        frequency_hz=10.0,
        sampling_rate_hz=1000.0000001,
    )
    src.total_samples_emitted = 100_000_000
    chunk = src.read_chunk(10)

    # Theoretical analytical value at t = 100,000,000 / 1000.0000001
    two_pi = 2.0 * np.pi
    t_expected = 100_000_000 / 1000.0000001
    expected_val = np.sin(two_pi * 10.0 * t_expected)

    diff = abs(chunk.data[0, 0] - expected_val)
    # Must preserve sampling_rate precision without rounding to 1000.0 Hz (where diff would be ~6.3e-4)
    assert diff < 1e-6
    assert chunk.data.dtype == np.float32


def test_synthetic_sample_source_rejects_clip_limits_rounding_to_identical_float32():
    from semg_dsp.source import SyntheticSampleSource

    # In float64, 1.0 < 1.0 + 1e-15 is True.
    # In float32, np.float32(1.0) == np.float32(1.0 + 1e-15) == 1.0.
    # Validation must cast to float32 before checking lower < upper.
    with pytest.raises(ValueError, match="clip_limits"):
        SyntheticSampleSource(waveform="saturation", clip_limits=(1.0, 1.0 + 1e-15))


def test_synthetic_sample_source_read_chunk_guards_counter_overflow():
    from semg_dsp.source import SyntheticSampleSource

    src = SyntheticSampleSource(waveform="sine")
    max_int64 = int(np.iinfo(np.int64).max)
    src.total_samples_emitted = max_int64 - 5

    with pytest.raises(OverflowError, match="signed 64-bit integer limit"):
        src.read_chunk(10)








