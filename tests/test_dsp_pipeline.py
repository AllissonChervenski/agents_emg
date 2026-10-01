"""Tests for semg_dsp.pipeline module contracts (T006, numeric_sensitive: true).

Requirements: FR-008, FR-009, FR-010, FR-011, FR-012.
Acceptance criteria: AC-003, AC-005, AC-007, AC-008, AC-009, AC-011.
Plan decisions: D-001, D-003, D-004, D-005, D-006.
"""

from pathlib import Path
import numpy as np
import pytest
from scipy.signal import sosfilt

from semg_dsp.source import ChunkData, SampleSource, SyntheticSampleSource
from semg_dsp.filter import CausalSosFilter
from semg_dsp.window import StatefulWindowBuffer

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "dsp" / "sos_test_filter.npz"
RTOL_SOS_FILTER_L1 = 1e-5
ATOL_SOS_FILTER_L1 = 1e-5


def _load_sos_fixture():
    with np.load(FIXTURE_PATH, allow_pickle=False) as data:
        return data["sos"].astype(np.float32), data["multi_tone_in"].astype(np.float32)


class ArraySampleSource(SampleSource):
    """Deterministic fixture-backed source with distinguishable channels."""

    def __init__(self, samples: np.ndarray) -> None:
        self.samples = samples
        self.num_channels = samples.shape[1]
        self.next_sample = 0

    def read_chunk(self, num_samples: int) -> ChunkData:
        start = self.next_sample
        end = start + num_samples
        if end > len(self.samples):
            raise ValueError("requested samples beyond fixture")
        self.next_sample = end
        return ChunkData(self.samples[start:end], start_sample_idx=start)

    def reset(self) -> None:
        self.next_sample = 0


def test_streaming_pipeline_init_and_properties():
    from semg_dsp.pipeline import StreamingPipeline

    sos, _ = _load_sos_fixture()
    source = SyntheticSampleSource(waveform="sine", sampling_rate_hz=1000.0, frequency_hz=50.0, num_channels=2)
    filt = CausalSosFilter(sos_coefficients=sos, num_channels=2)
    win = StatefulWindowBuffer(window_length=64, stride=16, num_channels=2)
    pipe = StreamingPipeline(source=source, filter_stage=filt, window_stage=win)

    assert pipe.source is source
    assert pipe.filter_stage is filt
    assert pipe.window_stage is win
    assert pipe.num_channels == 2


def test_streaming_pipeline_channel_mismatch_validation():
    from semg_dsp.pipeline import StreamingPipeline

    sos, _ = _load_sos_fixture()
    src_2ch = SyntheticSampleSource(waveform="sine", sampling_rate_hz=1000.0, frequency_hz=50.0, num_channels=2)
    filt_2ch = CausalSosFilter(sos_coefficients=sos, num_channels=2)
    win_1ch = StatefulWindowBuffer(window_length=64, stride=16, num_channels=1)

    with pytest.raises(ValueError, match="channel count mismatch"):
        StreamingPipeline(source=src_2ch, filter_stage=filt_2ch, window_stage=win_1ch)


def test_streaming_pipeline_vs_independent_scipy_oracle():
    """Validate end-to-end pipeline output against independent SciPy sosfilt + direct slicing oracle."""
    from semg_dsp.pipeline import StreamingPipeline

    sos, _ = _load_sos_fixture()
    fs = 1000.0
    freq = 60.0
    n_total = 200
    w = 32
    s = 8

    # 1. Independent Oracle: Generate pure mathematical signal and filter with SciPy
    t = (np.arange(n_total, dtype=np.float32) / fs)
    sine_ch0 = np.sin(2.0 * np.pi * freq * t).astype(np.float32)
    sine_ch1 = np.sin(2.0 * np.pi * freq * t).astype(np.float32)
    raw_samples = np.column_stack([sine_ch0, sine_ch1])

    # SciPy high-precision reference filtering
    scipy_filtered = sosfilt(sos, raw_samples, axis=0).astype(np.float32)

    # Independent slicing oracle:
    expected_windows = []
    cursor = 0
    while cursor + w <= n_total:
        expected_windows.append(scipy_filtered[cursor : cursor + w])
        cursor += s

    # 2. Actual StreamingPipeline Execution
    filt = CausalSosFilter(sos_coefficients=sos, num_channels=2)
    win = StatefulWindowBuffer(window_length=w, stride=s, num_channels=2)
    pipe = StreamingPipeline(
        source=SyntheticSampleSource(waveform="sine", sampling_rate_hz=fs, frequency_hz=freq, num_channels=2),
        filter_stage=filt,
        window_stage=win,
    )

    pipe_windows = []
    pipe_windows.extend(pipe.step(50))
    pipe_windows.extend(pipe.step(50))
    pipe_windows.extend(pipe.step(100))
    pipe_windows.extend(pipe.finalize())

    assert len(pipe_windows) == len(expected_windows)
    for i, (act, exp) in enumerate(zip(pipe_windows, expected_windows)):
        assert act.shape == (w, 2)
        assert act.dtype == np.float32
        np.testing.assert_allclose(act, exp, rtol=RTOL_SOS_FILTER_L1, atol=ATOL_SOS_FILTER_L1, err_msg=f"Window {i} deviation from SciPy oracle exceeds tolerance")


def test_streaming_pipeline_full_chunk_invariance():
    """Validate full-pipeline chunk invariance across arbitrary chunk partitioning (Run A vs Run B vs Run C)."""
    from semg_dsp.pipeline import StreamingPipeline

    sos, _ = _load_sos_fixture()
    fs = 1000.0
    freq = 45.0
    w = 32
    s = 8
    n_total = 200

    def run_pipeline(partitions):
        src = SyntheticSampleSource(waveform="multi_tone", sampling_rate_hz=fs, frequency_hz=freq, num_channels=2)
        filt = CausalSosFilter(sos_coefficients=sos, num_channels=2)
        win = StatefulWindowBuffer(window_length=w, stride=s, num_channels=2)
        pipe = StreamingPipeline(source=src, filter_stage=filt, window_stage=win)
        windows = []
        for chunk_len in partitions:
            windows.extend(pipe.step(chunk_len))
        windows.extend(pipe.finalize())
        return windows

    # Run A: Single monolithic chunk of 200 samples
    wins_a = run_pipeline([n_total])
    # Run B: Incremental chunks [1, 9, 30, 60, 100]
    wins_b = run_pipeline([1, 9, 30, 60, 100])
    # Run C: Irregular adversarial partitioning [3, 7, 13, 29, 31, 53, 44, 20]
    wins_c = run_pipeline([3, 7, 13, 29, 31, 53, 44, 20])

    assert len(wins_a) == len(wins_b) == len(wins_c)
    assert len(wins_a) > 0

    # Invariância bitwise exata entre execuções do próprio pipeline
    for i in range(len(wins_a)):
        np.testing.assert_array_equal(wins_b[i], wins_a[i], err_msg=f"Window {i} Run B vs Run A bitwise mismatch")
        np.testing.assert_array_equal(wins_c[i], wins_a[i], err_msg=f"Window {i} Run C vs Run A bitwise mismatch")


def test_streaming_pipeline_reset_determinism():
    """Validate that reset() restores all 3 pipeline stages to initial state."""
    from semg_dsp.pipeline import StreamingPipeline

    sos, _ = _load_sos_fixture()
    src = SyntheticSampleSource(waveform="sine", sampling_rate_hz=1000.0, frequency_hz=30.0, num_channels=2)
    filt = CausalSosFilter(sos_coefficients=sos, num_channels=2)
    win = StatefulWindowBuffer(window_length=32, stride=8, num_channels=2)
    pipe = StreamingPipeline(source=src, filter_stage=filt, window_stage=win)

    wins_run1 = pipe.step(100)
    assert len(wins_run1) > 0

    pipe.reset()
    assert pipe.source.total_samples_emitted == 0
    np.testing.assert_array_equal(pipe.filter_stage.state, np.zeros((2, 2, 2), dtype=np.float32))
    assert pipe.window_stage.buffered_samples == 0

    wins_run2 = pipe.step(100)
    assert len(wins_run2) == len(wins_run1)
    for i in range(len(wins_run1)):
        np.testing.assert_array_equal(wins_run2[i], wins_run1[i], err_msg=f"Window {i} not deterministic after reset")


def test_streaming_pipeline_multichannel_ordering_and_float32():
    """Validate multichannel column ordering is preserved end-to-end and strict float32 dtype."""
    from semg_dsp.pipeline import StreamingPipeline

    sos, _ = _load_sos_fixture()
    src = SyntheticSampleSource(waveform="dc", sampling_rate_hz=1000.0, amplitude=1.0, num_channels=2)
    filt = CausalSosFilter(sos_coefficients=sos, num_channels=2)
    win = StatefulWindowBuffer(window_length=16, stride=4, num_channels=2)
    pipe = StreamingPipeline(source=src, filter_stage=filt, window_stage=win)

    wins = pipe.step(40)
    assert len(wins) > 0
    for win in wins:
        assert win.shape == (16, 2)
        assert win.dtype == np.float32
        np.testing.assert_array_equal(win[:, 0], win[:, 1])


def test_streaming_pipeline_empty_step():
    from semg_dsp.pipeline import StreamingPipeline

    sos, _ = _load_sos_fixture()
    src = SyntheticSampleSource(waveform="sine", num_channels=2)
    filt = CausalSosFilter(sos_coefficients=sos, num_channels=2)
    win = StatefulWindowBuffer(window_length=16, stride=4, num_channels=2)
    pipe = StreamingPipeline(source=src, filter_stage=filt, window_stage=win)

    assert pipe.step(0) == []
    with pytest.raises(ValueError, match="num_samples"):
        pipe.step(-5)


def test_streaming_pipeline_distinct_channel_fixture_oracle():
    """TOL-SOS-FILTER-L1 + exact bitwise invariance across 3 chunk partitions using fixture with distinct channels."""
    from semg_dsp.pipeline import StreamingPipeline

    sos, samples = _load_sos_fixture()
    assert samples.shape == (200, 2)
    assert not np.array_equal(samples[:, 0], samples[:, 1])

    expected_filtered = sosfilt(sos.astype(np.float64), samples.astype(np.float64), axis=0)
    expected_windows = [expected_filtered[start : start + 32] for start in range(0, 169, 8)]
    partitions = ([200], [1, 9, 30, 60, 100], [3, 7, 13, 29, 31, 53, 44, 20])
    runs = []

    for sizes in partitions:
        source = ArraySampleSource(samples)
        filt = CausalSosFilter(sos_coefficients=sos, num_channels=2)
        win = StatefulWindowBuffer(window_length=32, stride=8, num_channels=2)
        pipeline = StreamingPipeline(source, filt, win)
        windows = []
        for size in sizes:
            windows.extend(pipeline.step(size))
        windows.extend(pipeline.finalize())
        assert source.next_sample == 200
        assert pipeline.filter_stage.state.dtype == np.float32
        assert pipeline.filter_stage.sos_coefficients.dtype == np.float32
        assert len(windows) == len(expected_windows) == 22
        for actual, expected in zip(windows, expected_windows):
            assert actual.shape == (32, 2)
            assert actual.dtype == np.float32
            np.testing.assert_allclose(
                actual,
                expected,
                rtol=RTOL_SOS_FILTER_L1,
                atol=ATOL_SOS_FILTER_L1,
            )
        runs.append(windows)

    # Invariância bitwise exata entre fragmentações
    for fragmented in runs[1:]:
        for actual, monolithic in zip(fragmented, runs[0]):
            assert actual.tobytes() == monolithic.tobytes()


def test_streaming_pipeline_state_continuity_finalize_pad_and_replay():
    """Source cursor, SciPy DF2T state, overlap, padding, and reset stay aligned."""
    from semg_dsp.pipeline import StreamingPipeline

    sos, samples = _load_sos_fixture()
    source = ArraySampleSource(samples)
    filt = CausalSosFilter(sos_coefficients=sos, num_channels=2)
    win = StatefulWindowBuffer(window_length=16, stride=5, num_channels=2, partial_policy="pad")
    pipeline = StreamingPipeline(source, filt, win)

    filtered_19, scipy_state = sosfilt(
        sos.astype(np.float64),
        samples[:19].astype(np.float64),
        axis=0,
        zi=np.zeros((len(sos), 2, 2), dtype=np.float64),
    )

    first_windows = pipeline.step(19)
    assert len(first_windows) == 1
    assert source.next_sample == 19
    assert pipeline.window_stage.buffered_samples == 14
    assert pipeline.filter_stage.state.shape == (len(sos), 2, 2)
    assert pipeline.filter_stage.state.dtype == np.float32
    np.testing.assert_allclose(
        pipeline.filter_stage.state,
        scipy_state,
        rtol=RTOL_SOS_FILTER_L1,
        atol=ATOL_SOS_FILTER_L1,
    )
    np.testing.assert_allclose(
        first_windows[0], filtered_19[:16],
        rtol=RTOL_SOS_FILTER_L1, atol=ATOL_SOS_FILTER_L1,
    )

    second_windows = pipeline.step(7)
    assert len(second_windows) == 2
    assert source.next_sample == 26
    assert pipeline.window_stage.buffered_samples == 11
    expected_filtered = sosfilt(sos.astype(np.float64), samples[:26].astype(np.float64), axis=0)
    for window, start in zip(second_windows, (5, 10)):
        np.testing.assert_allclose(
            window, expected_filtered[start : start + 16],
            rtol=RTOL_SOS_FILTER_L1, atol=ATOL_SOS_FILTER_L1,
        )

    padded = pipeline.finalize()
    assert len(padded) == 1
    assert padded[0].dtype == np.float32
    np.testing.assert_allclose(
        padded[0][:11], expected_filtered[15:26],
        rtol=RTOL_SOS_FILTER_L1, atol=ATOL_SOS_FILTER_L1,
    )
    np.testing.assert_array_equal(padded[0][11:], np.zeros((5, 2), dtype=np.float32))
    assert pipeline.window_stage.buffered_samples == 0
    assert pipeline.finalize() == []

    pipeline.reset()
    assert source.next_sample == 0
    assert pipeline.window_stage.buffered_samples == 0
    np.testing.assert_array_equal(pipeline.filter_stage.state, np.zeros_like(pipeline.filter_stage.state))
    replay = pipeline.step(19)
    assert len(replay) == 1
    assert replay[0].tobytes() == first_windows[0].tobytes()
    assert pipeline.window_stage.buffered_samples == 14


@pytest.mark.parametrize("source_channels,filter_channels,window_channels", [(1, 2, 2), (2, 1, 2), (2, 2, 1)])
def test_streaming_pipeline_rejects_each_stage_channel_mismatch(source_channels, filter_channels, window_channels):
    from semg_dsp.pipeline import StreamingPipeline

    sos, samples = _load_sos_fixture()
    source = ArraySampleSource(samples[:, :source_channels])
    filt = CausalSosFilter(sos_coefficients=sos[:, :], num_channels=filter_channels)
    window = StatefulWindowBuffer(16, 4, num_channels=window_channels)
    with pytest.raises(ValueError, match="channel"):
        StreamingPipeline(source, filt, window)


def test_streaming_pipeline_synthetic_source_phase_continuity():
    from semg_dsp.pipeline import StreamingPipeline

    sos, _ = _load_sos_fixture()
    source = SyntheticSampleSource(
        waveform="multi_tone",
        num_channels=2,
        sampling_rate_hz=1000.0,
        amplitude=(0.7, 0.2),
        frequency_hz=(43.0, 117.0),
        phase_rad=(0.1, -0.3),
    )
    filt = CausalSosFilter(sos_coefficients=sos, num_channels=2)
    win = StatefulWindowBuffer(window_length=16, stride=4, num_channels=2)
    pipeline = StreamingPipeline(source, filt, win)

    windows = pipeline.step(7) + pipeline.step(13) + pipeline.step(29)
    assert source.total_samples_emitted == 49
    assert len(windows) == 9

    time = np.arange(49, dtype=np.float64) / 1000.0
    independent_signal = (
        0.7 * np.sin(2 * np.pi * 43.0 * time + 0.1)
        + 0.2 * np.sin(2 * np.pi * 117.0 * time - 0.3)
    ).astype(np.float32)
    raw = np.column_stack([independent_signal, independent_signal])
    expected = sosfilt(sos.astype(np.float64), raw.astype(np.float64), axis=0)
    for index, window in enumerate(windows):
        assert window.dtype == np.float32
        np.testing.assert_allclose(
            window, expected[index * 4 : index * 4 + 16],
            rtol=RTOL_SOS_FILTER_L1, atol=ATOL_SOS_FILTER_L1,
        )
