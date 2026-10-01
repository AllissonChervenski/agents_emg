"""Tests for semg_dsp.window module contracts (T005, numeric_sensitive: true).

Requirements: FR-006, FR-007, FR-008, FR-010, FR-011.
Acceptance criteria: AC-004, AC-005, AC-006, AC-011.
Plan decisions: D-003, D-005, D-007.
"""

import numpy as np
import pytest

from semg_dsp.source import ChunkData


def test_window_buffer_init_and_properties():
    from semg_dsp.window import StatefulWindowBuffer
    buf = StatefulWindowBuffer(window_length=64, stride=16, num_channels=2, partial_policy="drop")
    assert buf.window_length == 64
    assert buf.stride == 16
    assert buf.num_channels == 2
    assert buf.partial_policy == "drop"
    assert buf.buffered_samples == 0


def test_window_buffer_overlap_and_advance_exact_sequence():
    """Validate that window = buffer[0:W] and buffer advances by stride S, preserving overlap."""
    from semg_dsp.window import StatefulWindowBuffer
    w = 8
    s = 3
    buf = StatefulWindowBuffer(window_length=w, stride=s, num_channels=2)

    # Create continuous ramp signal: shape (20, 2)
    samples = np.column_stack([
        np.arange(20, dtype=np.float32),
        np.arange(100, 120, dtype=np.float32),
    ])

    windows = buf.process_chunk(ChunkData(data=samples, start_sample_idx=0))

    # Independent slicing oracle:
    # window 0: samples[0:8]
    # window 1: samples[3:11]
    # window 2: samples[6:14]
    # window 3: samples[9:17]
    # window 4: samples[12:20]
    expected_slices = [
        samples[0:8],
        samples[3:11],
        samples[6:14],
        samples[9:17],
        samples[12:20],
    ]
    assert len(windows) == len(expected_slices)
    for i, (win, exp) in enumerate(zip(windows, expected_slices)):
        assert win.shape == (w, 2)
        assert win.dtype == np.float32
        np.testing.assert_array_equal(win, exp, err_msg=f"Window {i} index mismatch")

    # Residual: samples[15:20] has 5 samples remaining (5 < 8)
    assert buf.buffered_samples == 5


def test_window_buffer_chunk_invariance_and_adversarial_boundaries():
    """Validate chunk invariance: batch execution == arbitrary fragmented chunk execution (TOL-WINDOW-ACCUMULATION)."""
    from semg_dsp.window import StatefulWindowBuffer
    w = 16
    s = 4
    n_total = 200
    samples = np.random.RandomState(42).randn(n_total, 2).astype(np.float32)

    # 1. Full batch execution
    buf_full = StatefulWindowBuffer(window_length=w, stride=s, num_channels=2)
    windows_full = buf_full.process_chunk(ChunkData(data=samples, start_sample_idx=0))
    rem_full = buf_full.finalize()
    all_full = windows_full + rem_full

    # 2. Fragmented execution with adversarial boundaries [1, W-1, 1, S-1, 7, 13, 29, 31, 53, 44]
    buf_chunked = StatefulWindowBuffer(window_length=w, stride=s, num_channels=2)
    partition = [1, w - 1, 1, s - 1, 7, 13, 29, 31, 53, 44]
    assert sum(partition) < n_total
    partition.append(n_total - sum(partition))
    assert sum(partition) == n_total

    windows_chunked = []
    cursor = 0
    for chunk_len in partition:
        chunk_data = samples[cursor : cursor + chunk_len]
        c_wins = buf_chunked.process_chunk(ChunkData(data=chunk_data, start_sample_idx=cursor))
        windows_chunked.extend(c_wins)
        cursor += chunk_len

    rem_chunked = buf_chunked.finalize()
    all_chunked = windows_chunked + rem_chunked

    assert len(all_full) == len(all_chunked)
    # TOL-WINDOW-ACCUMULATION: rtol=0.0, atol=0.0 (bitwise exact identity)
    for i in range(len(all_full)):
        np.testing.assert_array_equal(all_chunked[i], all_full[i], err_msg=f"Window {i} chunk invariance failed")


def test_window_buffer_chunk_smaller_than_window():
    """Chunk smaller than window_length emits no window and retains all samples."""
    from semg_dsp.window import StatefulWindowBuffer
    buf = StatefulWindowBuffer(window_length=64, stride=16, num_channels=2)
    chunk = np.zeros((10, 2), dtype=np.float32)
    wins = buf.process_chunk(chunk)
    assert wins == []
    assert buf.buffered_samples == 10


def test_window_buffer_exact_window_chunk():
    """Chunk exactly window_length with stride=window_length emits exactly 1 window."""
    from semg_dsp.window import StatefulWindowBuffer
    buf = StatefulWindowBuffer(window_length=32, stride=32, num_channels=2)
    chunk = np.ones((32, 2), dtype=np.float32)
    wins = buf.process_chunk(chunk)
    assert len(wins) == 1
    assert wins[0].shape == (32, 2)
    assert buf.buffered_samples == 0


def test_window_buffer_multichannel_ordering():
    """Validate multichannel ordering: channel values remain in exact column positions."""
    from semg_dsp.window import StatefulWindowBuffer
    buf = StatefulWindowBuffer(window_length=8, stride=4, num_channels=3)
    ch0 = np.full((16, 1), 10.0, dtype=np.float32)
    ch1 = np.full((16, 1), 20.0, dtype=np.float32)
    ch2 = np.full((16, 1), 30.0, dtype=np.float32)
    data = np.hstack([ch0, ch1, ch2])

    wins = buf.process_chunk(data)
    assert len(wins) == 3
    for win in wins:
        assert win.shape == (8, 3)
        assert np.all(win[:, 0] == 10.0)
        assert np.all(win[:, 1] == 20.0)
        assert np.all(win[:, 2] == 30.0)


def test_window_buffer_finalize_drop_policy():
    """finalize under 'drop' policy discards incomplete tail and clears buffer."""
    from semg_dsp.window import StatefulWindowBuffer
    buf = StatefulWindowBuffer(window_length=16, stride=8, num_channels=1, partial_policy="drop")
    # 20 samples: window 0 (0..16, advances to 8). Residual: 8..20 (12 samples < 16)
    wins = buf.process_chunk(np.zeros((20, 1), dtype=np.float32))
    assert len(wins) == 1
    assert buf.buffered_samples == 12

    tail = buf.finalize()
    assert tail == []
    assert buf.buffered_samples == 0
    # Calling finalize again returns empty list
    assert buf.finalize() == []


def test_window_buffer_finalize_pad_policy():
    """finalize under 'pad' policy zero-pads incomplete tail up to window_length."""
    from semg_dsp.window import StatefulWindowBuffer
    buf = StatefulWindowBuffer(window_length=10, stride=5, num_channels=1, partial_policy="pad")
    # Feed 7 samples: no window emitted
    inp = np.full((7, 1), 5.0, dtype=np.float32)
    wins = buf.process_chunk(inp)
    assert wins == []
    assert buf.buffered_samples == 7

    tail = buf.finalize()
    assert len(tail) == 1
    assert tail[0].shape == (10, 1)
    assert tail[0].dtype == np.float32
    np.testing.assert_array_equal(tail[0][:7], inp)
    np.testing.assert_array_equal(tail[0][7:], np.zeros((3, 1), dtype=np.float32))
    assert buf.buffered_samples == 0


def test_window_buffer_reset():
    from semg_dsp.window import StatefulWindowBuffer
    buf = StatefulWindowBuffer(window_length=10, stride=5, num_channels=2)
    buf.process_chunk(np.zeros((7, 2), dtype=np.float32))
    assert buf.buffered_samples == 7
    buf.reset()
    assert buf.buffered_samples == 0

    # Fresh chunk after reset
    wins = buf.process_chunk(np.ones((10, 2), dtype=np.float32))
    assert len(wins) == 1
    np.testing.assert_array_equal(wins[0], np.ones((10, 2), dtype=np.float32))


def test_window_buffer_validations():
    from semg_dsp.window import StatefulWindowBuffer
    # window_length validations
    with pytest.raises(ValueError, match="window_length"):
        StatefulWindowBuffer(window_length=0, stride=1, num_channels=1)
    with pytest.raises(TypeError, match="window_length"):
        StatefulWindowBuffer(window_length=True, stride=1, num_channels=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="window_length"):
        StatefulWindowBuffer(window_length=8.5, stride=1, num_channels=1)  # type: ignore[arg-type]

    # stride validations (1 <= stride <= window_length)
    with pytest.raises(ValueError, match="stride"):
        StatefulWindowBuffer(window_length=8, stride=0, num_channels=1)
    with pytest.raises(ValueError, match="stride"):
        StatefulWindowBuffer(window_length=8, stride=9, num_channels=1)
    with pytest.raises(TypeError, match="stride"):
        StatefulWindowBuffer(window_length=8, stride=False, num_channels=1)  # type: ignore[arg-type]

    # num_channels validations
    with pytest.raises(ValueError, match="num_channels"):
        StatefulWindowBuffer(window_length=8, stride=4, num_channels=0)

    # partial_policy validations
    with pytest.raises(ValueError, match="partial_policy"):
        StatefulWindowBuffer(window_length=8, stride=4, num_channels=1, partial_policy="unknown")

    buf = StatefulWindowBuffer(window_length=8, stride=4, num_channels=2)
    # Non-2D input
    with pytest.raises(ValueError, match="2D"):
        buf.process_chunk(np.zeros(10, dtype=np.float32))
    # Channel mismatch
    with pytest.raises(ValueError, match="channels"):
        buf.process_chunk(np.zeros((10, 3), dtype=np.float32))
    # Non-float32 dtype
    with pytest.raises(TypeError, match="float32"):
        buf.process_chunk(np.zeros((10, 2), dtype=np.float64))
    # NaN / Inf rejection
    nan_chunk = np.zeros((10, 2), dtype=np.float32)
    nan_chunk[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN|finite"):
        buf.process_chunk(nan_chunk)
    inf_chunk = np.zeros((10, 2), dtype=np.float32)
    inf_chunk[0, 0] = np.inf
    with pytest.raises(ValueError, match="Inf|finite"):
        buf.process_chunk(inf_chunk)


def test_window_buffer_bounded_memory():
    """Validate that internal buffer memory remains strictly bounded during continuous streaming."""
    from semg_dsp.window import StatefulWindowBuffer
    w = 64
    s = 16
    buf = StatefulWindowBuffer(window_length=w, stride=s, num_channels=2)

    # Stream 10,000 samples in small chunks of 13 samples
    chunk = np.ones((13, 2), dtype=np.float32)
    for _ in range(500):
        buf.process_chunk(chunk)
        # Buffer must never hold more than window_length + chunk_size samples
        assert buf.buffered_samples < (w + 13)
