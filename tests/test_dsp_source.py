"""Tests for semg_dsp.source module contracts (T002).

Requirements: FR-001, FR-008
Acceptance criteria: AC-001, AC-005, AC-010
Plan decisions: D-001, D-003
"""

from dataclasses import FrozenInstanceError
import numpy as np
import pytest


def test_chunk_data_valid_2d_float32():
    from semg_dsp.source import ChunkData

    data = np.zeros((10, 2), dtype=np.float32)
    chunk = ChunkData(data=data, start_sample_idx=0)

    assert chunk.data.shape == (10, 2)
    assert chunk.data.dtype == np.float32
    assert chunk.num_samples == 10
    assert chunk.num_channels == 2
    assert chunk.start_sample_idx == 0


def test_chunk_data_rejects_non_2d_shape():
    from semg_dsp.source import ChunkData

    with pytest.raises(ValueError, match="2D"):
        ChunkData(data=np.zeros(10, dtype=np.float32))

    with pytest.raises(ValueError, match="2D"):
        ChunkData(data=np.zeros((10, 2, 2), dtype=np.float32))


def test_chunk_data_rejects_non_float32_dtype():
    from semg_dsp.source import ChunkData

    with pytest.raises(TypeError, match="float32"):
        ChunkData(data=np.zeros((10, 2), dtype=np.float64))

    with pytest.raises(TypeError, match="float32"):
        ChunkData(data=np.zeros((10, 2), dtype=np.int32))


def test_chunk_data_rejects_negative_start_index():
    from semg_dsp.source import ChunkData

    with pytest.raises(ValueError, match="non-negative"):
        ChunkData(data=np.zeros((10, 2), dtype=np.float32), start_sample_idx=-1)


def test_chunk_data_preserves_temporal_and_channel_layout():
    from semg_dsp.source import ChunkData

    raw = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    chunk = ChunkData(data=raw, start_sample_idx=42)

    np.testing.assert_array_equal(chunk.data, raw)
    assert chunk.start_sample_idx == 42
    assert chunk.data[0, 0] == 1.0
    assert chunk.data[0, 1] == 2.0
    assert chunk.data[2, 1] == 6.0


def test_chunk_data_is_immutable():
    from semg_dsp.source import ChunkData

    chunk = ChunkData(data=np.zeros((5, 1), dtype=np.float32), start_sample_idx=0)
    with pytest.raises(FrozenInstanceError):
        chunk.start_sample_idx = 10  # type: ignore[misc]


def test_chunk_data_buffer_is_read_only_and_cannot_be_unfrozen():
    from semg_dsp.source import ChunkData

    chunk = ChunkData(data=np.zeros((5, 1), dtype=np.float32), start_sample_idx=0)
    assert chunk.data.flags.writeable is False
    with pytest.raises(ValueError, match="read-only"):
        chunk.data[0, 0] = 99.0
    with pytest.raises(ValueError, match="cannot set WRITEABLE flag to True"):
        chunk.data.flags.writeable = True


def test_chunk_data_no_side_effects_on_caller_and_no_view_aliasing():
    from semg_dsp.source import ChunkData

    orig = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    chunk = ChunkData(data=orig, start_sample_idx=0)
    assert orig.flags.writeable is True
    orig[0, 0] = 99.0
    assert chunk.data[0, 0] == 1.0


def test_chunk_data_equality_and_hash_including_signed_zeros():
    from semg_dsp.source import ChunkData

    arr1 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    arr2 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    arr3 = np.array([[1.0, 2.0], [3.0, 5.0]], dtype=np.float32)

    c1 = ChunkData(data=arr1, start_sample_idx=0)
    c2 = ChunkData(data=arr2, start_sample_idx=0)
    c3 = ChunkData(data=arr3, start_sample_idx=0)
    c4 = ChunkData(data=arr1, start_sample_idx=1)

    assert c1 == c2
    assert c1 != c3
    assert c1 != c4
    assert c1 != "not_a_chunk"
    assert hash(c1) == hash(c2)
    assert len({c1, c2, c3, c4}) == 3

    # Signed zero invariant: +0.0 vs -0.0
    c_pos0 = ChunkData(data=np.array([[0.0, 1.0]], dtype=np.float32), start_sample_idx=0)
    c_neg0 = ChunkData(data=np.array([[-0.0, 1.0]], dtype=np.float32), start_sample_idx=0)
    assert c_pos0 == c_neg0
    assert hash(c_pos0) == hash(c_neg0)
    assert len({c_pos0, c_neg0}) == 1


def test_chunk_data_start_sample_idx_validation():
    from semg_dsp.source import ChunkData

    # Must reject booleans
    with pytest.raises(TypeError, match="integer"):
        ChunkData(data=np.zeros((2, 1), dtype=np.float32), start_sample_idx=True)

    with pytest.raises(TypeError, match="integer"):
        ChunkData(data=np.zeros((2, 1), dtype=np.float32), start_sample_idx=False)

    # Must reject non-integers
    with pytest.raises(TypeError, match="integer"):
        ChunkData(data=np.zeros((2, 1), dtype=np.float32), start_sample_idx=1.5)  # type: ignore[arg-type]

    # Must accept numpy integers and normalize to Python int
    c = ChunkData(data=np.zeros((2, 1), dtype=np.float32), start_sample_idx=np.int64(7))
    assert c.start_sample_idx == 7
    assert type(c.start_sample_idx) is int


def test_chunk_data_rejects_non_ndarray():
    from semg_dsp.source import ChunkData

    with pytest.raises(TypeError, match="ndarray"):
        ChunkData(data=[[1.0, 2.0], [3.0, 4.0]], start_sample_idx=0)  # type: ignore[arg-type]


def test_sample_source_cannot_be_instantiated_directly():
    from semg_dsp.source import SampleSource

    with pytest.raises(TypeError):
        SampleSource()  # type: ignore[abstract]


def test_sample_source_incomplete_subclass_raises():
    from semg_dsp.source import SampleSource

    class IncompleteSource(SampleSource):
        def reset(self) -> None:
            pass

    with pytest.raises(TypeError):
        IncompleteSource()  # type: ignore[abstract]


def test_sample_source_subclass_protocol():
    from semg_dsp.source import ChunkData, SampleSource

    class DummySource(SampleSource):
        def __init__(self):
            self.cursor = 0

        def read_chunk(self, num_samples: int) -> ChunkData:
            chunk = ChunkData(
                data=np.zeros((num_samples, 2), dtype=np.float32),
                start_sample_idx=self.cursor,
            )
            self.cursor += num_samples
            return chunk

        def reset(self) -> None:
            self.cursor = 0

    src = DummySource()
    c1 = src.read_chunk(5)
    assert isinstance(c1, ChunkData)
    assert c1.num_samples == 5
    assert c1.start_sample_idx == 0

    c2 = src.read_chunk(3)
    assert c2.num_samples == 3
    assert c2.start_sample_idx == 5

    src.reset()
    c3 = src.read_chunk(2)
    assert c3.start_sample_idx == 0
