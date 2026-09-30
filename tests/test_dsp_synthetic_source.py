"""T003 RED: independent L0 oracle for SyntheticSampleSource.

FR-002, FR-008, FR-010, FR-011; AC-001, AC-005, AC-006, AC-011.
The fixture is generated from analytical formulas without importing semg_dsp.
TOL-ANALYTICAL-L0 is PROVISIONAL: rtol=1e-6, atol=1e-6.
"""

from pathlib import Path

import numpy as np
import pytest


FIXTURE = Path(__file__).parent / "fixtures" / "dsp" / "l0_analytical_cases.npz"
ANALYTICAL_RTOL = 1e-6
ANALYTICAL_ATOL = 1e-6


@pytest.fixture(scope="module")
def l0_cases():
    with np.load(FIXTURE, allow_pickle=False) as cases:
        yield cases


@pytest.mark.parametrize(
    ("waveform", "parameters"),
    [
        ("zeros", {}),
        ("dc", {"dc_offset": 2.5}),
        ("impulse", {"event_sample_idx": 15}),
        ("step", {"event_sample_idx": 30}),
        (
            "sine",
            {"amplitude": 1.5, "frequency_hz": 10.0, "phase_rad": np.pi / 6.0},
        ),
        (
            "multi_tone",
            {
                "amplitude": [1.0, 0.5, 0.25],
                "frequency_hz": [5.0, 25.0, 60.0],
                "phase_rad": [0.0, np.pi / 4.0, np.pi / 3.0],
            },
        ),
        (
            "saturation",
            {"amplitude": 2.5, "frequency_hz": 5.0, "clip_limits": (-1.0, 1.0)},
        ),
    ],
)
def test_l0_waveforms_match_independent_fixture(waveform, parameters, l0_cases):
    from semg_dsp.source import ChunkData, SyntheticSampleSource

    source = SyntheticSampleSource(
        waveform=waveform, num_channels=2, sampling_rate_hz=1000.0, **parameters
    )
    chunk = source.read_chunk(200)
    expected = l0_cases[waveform]

    assert isinstance(chunk, ChunkData)
    assert chunk.start_sample_idx == 0
    assert chunk.data.shape == expected.shape == (200, 2)
    assert chunk.data.dtype == expected.dtype == np.float32
    assert np.isfinite(chunk.data).all()
    assert source.total_samples_emitted == 200

    if waveform in {"zeros", "dc", "impulse", "step"}:
        np.testing.assert_array_equal(chunk.data, expected)
    else:
        # TOL-ANALYTICAL-L0: float32 against independent analytical fixture.
        np.testing.assert_allclose(
            chunk.data, expected, rtol=ANALYTICAL_RTOL, atol=ANALYTICAL_ATOL
        )

    if waveform == "saturation":
        lower, upper = parameters["clip_limits"]
        for limit in (lower, upper):
            mask = expected == limit
            assert mask.any()
            np.testing.assert_array_equal(chunk.data[mask], expected[mask])
        assert np.all((chunk.data >= lower) & (chunk.data <= upper))


def test_chunk_partition_preserves_phase_and_sample_indices(l0_cases):
    from semg_dsp.source import SyntheticSampleSource

    parameters = {
        "waveform": "sine",
        "num_channels": 2,
        "sampling_rate_hz": 1000.0,
        "amplitude": 1.5,
        "frequency_hz": 10.0,
        "phase_rad": np.pi / 6.0,
    }
    split = SyntheticSampleSource(**parameters)
    first = split.read_chunk(100)
    second = split.read_chunk(100)
    whole = SyntheticSampleSource(**parameters).read_chunk(200)

    assert (first.start_sample_idx, second.start_sample_idx) == (0, 100)
    assert split.total_samples_emitted == 200
    np.testing.assert_array_equal(np.concatenate((first.data, second.data)), whole.data)
    np.testing.assert_allclose(
        whole.data, l0_cases["sine"], rtol=ANALYTICAL_RTOL, atol=ANALYTICAL_ATOL
    )

    fragmented = SyntheticSampleSource(**parameters)
    lengths = (1, 9, 30, 60, 100)
    chunks = [fragmented.read_chunk(length) for length in lengths]
    assert [chunk.start_sample_idx for chunk in chunks] == [0, 1, 10, 40, 100]
    np.testing.assert_array_equal(
        np.concatenate([chunk.data for chunk in chunks]), whole.data
    )


def test_reset_restarts_counter_and_initial_sequence():
    from semg_dsp.source import SyntheticSampleSource

    source = SyntheticSampleSource(
        waveform="multi_tone",
        num_channels=2,
        sampling_rate_hz=1000.0,
        amplitude=[1.0, 0.5, 0.25],
        frequency_hz=[5.0, 25.0, 60.0],
        phase_rad=[0.0, np.pi / 4.0, np.pi / 3.0],
    )
    initial = source.read_chunk(64)
    source.read_chunk(17)
    assert source.total_samples_emitted == 81

    source.reset()
    assert source.total_samples_emitted == 0
    repeated = source.read_chunk(64)
    assert repeated.start_sample_idx == 0
    assert source.total_samples_emitted == 64
    np.testing.assert_array_equal(repeated.data, initial.data)


@pytest.mark.parametrize(
    ("parameters", "error_name"),
    [
        ({"waveform": "unknown"}, "waveform"),
        ({"sampling_rate_hz": 0.0}, "sampling_rate_hz"),
        ({"sampling_rate_hz": -1000.0}, "sampling_rate_hz"),
        ({"num_channels": 0}, "num_channels"),
        ({"num_channels": -1}, "num_channels"),
        ({"clip_limits": (1.0, -1.0)}, "clip_limits"),
        ({"clip_limits": (1.0, 1.0)}, "clip_limits"),
    ],
)
def test_invalid_parameters_raise_value_error(parameters, error_name):
    from semg_dsp.source import SyntheticSampleSource

    with pytest.raises(ValueError, match=error_name):
        SyntheticSampleSource(**parameters)
