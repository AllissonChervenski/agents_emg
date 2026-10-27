"""Tests for Frozen Dataset Views (refined default vs stimulus secondary) and pair isolation."""

import numpy as np
import pytest

from semg_dataset.contract import (
    AlignmentRecord,
    AlignmentStatus,
    DatasetView,
    FrozenViewManager,
    RecordingData,
)


@pytest.fixture
def sample_recording() -> RecordingData:
    """Create a minimal synthetic RecordingData instance."""
    n = 100
    emg = np.random.randn(n, 12).astype(np.float32) * 0.001
    stim = np.zeros(n, dtype=np.int8)
    stim[20:40] = 5
    rep = np.zeros(n, dtype=np.int8)
    rep[20:40] = 1

    restim = np.zeros(n, dtype=np.int8)
    restim[25:42] = 5
    rerep = np.zeros(n, dtype=np.int8)
    rerep[25:42] = 1

    alignment = AlignmentRecord(
        status=AlignmentStatus.IDENTICAL,
        original_emg_length=n,
        original_refined_length=n,
        delta_samples=0,
        start_anchored_proven=True,
    )

    return RecordingData(
        subject_id="S1",
        exercise_id="E1",
        emg=emg,
        stimulus=stim,
        repetition=rep,
        restimulus=restim,
        rerepetition=rerep,
        alignment=alignment,
    )


def test_refined_view_exports_restimulus_and_rerepetition(sample_recording: RecordingData) -> None:
    """The 'refined' view must export restimulus and rerepetition pairs exclusively."""
    view = sample_recording.get_view("refined")

    assert isinstance(view, DatasetView)
    assert view.view_id == "refined"
    assert np.array_equal(view.signals, sample_recording.emg)
    assert np.array_equal(view.labels, sample_recording.restimulus)
    assert np.array_equal(view.repetitions, sample_recording.rerepetition)
    assert isinstance(view.view_hash, str)
    assert len(view.view_hash) == 64


def test_stimulus_view_exports_stimulus_and_repetition(sample_recording: RecordingData) -> None:
    """The 'stimulus' view must export stimulus and repetition pairs exclusively."""
    view = sample_recording.get_view("stimulus")

    assert isinstance(view, DatasetView)
    assert view.view_id == "stimulus"
    assert np.array_equal(view.signals, sample_recording.emg)
    assert np.array_equal(view.labels, sample_recording.stimulus)
    assert np.array_equal(view.repetitions, sample_recording.repetition)
    assert isinstance(view.view_hash, str)
    assert len(view.view_hash) == 64


def test_view_hashes_differ_between_refined_and_stimulus(sample_recording: RecordingData) -> None:
    """Refined and stimulus views must generate distinct SHA-256 hashes."""
    view_refined = sample_recording.get_view("refined")
    view_stim = sample_recording.get_view("stimulus")

    assert view_refined.view_hash != view_stim.view_hash


def test_view_hash_is_deterministic(sample_recording: RecordingData) -> None:
    """Consecutive view generations on identical data must yield identical view_hash."""
    view1 = sample_recording.get_view("refined")
    view2 = sample_recording.get_view("refined")

    assert view1.view_hash == view2.view_hash


def test_invalid_view_id_raises_value_error(sample_recording: RecordingData) -> None:
    """Requesting an unknown view identifier must raise ValueError."""
    with pytest.raises(ValueError, match="Unknown view_id 'hybrid'"):
        sample_recording.get_view("hybrid")


def test_cross_pair_mixing_is_strictly_prohibited() -> None:
    """Attempting to construct a view mixing stimulus and rerepetition must raise ValueError."""
    n = 50
    signals = np.zeros((n, 12), dtype=np.float32)
    stimulus = np.ones(n, dtype=np.int8)
    rerepetition = np.ones(n, dtype=np.int8)

    with pytest.raises(ValueError, match="[Cc]ross-pair mixing"):
        FrozenViewManager.create_view_explicit(
            view_id="illegal_mixed",
            signals=signals,
            labels=stimulus,
            repetitions=rerepetition,
            label_source="stimulus",
            repetition_source="rerepetition",
        )
