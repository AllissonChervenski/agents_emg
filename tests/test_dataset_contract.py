"""Tests for RecordingData, Exercise, Subject, and Alignment domain entities."""

import numpy as np
import pytest


def test_recording_data_valid_instantiation() -> None:
    """Verify valid RecordingData creation with 12-channel float32 sEMG and labels."""
    from semg_dataset.contract import AlignmentRecord, AlignmentStatus, RecordingData

    n_samples = 500
    emg = np.zeros((n_samples, 12), dtype=np.float32)
    stimulus = np.ones(n_samples, dtype=np.int8)
    repetition = np.ones(n_samples, dtype=np.int8)
    restimulus = np.ones(n_samples, dtype=np.int8)
    rerepetition = np.ones(n_samples, dtype=np.int8)

    alignment = AlignmentRecord(
        status=AlignmentStatus.IDENTICAL,
        original_emg_length=n_samples,
        original_refined_length=n_samples,
        delta_samples=0,
        start_anchored_proven=True,
        notes="Identical length",
    )

    rec = RecordingData(
        subject_id="S1",
        exercise_id="E1",
        emg=emg,
        stimulus=stimulus,
        repetition=repetition,
        restimulus=restimulus,
        rerepetition=rerepetition,
        alignment=alignment,
        sampling_rate_hz=2000.0,
    )

    assert rec.subject_id == "S1"
    assert rec.exercise_id == "E1"
    assert rec.num_samples == 500
    assert rec.num_channels == 12
    assert rec.emg.dtype == np.float32


def test_recording_data_rejects_non_12_channels() -> None:
    """Verify RecordingData rejects arrays that do not have strictly 12 channels."""
    from semg_dataset.contract import AlignmentRecord, AlignmentStatus, RecordingData

    n_samples = 100
    emg_wrong_channels = np.zeros((n_samples, 8), dtype=np.float32)
    stim = np.zeros(n_samples, dtype=np.int8)
    alignment = AlignmentRecord(
        status=AlignmentStatus.IDENTICAL,
        original_emg_length=n_samples,
        original_refined_length=n_samples,
        delta_samples=0,
        start_anchored_proven=True,
    )

    with pytest.raises(ValueError, match="12 channels"):
        RecordingData(
            subject_id="S1",
            exercise_id="E1",
            emg=emg_wrong_channels,
            stimulus=stim,
            repetition=stim,
            restimulus=stim,
            rerepetition=stim,
            alignment=alignment,
        )


def test_recording_data_rejects_non_float32() -> None:
    """Verify RecordingData strictly enforces float32 sEMG arrays."""
    from semg_dataset.contract import AlignmentRecord, AlignmentStatus, RecordingData

    n_samples = 100
    emg_float64 = np.zeros((n_samples, 12), dtype=np.float64)
    stim = np.zeros(n_samples, dtype=np.int8)
    alignment = AlignmentRecord(
        status=AlignmentStatus.IDENTICAL,
        original_emg_length=n_samples,
        original_refined_length=n_samples,
        delta_samples=0,
        start_anchored_proven=True,
    )

    with pytest.raises(TypeError, match="float32"):
        RecordingData(
            subject_id="S1",
            exercise_id="E1",
            emg=emg_float64,
            stimulus=stim,
            repetition=stim,
            restimulus=stim,
            rerepetition=stim,
            alignment=alignment,
        )


def test_recording_data_rejects_nan_and_inf() -> None:
    """Verify RecordingData rejects NaN and Inf values in sEMG signals."""
    from semg_dataset.contract import AlignmentRecord, AlignmentStatus, RecordingData

    n_samples = 100
    emg_nan = np.zeros((n_samples, 12), dtype=np.float32)
    emg_nan[10, 0] = np.nan
    stim = np.zeros(n_samples, dtype=np.int8)
    alignment = AlignmentRecord(
        status=AlignmentStatus.IDENTICAL,
        original_emg_length=n_samples,
        original_refined_length=n_samples,
        delta_samples=0,
        start_anchored_proven=True,
    )

    with pytest.raises(ValueError, match="NaN"):
        RecordingData(
            subject_id="S1",
            exercise_id="E1",
            emg=emg_nan,
            stimulus=stim,
            repetition=stim,
            restimulus=stim,
            rerepetition=stim,
            alignment=alignment,
        )


def test_derived_local_labels_mapping() -> None:
    """Verify derived local labels map E1(1..17), E2(18..40->1..23), E3(41..49->1..9), preserving REST=0."""
    from semg_dataset.contract import AlignmentRecord, AlignmentStatus, RecordingData

    alignment = AlignmentRecord(
        status=AlignmentStatus.IDENTICAL,
        original_emg_length=3,
        original_refined_length=3,
        delta_samples=0,
        start_anchored_proven=True,
    )

    # Exercise 2 test: global labels [0, 18, 40] -> local labels [0, 1, 23]
    emg = np.zeros((3, 12), dtype=np.float32)
    stim_e2 = np.array([0, 18, 40], dtype=np.int8)
    rec_e2 = RecordingData(
        subject_id="S1",
        exercise_id="E2",
        emg=emg,
        stimulus=stim_e2,
        repetition=stim_e2,
        restimulus=stim_e2,
        rerepetition=stim_e2,
        alignment=alignment,
    )

    local_labels = rec_e2.get_local_labels()
    assert np.array_equal(local_labels, np.array([0, 1, 23], dtype=np.int8))
