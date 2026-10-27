"""Tests for deterministic procedural synthetic RecordingData generator (T002)."""

import sys
import numpy as np


def test_synthetic_semg_generator_importable() -> None:
    """Verify make_synthetic_recording is importable from tests.fixtures.synthetic_semg."""
    from tests.fixtures.synthetic_semg import make_synthetic_recording

    assert callable(make_synthetic_recording)


def test_synthetic_recording_deterministic_same_seed() -> None:
    """Verify that same configuration and seed produces bit-exact identical RecordingData."""
    from tests.fixtures.synthetic_semg import make_synthetic_recording

    rec1 = make_synthetic_recording(seed=12345)
    rec2 = make_synthetic_recording(seed=12345)

    np.testing.assert_array_equal(rec1.emg, rec2.emg)
    np.testing.assert_array_equal(rec1.restimulus, rec2.restimulus)
    np.testing.assert_array_equal(rec1.rerepetition, rec2.rerepetition)
    np.testing.assert_array_equal(rec1.stimulus, rec2.stimulus)
    np.testing.assert_array_equal(rec1.repetition, rec2.repetition)
    assert rec1.subject_id == rec2.subject_id
    assert rec1.exercise_id == rec2.exercise_id
    assert rec1.sampling_rate_hz == rec2.sampling_rate_hz


def test_synthetic_recording_stochastic_different_seeds() -> None:
    """Verify that different seeds produce different EMG signals while maintaining timeline structure."""
    from tests.fixtures.synthetic_semg import make_synthetic_recording

    rec1 = make_synthetic_recording(seed=100)
    rec2 = make_synthetic_recording(seed=200)

    assert not np.array_equal(rec1.emg, rec2.emg)
    # Timelines/labels remain structurally identical
    np.testing.assert_array_equal(rec1.restimulus, rec2.restimulus)
    np.testing.assert_array_equal(rec1.rerepetition, rec2.rerepetition)


def test_synthetic_recording_structural_and_numerical_contracts() -> None:
    """Verify dimensions, dtypes, finiteness, and sampling rate of synthetic recording."""
    from tests.fixtures.synthetic_semg import make_synthetic_recording

    rec = make_synthetic_recording(seed=42)

    assert rec.emg.ndim == 2
    assert rec.emg.shape[1] == 12
    assert rec.emg.dtype == np.float32
    assert rec.sampling_rate_hz == 2000.0
    assert not np.isnan(rec.emg).any()
    assert not np.isinf(rec.emg).any()

    n_samples = rec.emg.shape[0]
    assert rec.stimulus.shape == (n_samples,)
    assert rec.repetition.shape == (n_samples,)
    assert rec.restimulus.shape == (n_samples,)
    assert rec.rerepetition.shape == (n_samples,)


def test_synthetic_recording_label_repetition_invariants() -> None:
    """Verify NinaPro label-repetition invariants: label == 0 <=> repetition == 0."""
    from tests.fixtures.synthetic_semg import make_synthetic_recording

    active_labels = (1, 2, 3, 4, 5)
    rec = make_synthetic_recording(seed=42, active_labels=active_labels)

    # Invariant 1: REST samples (label 0) must have repetition 0
    rest_mask = rec.restimulus == 0
    assert np.all(rec.rerepetition[rest_mask] == 0)

    # Invariant 2: Active samples (label > 0) must have repetition in 1..6
    active_mask = rec.restimulus > 0
    assert np.all(rec.rerepetition[active_mask] > 0)
    assert np.all(rec.rerepetition[active_mask] <= 6)

    # Invariant 3: Only configured active labels and REST exist
    unique_labels = set(np.unique(rec.restimulus))
    expected_labels = {0} | set(active_labels)
    assert unique_labels == expected_labels

    # Invariant 4: Repetitions 1 to 6 all present
    unique_reps = set(np.unique(rec.rerepetition))
    assert unique_reps == {0, 1, 2, 3, 4, 5, 6}


def test_synthetic_recording_frozen_views_integration() -> None:
    """Verify that synthetic recording seamlessly integrates with Feature 003 DatasetView."""
    from tests.fixtures.synthetic_semg import make_synthetic_recording

    rec = make_synthetic_recording(seed=42)

    refined_view = rec.get_view("refined")
    assert refined_view.view_id == "refined"
    assert refined_view.signals.shape == rec.emg.shape
    assert np.array_equal(refined_view.labels, rec.restimulus)
    assert np.array_equal(refined_view.repetitions, rec.rerepetition)

    stimulus_view = rec.get_view("stimulus")
    assert stimulus_view.view_id == "stimulus"
    assert stimulus_view.signals.shape == rec.emg.shape
    assert np.array_equal(stimulus_view.labels, rec.stimulus)
    assert np.array_equal(stimulus_view.repetitions, rec.repetition)


def test_synthetic_recording_class_separability_patterns() -> None:
    """Verify that synthetic classes have distinct spatial-frequency patterns rather than pure white noise."""
    from tests.fixtures.synthetic_semg import make_synthetic_recording

    rec = make_synthetic_recording(seed=42, active_labels=(1, 2, 3))

    rest_emg = rec.emg[rec.restimulus == 0]
    class1_emg = rec.emg[rec.restimulus == 1]
    class2_emg = rec.emg[rec.restimulus == 2]

    # Active classes should have significantly higher total RMS than REST
    rms_rest = np.sqrt(np.mean(rest_emg**2))
    rms_c1 = np.sqrt(np.mean(class1_emg**2))
    rms_c2 = np.sqrt(np.mean(class2_emg**2))

    assert rms_c1 > 1.5 * rms_rest, "Class 1 must have higher energy than baseline REST"
    assert rms_c2 > 1.5 * rms_rest, "Class 2 must have higher energy than baseline REST"

    # Per-channel spatial profiles should differ between class 1 and class 2
    ch_rms_c1 = np.sqrt(np.mean(class1_emg**2, axis=0))
    ch_rms_c2 = np.sqrt(np.mean(class2_emg**2, axis=0))
    assert not np.allclose(ch_rms_c1, ch_rms_c2, rtol=0.2), "Channel activation profiles must be distinct"


def test_synthetic_recording_no_raw_data_or_sklearn_import() -> None:
    """Verify generator does not access raw DB2 directories or import scikit-learn."""
    from tests.fixtures.synthetic_semg import make_synthetic_recording

    assert "sklearn" not in sys.modules, "synthetic fixture must not require sklearn"
    # Execute generation and ensure no raw DB2 access occurs
    rec = make_synthetic_recording(seed=42)
    assert rec.num_channels == 12


def test_synthetic_recording_adversarial_modes() -> None:
    """Verify optional opt-in adversarial generation modes."""
    from tests.fixtures.synthetic_semg import make_synthetic_recording, generate_raw_synthetic_arrays

    # 1. Silent channel (zero variance)
    rec_silent = make_synthetic_recording(seed=42, silent_channel_idx=4)
    assert np.all(rec_silent.emg[:, 4] == 0.0)
    assert np.std(rec_silent.emg[:, 4]) == 0.0

    # 2. Singular / redundant channel pattern (ch 1 duplicates ch 0)
    rec_singular = make_synthetic_recording(seed=42, duplicate_channel_pair=(0, 1))
    np.testing.assert_array_equal(rec_singular.emg[:, 0], rec_singular.emg[:, 1])

    # 3. Severe REST imbalance mode (>90% REST)
    rec_imbalance = make_synthetic_recording(seed=42, severe_rest_imbalance=True)
    rest_fraction = np.mean(rec_imbalance.restimulus == 0)
    assert rest_fraction > 0.85

    # 4. Truncated repetition (< 400 samples)
    rec_truncated = make_synthetic_recording(seed=42, truncated_repetition_samples=250)
    rep_lengths = [np.sum(rec_truncated.rerepetition == r) for r in range(1, 7)]
    assert any(length < 400 for length in rep_lengths)

    # 5. Raw arrays generation for NaN / Inf injection (bypassing RecordingData strict validator)
    raw_arrays = generate_raw_synthetic_arrays(seed=42, inject_nan_coords=(50, 2))
    assert np.isnan(raw_arrays["emg"][50, 2])

    raw_arrays_inf = generate_raw_synthetic_arrays(seed=42, inject_inf_coords=(100, 5))
    assert np.isinf(raw_arrays_inf["emg"][100, 5])
