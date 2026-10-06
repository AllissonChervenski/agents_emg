"""Unit tests for steady-state pure window segmenter (FR-012.2, FR-012.3, SC-002, Decision-2).

Validates:
- Pure window segmentation (W=400 samples, S=100 samples, 20 Hz decision rate).
- Strict exclusion of all label transition windows (REST <-> ACTIVE, ACTIVE <-> ACTIVE).
- Strict exclusion of repetition unit boundary crossing windows.
- Invariant: repetition_crossing_windows_admitted == 0.
- TransitionAudit accuracy and accounting.
"""

from __future__ import annotations

import numpy as np
import pytest

from semg_baseline.window import PureWindowSegmenter, TransitionAudit
from semg_dataset.manifest import associate_rest_to_repetition_units
from tests.fixtures.synthetic_semg import make_synthetic_recording


def test_pure_windowing_synthetic_recording() -> None:
    """PureWindowSegmenter must extract pure windows and audit boundaries on RecordingData."""
    rec = make_synthetic_recording(seed=42)
    view = rec.get_view("refined")

    segmenter = PureWindowSegmenter(window_len=400, stride=100)
    windows, labels, audit = segmenter.segment(
        emg=view.signals,
        labels=view.labels,
        repetitions=view.repetitions,
    )

    assert isinstance(audit, TransitionAudit)
    assert windows.ndim == 3
    assert windows.shape[1] == 400
    assert windows.shape[2] == 12
    assert windows.dtype == np.float32
    assert len(labels) == windows.shape[0]

    # Verify audit invariants
    assert audit.repetition_crossing_windows_admitted == 0
    assert audit.admitted_window_count == windows.shape[0]
    assert audit.discarded_window_count == (
        audit.discarded_label_transitions + audit.discarded_repetition_crossings
    )
    assert audit.total_candidate_windows == (
        audit.admitted_window_count + audit.discarded_window_count
    )
    assert audit.admitted_window_count > 0
    assert audit.discarded_window_count > 0


def test_every_admitted_window_is_strictly_pure() -> None:
    """Every single admitted window must have 100% uniform label and repetition unit."""
    rec = make_synthetic_recording(seed=99)
    view = rec.get_view("refined")
    units = associate_rest_to_repetition_units(view.repetitions)

    segmenter = PureWindowSegmenter(window_len=400, stride=100)
    windows, labels, audit = segmenter.segment(
        emg=view.signals,
        labels=view.labels,
        repetitions=view.repetitions,
    )

    assert len(audit.admitted_start_indices) == windows.shape[0]

    for k, start_idx in enumerate(audit.admitted_start_indices):
        end_idx = start_idx + 400
        win_labels = view.labels[start_idx:end_idx]
        win_units = units[start_idx:end_idx]

        # Uniform label
        assert np.all(win_labels == labels[k]), f"Admitted window {k} has mixed labels"
        # Uniform repetition unit
        assert np.all(win_units == win_units[0]), f"Admitted window {k} crosses repetition unit"


def test_boundary_label_transition_exclusion() -> None:
    """A window spanning a transition from REST to active gesture must be discarded."""
    # 200 samples REST (0), 200 samples GESTURE 1 (1)
    emg = np.ones((400, 12), dtype=np.float32)
    labels = np.array([0] * 200 + [1] * 200, dtype=np.int16)
    repetitions = np.array([0] * 200 + [1] * 200, dtype=np.int8)

    segmenter = PureWindowSegmenter(window_len=400, stride=100)
    windows, out_labels, audit = segmenter.segment(emg, labels, repetitions)

    assert windows.shape[0] == 0
    assert audit.admitted_window_count == 0
    assert audit.discarded_label_transitions == 1
    assert audit.repetition_crossing_windows_admitted == 0


def test_repetition_crossing_exclusion_even_with_same_label() -> None:
    """A window spanning two repetition units (e.g. rep 1 rest into rep 2 rest) must be discarded."""
    # 400 samples all REST (label 0), but repetition unit 1 transitions to unit 2 at sample 200
    emg = np.ones((400, 12), dtype=np.float32)
    labels = np.zeros(400, dtype=np.int16)
    # Explicitly pass repetition units crossing at sample 200
    repetition_units = np.array([1] * 200 + [2] * 200, dtype=np.int32)
    repetitions = np.zeros(400, dtype=np.int8)

    segmenter = PureWindowSegmenter(window_len=400, stride=100)
    windows, out_labels, audit = segmenter.segment(
        emg, labels, repetitions, repetition_units=repetition_units
    )

    assert windows.shape[0] == 0
    assert audit.admitted_window_count == 0
    assert audit.discarded_repetition_crossings == 1
    assert audit.repetition_crossing_windows_admitted == 0


def test_truncated_input_shorter_than_window() -> None:
    """Input with fewer samples than window_len must admit zero windows and log dropped tail."""
    emg = np.zeros((350, 12), dtype=np.float32)
    labels = np.zeros(350, dtype=np.int16)
    repetitions = np.zeros(350, dtype=np.int8)

    segmenter = PureWindowSegmenter(window_len=400, stride=100)
    windows, out_labels, audit = segmenter.segment(emg, labels, repetitions)

    assert windows.shape == (0, 400, 12)
    assert audit.admitted_window_count == 0
    assert audit.total_candidate_windows == 0
    assert audit.dropped_tail_samples == 350


def test_invalid_shapes_and_parameters_raise_value_error() -> None:
    """Invalid parameter values or mismatched array dimensions must raise ValueError."""
    with pytest.raises(ValueError, match="window_len must be positive"):
        PureWindowSegmenter(window_len=0, stride=100)

    with pytest.raises(ValueError, match="stride must be positive"):
        PureWindowSegmenter(window_len=400, stride=-1)

    segmenter = PureWindowSegmenter(window_len=400, stride=100)
    emg = np.zeros((500, 12), dtype=np.float32)
    labels = np.zeros(400, dtype=np.int16)  # length mismatch
    repetitions = np.zeros(500, dtype=np.int8)

    with pytest.raises(ValueError, match="Length mismatch"):
        segmenter.segment(emg, labels, repetitions)
