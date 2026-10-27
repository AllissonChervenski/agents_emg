"""Guardrails for real S1/E1+E2+E3 adapter; does not read NinaPro data."""
from __future__ import annotations

import numpy as np
import pytest

from semg_baseline.feature004_multi_exercise import (
    EXERCISE_LABELS,
    MultiExerciseSubject,
    MultiExerciseWithinSubjectRunner,
)
from semg_baseline.within_subject import WithinSubjectRunner
from semg_dataset.contract import RecordingData


def stub_recording(subject_id: str, exercise_id: str) -> RecordingData:
    """A bare object is safe because the inherited preprocessing is monkeypatched."""
    rec = object.__new__(RecordingData)
    rec.subject_id = subject_id
    rec.exercise_id = exercise_id
    return rec


def test_multi_exercise_aggregates_only_admitted_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, tuple[int, ...], bool]] = []

    def fake_prepare(self, recording, repetitions, validate_all_required=False):
        calls.append((recording.exercise_id, tuple(repetitions), validate_all_required))
        exercise_labels = np.array(sorted(EXERCISE_LABELS[recording.exercise_id]), dtype=np.int32)
        windows = np.zeros((len(exercise_labels), self.window_len, self.num_channels), dtype=np.float32)
        units = np.full(len(exercise_labels), repetitions[0], dtype=np.int32)
        return windows, exercise_labels, units, [recording.exercise_id]

    monkeypatch.setattr(WithinSubjectRunner, "prepare_repetition_windows", fake_prepare)
    source = MultiExerciseSubject("S1", stub_recording)
    runner = MultiExerciseWithinSubjectRunner(candidate_grid=(1.0,))
    windows, labels, units, audits = runner.prepare_repetition_windows(
        source, [1, 3, 4, 6], validate_all_required=True
    )
    assert calls == [(ex, (1, 3, 4, 6), True) for ex in ("E1", "E2", "E3")]
    assert windows.shape == (52, 400, 12)  # REST is represented independently in each exercise
    assert set(labels.tolist()) == set(range(50))
    assert units.tolist() == [1] * 52
    assert audits == ["E1", "E2", "E3"]


def test_multi_exercise_does_not_process_test_during_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[tuple[str, tuple[int, ...]]] = []

    def fake_prepare(self, recording, repetitions, validate_all_required=False):
        seen.append((recording.exercise_id, tuple(repetitions)))
        labels = np.array(sorted(EXERCISE_LABELS[recording.exercise_id]), dtype=np.int32)
        w = np.zeros((len(labels), self.window_len, self.num_channels), dtype=np.float32)
        u = np.full(len(labels), repetitions[0], dtype=np.int32)
        return w, labels, u, []

    monkeypatch.setattr(WithinSubjectRunner, "prepare_repetition_windows", fake_prepare)
    runner = MultiExerciseWithinSubjectRunner()
    source = MultiExerciseSubject("S1", stub_recording)
    for fold in runner.get_fold_definitions():
        runner.prepare_repetition_windows(source, fold.train_repetitions)
        runner.prepare_repetition_windows(source, (fold.val_repetition,))
    assert seen
    assert all(not ({2, 5} & set(reps)) for _, reps in seen)


def test_multi_exercise_fails_on_wrong_label_and_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_prepare(self, recording, repetitions, validate_all_required=False):
        labels = np.array([0, 49], dtype=np.int32) if recording.exercise_id == "E1" else np.array(
            sorted(EXERCISE_LABELS[recording.exercise_id]), dtype=np.int32
        )
        w = np.zeros((len(labels), self.window_len, self.num_channels), dtype=np.float32)
        u = np.ones(len(labels), dtype=np.int32)
        return w, labels, u, []

    monkeypatch.setattr(WithinSubjectRunner, "prepare_repetition_windows", fake_prepare)
    runner = MultiExerciseWithinSubjectRunner()
    with pytest.raises(ValueError, match="Unexpected/non-global labels"):
        runner.prepare_repetition_windows(MultiExerciseSubject("S1", stub_recording), [1])
    with pytest.raises(ValueError, match="identity mismatch"):
        runner.prepare_repetition_windows(MultiExerciseSubject("S1", lambda _s, e: stub_recording("S2", e)), [1])


def test_multi_exercise_rejects_incomplete_admitted_vocabulary(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_prepare(self, recording, repetitions, validate_all_required=False):
        labels = np.array(sorted(EXERCISE_LABELS[recording.exercise_id] - {49}), dtype=np.int32)
        w = np.zeros((len(labels), self.window_len, self.num_channels), dtype=np.float32)
        u = np.ones(len(labels), dtype=np.int32)
        return w, labels, u, []

    monkeypatch.setattr(WithinSubjectRunner, "prepare_repetition_windows", fake_prepare)
    with pytest.raises(ValueError, match="50-class vocabulary"):
        MultiExerciseWithinSubjectRunner().prepare_repetition_windows(
            MultiExerciseSubject("S1", stub_recording), [1]
        )


def test_multi_exercise_rejects_bad_pilot_configuration() -> None:
    with pytest.raises(ValueError, match="Invalid NinaPro"):
        MultiExerciseSubject("SYNTH_S1", stub_recording)
    with pytest.raises(ValueError, match="num_classes=50"):
        MultiExerciseWithinSubjectRunner(num_classes=4)
    with pytest.raises(ValueError, match="num_classes=50"):
        MultiExerciseWithinSubjectRunner(view_name="stimulus")
    with pytest.raises(ValueError, match="Repetition selection"):
        MultiExerciseWithinSubjectRunner().prepare_repetition_windows(
            MultiExerciseSubject("S1", stub_recording), [1, 1]
        )


def test_real_pilot_invokes_unmodified_base_run(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def fake_run(self, recording):
        calls.append((type(recording).__name__, recording.subject_id))
        return "SENTINEL"

    monkeypatch.setattr(WithinSubjectRunner, "run", fake_run)
    r = MultiExerciseWithinSubjectRunner().run_subject("S1", stub_recording)
    assert r == "SENTINEL"
    assert calls == [("MultiExerciseSubject", "S1")]
