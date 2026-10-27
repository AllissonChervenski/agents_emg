"""Behavioral tests for exploratory per-exercise report, no real data needed."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from semg_baseline.feature004_exercise_metrics import stratify_final_test
from semg_baseline.feature004_multi_exercise import (
    EXERCISE_LABELS,
    MultiExerciseSubject,
    MultiExerciseWithinSubjectRunner,
)
from semg_baseline.within_subject import WithinSubjectRunner
from semg_dataset.contract import RecordingData


def _example():
    origins = []
    truth = []
    for ex, labels in EXERCISE_LABELS.items():
        for y in sorted(labels):
            truth.append(y)
            origins.append(ex)
    truth = np.array(truth, dtype=np.int32)
    pred = truth.copy()
    # Preserve true REST origin. Fail E1 REST and mispredict one E2 gesture as E1.
    pred[0] = 18
    e2_active_start = len(EXERCISE_LABELS["E1"]) + 1
    pred[e2_active_start] = 1
    result = SimpleNamespace(
        subject_id="S1", selected_r=0.5, test_evaluation_count=1,
        test_predictions=pred, test_labels=truth,
        test_accuracy=float(np.mean(pred == truth)),
        test_balanced_accuracy=0.5,
        test_macro_f1=0.4,
    )
    return result, np.array(origins, dtype="<U2")


def _stub_recording(subject_id: str, exercise_id: str) -> RecordingData:
    rec = object.__new__(RecordingData)
    rec.subject_id = subject_id
    rec.exercise_id = exercise_id
    return rec


def _stub_preparation(self, rec, reps, validate_all_required=False):
    labels = np.array(sorted(EXERCISE_LABELS[rec.exercise_id]), dtype=np.int32)
    windows = np.zeros((len(labels), self.window_len, self.num_channels), dtype=np.float32)
    units = np.full(len(labels), int(reps[0]), dtype=np.int32)
    return windows, labels, units, []


def test_stratified_global_and_rest_origin_are_consistent():
    result, origin = _example()
    report = stratify_final_test(result, origin)
    assert report.test_windows == 52
    assert report.test_evaluation_count == 1
    assert report.global_top1_accuracy == pytest.approx(50 / 52)
    assert [part.exercise_id for part in report.by_exercise] == ["E1", "E2", "E3"]
    e1, e2, e3 = report.by_exercise
    assert e1.rest_windows == e2.rest_windows == e3.rest_windows == 1
    assert e1.rest_false_alarm_rate == 1.0
    assert e1.rest_recall == 0.0
    assert e2.rest_false_alarm_rate == e3.rest_false_alarm_rate == 0.0
    assert e2.outside_exercise_prediction_count == 1
    assert e1.outside_exercise_prediction_count == 1
    assert e3.top1_accuracy == 1.0
    assert e1.active_top1_accuracy == 1.0
    assert report.to_dict()["evidence_status"] == "EXPLORATORY_NOT_SCIENTIFIC_ACCEPTANCE"


def test_wrong_origins_and_class_coverage_rejected():
    result, origins = _example()
    bad = origins.copy()
    bad[0] = "E3"  # E3 true vocabulary no longer matches attribution
    with pytest.raises(ValueError, match="missing/invalid true TEST labels"):
        stratify_final_test(result, bad)
    with pytest.raises(ValueError, match="exactly E1, E2, E3"):
        stratify_final_test(result, np.full(len(origins), "E1"))


def test_bad_prediction_shapes_values_and_count_rejected():
    result, origin = _example()
    result.test_evaluation_count = 2
    with pytest.raises(ValueError, match="Exactly one"):
        stratify_final_test(result, origin)
    result.test_evaluation_count = 1
    with pytest.raises(ValueError, match="equal nonzero length"):
        stratify_final_test(result, origin[:-1])
    result.test_predictions = result.test_predictions.copy()
    result.test_predictions[0] = 50
    with pytest.raises(ValueError, match="outside 0..49"):
        stratify_final_test(result, origin)


def test_global_accuracy_mismatch_rejected():
    result, origin = _example()
    result.test_accuracy = 1.0
    with pytest.raises(ValueError, match="disagrees"):
        stratify_final_test(result, origin)


def test_origin_captured_only_during_final_test(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(WithinSubjectRunner, "prepare_repetition_windows", _stub_preparation)
    r = MultiExerciseWithinSubjectRunner()
    source = MultiExerciseSubject("S1", _stub_recording)
    with pytest.raises(RuntimeError, match="unavailable"):
        _ = r.test_exercise_ids
    r.prepare_repetition_windows(source, (1, 3, 4, 6))
    r.prepare_repetition_windows(source, (2, 5))
    with pytest.raises(RuntimeError, match="unavailable"):
        _ = r.test_exercise_ids

    def fake_evaluate(self, recording, trainer, extractor, scaler):
        windows, labels, _, _ = self.prepare_repetition_windows(recording, (2, 5))
        self._test_evaluation_count += 1
        return (1.0, 1.0, 1.0, labels.copy(), labels.copy())

    monkeypatch.setattr(WithinSubjectRunner, "evaluate_test", fake_evaluate)
    r.evaluate_test(source, None, None, None)
    origin = r.test_exercise_ids
    assert origin.tolist() == (["E1"] * 18 + ["E2"] * 24 + ["E3"] * 10)
    origin[0] = "E3"  # property must never expose writable internal origin
    assert r.test_exercise_ids[0] == "E1"
    assert r.test_evaluation_count == 1


def test_failed_final_evaluation_clears_origin(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(WithinSubjectRunner, "prepare_repetition_windows", _stub_preparation)
    source = MultiExerciseSubject("S1", _stub_recording)
    r = MultiExerciseWithinSubjectRunner()

    def fail_after_preparing(self, recording, trainer, extractor, scaler):
        self.prepare_repetition_windows(recording, (2, 5))
        raise RuntimeError("training failed")

    monkeypatch.setattr(WithinSubjectRunner, "evaluate_test", fail_after_preparing)
    with pytest.raises(RuntimeError, match="training failed"):
        r.evaluate_test(source, None, None, None)
    with pytest.raises(RuntimeError, match="unavailable"):
        _ = r.test_exercise_ids
