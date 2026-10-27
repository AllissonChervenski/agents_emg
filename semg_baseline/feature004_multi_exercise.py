"""Exploratory NinaPro DB2 multi-exercise Within-Subject adapter.

Reuses the accepted WithinSubjectRunner without altering its LOTO, calibrator,
scaler, LDA, or single-TEST-evaluation logic. Processes E1/E2/E3 individually
before pooling *segmented* windows. Never joins raw signals across recordings.

Important boundary: the archive loader reads the entire recording, including
TEST samples, into memory. The inherited prepare_repetition_windows() then
filters/segments only requested repetition units. Thus this adapter preserves
processing/selection isolation, NOT physical-I/O-level zero TEST access.

The scientific acceptance, full manifest provenance and real cross-subject
benchmark are NOT performed by this adapter.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, cast
import re

import numpy as np

from semg_baseline.within_subject import WithinSubjectResult, WithinSubjectRunner
from semg_baseline.window import TransitionAudit
from semg_dataset.contract import RecordingData


EXERCISE_LABELS: dict[str, frozenset[int]] = {
    "E1": frozenset({0, *range(1, 18)}),
    "E2": frozenset({0, *range(18, 41)}),
    "E3": frozenset({0, *range(41, 50)}),
}
GLOBAL_LABELS: frozenset[int] = frozenset(range(50))


@dataclass(frozen=True)
class MultiExerciseSubject:
    """Lightweight *lazy* source; no synthetic or concatenated RecordingData."""

    subject_id: str
    load_recording: Callable[[str, str], RecordingData]

    def __post_init__(self) -> None:
        if re.fullmatch(r"S(?:[1-9]|[1-3][0-9]|40)", self.subject_id) is None:
            raise ValueError(f"Invalid NinaPro DB2 subject: {self.subject_id!r}")
        if not callable(self.load_recording):
            raise TypeError("load_recording must be callable")


class MultiExerciseWithinSubjectRunner(WithinSubjectRunner):
    """Run the inherited 50-class LOTO procedure on E1/E2/E3 windows.

    Existing one-recording behavior is preserved when passed a RecordingData.
    MultiExerciseSubject is accepted solely by this subclass. There is no
    raw-array concatenation and no cross-exercise DSP/filter state.
    """

    def __init__(self, **kwargs: Any) -> None:
        # Canonical taxonomic/view constraints are not configurable in real pilot.
        if "num_classes" in kwargs or "view_name" in kwargs:
            raise ValueError("Real pilot fixes num_classes=50 and view_name='refined'")
        super().__init__(num_classes=50, view_name="refined", **kwargs)
        # Set only when the final TEST windows are prepared, never during LOTO.
        self._test_exercise_ids: np.ndarray | None = None
        self._capturing_final_test = False

    @property
    def test_exercise_ids(self) -> np.ndarray:
        """Exercise origin aligned 1:1 with the final TEST result arrays.

        Available only after a successful multi-exercise TEST preparation.
        The caller receives a copy to protect the recorded order.
        """
        if self._test_exercise_ids is None:
            raise RuntimeError("TEST exercise attribution unavailable; run the real pilot first")
        return self._test_exercise_ids.copy()

    def prepare_repetition_windows(
        self,
        recording: RecordingData | MultiExerciseSubject,
        repetitions: Sequence[int],
        validate_all_required: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[TransitionAudit]]:
        if not isinstance(recording, MultiExerciseSubject):
            return super().prepare_repetition_windows(
                recording, repetitions, validate_all_required=validate_all_required
            )

        selected_reps = tuple(int(r) for r in repetitions)
        is_final_test = self._capturing_final_test and selected_reps == self.TEST_REPETITIONS
        if is_final_test:
            # If the preparation fails, never expose stale attribution.
            self._test_exercise_ids = None
        if not selected_reps or len(set(selected_reps)) != len(selected_reps):
            raise ValueError("Repetition selection must be nonempty and unique")
        if not set(selected_reps) <= set(self.ALL_REQUIRED_REPETITIONS):
            raise ValueError("Unrecognized repetition unit")

        windows_parts: list[np.ndarray] = []
        label_parts: list[np.ndarray] = []
        rep_parts: list[np.ndarray] = []
        test_origin_parts: list[np.ndarray] = []
        audits: list[TransitionAudit] = []

        for exercise_id, allowed_labels in EXERCISE_LABELS.items():
            rec = recording.load_recording(recording.subject_id, exercise_id)
            if not isinstance(rec, RecordingData):
                raise TypeError(f"Loader did not return RecordingData for {exercise_id}")
            if rec.subject_id != recording.subject_id or rec.exercise_id != exercise_id:
                raise ValueError(f"Subject/exercise identity mismatch for {exercise_id}")

            # The *existing* filtering and segmentation remain independent per
            # recording and per repetition unit. TEST rep arrays are not sliced
            # or filtered during validation (but are read by the archive loader).
            windows, labels, rep_units, exercise_audits = super().prepare_repetition_windows(
                rec, selected_reps, validate_all_required=validate_all_required
            )
            observed = {int(x) for x in np.unique(labels)}
            if not observed <= allowed_labels:
                raise ValueError(
                    f"Unexpected/non-global labels in {exercise_id}: "
                    f"{sorted(observed - allowed_labels)}"
                )
            if len(labels) == 0:
                raise ValueError(f"No admitted {exercise_id} windows for repetitions {selected_reps}")
            windows_parts.append(windows)
            label_parts.append(labels)
            rep_parts.append(rep_units)
            if is_final_test:
                # The adapter concatenates windows E1, E2, E3 in this order.
                # Preserve true origin also for REST (label 0), which cannot
                # be attributed to an exercise from the label alone.
                test_origin_parts.append(np.full(len(labels), exercise_id, dtype="<U2"))
            audits.extend(exercise_audits)
            del rec

        joined_labels = np.concatenate(label_parts)
        if {int(x) for x in np.unique(joined_labels)} != GLOBAL_LABELS:
            raise ValueError(
                "Global 50-class vocabulary not present in admitted windows "
                f"for repetition selection {selected_reps}; refusing silent shrinkage"
            )
        if is_final_test:
            origin = np.concatenate(test_origin_parts)
            if len(origin) != len(joined_labels):
                raise RuntimeError("Exercise attribution does not match admitted TEST windows")
            self._test_exercise_ids = origin
        return (
            np.concatenate(windows_parts, axis=0),
            joined_labels,
            np.concatenate(rep_parts),
            audits,
        )

    def evaluate_test(
        self,
        recording: RecordingData | MultiExerciseSubject,
        trainer: Any,
        extractor: Any,
        scaler: Any,
    ) -> tuple[float, float, float, np.ndarray, np.ndarray]:
        """Capture exercise origin only while the inherited TEST evaluation runs.

        The original evaluation and its single event counter are unchanged.
        """
        if self._capturing_final_test:
            raise RuntimeError("Nested final TEST evaluation is forbidden")
        self._test_exercise_ids = None
        self._capturing_final_test = True
        try:
            return super().evaluate_test(cast(RecordingData, recording), trainer, extractor, scaler)
        except BaseException:
            self._test_exercise_ids = None
            raise
        finally:
            self._capturing_final_test = False

    def run_subject(
        self,
        subject_id: str,
        load_recording: Callable[[str, str], RecordingData],
    ) -> WithinSubjectResult:
        # A second invocation must never reuse attribution from an earlier run.
        self._test_exercise_ids = None
        subject = MultiExerciseSubject(subject_id=subject_id, load_recording=load_recording)
        # The inherited run() accesses only .subject_id and calls the overridden
        # prepare_repetition_windows(); it never requires .get_view() here.
        # The cast is localized to this adapter and never escapes into callers.
        result = super().run(cast(RecordingData, subject))
        if isinstance(result, WithinSubjectResult):
            if result.test_evaluation_count != 1:
                raise RuntimeError("Expected one, and only one, final TEST evaluation")
            if self._test_exercise_ids is None or len(self._test_exercise_ids) != len(result.test_labels):
                raise RuntimeError("Missing or misaligned TEST exercise attribution")
        return result
