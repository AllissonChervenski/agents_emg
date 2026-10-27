"""Exploratory per-exercise TEST metrics for the 50-class real NinaPro DB2 pilot.

Uses predictions already produced by the single final TEST evaluation. Tracks
REST by recording origin instead of guessing it from its shared label (0).
This is NOT scientific acceptance and provides no provenance seal.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from semg_baseline.feature004_multi_exercise import EXERCISE_LABELS
from semg_baseline.within_subject import WithinSubjectResult


@dataclass(frozen=True)
class ExerciseTestMetrics:
    exercise_id: str
    test_windows: int
    rest_windows: int
    active_windows: int
    top1_accuracy: float
    balanced_accuracy: float
    macro_f1: float
    active_top1_accuracy: float
    rest_recall: float
    rest_false_alarm_rate: float
    active_to_rest_count: int
    active_to_rest_rate: float
    outside_exercise_prediction_count: int
    outside_exercise_prediction_rate: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExerciseStratifiedReport:
    subject_id: str
    selected_r: float
    test_evaluation_count: int
    test_windows: int
    global_top1_accuracy: float
    global_balanced_accuracy: float
    global_macro_f1: float
    by_exercise: tuple[ExerciseTestMetrics, ...]
    evidence_status: str = "EXPLORATORY_NOT_SCIENTIFIC_ACCEPTANCE"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["limitations"] = [
            "REST (label 0) attributed by recorded E1/E2/E3 window origin, not predicted label",
            "All predictions come from one global 50-class model and one TEST evaluation",
            "Per-exercise balanced accuracy and macro-F1 average over each exercise's original label vocabulary",
            "Outside-exercise predictions are counted as errors; no label remapping",
            "Positive S1 REST/session confounding remains unresolved",
            "No independently anchored dataset provenance or scientific sign-off",
        ]
        return data


def _metrics_for_exercise(exercise: str, truth: np.ndarray, pred: np.ndarray) -> ExerciseTestMetrics:
    allowed = EXERCISE_LABELS[exercise]
    observed = set(int(x) for x in np.unique(truth))
    if observed != allowed:
        raise ValueError(f"{exercise}: missing/invalid true TEST labels: {sorted(allowed - observed)}")

    recalls: list[float] = []
    f1s: list[float] = []
    for label in sorted(allowed):
        positives = truth == label
        predicted = pred == label
        tp = int(np.count_nonzero(positives & predicted))
        fn = int(np.count_nonzero(positives & ~predicted))
        fp = int(np.count_nonzero(~positives & predicted))
        recalls.append(tp / (tp + fn))  # present in truth; denominator > 0
        denominator = 2 * tp + fp + fn
        f1s.append(2 * tp / denominator if denominator else 0.0)

    rest = truth == 0
    active = ~rest
    n_rest = int(np.count_nonzero(rest))
    n_active = int(np.count_nonzero(active))
    if n_rest == 0 or n_active == 0:
        raise ValueError(f"{exercise}: REST and active windows are both required")
    to_rest = int(np.count_nonzero(active & (pred == 0)))
    outside = int(np.count_nonzero(~np.isin(pred, list(allowed))))
    return ExerciseTestMetrics(
        exercise_id=exercise,
        test_windows=len(truth),
        rest_windows=n_rest,
        active_windows=n_active,
        top1_accuracy=float(np.mean(truth == pred)),
        balanced_accuracy=float(np.mean(recalls)),
        macro_f1=float(np.mean(f1s)),
        active_top1_accuracy=float(np.mean(pred[active] == truth[active])),
        rest_recall=float(np.mean(pred[rest] == 0)),
        rest_false_alarm_rate=float(np.mean(pred[rest] != 0)),
        active_to_rest_count=to_rest,
        active_to_rest_rate=to_rest / n_active,
        outside_exercise_prediction_count=outside,
        outside_exercise_prediction_rate=outside / len(truth),
    )


def stratify_final_test(
    result: WithinSubjectResult,
    test_exercise_ids: np.ndarray,
) -> ExerciseStratifiedReport:
    """Summarize the *existing* TEST prediction arrays; never refit or repredict."""
    if result.test_evaluation_count != 1:
        raise ValueError("Exactly one TEST evaluation must have occurred")
    truth = np.asarray(result.test_labels)
    pred = np.asarray(result.test_predictions)
    origin = np.asarray(test_exercise_ids)
    if truth.ndim != 1 or pred.ndim != 1 or origin.ndim != 1:
        raise ValueError("TEST labels, predictions, and origins must be one-dimensional")
    if not (len(truth) == len(pred) == len(origin)) or len(truth) == 0:
        raise ValueError("TEST labels, predictions, and origins must have equal nonzero length")
    if truth.dtype.kind not in "iu" or pred.dtype.kind not in "iu":
        raise ValueError("TEST predictions and labels must be integer arrays")
    if np.any(truth < 0) or np.any(truth > 49) or np.any(pred < 0) or np.any(pred > 49):
        raise ValueError("Unexpected global label outside 0..49")
    if set(origin.tolist()) != set(EXERCISE_LABELS):
        raise ValueError("TEST origin must contain exactly E1, E2, E3")
    measured_global_accuracy = float(np.mean(pred == truth))
    if not np.isclose(measured_global_accuracy, result.test_accuracy, rtol=0, atol=1e-12):
        raise ValueError("Original TEST accuracy disagrees with prediction arrays")

    by_exercise = tuple(
        _metrics_for_exercise(ex, truth[origin == ex], pred[origin == ex])
        for ex in EXERCISE_LABELS
    )
    if sum(ex.test_windows for ex in by_exercise) != len(truth):
        raise RuntimeError("Unassigned TEST windows")
    return ExerciseStratifiedReport(
        subject_id=result.subject_id,
        selected_r=result.selected_r,
        test_evaluation_count=result.test_evaluation_count,
        test_windows=len(truth),
        global_top1_accuracy=measured_global_accuracy,
        global_balanced_accuracy=float(result.test_balanced_accuracy),
        global_macro_f1=float(result.test_macro_f1),
        by_exercise=by_exercise,
    )
