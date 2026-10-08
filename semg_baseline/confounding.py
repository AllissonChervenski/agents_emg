"""Exercise Confounding Auditor for NinaPro DB2 baseline (FR-007.3, FR-007.5, SC-005, Decision-4).

Implements the mandatory diagnostic audit evaluating recording/session bias:
- Trains an auxiliary Linear Discriminant Analysis classifier to predict exercise_id in {E1, E2, E3}
  strictly from resting windows (y == 0) under identical split constraints.
- Establishes canonical chance level = 1/3 (0.3333333333333333).
- Calculates confounding delta = rest_exercise_accuracy - chance_level.
- Identifies confounding if delta > 0.05.
- Provides standardized scientific interpretations:
    * Near chance (delta <= 0.05): "nenhuma evidência de confounding detectável por este diagnóstico"
    * Above chance (delta > 0.05): "evidência de confounding de sessão/exercício detectada; deve acompanhar explicitamente a interpretação dos resultados globais"
- Strictly non-blocking: secondary and diagnostic audits contextualize results without excluding recordings.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score

from semg_baseline.features import TimeDomainFeatureExtractor
from semg_baseline.filter import CausalBandpassFilter
from semg_baseline.lda import CanonicalLdaTrainer, PriorType
from semg_baseline.scaling import FeatureScaler
from semg_baseline.thresholds import ThresholdCalibrator
from semg_baseline.window import PureWindowSegmenter
from semg_dataset.contract import RecordingData
from semg_dataset.manifest import associate_rest_to_repetition_units


@dataclass(frozen=True)
class ExerciseConfoundingReport:
    """Diagnostic audit record evaluating recording/session bias across exercise recordings."""

    audit_id: str
    protocol_id: str
    rest_exercise_accuracy: float
    chance_level: float = 1.0 / 3.0
    confounding_delta: float = 0.0
    confounding_indicated: bool = False
    scientific_interpretation: str = ""

    def __post_init__(self) -> None:
        """Validate value ranges and structural invariants."""
        if not (0.0 <= self.rest_exercise_accuracy <= 1.0):
            raise ValueError(
                f"rest_exercise_accuracy must be in [0.0, 1.0]; got {self.rest_exercise_accuracy}"
            )
        if not (0.0 <= self.chance_level <= 1.0):
            raise ValueError(f"chance_level must be in [0.0, 1.0]; got {self.chance_level}")

    @property
    def interpretation(self) -> str:
        """Alias for scientific_interpretation conforming to evaluation_report_schema.json."""
        return self.scientific_interpretation

    def to_diagnostic_dict(self) -> dict[str, Any]:
        """Convert to dictionary strictly conforming to evaluation_report_schema.json confounding_diagnostic."""
        return {
            "rest_exercise_accuracy": float(self.rest_exercise_accuracy),
            "chance_level": float(self.chance_level),
            "confounding_delta": float(self.confounding_delta),
            "confounding_indicated": bool(self.confounding_indicated),
            "interpretation": str(self.scientific_interpretation),
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert complete audit record to dictionary."""
        return {
            "audit_id": self.audit_id,
            "protocol_id": self.protocol_id,
            "rest_exercise_accuracy": float(self.rest_exercise_accuracy),
            "chance_level": float(self.chance_level),
            "confounding_delta": float(self.confounding_delta),
            "confounding_indicated": bool(self.confounding_indicated),
            "scientific_interpretation": str(self.scientific_interpretation),
        }


class ExerciseConfoundingAuditor:
    """Executes the mandatory diagnostic audit predicting exercise_id from rest windows."""

    CHANCE_LEVEL: float = 1.0 / 3.0
    CONFOUNDING_THRESHOLD: float = 0.05
    INTERPRETATION_CLEAN: str = (
        "nenhuma evidência de confounding detectável por este diagnóstico"
    )
    INTERPRETATION_CONFOUNDED: str = (
        "evidência de confounding de sessão/exercício detectada; "
        "deve acompanhar explicitamente a interpretação dos resultados globais"
    )
    EXERCISE_NAME_TO_INT: dict[str, int] = {"E1": 0, "E2": 1, "E3": 2}

    def __init__(
        self,
        filter_low_hz: float = 20.0,
        filter_high_hz: float = 450.0,
        fs: float = 2000.0,
        filter_order: int = 4,
        num_channels: int = 12,
        window_len: int = 400,
        stride: int = 100,
        view_name: str = "refined",
    ) -> None:
        self.filter_low_hz = float(filter_low_hz)
        self.filter_high_hz = float(filter_high_hz)
        self.fs = float(fs)
        self.filter_order = int(filter_order)
        self.num_channels = int(num_channels)
        self.window_len = int(window_len)
        self.stride = int(stride)
        self.view_name = view_name
        self.segmenter = PureWindowSegmenter(window_len=self.window_len, stride=self.stride)

    def audit_from_features(
        self,
        X_rest_train: np.ndarray,
        y_exercise_train: np.ndarray,
        X_rest_test: np.ndarray,
        y_exercise_test: np.ndarray,
        *,
        audit_id: str = "confounding_audit",
        protocol_id: str = "within_subject",
    ) -> ExerciseConfoundingReport:
        """Train auxiliary regularized LDA on train REST features and evaluate on test REST features."""
        X_tr = np.asarray(X_rest_train, dtype=np.float64)
        y_tr = np.asarray(y_exercise_train, dtype=np.int32)
        X_te = np.asarray(X_rest_test, dtype=np.float64)
        y_te = np.asarray(y_exercise_test, dtype=np.int32)

        if len(X_tr) == 0 or len(X_te) == 0:
            raise ValueError("Train and test REST samples must be non-empty")

        if len(X_tr) != len(y_tr) or len(X_te) != len(y_te):
            raise ValueError("Sample count mismatch between features and labels")

        if X_tr.ndim != 2 or X_te.ndim != 2:
            raise ValueError("Feature matrices must be 2D")

        if X_tr.shape[1] != X_te.shape[1]:
            raise ValueError(
                f"Feature dimension mismatch: train has {X_tr.shape[1]}, test has {X_te.shape[1]}"
            )

        if not np.all(np.isfinite(X_tr)) or not np.all(np.isfinite(X_te)):
            raise ValueError("Features must not contain NaN or Inf")

        unique_classes = np.unique(y_tr)
        if len(unique_classes) < 2:
            raise ValueError("Confounding audit requires at least 2 distinct exercise classes")

        # Map classes to contiguous 0..K-1 for CanonicalLdaTrainer vocabulary compliance
        class_map = {cls_val: idx for idx, cls_val in enumerate(unique_classes)}
        y_tr_mapped = np.array([class_map[c] for c in y_tr], dtype=np.int32)
        y_te_mapped = np.array([class_map.get(c, -1) for c in y_te], dtype=np.int32)

        # Fit train-only scaler and scale features
        scaler = FeatureScaler().fit(X_tr)
        X_tr_scaled = scaler.transform(X_tr)
        X_te_scaled = scaler.transform(X_te)

        # Fit auxiliary LDA with uniform priors
        trainer = CanonicalLdaTrainer(
            prior_type=PriorType.UNIFORM,
            num_classes=len(unique_classes),
        )
        trainer.fit(X_tr_scaled, y_tr_mapped)

        # Predict test exercise labels
        preds_mapped = trainer.predict(X_te_scaled)
        accuracy = float(accuracy_score(y_te_mapped, preds_mapped))
        delta = float(accuracy - self.CHANCE_LEVEL)
        indicated = bool(delta > self.CONFOUNDING_THRESHOLD)
        interp = self.INTERPRETATION_CONFOUNDED if indicated else self.INTERPRETATION_CLEAN

        return ExerciseConfoundingReport(
            audit_id=audit_id,
            protocol_id=protocol_id,
            rest_exercise_accuracy=accuracy,
            chance_level=self.CHANCE_LEVEL,
            confounding_delta=delta,
            confounding_indicated=indicated,
            scientific_interpretation=interp,
        )

    def extract_rest_windows(
        self,
        recording: RecordingData,
        repetitions: Sequence[int],
    ) -> np.ndarray:
        """Filter continuous EMG and extract pure steady-state REST windows for specified repetitions."""
        view = recording.get_view(self.view_name)
        units = associate_rest_to_repetition_units(view.repetitions)

        windows_list: list[np.ndarray] = []

        for rep in repetitions:
            rep_mask = units == rep
            if not np.any(rep_mask):
                continue

            rep_signals = view.signals[rep_mask]
            rep_labels = view.labels[rep_mask]
            rep_original = view.repetitions[rep_mask]
            rep_units_slice = units[rep_mask]

            bp_filter = CausalBandpassFilter(
                low_hz=self.filter_low_hz,
                high_hz=self.filter_high_hz,
                fs=self.fs,
                order=self.filter_order,
                num_channels=self.num_channels,
            )
            filtered_rep = bp_filter.process(rep_signals)

            win_r, lbl_r, _ = self.segmenter.segment(
                emg=filtered_rep,
                labels=rep_labels,
                repetitions=rep_original,
                repetition_units=rep_units_slice,
            )

            # Filter strictly for REST windows (label == 0)
            rest_mask = lbl_r == 0
            if np.any(rest_mask):
                windows_list.append(win_r[rest_mask])

        if not windows_list:
            return np.empty((0, self.window_len, self.num_channels), dtype=np.float32)

        return np.vstack(windows_list)

    def audit_within_subject(
        self,
        subject_id: str,
        recordings: Mapping[str, RecordingData] | Sequence[RecordingData],
        *,
        candidate_r: float = 1.0,
        train_repetitions: Sequence[int] = (1, 3, 4, 6),
        test_repetitions: Sequence[int] = (2, 5),
    ) -> ExerciseConfoundingReport:
        """Execute Within-Subject confounding audit across E1, E2, E3 recordings."""
        rec_map: dict[str, RecordingData] = {}
        if isinstance(recordings, Mapping):
            rec_map = {str(k): v for k, v in recordings.items()}
        else:
            rec_map = {rec.exercise_id: rec for rec in recordings}

        train_windows_list: list[np.ndarray] = []
        train_labels_list: list[np.ndarray] = []
        test_windows_list: list[np.ndarray] = []
        test_labels_list: list[np.ndarray] = []

        for ex_name, ex_int in sorted(self.EXERCISE_NAME_TO_INT.items()):
            if ex_name not in rec_map:
                continue
            rec = rec_map[ex_name]

            tr_win = self.extract_rest_windows(rec, repetitions=train_repetitions)
            if len(tr_win) > 0:
                train_windows_list.append(tr_win)
                train_labels_list.append(np.full(len(tr_win), ex_int, dtype=np.int32))

            te_win = self.extract_rest_windows(rec, repetitions=test_repetitions)
            if len(te_win) > 0:
                test_windows_list.append(te_win)
                test_labels_list.append(np.full(len(te_win), ex_int, dtype=np.int32))

        if not train_windows_list or not test_windows_list:
            raise ValueError("Insufficient REST windows extracted for within-subject audit")

        all_tr_windows = np.vstack(train_windows_list)
        y_tr = np.concatenate(train_labels_list)
        all_te_windows = np.vstack(test_windows_list)
        y_te = np.concatenate(test_labels_list)

        # Calibrate deadzone threshold from train rest windows
        calibrator = ThresholdCalibrator()
        threshold_cal = calibrator.calibrate(
            train_emg=all_tr_windows,
            train_labels=np.zeros(len(all_tr_windows), dtype=int),
            rest_label=0,
            r_multiplier=candidate_r,
        )

        extractor = TimeDomainFeatureExtractor(
            epsilon=threshold_cal.epsilon_channels,
            num_channels=self.num_channels,
        )
        X_tr = extractor.extract(all_tr_windows)
        X_te = extractor.extract(all_te_windows)

        return self.audit_from_features(
            X_rest_train=X_tr,
            y_exercise_train=y_tr,
            X_rest_test=X_te,
            y_exercise_test=y_te,
            audit_id=f"confounding_audit_ws_{subject_id}",
            protocol_id="within_subject",
        )

    def audit_cross_subject(
        self,
        fold_id: int,
        train_recordings: Sequence[RecordingData],
        test_recordings: Sequence[RecordingData],
        *,
        candidate_r: float = 1.0,
        train_repetitions: Sequence[int] = (1, 3, 4, 6),
        test_repetitions: Sequence[int] = (2, 5),
    ) -> ExerciseConfoundingReport:
        """Execute Cross-Subject confounding audit across multi-subject recording sets."""
        train_windows_list: list[np.ndarray] = []
        train_labels_list: list[np.ndarray] = []
        test_windows_list: list[np.ndarray] = []
        test_labels_list: list[np.ndarray] = []

        for rec in train_recordings:
            ex_int = self.EXERCISE_NAME_TO_INT.get(rec.exercise_id)
            if ex_int is None:
                continue
            tr_win = self.extract_rest_windows(rec, repetitions=train_repetitions)
            if len(tr_win) > 0:
                train_windows_list.append(tr_win)
                train_labels_list.append(np.full(len(tr_win), ex_int, dtype=np.int32))

        for rec in test_recordings:
            ex_int = self.EXERCISE_NAME_TO_INT.get(rec.exercise_id)
            if ex_int is None:
                continue
            te_win = self.extract_rest_windows(rec, repetitions=test_repetitions)
            if len(te_win) > 0:
                test_windows_list.append(te_win)
                test_labels_list.append(np.full(len(te_win), ex_int, dtype=np.int32))

        if not train_windows_list or not test_windows_list:
            raise ValueError("Insufficient REST windows extracted for cross-subject audit")

        all_tr_windows = np.vstack(train_windows_list)
        y_tr = np.concatenate(train_labels_list)
        all_te_windows = np.vstack(test_windows_list)
        y_te = np.concatenate(test_labels_list)

        calibrator = ThresholdCalibrator()
        threshold_cal = calibrator.calibrate(
            train_emg=all_tr_windows,
            train_labels=np.zeros(len(all_tr_windows), dtype=int),
            rest_label=0,
            r_multiplier=candidate_r,
        )

        extractor = TimeDomainFeatureExtractor(
            epsilon=threshold_cal.epsilon_channels,
            num_channels=self.num_channels,
        )
        X_tr = extractor.extract(all_tr_windows)
        X_te = extractor.extract(all_te_windows)

        return self.audit_from_features(
            X_rest_train=X_tr,
            y_exercise_train=y_tr,
            X_rest_test=X_te,
            y_exercise_test=y_te,
            audit_id=f"confounding_audit_cs_fold_{fold_id}",
            protocol_id="cross_subject",
        )
