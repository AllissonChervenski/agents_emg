"""Shuffled-Label Negative Control for NinaPro DB2 baseline (FR-005.8, SC-005, SC-006, Decision-5).

Implements the automated negative control verification that validates baseline integrity
against data leakage and evaluation errors:
- Permutes TRAIN labels deterministically under a fixed random seed.
- Preserves continuous signal preprocessing, windowing, and TEST labels untouched.
- Evaluates trained model against pre-registered mathematical failure threshold:
    * Nominal chance: 2.0% (1/50) for 50-class baseline.
    * Statistical failure upper bound: <= 5.0% (0.05).
- Fail-closed: performance exceeding the threshold flags a blocking data leakage or evaluation bug.
- Produces structured evidence and integrates with EvaluationReportGenerator.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from semg_baseline.evaluator import EvaluationReport, EvaluationReportGenerator
from semg_baseline.features import TimeDomainFeatureExtractor
from semg_baseline.filter import CausalBandpassFilter
from semg_baseline.lda import CanonicalLdaTrainer, PriorType
from semg_baseline.scaling import FeatureScaler
from semg_baseline.thresholds import ThresholdCalibrator
from semg_baseline.window import PureWindowSegmenter
from semg_dataset.contract import RecordingData
from semg_dataset.manifest import associate_rest_to_repetition_units


@dataclass(frozen=True)
class ShuffledLabelControlResult:
    """Structured outcome of a shuffled-label negative control verification run."""

    control_id: str
    protocol_id: str
    partition_id: str
    num_classes: int
    seed: int
    test_top1_accuracy: float
    test_balanced_accuracy: float
    test_macro_f1: float
    chance_level: float
    failure_threshold: float = 0.05
    passed: bool = True
    diagnostic_message: str = ""
    report: EvaluationReport | None = None

    def __post_init__(self) -> None:
        """Validate value ranges and structural invariants."""
        if not (0.0 <= self.test_top1_accuracy <= 1.0):
            raise ValueError(
                f"test_top1_accuracy must be in [0.0, 1.0]; got {self.test_top1_accuracy}"
            )
        if not (0.0 <= self.test_balanced_accuracy <= 1.0):
            raise ValueError(
                f"test_balanced_accuracy must be in [0.0, 1.0]; got {self.test_balanced_accuracy}"
            )
        if not (0.0 <= self.test_macro_f1 <= 1.0):
            raise ValueError(f"test_macro_f1 must be in [0.0, 1.0]; got {self.test_macro_f1}")
        if not (0.0 <= self.chance_level <= 1.0):
            raise ValueError(f"chance_level must be in [0.0, 1.0]; got {self.chance_level}")
        if not (0.0 <= self.failure_threshold <= 1.0):
            raise ValueError(
                f"failure_threshold must be in [0.0, 1.0]; got {self.failure_threshold}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary representation."""
        return {
            "control_id": self.control_id,
            "protocol_id": self.protocol_id,
            "partition_id": self.partition_id,
            "num_classes": self.num_classes,
            "seed": self.seed,
            "test_top1_accuracy": float(self.test_top1_accuracy),
            "test_balanced_accuracy": float(self.test_balanced_accuracy),
            "test_macro_f1": float(self.test_macro_f1),
            "chance_level": float(self.chance_level),
            "failure_threshold": float(self.failure_threshold),
            "passed": bool(self.passed),
            "diagnostic_message": self.diagnostic_message,
        }


class ShuffledLabelNegativeControl:
    """Executes the automated shuffled-label negative control verification."""

    DEFAULT_FAILURE_THRESHOLD_50_CLASS: float = 0.05
    DEFAULT_SEED: int = 42

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
        self.evaluator = EvaluationReportGenerator()

    def run_from_features(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        *,
        num_classes: int = 50,
        seed: int = 42,
        failure_threshold: float = 0.05,
        protocol_id: str = "within_subject",
        partition_id: str = "synthetic",
        enforce_pass: bool = False,
        model_taxonomy: str = "global_50_class",
    ) -> ShuffledLabelControlResult:
        """Permute train labels under seed, train LDA, and evaluate on test features."""
        X_tr = np.asarray(X_train, dtype=np.float64)
        y_tr = np.asarray(y_train, dtype=np.int32).copy()
        X_te = np.asarray(X_test, dtype=np.float64)
        y_te = np.asarray(y_test, dtype=np.int32).copy()

        if len(X_tr) == 0 or len(X_te) == 0:
            raise ValueError("Train and test sample sets must be non-empty")

        if len(X_tr) != len(y_tr) or len(X_te) != len(y_te):
            raise ValueError("Sample count mismatch between features and labels")

        if X_tr.ndim != 2 or X_te.ndim != 2:
            raise ValueError("Feature matrices must be 2D")

        if X_tr.shape[1] != X_te.shape[1]:
            raise ValueError("Feature dimension mismatch between train and test")

        if not np.all(np.isfinite(X_tr)) or not np.all(np.isfinite(X_te)):
            raise ValueError("Features must not contain NaN or Inf")

        if not (0.0 <= failure_threshold <= 1.0):
            raise ValueError(f"failure_threshold must be in [0.0, 1.0]; got {failure_threshold}")

        # Deterministically permute train labels under fixed seed
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(y_tr))
        y_tr_shuffled = y_tr[perm]

        # Fit train-only scaler and scale features
        scaler = FeatureScaler().fit(X_tr)
        X_tr_scaled = scaler.transform(X_tr)
        X_te_scaled = scaler.transform(X_te)

        # Fit regularized LDA classifier with uniform priors on permuted labels
        trainer = CanonicalLdaTrainer(
            prior_type=PriorType.UNIFORM,
            num_classes=num_classes,
        )
        trainer.fit(X_tr_scaled, y_tr_shuffled)

        # Predict test labels
        y_pred = trainer.predict(X_te_scaled)

        # Generate structured evaluation report (T017)
        report = self.evaluator.generate_report(
            y_true=y_te,
            y_pred=y_pred,
            report_id=f"report_neg_control_{protocol_id}_{partition_id}_{seed}",
            job_id="job_negative_control",
            job_fingerprint="0" * 64,
            protocol_id=protocol_id,
            partition_id=partition_id,
            model_taxonomy=model_taxonomy,
        )

        chance_level = 1.0 / float(num_classes)
        top1 = float(report.top1_accuracy)
        bal_acc = float(report.balanced_accuracy)
        macro_f1 = float(report.macro_f1)
        passed = bool(top1 <= failure_threshold)

        diag_msg = (
            f"Negative control {'PASSED' if passed else 'FAILED'}: "
            f"accuracy {top1:.4f} {'<=' if passed else '>'} threshold {failure_threshold:.4f} "
            f"(nominal chance: {chance_level:.4f}, seed: {seed})"
        )

        if enforce_pass and not passed:
            raise AssertionError(f"Shuffled-label negative control FAILED: {diag_msg}")

        return ShuffledLabelControlResult(
            control_id=f"shuffled_control_{protocol_id}_{partition_id}_{seed}",
            protocol_id=protocol_id,
            partition_id=partition_id,
            num_classes=num_classes,
            seed=seed,
            test_top1_accuracy=top1,
            test_balanced_accuracy=bal_acc,
            test_macro_f1=macro_f1,
            chance_level=chance_level,
            failure_threshold=failure_threshold,
            passed=passed,
            diagnostic_message=diag_msg,
            report=report,
        )

    def prepare_repetition_windows(
        self,
        recording: RecordingData,
        repetitions: Sequence[int],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Filter continuous EMG and segment pure steady-state windows for repetitions."""
        view = recording.get_view(self.view_name)
        units = associate_rest_to_repetition_units(view.repetitions)

        windows_list: list[np.ndarray] = []
        labels_list: list[np.ndarray] = []

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

            if len(win_r) > 0:
                windows_list.append(win_r)
                labels_list.append(lbl_r)

        if not windows_list:
            return (
                np.empty((0, self.window_len, self.num_channels), dtype=np.float32),
                np.empty(0, dtype=np.int32),
            )

        return np.vstack(windows_list), np.concatenate(labels_list)

    def run_within_subject(
        self,
        recording: RecordingData,
        *,
        num_classes: int = 50,
        seed: int = 42,
        failure_threshold: float = 0.05,
        candidate_r: float = 1.0,
        train_repetitions: Sequence[int] = (1, 3, 4, 6),
        test_repetitions: Sequence[int] = (2, 5),
        enforce_pass: bool = False,
        model_taxonomy: str = "global_50_class",
    ) -> ShuffledLabelControlResult:
        """Execute Within-Subject negative control by permuting train repetition labels."""
        tr_windows, tr_labels = self.prepare_repetition_windows(
            recording, repetitions=train_repetitions
        )
        te_windows, te_labels = self.prepare_repetition_windows(
            recording, repetitions=test_repetitions
        )

        if len(tr_windows) == 0 or len(te_windows) == 0:
            raise ValueError("Insufficient windows extracted for within-subject negative control")

        # Calibrate deadzone thresholds strictly from train rest windows
        calibrator = ThresholdCalibrator()
        threshold_cal = calibrator.calibrate(
            train_emg=tr_windows,
            train_labels=tr_labels,
            rest_label=0,
            r_multiplier=candidate_r,
        )

        extractor = TimeDomainFeatureExtractor(
            epsilon=threshold_cal.epsilon_channels,
            num_channels=self.num_channels,
        )
        X_tr = extractor.extract(tr_windows)
        X_te = extractor.extract(te_windows)

        return self.run_from_features(
            X_train=X_tr,
            y_train=tr_labels,
            X_test=X_te,
            y_test=te_labels,
            num_classes=num_classes,
            seed=seed,
            failure_threshold=failure_threshold,
            protocol_id="within_subject",
            partition_id=recording.subject_id,
            enforce_pass=enforce_pass,
            model_taxonomy=model_taxonomy,
        )
