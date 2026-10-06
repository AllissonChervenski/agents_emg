"""Within-Subject validation protocol and benchmark execution engine.

Implements FR-005.7, FR-008, SC-003, Decision-3, Decision-5:
- Exactly four leave-one-training-repetition-out (LOTO) validation folds over repetitions [1, 3, 4, 6].
    Fold 1: Val=[1], Train=[3, 4, 6]
    Fold 2: Val=[3], Train=[1, 4, 6]
    Fold 3: Val=[4], Train=[1, 3, 6]
    Fold 4: Val=[6], Train=[1, 3, 4]
- Disjoint partition separation: TRAIN ∩ VAL = ∅, TRAIN ∩ TEST = ∅, VAL ∩ TEST = ∅.
- Test repetitions [2, 5] strictly invisible (zero access) during validation / configuration selection (FR-005.7).
- Final model fitted on all training repetitions [1, 3, 4, 6].
- Final test evaluated strictly once on test repetitions [2, 5] (test_evaluation_count == 1).
- Preprocessing anti-leakage: noise deadzones and feature scaling computed solely from fold train.
- Candidate configuration selection over R in candidate grid strictly derived from training pool validation.
- Deterministic reproducibility under identical inputs and configurations.
- Fail-closed validation for missing required repetitions (FR-011.5).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from semg_baseline.features import TimeDomainFeatureExtractor
from semg_baseline.filter import CausalBandpassFilter
from semg_baseline.lda import CanonicalLdaTrainer, PriorType
from semg_baseline.scaling import FeatureScaler
from semg_baseline.thresholds import ThresholdCalibration, ThresholdCalibrator
from semg_baseline.window import PureWindowSegmenter, TransitionAudit
from semg_dataset.contract import RecordingData


@dataclass(frozen=True)
class FoldPartitionDef:
    """Mathematical definition of a single Within-Subject LOTO partition."""

    fold_id: int
    val_repetition: int
    train_repetitions: list[int]


@dataclass(frozen=True)
class FoldResult:
    """Outcome and fitted preprocessing evidence for a single validation fold."""

    fold_id: int
    val_repetition: int
    train_repetitions: list[int]
    r_multiplier: float
    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    n_train_samples: int
    n_val_samples: int
    threshold_calibration: ThresholdCalibration | None = None
    scaler_mean: np.ndarray | None = None
    scaler_scale: np.ndarray | None = None


@dataclass(frozen=True)
class ValidationSelectionResult:
    """Summary of cross-validation performance for a candidate R multiplier."""

    candidate_r: float
    mean_accuracy: float
    mean_balanced_accuracy: float
    mean_macro_f1: float
    fold_results: list[FoldResult] = field(default_factory=list)


@dataclass(frozen=True)
class WithinSubjectResult:
    """Structured evidence and metrics record for the complete Within-Subject protocol."""

    subject_id: str
    selected_r: float
    validation_folds: list[FoldResult]
    selection_summary: list[ValidationSelectionResult]
    final_train_repetitions: list[int]
    final_test_repetitions: list[int]
    final_model_trainer: CanonicalLdaTrainer
    test_accuracy: float
    test_balanced_accuracy: float
    test_macro_f1: float
    test_predictions: np.ndarray
    test_labels: np.ndarray
    test_evaluation_count: int
    n_final_train_samples: int
    final_scaler_mean: np.ndarray
    final_scaler_scale: np.ndarray


class WithinSubjectRunner:
    """Executes the canonical Within-Subject LOTO validation and single test evaluation protocol."""

    TRAINING_REPETITIONS: tuple[int, ...] = (1, 3, 4, 6)
    TEST_REPETITIONS: tuple[int, ...] = (2, 5)
    ALL_REQUIRED_REPETITIONS: tuple[int, ...] = (1, 2, 3, 4, 5, 6)

    def __init__(
        self,
        candidate_grid: Sequence[float] = (0.5, 1.0, 1.5, 2.0, 3.0),
        num_classes: int | None = None,
        prior_type: PriorType | str = PriorType.UNIFORM,
        view_name: str = "refined",
        filter_low_hz: float = 20.0,
        filter_high_hz: float = 450.0,
        fs: float = 2000.0,
        filter_order: int = 4,
        window_len: int = 400,
        stride: int = 100,
        num_channels: int = 12,
        access_callback: Callable[[int], None] | None = None,
        selection_metric: str = "balanced_accuracy",
    ) -> None:
        self.candidate_grid = tuple(float(r) for r in candidate_grid)
        self.num_classes = int(num_classes) if num_classes is not None else None
        self.prior_type = prior_type if isinstance(prior_type, PriorType) else PriorType(str(prior_type).lower())
        self.view_name = str(view_name)
        self.filter_low_hz = float(filter_low_hz)
        self.filter_high_hz = float(filter_high_hz)
        self.fs = float(fs)
        self.filter_order = int(filter_order)
        self.window_len = int(window_len)
        self.stride = int(stride)
        self.num_channels = int(num_channels)
        self.access_callback = access_callback
        self.selection_metric = str(selection_metric)

        self.segmenter = PureWindowSegmenter(
            window_len=self.window_len,
            stride=self.stride,
            fs=self.fs,
        )

    def _resolve_num_classes(self, labels: np.ndarray) -> int:
        """Resolve effective class count: explicitly configured or inferred from labels."""
        if self.num_classes is not None:
            return self.num_classes
        return int(np.max(labels)) + 1

    @classmethod
    def get_fold_definitions(cls) -> list[FoldPartitionDef]:
        """Return canonical 4 leave-one-training-repetition-out fold partition definitions (FR-008)."""
        return [
            FoldPartitionDef(fold_id=1, val_repetition=1, train_repetitions=[3, 4, 6]),
            FoldPartitionDef(fold_id=2, val_repetition=3, train_repetitions=[1, 4, 6]),
            FoldPartitionDef(fold_id=3, val_repetition=4, train_repetitions=[1, 3, 6]),
            FoldPartitionDef(fold_id=4, val_repetition=6, train_repetitions=[1, 3, 4]),
        ]

    def _prepare_windows(
        self, recording: RecordingData
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, TransitionAudit]:
        """Filter continuous EMG and segment pure steady-state windows with repetition tracking."""
        view = recording.get_view(self.view_name)

        bp_filter = CausalBandpassFilter(
            low_hz=self.filter_low_hz,
            high_hz=self.filter_high_hz,
            fs=self.fs,
            order=self.filter_order,
            num_channels=self.num_channels,
        )
        filtered_emg = bp_filter.process(view.signals)

        windows, labels, audit = self.segmenter.segment(
            emg=filtered_emg,
            labels=view.labels,
            repetitions=view.repetitions,
        )

        repetition_units = np.array(audit.admitted_repetition_units, dtype=np.int32)

        # Fail-closed validation: all required repetitions 1..6 must be present
        present_reps = set(np.unique(repetition_units).tolist())
        missing_reps = set(self.ALL_REQUIRED_REPETITIONS) - present_reps
        if missing_reps:
            raise ValueError(
                f"Required repetition(s) missing from admitted windows in subject {recording.subject_id}: "
                f"{sorted(missing_reps)}"
            )

        return windows, labels, repetition_units, audit

    def count_admitted_windows(
        self, recording: RecordingData, repetitions: Sequence[int]
    ) -> int:
        """Count admitted windows matching specified repetitions."""
        _, _, repetition_units, _ = self._prepare_windows(recording)
        mask = np.isin(repetition_units, list(repetitions))
        return int(np.sum(mask))

    def run_validation_selection(
        self, recording: RecordingData
    ) -> list[ValidationSelectionResult]:
        """Execute internal 4-fold validation across candidate R values without test set exposure (FR-005.7)."""
        windows, labels, repetition_units, _ = self._prepare_windows(recording)
        fold_defs = self.get_fold_definitions()

        selection_results: list[ValidationSelectionResult] = []

        for candidate_r in self.candidate_grid:
            fold_results: list[FoldResult] = []

            for fold in fold_defs:
                # 1. Structural partition extraction: strictly train repetitions
                train_mask = np.isin(repetition_units, fold.train_repetitions)
                train_windows = windows[train_mask]
                train_labels = labels[train_mask]
                train_rep_units = repetition_units[train_mask]

                # Instrument access callback
                if self.access_callback is not None:
                    for r in train_rep_units:
                        self.access_callback(int(r))

                # 2. Threshold calibration: strictly on fold train rest windows
                calibrator = ThresholdCalibrator(
                    candidate_grid=(candidate_r,),
                    default_r=candidate_r,
                    num_channels=self.num_channels,
                )
                calibration = calibrator.calibrate(
                    train_emg=train_windows,
                    train_labels=train_labels,
                    rest_label=0,
                    r_multiplier=candidate_r,
                )

                # 3. Feature extraction: strictly on fold train windows
                extractor = TimeDomainFeatureExtractor(
                    epsilon=calibration.epsilon_channels,
                    num_channels=self.num_channels,
                )
                X_train_raw = extractor.extract(train_windows)

                # 4. Feature scaling: strictly on fold train features
                scaler = FeatureScaler()
                scaler.fit(X_train_raw)
                X_train_scaled = scaler.transform(X_train_raw)

                # 5. LDA fit: strictly on fold train scaled features
                trainer = CanonicalLdaTrainer(
                    prior_type=self.prior_type,
                    num_classes=self._resolve_num_classes(train_labels),
                )
                trainer.fit(X_train_scaled, train_labels)

                # 6. Validation evaluation: strictly on fold validation repetition
                val_mask = repetition_units == fold.val_repetition
                val_windows = windows[val_mask]
                val_labels = labels[val_mask]
                val_rep_units = repetition_units[val_mask]

                if self.access_callback is not None:
                    for r in val_rep_units:
                        self.access_callback(int(r))

                X_val_raw = extractor.extract(val_windows)
                X_val_scaled = scaler.transform(X_val_raw)
                val_preds = trainer.predict(X_val_scaled)

                acc = float(accuracy_score(val_labels, val_preds))
                bal_acc = float(balanced_accuracy_score(val_labels, val_preds))
                m_f1 = float(f1_score(val_labels, val_preds, average="macro", zero_division=0))

                fold_results.append(
                    FoldResult(
                        fold_id=fold.fold_id,
                        val_repetition=fold.val_repetition,
                        train_repetitions=list(fold.train_repetitions),
                        r_multiplier=candidate_r,
                        accuracy=acc,
                        balanced_accuracy=bal_acc,
                        macro_f1=m_f1,
                        n_train_samples=len(train_labels),
                        n_val_samples=len(val_labels),
                        threshold_calibration=calibration,
                        scaler_mean=scaler.mean_train.copy() if scaler.mean_train is not None else None,
                        scaler_scale=scaler.scale_train.copy() if scaler.scale_train is not None else None,
                    )
                )

            mean_acc = float(np.mean([f.accuracy for f in fold_results]))
            mean_bal_acc = float(np.mean([f.balanced_accuracy for f in fold_results]))
            mean_macro_f1 = float(np.mean([f.macro_f1 for f in fold_results]))

            selection_results.append(
                ValidationSelectionResult(
                    candidate_r=candidate_r,
                    mean_accuracy=mean_acc,
                    mean_balanced_accuracy=mean_bal_acc,
                    mean_macro_f1=mean_macro_f1,
                    fold_results=fold_results,
                )
            )

        return selection_results

    def _select_best_r(
        self, selection_summary: list[ValidationSelectionResult]
    ) -> tuple[float, list[FoldResult]]:
        """Select best candidate R multiplier using internal validation results."""
        if not selection_summary:
            raise ValueError("Empty selection summary: no candidate configurations evaluated.")

        best_candidate = selection_summary[0].candidate_r
        best_score = -1.0
        best_folds = selection_summary[0].fold_results

        for res in selection_summary:
            if self.selection_metric == "accuracy":
                score = res.mean_accuracy
            elif self.selection_metric == "macro_f1":
                score = res.mean_macro_f1
            else:
                score = res.mean_balanced_accuracy

            if score > best_score:
                best_score = score
                best_candidate = res.candidate_r
                best_folds = res.fold_results

        return best_candidate, best_folds

    def run(self, recording: RecordingData) -> WithinSubjectResult:
        """Execute full Within-Subject benchmark: LOTO validation, configuration selection, and final test."""
        # Step 1: Internal validation selection across candidate grid (zero access to test repetitions)
        selection_summary = self.run_validation_selection(recording)
        selected_r, validation_folds = self._select_best_r(selection_summary)

        # Step 2: Prepare windows
        windows, labels, repetition_units, _ = self._prepare_windows(recording)

        # Step 3: Final training fit strictly on all training pool repetitions [1, 3, 4, 6]
        train_mask = np.isin(repetition_units, list(self.TRAINING_REPETITIONS))
        train_windows = windows[train_mask]
        train_labels = labels[train_mask]

        calibrator = ThresholdCalibrator(
            candidate_grid=(selected_r,),
            default_r=selected_r,
            num_channels=self.num_channels,
        )
        final_calibration = calibrator.calibrate(
            train_emg=train_windows,
            train_labels=train_labels,
            rest_label=0,
            r_multiplier=selected_r,
        )

        extractor = TimeDomainFeatureExtractor(
            epsilon=final_calibration.epsilon_channels,
            num_channels=self.num_channels,
        )
        X_train_raw = extractor.extract(train_windows)

        scaler = FeatureScaler()
        scaler.fit(X_train_raw)
        X_train_scaled = scaler.transform(X_train_raw)

        final_trainer = CanonicalLdaTrainer(
            prior_type=self.prior_type,
            num_classes=self._resolve_num_classes(train_labels),
        )
        final_trainer.fit(X_train_scaled, train_labels)

        # Step 4: Final test evaluation strictly once on test repetitions [2, 5]
        test_mask = np.isin(repetition_units, list(self.TEST_REPETITIONS))
        test_windows = windows[test_mask]
        test_labels = labels[test_mask]

        X_test_raw = extractor.extract(test_windows)
        X_test_scaled = scaler.transform(X_test_raw)
        test_preds = final_trainer.predict(X_test_scaled)

        test_acc = float(accuracy_score(test_labels, test_preds))
        test_bal_acc = float(balanced_accuracy_score(test_labels, test_preds))
        test_macro_f1 = float(f1_score(test_labels, test_preds, average="macro", zero_division=0))

        assert scaler.mean_train is not None and scaler.scale_train is not None

        return WithinSubjectResult(
            subject_id=recording.subject_id,
            selected_r=selected_r,
            validation_folds=validation_folds,
            selection_summary=selection_summary,
            final_train_repetitions=list(self.TRAINING_REPETITIONS),
            final_test_repetitions=list(self.TEST_REPETITIONS),
            final_model_trainer=final_trainer,
            test_accuracy=test_acc,
            test_balanced_accuracy=test_bal_acc,
            test_macro_f1=test_macro_f1,
            test_predictions=test_preds,
            test_labels=test_labels,
            test_evaluation_count=1,
            n_final_train_samples=len(train_labels),
            final_scaler_mean=scaler.mean_train.copy(),
            final_scaler_scale=scaler.scale_train.copy(),
        )
