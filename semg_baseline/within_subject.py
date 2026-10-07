"""Within-Subject validation protocol and benchmark execution engine.

Implements FR-005.7, FR-008, SC-003, Decision-3, Decision-5:
- Exactly four leave-one-training-repetition-out (LOTO) validation folds over repetitions [1, 3, 4, 6].
    Fold 1: Val=[1], Train=[3, 4, 6]
    Fold 2: Val=[3], Train=[1, 4, 6]
    Fold 3: Val=[4], Train=[1, 3, 6]
    Fold 4: Val=[6], Train=[1, 3, 4]
- Disjoint partition separation: TRAIN ∩ VAL = ∅, TRAIN ∩ TEST = ∅, VAL ∩ TEST = ∅.
- Structural test isolation: Test repetitions [2, 5] receive strictly zero raw access,
  filtering, segmentation, or prediction during validation / configuration selection (FR-005.7).
- Independent causal filtering per repetition unit eliminating temporal filter state leakage.
- Final model fitted strictly on all training repetitions [1, 3, 4, 6].
- Final test evaluated strictly once on test repetitions [2, 5] (test_evaluation_count == 1).
- Preprocessing anti-leakage: noise deadzones and feature scaling computed solely from fold train.
- Candidate configuration selection over R in candidate grid strictly derived from training pool validation.
- Formally governed selection metric with deterministic tie-break (lowest candidate R).
- Fail-closed vocabulary enforcement preventing silent class count shrinkage.
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
from semg_dataset.manifest import associate_rest_to_repetition_units


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
    CANONICAL_SELECTION_METRIC: str = "balanced_accuracy"
    CANONICAL_VOCABULARIES: dict[str, int] = {
        "global_50_class": 50,
        "exercise_E1": 18,
        "exercise_E2": 24,
        "exercise_E3": 10,
    }

    def __init__(
        self,
        candidate_grid: Sequence[float] = (0.5, 1.0, 1.5, 2.0, 3.0),
        num_classes: int | str = 50,
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
        raw_access_callback: Callable[[int, int], None] | None = None,
        selection_metric: str = "balanced_accuracy",
    ) -> None:
        self.candidate_grid = tuple(float(r) for r in candidate_grid)
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
        self.raw_access_callback = raw_access_callback

        # Fail-closed selection metric validation (frozen canonical metric)
        metric_str = str(selection_metric).lower()
        if metric_str != self.CANONICAL_SELECTION_METRIC:
            raise ValueError(
                f"Non-canonical selection_metric '{selection_metric}' is prohibited. "
                f"The canonical Within-Subject benchmark strictly mandates "
                f"selection_metric='{self.CANONICAL_SELECTION_METRIC}'."
            )
        self.selection_metric = metric_str

        # Vocabulary configuration
        if isinstance(num_classes, str) and num_classes.lower() == "auto":
            self.num_classes: int | str = "auto"
        elif isinstance(num_classes, str) and num_classes in self.CANONICAL_VOCABULARIES:
            self.num_classes = self.CANONICAL_VOCABULARIES[num_classes]
        else:
            try:
                val = int(num_classes)
            except (ValueError, TypeError):
                raise ValueError(
                    f"Invalid num_classes '{num_classes}'. Must be an integer >= 2, "
                    f"'auto', or one of {list(self.CANONICAL_VOCABULARIES.keys())}."
                ) from None
            if val < 2:
                raise ValueError(f"num_classes must be >= 2, got {val}")
            self.num_classes = val

        self.segmenter = PureWindowSegmenter(
            window_len=self.window_len,
            stride=self.stride,
            fs=self.fs,
        )

        # Execution event audit counters
        self._test_evaluation_count: int = 0
        self._accessed_raw_repetitions: list[int] = []

    @property
    def test_evaluation_count(self) -> int:
        """Return the real number of test evaluations executed."""
        return self._test_evaluation_count

    @property
    def accessed_raw_repetitions(self) -> list[int]:
        """Return audit history of repetition units whose raw signals were accessed."""
        return list(self._accessed_raw_repetitions)

    def _resolve_num_classes(self, labels: np.ndarray) -> int:
        """Resolve nominal vocabulary size.

        In canonical benchmark mode, vocabulary size is strictly explicit and fixed.
        Dynamic inference from labels is permitted only when explicitly configured
        with num_classes='auto' for controlled procedural test fixtures.
        """
        if self.num_classes == "auto":
            return int(np.max(labels)) + 1
        return int(self.num_classes)

    @classmethod
    def get_fold_definitions(cls) -> list[FoldPartitionDef]:
        """Return canonical 4 leave-one-training-repetition-out fold partition definitions (FR-008)."""
        return [
            FoldPartitionDef(fold_id=1, val_repetition=1, train_repetitions=[3, 4, 6]),
            FoldPartitionDef(fold_id=2, val_repetition=3, train_repetitions=[1, 4, 6]),
            FoldPartitionDef(fold_id=3, val_repetition=4, train_repetitions=[1, 3, 6]),
            FoldPartitionDef(fold_id=4, val_repetition=6, train_repetitions=[1, 3, 4]),
        ]

    def prepare_repetition_windows(
        self,
        recording: RecordingData,
        repetitions: Sequence[int],
        validate_all_required: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[TransitionAudit]]:
        """Filter continuous EMG and segment pure steady-state windows strictly for specified repetitions.

        Enforces structural zero-leakage partition isolation:
        - Never reads, slices, or filters signals from unrequested repetitions.
        - Filters each repetition unit independently to eliminate causal filter memory leakage across repetitions.
        - Validates presence of all required repetitions when requested (FR-011.5).

        Returns:
            windows: Array of shape (N_admitted, window_len, num_channels) in float32.
            labels: Array of shape (N_admitted,) containing integer gesture labels.
            repetition_units: Array of shape (N_admitted,) containing integer repetition units.
            audits: List of TransitionAudit records per admitted repetition unit.
        """
        view = recording.get_view(self.view_name)
        units = associate_rest_to_repetition_units(view.repetitions)

        # Fail-closed validation for required repetitions
        if validate_all_required:
            present_reps = set(np.unique(units).tolist())
            missing_reps = set(self.ALL_REQUIRED_REPETITIONS) - present_reps
            if missing_reps:
                raise ValueError(
                    f"Required repetition(s) missing from subject {recording.subject_id}: "
                    f"{sorted(missing_reps)}"
                )

        windows_list: list[np.ndarray] = []
        labels_list: list[np.ndarray] = []
        repetition_units_list: list[np.ndarray] = []
        audits_list: list[TransitionAudit] = []

        for rep in repetitions:
            rep_mask = units == rep
            if not np.any(rep_mask):
                continue

            # Structural isolation: access raw signals strictly for repetition 'rep'
            rep_signals = view.signals[rep_mask]
            rep_labels = view.labels[rep_mask]
            rep_original = view.repetitions[rep_mask]
            rep_units_slice = units[rep_mask]

            # Record raw access audit
            self._accessed_raw_repetitions.append(rep)
            if self.raw_access_callback is not None:
                self.raw_access_callback(rep, len(rep_signals))

            # Filter repetition unit independently (fresh filter state: zero inter-repetition leakage)
            bp_filter = CausalBandpassFilter(
                low_hz=self.filter_low_hz,
                high_hz=self.filter_high_hz,
                fs=self.fs,
                order=self.filter_order,
                num_channels=self.num_channels,
            )
            filtered_rep = bp_filter.process(rep_signals)

            win_r, lbl_r, audit_r = self.segmenter.segment(
                emg=filtered_rep,
                labels=rep_labels,
                repetitions=rep_original,
                repetition_units=rep_units_slice,
            )

            if len(win_r) > 0:
                windows_list.append(win_r)
                labels_list.append(lbl_r)
                repetition_units_list.append(
                    np.array(audit_r.admitted_repetition_units, dtype=np.int32)
                )
                audits_list.append(audit_r)

        if not windows_list:
            empty_windows = np.empty((0, self.window_len, self.num_channels), dtype=np.float32)
            empty_labels = np.empty(0, dtype=view.labels.dtype)
            empty_units = np.empty(0, dtype=np.int32)
            return empty_windows, empty_labels, empty_units, audits_list

        windows = np.vstack(windows_list)
        labels = np.concatenate(labels_list)
        repetition_units = np.concatenate(repetition_units_list)

        return windows, labels, repetition_units, audits_list

    def count_admitted_windows(
        self, recording: RecordingData, repetitions: Sequence[int]
    ) -> int:
        """Count admitted windows matching specified repetitions."""
        windows, _, _, _ = self.prepare_repetition_windows(recording, repetitions=repetitions)
        return len(windows)

    def run_validation_selection(
        self, recording: RecordingData
    ) -> list[ValidationSelectionResult]:
        """Execute internal 4-fold validation across candidate R values without test set exposure (FR-005.7).

        Structural Invariant:
        - Only TRAINING_REPETITIONS [1, 3, 4, 6] are loaded, filtered, or segmented.
        - TEST_REPETITIONS [2, 5] receive strictly zero raw access or preprocessing.
        """
        windows, labels, repetition_units, _ = self.prepare_repetition_windows(
            recording=recording,
            repetitions=self.TRAINING_REPETITIONS,
            validate_all_required=True,
        )
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

                # Instrument window access callback
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
        """Select best candidate R multiplier using internal validation results.

        Canonical Selection Policy:
        - Frozen metric: mean Balanced Accuracy across the 4 LOTO folds.
        - Deterministic tie-break: lowest candidate R wins numerically,
          completely independent of input summary order.
        """
        if not selection_summary:
            raise ValueError("Empty selection summary: no candidate configurations evaluated.")

        # Sort candidate results deterministically by:
        # primary key: descending score (higher score is better)
        # secondary key: ascending candidate_r (lowest R wins ties)
        # Float scores are rounded to 12 decimal places to prevent spurious IEEE-754 precision artifacts.
        def _candidate_sort_key(res: ValidationSelectionResult) -> tuple[float, float]:
            score = round(float(res.mean_balanced_accuracy), 12)
            return (-score, float(res.candidate_r))

        best_result = min(selection_summary, key=_candidate_sort_key)
        return best_result.candidate_r, best_result.fold_results

    def evaluate_test(
        self,
        recording: RecordingData,
        trainer: CanonicalLdaTrainer,
        extractor: TimeDomainFeatureExtractor,
        scaler: FeatureScaler,
    ) -> tuple[float, float, float, np.ndarray, np.ndarray]:
        """Execute final test evaluation strictly once on test repetitions [2, 5].

        Increments self._test_evaluation_count upon successful evaluation completion.
        """
        test_windows, test_labels, test_rep_units, _ = self.prepare_repetition_windows(
            recording=recording,
            repetitions=self.TEST_REPETITIONS,
            validate_all_required=False,
        )

        if self.access_callback is not None:
            for r in test_rep_units:
                self.access_callback(int(r))

        X_test_raw = extractor.extract(test_windows)
        X_test_scaled = scaler.transform(X_test_raw)
        test_preds = trainer.predict(X_test_scaled)

        test_acc = float(accuracy_score(test_labels, test_preds))
        test_bal_acc = float(balanced_accuracy_score(test_labels, test_preds))
        test_macro_f1 = float(f1_score(test_labels, test_preds, average="macro", zero_division=0))

        # Real auditable event counter increments strictly after successful evaluation
        self._test_evaluation_count += 1

        return test_acc, test_bal_acc, test_macro_f1, test_preds, test_labels

    def run(self, recording: RecordingData) -> WithinSubjectResult:
        """Execute full Within-Subject benchmark: LOTO validation, configuration selection, and final test."""
        # Reset execution audit counters for this run
        self._test_evaluation_count = 0
        self._accessed_raw_repetitions.clear()

        # Step 1: Internal validation selection across candidate grid (zero access to test repetitions)
        selection_summary = self.run_validation_selection(recording)
        selected_r, validation_folds = self._select_best_r(selection_summary)

        # Step 2: Prepare windows for final training fit strictly on all training pool repetitions [1, 3, 4, 6]
        train_windows, train_labels, _, _ = self.prepare_repetition_windows(
            recording=recording,
            repetitions=self.TRAINING_REPETITIONS,
            validate_all_required=False,
        )

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

        # Step 3: Final test evaluation strictly once on test repetitions [2, 5]
        test_acc, test_bal_acc, test_macro_f1, test_preds, test_labels = self.evaluate_test(
            recording=recording,
            trainer=final_trainer,
            extractor=extractor,
            scaler=scaler,
        )

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
            test_evaluation_count=self._test_evaluation_count,
            n_final_train_samples=len(train_labels),
            final_scaler_mean=scaler.mean_train.copy(),
            final_scaler_scale=scaler.scale_train.copy(),
        )
