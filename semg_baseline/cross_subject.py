"""Cross-Subject 5-fold validation protocol and benchmark execution engine.

Implements FR-009, FR-012.4, SC-004, Decision-5, Decision-6:
- Consumes strictly the 5 frozen Cross-Subject folds from Feature 003.
- Manifest identity validation: rejection of unauthorized subject permutations, mutations, or overlaps.
- Strict 24 train / 8 val / 8 test subject partition mapping per fold.
- Mathematical disjointness: TRAIN ∩ VAL = ∅, TRAIN ∩ TEST = ∅, VAL ∩ TEST = ∅.
- Complete subject preservation: TRAIN ∪ VAL ∪ TEST = exactly all 40 subjects.
- Prohibition of random fold generation, subject shuffling, or sklearn splitters.
- Structural test isolation: TEST subjects receive strictly zero raw access, filtering,
  segmentation, or prediction during hyperparameter / R selection.
- Immunity of selection phase to NaN-poisoned TEST subjects.
- Three-tier role segregation:
    * TRAIN (24 subjects): sole source for ThresholdCalibrator.fit, FeatureScaler.fit, CanonicalLdaTrainer.fit.
    * VAL (8 subjects): sole source for R selection via transform() and predict().
    * TEST (8 subjects): strictly invisible during selection; evaluated exactly once post-freeze.
- Selection of R governed by canonical Balanced Accuracy on VAL with order-independent lowest-R tie-break.
- Final test evaluation executed strictly once per fold (accumulating to 5 evaluations across all 5 folds).
- Multi-fold aggregate summary contract: mean, median, and standard deviation for accuracy,
  balanced_accuracy, and macro_f1.
- Fail-closed vocabulary enforcement preventing silent class count shrinkage.
- Deterministic reproducibility under identical inputs and configurations.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from semg_baseline.features import TimeDomainFeatureExtractor
from semg_baseline.filter import CausalBandpassFilter
from semg_baseline.lda import CanonicalLdaTrainer, PriorType
from semg_baseline.scaling import FeatureScaler
from semg_baseline.thresholds import ThresholdCalibration, ThresholdCalibrator
from semg_baseline.window import PureWindowSegmenter
from semg_baseline.within_subject import ValidationSelectionResult
from semg_dataset.contract import RecordingData
from semg_dataset.manifest import associate_rest_to_repetition_units
from semg_dataset.splits import CrossSubjectSplitter, FoldPartition, SplitManifest


@dataclass(frozen=True)
class MetricStats:
    """Summary statistics (mean, median, standard deviation) for a scalar metric."""

    mean: float
    median: float
    std: float


@dataclass(frozen=True)
class CrossSubjectAggregateMetrics:
    """Multi-fold statistical aggregate summary across all evaluated folds."""

    accuracy: MetricStats
    balanced_accuracy: MetricStats
    macro_f1: MetricStats


@dataclass(frozen=True)
class CrossSubjectFoldResult:
    """Outcome, predictions, and fitted parameters for a single Cross-Subject fold."""

    fold_id: int
    train_subjects: list[str]
    val_subjects: list[str]
    test_subjects: list[str]
    selected_r: float
    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    test_predictions: np.ndarray
    test_labels: np.ndarray
    n_train_samples: int
    n_val_samples: int
    n_test_samples: int
    final_model_trainer: CanonicalLdaTrainer
    threshold_calibration: ThresholdCalibration | None
    scaler_mean: np.ndarray | None
    scaler_scale: np.ndarray | None
    test_evaluation_count: int
    selection_summary: list[ValidationSelectionResult] = field(default_factory=list)


@dataclass(frozen=True)
class CrossSubjectBenchmarkResult:
    """Consolidated multi-fold benchmark results and statistical aggregate."""

    folds: list[CrossSubjectFoldResult]
    aggregate: CrossSubjectAggregateMetrics
    total_test_evaluations: int


class CrossSubjectRunner:
    """Executes the canonical 5-fold Cross-Subject benchmark evaluation protocol."""

    CANONICAL_SELECTION_METRIC: str = "balanced_accuracy"
    CANONICAL_VOCABULARIES: dict[str, int] = {
        "global_50_class": 50,
        "exercise_E1": 18,
        "exercise_E2": 24,
        "exercise_E3": 10,
    }

    def __init__(
        self,
        split_manifest: SplitManifest | None = None,
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
        subject_access_callback: Callable[[str], None] | None = None,
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
        self.subject_access_callback = subject_access_callback

        # Fail-closed selection metric validation
        metric_str = str(selection_metric).lower()
        if metric_str != self.CANONICAL_SELECTION_METRIC:
            raise ValueError(
                f"Non-canonical selection_metric '{selection_metric}' is prohibited. "
                f"The canonical Cross-Subject benchmark strictly mandates "
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

        # Manifest validation & establishment
        manifest = split_manifest if split_manifest is not None else CrossSubjectSplitter().generate_splits()
        self._validate_split_manifest(manifest)
        self._split_manifest = manifest

        # Execution event audit counter
        self._test_evaluation_count: int = 0

    @property
    def number_of_folds(self) -> int:
        """Return the number of folds in the split manifest."""
        return len(self._split_manifest.folds)

    @property
    def split_manifest(self) -> SplitManifest:
        """Return the immutable split manifest."""
        return self._split_manifest

    @property
    def test_evaluation_count(self) -> int:
        """Return the real number of test evaluations executed."""
        return self._test_evaluation_count

    def _validate_split_manifest(self, manifest: SplitManifest) -> None:
        """Validate split manifest conformity, 24/8/8 counts, disjointness, and Feature 003 identity."""
        if manifest.protocol != "cross_subject":
            raise ValueError(
                f"Invalid protocol '{manifest.protocol}' in split manifest; expected 'cross_subject'"
            )

        if len(manifest.folds) != 5:
            raise ValueError(f"CrossSubjectRunner requires exactly 5 folds; got {len(manifest.folds)}")

        all_expected_subjects = {f"S{i}" for i in range(1, 41)}

        for fold in manifest.folds:
            if len(fold.train_subjects) != 24:
                raise ValueError(
                    f"Fold {fold.fold_id} must contain exactly 24 train subjects; got {len(fold.train_subjects)}"
                )
            if len(fold.val_subjects) != 8:
                raise ValueError(
                    f"Fold {fold.fold_id} must contain exactly 8 val subjects; got {len(fold.val_subjects)}"
                )
            if len(fold.test_subjects) != 8:
                raise ValueError(
                    f"Fold {fold.fold_id} must contain exactly 8 test subjects; got {len(fold.test_subjects)}"
                )

            train_set = set(fold.train_subjects)
            val_set = set(fold.val_subjects)
            test_set = set(fold.test_subjects)

            if train_set & val_set:
                raise ValueError(
                    f"Leakage detected: TRAIN and VAL partition overlap in fold {fold.fold_id}"
                )
            if train_set & test_set:
                raise ValueError(
                    f"Leakage detected: TRAIN and TEST partition overlap in fold {fold.fold_id}"
                )
            if val_set & test_set:
                raise ValueError(
                    f"Leakage detected: VAL and TEST partition overlap in fold {fold.fold_id}"
                )

            if (train_set | val_set | test_set) != all_expected_subjects:
                raise ValueError(
                    f"Fold {fold.fold_id} does not span all 40 subjects exactly"
                )

        # Manifest identity validation against frozen Feature 003 canonical splits
        canonical = CrossSubjectSplitter().generate_splits()
        for fold, canonical_fold in zip(manifest.folds, canonical.folds):
            if (
                fold.fold_id != canonical_fold.fold_id
                or fold.train_subjects != canonical_fold.train_subjects
                or fold.val_subjects != canonical_fold.val_subjects
                or fold.test_subjects != canonical_fold.test_subjects
            ):
                raise ValueError(
                    f"Split manifest identity mismatch with Feature 003 canonical splits in fold {fold.fold_id}"
                )

    def _resolve_num_classes(self, labels: np.ndarray) -> int:
        """Resolve nominal vocabulary size."""
        if self.num_classes == "auto":
            return int(np.max(labels)) + 1
        return int(self.num_classes)

    def _get_fold_by_id(self, fold_id: int) -> FoldPartition:
        """Retrieve fold definition matching fold_id."""
        for fold in self._split_manifest.folds:
            if fold.fold_id == fold_id:
                return fold
        raise ValueError(f"Fold id {fold_id} not found in split manifest")

    def _prepare_subject_windows(
        self,
        subject_id: str,
        recording_provider: Mapping[str, RecordingData] | Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Load, causal-filter, and segment pure steady-state windows for a single subject."""
        if not isinstance(recording_provider, Mapping) and not hasattr(recording_provider, "__getitem__"):
            raise TypeError("recording_provider must support key indexing by subject_id")

        if subject_id not in recording_provider:
            raise KeyError(f"Subject '{subject_id}' not found in recording provider")

        if self.subject_access_callback is not None:
            self.subject_access_callback(subject_id)

        recording: RecordingData = recording_provider[subject_id]
        view = recording.get_view(self.view_name)

        # Fail immediately if raw signals contain NaN or Inf
        if not np.all(np.isfinite(view.signals)):
            raise ValueError(f"Recording signals for subject {subject_id} contain NaN or Inf values")

        units = associate_rest_to_repetition_units(view.repetitions)
        unique_units = np.unique(units)

        windows_list: list[np.ndarray] = []
        labels_list: list[np.ndarray] = []

        for rep in unique_units:
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
            empty_windows = np.empty((0, self.window_len, self.num_channels), dtype=np.float32)
            empty_labels = np.empty(0, dtype=view.labels.dtype)
            return empty_windows, empty_labels

        windows = np.vstack(windows_list)
        labels = np.concatenate(labels_list)
        return windows, labels

    def _prepare_partition_windows(
        self,
        subjects: Sequence[str],
        recording_provider: Mapping[str, RecordingData] | Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Aggregate analysis windows and labels across a sequence of subjects."""
        windows_list: list[np.ndarray] = []
        labels_list: list[np.ndarray] = []

        for sub_id in subjects:
            w, l = self._prepare_subject_windows(sub_id, recording_provider)
            if len(w) > 0:
                windows_list.append(w)
                labels_list.append(l)

        if not windows_list:
            empty_windows = np.empty((0, self.window_len, self.num_channels), dtype=np.float32)
            empty_labels = np.empty(0, dtype=np.int32)
            return empty_windows, empty_labels

        windows = np.vstack(windows_list)
        labels = np.concatenate(labels_list)
        return windows, labels

    def _evaluate_candidate_grid(
        self,
        train_windows: np.ndarray,
        train_labels: np.ndarray,
        val_windows: np.ndarray,
        val_labels: np.ndarray,
    ) -> list[ValidationSelectionResult]:
        """Evaluate candidate R multipliers statelessly on VAL after fitting strictly on TRAIN."""
        selection_summary: list[ValidationSelectionResult] = []

        for candidate_r in self.candidate_grid:
            # 1. Calibrate thresholds strictly on TRAIN rest windows
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

            # 2. Extract TRAIN features
            extractor = TimeDomainFeatureExtractor(
                epsilon=calibration.epsilon_channels,
                num_channels=self.num_channels,
            )
            X_train_raw = extractor.extract(train_windows)

            # 3. Fit scaler strictly on TRAIN features
            scaler = FeatureScaler()
            scaler.fit(X_train_raw)
            X_train_scaled = scaler.transform(X_train_raw)

            # 4. Fit Canonical LDA strictly on TRAIN scaled features
            trainer = CanonicalLdaTrainer(
                prior_type=self.prior_type,
                num_classes=self._resolve_num_classes(train_labels),
            )
            trainer.fit(X_train_scaled, train_labels)

            # 5. Evaluate statelessly on VAL features
            X_val_raw = extractor.extract(val_windows)
            X_val_scaled = scaler.transform(X_val_raw)
            val_preds = trainer.predict(X_val_scaled)

            acc = float(accuracy_score(val_labels, val_preds))
            bal_acc = float(balanced_accuracy_score(val_labels, val_preds))
            macro_f1 = float(f1_score(val_labels, val_preds, average="macro", zero_division=0))

            selection_summary.append(
                ValidationSelectionResult(
                    candidate_r=candidate_r,
                    mean_accuracy=acc,
                    mean_balanced_accuracy=bal_acc,
                    mean_macro_f1=macro_f1,
                )
            )

        return selection_summary

    def _select_best_r(
        self, selection_summary: list[ValidationSelectionResult]
    ) -> float:
        """Select best candidate R multiplier using VAL results.

        Canonical Selection Policy:
        - Primary criterion: highest mean Balanced Accuracy on VAL.
        - Tie-break: lowest candidate R numerically, independent of input order.
        """
        if not selection_summary:
            raise ValueError("Empty selection summary: no candidate configurations evaluated.")

        def _candidate_sort_key(res: ValidationSelectionResult) -> tuple[float, float]:
            score = round(float(res.mean_balanced_accuracy), 12)
            return (-score, float(res.candidate_r))

        best_result = min(selection_summary, key=_candidate_sort_key)
        return float(best_result.candidate_r)

    def run_validation_selection_for_fold(
        self,
        fold_id: int,
        recording_provider: Mapping[str, RecordingData] | Any,
    ) -> tuple[float, list[ValidationSelectionResult]]:
        """Run candidate selection for a single fold without any TEST exposure (FR-005.7, Decision-6)."""
        fold = self._get_fold_by_id(fold_id)

        # Structural zero-leakage isolation: only TRAIN (24) and VAL (8) subjects accessed
        train_windows, train_labels = self._prepare_partition_windows(
            fold.train_subjects, recording_provider
        )
        val_windows, val_labels = self._prepare_partition_windows(
            fold.val_subjects, recording_provider
        )

        selection_summary = self._evaluate_candidate_grid(
            train_windows=train_windows,
            train_labels=train_labels,
            val_windows=val_windows,
            val_labels=val_labels,
        )
        selected_r = self._select_best_r(selection_summary)
        return selected_r, selection_summary

    def run_fold(
        self,
        fold_id: int,
        recording_provider: Mapping[str, RecordingData] | Any,
    ) -> CrossSubjectFoldResult:
        """Execute complete single-fold evaluation: TRAIN fit, VAL selection, and single TEST evaluation."""
        fold = self._get_fold_by_id(fold_id)

        # 1. Load TRAIN (24) and VAL (8) subjects
        train_windows, train_labels = self._prepare_partition_windows(
            fold.train_subjects, recording_provider
        )
        val_windows, val_labels = self._prepare_partition_windows(
            fold.val_subjects, recording_provider
        )

        # 2. Candidate selection on VAL
        selection_summary = self._evaluate_candidate_grid(
            train_windows=train_windows,
            train_labels=train_labels,
            val_windows=val_windows,
            val_labels=val_labels,
        )
        selected_r = self._select_best_r(selection_summary)

        # 3. Final model refit strictly on all 24 TRAIN subjects
        final_calibrator = ThresholdCalibrator(
            candidate_grid=(selected_r,),
            default_r=selected_r,
            num_channels=self.num_channels,
        )
        final_calibration = final_calibrator.calibrate(
            train_emg=train_windows,
            train_labels=train_labels,
            rest_label=0,
            r_multiplier=selected_r,
        )

        final_extractor = TimeDomainFeatureExtractor(
            epsilon=final_calibration.epsilon_channels,
            num_channels=self.num_channels,
        )
        X_train_raw = final_extractor.extract(train_windows)

        final_scaler = FeatureScaler()
        final_scaler.fit(X_train_raw)
        X_train_scaled = final_scaler.transform(X_train_raw)

        final_trainer = CanonicalLdaTrainer(
            prior_type=self.prior_type,
            num_classes=self._resolve_num_classes(train_labels),
        )
        final_trainer.fit(X_train_scaled, train_labels)

        # 4. Single frozen TEST evaluation (strictly 8 TEST subjects)
        test_windows, test_labels = self._prepare_partition_windows(
            fold.test_subjects, recording_provider
        )

        X_test_raw = final_extractor.extract(test_windows)
        X_test_scaled = final_scaler.transform(X_test_raw)
        test_preds = final_trainer.predict(X_test_scaled)

        test_acc = float(accuracy_score(test_labels, test_preds))
        test_bal_acc = float(balanced_accuracy_score(test_labels, test_preds))
        test_macro_f1 = float(f1_score(test_labels, test_preds, average="macro", zero_division=0))

        # Test evaluation counter increments strictly upon successful completion
        self._test_evaluation_count += 1

        return CrossSubjectFoldResult(
            fold_id=fold.fold_id,
            train_subjects=list(fold.train_subjects),
            val_subjects=list(fold.val_subjects),
            test_subjects=list(fold.test_subjects),
            selected_r=selected_r,
            accuracy=test_acc,
            balanced_accuracy=test_bal_acc,
            macro_f1=test_macro_f1,
            test_predictions=test_preds,
            test_labels=test_labels,
            n_train_samples=len(train_labels),
            n_val_samples=len(val_labels),
            n_test_samples=len(test_labels),
            final_model_trainer=final_trainer,
            threshold_calibration=final_calibration,
            scaler_mean=final_scaler.mean_train.copy() if final_scaler.mean_train is not None else None,
            scaler_scale=final_scaler.scale_train.copy() if final_scaler.scale_train is not None else None,
            test_evaluation_count=1,
            selection_summary=selection_summary,
        )

    def run_all_folds(
        self,
        recording_provider: Mapping[str, RecordingData] | Any,
    ) -> CrossSubjectBenchmarkResult:
        """Execute complete 5-fold Cross-Subject benchmark and compute statistical aggregate summary."""
        self._test_evaluation_count = 0
        fold_results: list[CrossSubjectFoldResult] = []

        for fold in self._split_manifest.folds:
            res = self.run_fold(fold_id=fold.fold_id, recording_provider=recording_provider)
            fold_results.append(res)

        accuracies = [f.accuracy for f in fold_results]
        bal_accs = [f.balanced_accuracy for f in fold_results]
        f1s = [f.macro_f1 for f in fold_results]

        agg = CrossSubjectAggregateMetrics(
            accuracy=MetricStats(
                mean=float(np.mean(accuracies)),
                median=float(np.median(accuracies)),
                std=float(np.std(accuracies, ddof=0)),
            ),
            balanced_accuracy=MetricStats(
                mean=float(np.mean(bal_accs)),
                median=float(np.median(bal_accs)),
                std=float(np.std(bal_accs, ddof=0)),
            ),
            macro_f1=MetricStats(
                mean=float(np.mean(f1s)),
                median=float(np.median(f1s)),
                std=float(np.std(f1s, ddof=0)),
            ),
        )

        return CrossSubjectBenchmarkResult(
            folds=fold_results,
            aggregate=agg,
            total_test_evaluations=len(fold_results),
        )


__all__ = [
    "CrossSubjectAggregateMetrics",
    "CrossSubjectBenchmarkResult",
    "CrossSubjectFoldResult",
    "CrossSubjectRunner",
    "MetricStats",
]
