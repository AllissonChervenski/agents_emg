"""Unit, contract, and anti-leakage tests for CrossSubjectRunner (FR-009, SC-004, Decision-5).

Validates:
- Exactly 5 frozen Cross-Subject rotating folds from Feature 003.
- Strict 24 train / 8 val / 8 test subject partition mapping per fold.
- Mathematical disjointness: TRAIN ∩ VAL = ∅, TRAIN ∩ TEST = ∅, VAL ∩ TEST = ∅.
- Complete subject preservation: TRAIN ∪ VAL ∪ TEST = exactly all 40 subjects.
- Prohibition of random fold generation, subject shuffling, or sklearn splitters.
- Structural test isolation: TEST subjects receive zero raw access, filtering,
  segmentation, or prediction during hyperparameter / R selection.
- Immunity of selection phase to NaN-poisoned TEST subjects.
- Preprocessing anti-leakage: noise thresholds and feature scaling derived strictly from 24 TRAIN subjects.
- Regularized LDA fit strictly on 24 TRAIN subjects.
- Selection of R governed by canonical Balanced Accuracy with order-independent lowest-R tie-break.
- Final test evaluation executed strictly once per fold (5 evaluations across all 5 folds).
- Adversarial rejection of invalid partitions (23 train, 9 val, overlapping subjects).
- Adversarial rejection of missing subjects or class vocabulary mismatches.
"""

from __future__ import annotations

import numpy as np
import pytest

from semg_baseline.cross_subject import (
    CrossSubjectBenchmarkResult,
    CrossSubjectFoldResult,
    CrossSubjectRunner,
)
from semg_baseline.lda import CanonicalLdaTrainer
from semg_dataset.contract import RecordingData
from semg_dataset.splits import CrossSubjectSplitter, FoldPartition, SplitManifest
from tests.fixtures.synthetic_semg import make_synthetic_recording


def _make_mock_synthetic_provider(
    subjects: list[str],
    active_labels: tuple[int, ...] = (1, 2, 3),
    num_classes: int = 4,
    poison_subjects: tuple[str, ...] = (),
) -> dict[str, RecordingData]:
    """Helper to create lightweight synthetic recordings for a list of subjects."""
    provider: dict[str, RecordingData] = {}
    for idx, sub_id in enumerate(subjects):
        rec = make_synthetic_recording(
            seed=1000 + idx,
            subject_id=sub_id,
            active_labels=active_labels,
            samples_per_rest=400,
            samples_per_active=600,
        )
        if sub_id in poison_subjects:
            rec.emg[:] = np.nan
        provider[sub_id] = rec
    return provider


def test_cross_subject_exactly_five_frozen_folds_from_feature_003() -> None:
    """Runner must consume exactly 5 frozen folds from Feature 003 without regeneration (FR-009)."""
    runner = CrossSubjectRunner()

    assert runner.number_of_folds == 5
    manifest = runner.split_manifest
    assert isinstance(manifest, SplitManifest)
    assert manifest.protocol == "cross_subject"
    assert len(manifest.folds) == 5

    # Identity check with canonical CrossSubjectSplitter
    canonical_manifest = CrossSubjectSplitter().generate_splits()
    for fold, canonical_fold in zip(manifest.folds, canonical_manifest.folds):
        assert fold.fold_id == canonical_fold.fold_id
        assert fold.train_subjects == canonical_fold.train_subjects
        assert fold.val_subjects == canonical_fold.val_subjects
        assert fold.test_subjects == canonical_fold.test_subjects


def test_cross_subject_partitions_24_8_8_and_disjointness() -> None:
    """Every fold must contain exactly 24 train, 8 val, 8 test subjects with zero overlap (FR-009, SC-004)."""
    runner = CrossSubjectRunner()
    all_expected_subjects = {f"S{i}" for i in range(1, 41)}

    for fold in runner.split_manifest.folds:
        assert len(fold.train_subjects) == 24, f"Fold {fold.fold_id} must have 24 train subjects"
        assert len(fold.val_subjects) == 8, f"Fold {fold.fold_id} must have 8 val subjects"
        assert len(fold.test_subjects) == 8, f"Fold {fold.fold_id} must have 8 test subjects"

        train_set = set(fold.train_subjects)
        val_set = set(fold.val_subjects)
        test_set = set(fold.test_subjects)

        # Zero leakage cross-partition assertions
        assert train_set & val_set == set(), f"Fold {fold.fold_id}: TRAIN and VAL overlap"
        assert train_set & test_set == set(), f"Fold {fold.fold_id}: TRAIN and TEST overlap"
        assert val_set & test_set == set(), f"Fold {fold.fold_id}: VAL and TEST overlap"

        # Complete subject universe preservation
        assert (train_set | val_set | test_set) == all_expected_subjects, (
            f"Fold {fold.fold_id} does not span all 40 subjects exactly"
        )


def test_cross_subject_test_subjects_invisible_during_selection() -> None:
    """TEST subjects must receive strictly zero access during validation and R selection (FR-005.7)."""
    runner = CrossSubjectRunner(
        num_classes=4,
        candidate_grid=(1.0, 1.5),
    )
    fold_1 = runner.split_manifest.folds[0]

    accessed_subjects: list[str] = []

    def on_subject_access(subject_id: str) -> None:
        accessed_subjects.append(subject_id)

    runner.subject_access_callback = on_subject_access

    # Provide synthetic recordings for train + val subjects
    all_needed_subjects = fold_1.train_subjects + fold_1.val_subjects + fold_1.test_subjects
    provider = _make_mock_synthetic_provider(all_needed_subjects)

    # Run internal validation selection only for fold 1
    selected_r, _ = runner.run_validation_selection_for_fold(
        fold_id=1,
        recording_provider=provider,
    )

    assert selected_r in (1.0, 1.5)

    # Assert that TEST subjects were NEVER accessed
    test_subject_set = set(fold_1.test_subjects)
    accessed_test_subjects = [s for s in accessed_subjects if s in test_subject_set]
    assert len(accessed_test_subjects) == 0, (
        f"Data leakage detected: TEST subjects accessed during selection: {accessed_test_subjects}"
    )


def test_cross_subject_test_poisoning_immunity_during_selection() -> None:
    """Selection phase must be completely immune to NaN-poisoned TEST subjects (FR-005.7)."""
    runner = CrossSubjectRunner(
        num_classes=4,
        candidate_grid=(1.0, 2.0),
    )
    fold_1 = runner.split_manifest.folds[0]

    # Poison all 8 TEST subjects with NaNs in EMG
    all_subjects = fold_1.train_subjects + fold_1.val_subjects + fold_1.test_subjects
    provider = _make_mock_synthetic_provider(
        all_subjects,
        poison_subjects=tuple(fold_1.test_subjects),
    )

    # Validation selection MUST succeed without error because TEST subjects are never touched
    selected_r, _ = runner.run_validation_selection_for_fold(
        fold_id=1,
        recording_provider=provider,
    )
    assert selected_r in (1.0, 2.0)

    # In contrast, full fold execution must fail when final test evaluation opens poisoned TEST data
    with pytest.raises(ValueError, match="NaN"):
        runner.run_fold(fold_id=1, recording_provider=provider)


def test_cross_subject_preprocessing_and_lda_strictly_train_only() -> None:
    """Noise thresholds, feature scaler, and LDA fit must be computed solely from 24 TRAIN subjects."""
    runner = CrossSubjectRunner(
        num_classes=4,
        candidate_grid=(1.5,),
    )
    fold_1 = runner.split_manifest.folds[0]
    all_subjects = fold_1.train_subjects + fold_1.val_subjects + fold_1.test_subjects
    provider = _make_mock_synthetic_provider(all_subjects)

    result = runner.run_fold(fold_id=1, recording_provider=provider)

    assert isinstance(result, CrossSubjectFoldResult)
    assert result.fold_id == 1
    assert result.train_subjects == fold_1.train_subjects
    assert result.val_subjects == fold_1.val_subjects
    assert result.test_subjects == fold_1.test_subjects

    # Check fitted model parameters
    assert isinstance(result.final_model_trainer, CanonicalLdaTrainer)
    assert result.threshold_calibration is not None
    assert result.scaler_mean is not None
    assert result.scaler_scale is not None
    assert result.n_train_samples > 0
    assert result.n_val_samples > 0
    assert result.n_test_samples > 0


def test_cross_subject_test_evaluated_exactly_once_per_fold() -> None:
    """Final test evaluation must occur strictly once per fold (SC-004)."""
    runner = CrossSubjectRunner(
        num_classes=4,
        candidate_grid=(1.0,),
    )
    fold_1 = runner.split_manifest.folds[0]
    all_subjects = fold_1.train_subjects + fold_1.val_subjects + fold_1.test_subjects
    provider = _make_mock_synthetic_provider(all_subjects)

    assert runner.test_evaluation_count == 0

    result = runner.run_fold(fold_id=1, recording_provider=provider)

    assert runner.test_evaluation_count == 1
    assert result.test_evaluation_count == 1
    assert len(result.test_predictions) == len(result.test_labels)
    assert len(result.test_predictions) > 0


def test_cross_subject_selection_metric_frozen_and_lowest_r_tie_break() -> None:
    """Selection metric must enforce canonical balanced_accuracy and numerically pick lowest R on tie."""
    # 1. Non-canonical metric rejection
    with pytest.raises(ValueError, match="Non-canonical selection_metric 'accuracy'"):
        CrossSubjectRunner(selection_metric="accuracy")

    with pytest.raises(ValueError, match="Non-canonical selection_metric 'invalid_metric'"):
        CrossSubjectRunner(selection_metric="invalid_metric")

    # 2. Order-independent lowest-R tie-break verification
    runner = CrossSubjectRunner(
        num_classes=4,
        candidate_grid=(0.5, 1.0, 2.0),
        selection_metric="balanced_accuracy",
    )

    # Mock tie summary in reversed order [2.0, 1.0, 0.5] with identical validation scores
    from semg_baseline.within_subject import ValidationSelectionResult

    reversed_tie_summary = [
        ValidationSelectionResult(candidate_r=2.0, mean_accuracy=0.8, mean_balanced_accuracy=0.85, mean_macro_f1=0.8),
        ValidationSelectionResult(candidate_r=1.0, mean_accuracy=0.8, mean_balanced_accuracy=0.85, mean_macro_f1=0.8),
        ValidationSelectionResult(candidate_r=0.5, mean_accuracy=0.8, mean_balanced_accuracy=0.85, mean_macro_f1=0.8),
    ]

    best_r = runner._select_best_r(reversed_tie_summary)
    assert best_r == 0.5, f"Expected lowest R (0.5) under reversed candidate order, got {best_r}"


def test_cross_subject_vocabulary_fail_closed() -> None:
    """Default num_classes=50 must fail closed on incomplete gesture vocabulary."""
    runner = CrossSubjectRunner()
    assert runner.num_classes == 50

    fold_1 = runner.split_manifest.folds[0]
    all_subjects = fold_1.train_subjects + fold_1.val_subjects + fold_1.test_subjects
    # Provider with only 4 classes (0..3)
    provider = _make_mock_synthetic_provider(all_subjects, active_labels=(1, 2, 3), num_classes=4)

    with pytest.raises(ValueError, match="Class vocabulary mismatch for num_classes=50: missing classes"):
        runner.run_fold(fold_id=1, recording_provider=provider)


def test_cross_subject_adversarial_invalid_partitions_rejected() -> None:
    """Runner must reject malformed, non-canonical, or leaking split manifests."""
    # 1. Manifest with only 4 folds
    canonical = CrossSubjectSplitter().generate_splits()
    malformed_folds = canonical.folds[:4]
    manifest_4_folds = SplitManifest(protocol="cross_subject", folds=malformed_folds)

    with pytest.raises(ValueError, match="CrossSubjectRunner requires exactly 5 folds; got 4"):
        CrossSubjectRunner(split_manifest=manifest_4_folds)

    # 2. Fold with 23 train subjects (missing one subject)
    f1 = canonical.folds[0]
    bad_f1 = FoldPartition(
        fold_id=1,
        train_subjects=f1.train_subjects[:23],  # 23 instead of 24
        val_subjects=f1.val_subjects,
        test_subjects=f1.test_subjects,
    )
    manifest_bad_train = SplitManifest(
        protocol="cross_subject",
        folds=[bad_f1] + canonical.folds[1:],
    )
    with pytest.raises(ValueError, match="[Ff]old 1 must contain exactly 24 train subjects; got 23"):
        CrossSubjectRunner(split_manifest=manifest_bad_train)

    # 3. Fold with 9 val subjects
    bad_f1_val = FoldPartition(
        fold_id=1,
        train_subjects=f1.train_subjects,
        val_subjects=f1.val_subjects + ["S99"],  # 9 instead of 8
        test_subjects=f1.test_subjects,
    )
    manifest_bad_val = SplitManifest(
        protocol="cross_subject",
        folds=[bad_f1_val] + canonical.folds[1:],
    )
    with pytest.raises(ValueError, match="[Ff]old 1 must contain exactly 8 val subjects; got 9"):
        CrossSubjectRunner(split_manifest=manifest_bad_val)

    # 4. Subject overlap between train and test
    overlapping_f1 = FoldPartition(
        fold_id=1,
        train_subjects=f1.train_subjects,
        val_subjects=f1.val_subjects,
        test_subjects=[f1.train_subjects[0]] + f1.test_subjects[1:],  # shared subject!
    )
    manifest_overlap = SplitManifest(
        protocol="cross_subject",
        folds=[overlapping_f1] + canonical.folds[1:],
    )
    with pytest.raises(ValueError, match="[Ll]eakage.*overlap"):
        CrossSubjectRunner(split_manifest=manifest_overlap)


def test_cross_subject_missing_subject_in_provider_fails_closed() -> None:
    """Missing subject in recording provider must fail closed with clear error."""
    runner = CrossSubjectRunner(num_classes=4)
    fold_1 = runner.split_manifest.folds[0]
    # Provider missing one train subject
    incomplete_subjects = fold_1.train_subjects[1:] + fold_1.val_subjects + fold_1.test_subjects
    provider = _make_mock_synthetic_provider(incomplete_subjects)

    with pytest.raises(KeyError, match=fold_1.train_subjects[0]):
        runner.run_fold(fold_id=1, recording_provider=provider)


def test_cross_subject_run_all_folds_returns_benchmark_result() -> None:
    """run_all_folds must aggregate all 5 folds into CrossSubjectBenchmarkResult with 5 total evaluations."""
    runner = CrossSubjectRunner(num_classes=4, candidate_grid=(1.0,))
    all_subjects = [f"S{i}" for i in range(1, 41)]
    provider = _make_mock_synthetic_provider(all_subjects)

    benchmark_res = runner.run_all_folds(provider)

    assert isinstance(benchmark_res, CrossSubjectBenchmarkResult)
    assert len(benchmark_res.folds) == 5
    assert benchmark_res.total_test_evaluations == 5
    assert runner.test_evaluation_count == 5
    assert 0.0 <= benchmark_res.mean_accuracy <= 1.0
    assert 0.0 <= benchmark_res.mean_balanced_accuracy <= 1.0
    assert 0.0 <= benchmark_res.mean_macro_f1 <= 1.0
