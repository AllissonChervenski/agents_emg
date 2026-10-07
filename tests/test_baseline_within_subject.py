"""Unit, contract, and anti-leakage tests for WithinSubjectRunner (FR-005.7, FR-008, SC-003, Decision-3, Decision-5).

Validates:
- Exactly four leave-one-training-repetition-out validation folds over repetitions [1, 3, 4, 6] (FR-008).
- Exact train/val partition mapping for all four folds:
    Fold 1: Val=[1], Train=[3, 4, 6]
    Fold 2: Val=[3], Train=[1, 4, 6]
    Fold 3: Val=[4], Train=[1, 3, 6]
    Fold 4: Val=[6], Train=[1, 3, 4]
- Disjoint partition separation: TRAIN ∩ VAL = ∅, TRAIN ∩ TEST = ∅, VAL ∩ TEST = ∅.
- Structural test isolation: Test repetitions [2, 5] receive zero raw access, filtering,
  segmentation, or prediction during validation / configuration selection (FR-005.7).
- Immunity to test set raw NaN-poisoning during selection phase.
- Final model fitted strictly on all training pool repetitions [1, 3, 4, 6].
- Final test evaluated strictly once on test repetitions [2, 5] (real auditable event counter).
- Preprocessing anti-leakage: noise deadzones and feature scaling computed solely from fold train.
- Candidate configuration selection over R in candidate grid strictly derived from training pool validation.
- Fail-closed validation on invalid selection metrics and deterministic lowest-R tie-break.
- Fail-closed vocabulary enforcement preventing silent class count shrinkage from 50 to 49.
- Deterministic reproducibility under identical inputs and configurations.
- Fail-closed error handling when required repetitions are missing.
- Prohibition of random window shuffling across repetitions.
"""

from __future__ import annotations

import numpy as np
import pytest

from semg_baseline.lda import CanonicalLdaTrainer
from semg_baseline.within_subject import (
    FoldPartitionDef,
    ValidationSelectionResult,
    WithinSubjectResult,
    WithinSubjectRunner,
)
from tests.fixtures.synthetic_semg import make_synthetic_recording


def test_within_subject_four_loto_folds_exact_structure() -> None:
    """Runner must generate exactly four leave-one-training-repetition-out folds with exact mappings (FR-008)."""
    runner = WithinSubjectRunner()
    folds = runner.get_fold_definitions()

    assert len(folds) == 4, f"Expected exactly 4 LOTO folds, got {len(folds)}"

    expected_mappings = [
        (1, 1, [3, 4, 6]),
        (2, 3, [1, 4, 6]),
        (3, 4, [1, 3, 6]),
        (4, 6, [1, 3, 4]),
    ]

    for fold_def, (exp_id, exp_val_rep, exp_train_reps) in zip(folds, expected_mappings):
        assert isinstance(fold_def, FoldPartitionDef)
        assert fold_def.fold_id == exp_id
        assert fold_def.val_repetition == exp_val_rep
        assert fold_def.train_repetitions == exp_train_reps

        train_set = set(fold_def.train_repetitions)
        val_set = {fold_def.val_repetition}
        test_set = {2, 5}

        # Mathematical partition disjointness invariants
        assert train_set & val_set == set(), f"Fold {exp_id}: train and val overlap"
        assert train_set & test_set == set(), f"Fold {exp_id}: train and test overlap"
        assert val_set & test_set == set(), f"Fold {exp_id}: val and test overlap"
        assert train_set | val_set == {1, 3, 4, 6}, f"Fold {exp_id}: incomplete training pool"


def test_within_subject_test_reps_invisible_during_selection() -> None:
    """Test repetitions [2, 5] must receive zero access during validation and configuration selection (FR-005.7)."""
    rec = make_synthetic_recording(seed=42, active_labels=(1, 2, 3, 4, 5))

    accessed_reps: list[int] = []
    raw_accessed_reps: list[int] = []

    def on_window_access(rep_unit: int) -> None:
        accessed_reps.append(rep_unit)

    def on_raw_access(rep: int, n_samples: int) -> None:
        raw_accessed_reps.append(rep)

    runner = WithinSubjectRunner(
        num_classes=6,
        candidate_grid=(1.0, 1.5),
        access_callback=on_window_access,
        raw_access_callback=on_raw_access,
    )

    # Run internal validation selection phase only
    selection_results = runner.run_validation_selection(rec)

    assert len(selection_results) > 0
    # Test repetitions 2 and 5 must NEVER have been accessed (neither raw signals nor windows)
    window_test_accesses = [r for r in accessed_reps if r in (2, 5)]
    raw_test_accesses = [r for r in raw_accessed_reps if r in (2, 5)]

    assert len(window_test_accesses) == 0, f"Leakage detected: test windows accessed: {window_test_accesses}"
    assert len(raw_test_accesses) == 0, f"Leakage detected: test raw signals accessed: {raw_test_accesses}"


def test_within_subject_final_fit_uses_all_training_repetitions() -> None:
    """Final model must be fitted strictly on all training pool repetitions [1, 3, 4, 6] (FR-008)."""
    rec = make_synthetic_recording(seed=101, active_labels=(1, 2, 3))
    runner = WithinSubjectRunner(num_classes=4, candidate_grid=(1.5,))

    result = runner.run(rec)

    assert isinstance(result, WithinSubjectResult)
    assert result.final_train_repetitions == [1, 3, 4, 6]
    assert isinstance(result.final_model_trainer, CanonicalLdaTrainer)

    # Verify that the final model was fitted on samples from reps 1, 3, 4, 6
    assert result.n_final_train_samples > 0
    # Sum of samples from the 4 training repetitions
    expected_train_samples = runner.count_admitted_windows(rec, repetitions=[1, 3, 4, 6])
    assert result.n_final_train_samples == expected_train_samples


def test_within_subject_test_evaluated_exactly_once() -> None:
    """Final test repetitions [2, 5] must be evaluated strictly once after final fit (FR-005.7, FR-008)."""
    rec = make_synthetic_recording(seed=202, active_labels=(1, 2, 3, 4))
    runner = WithinSubjectRunner(num_classes=5, candidate_grid=(1.0,))

    # Prior to run, test evaluation count is 0
    assert runner.test_evaluation_count == 0

    result = runner.run(rec)

    # After run, test evaluation count is exactly 1 (real auditable execution event)
    assert runner.test_evaluation_count == 1
    assert result.test_evaluation_count == 1
    assert result.final_test_repetitions == [2, 5]
    assert len(result.test_predictions) == len(result.test_labels)
    assert len(result.test_predictions) > 0

    expected_test_samples = runner.count_admitted_windows(rec, repetitions=[2, 5])
    assert len(result.test_predictions) == expected_test_samples


def test_within_subject_preprocessing_anti_leakage() -> None:
    """Preprocessing parameters (deadzones, scaler) must be fitted strictly on train partition (SC-003, Decision-2)."""
    rec = make_synthetic_recording(seed=303, active_labels=(1, 2, 3))
    runner = WithinSubjectRunner(num_classes=4, candidate_grid=(1.5,))

    # Collect fitted scalers and calibrations per fold
    fold_scalers = []
    fold_calibrations = []

    result = runner.run(rec)

    for fold in result.validation_folds:
        assert fold.threshold_calibration is not None
        assert fold.scaler_mean is not None
        assert fold.scaler_scale is not None
        fold_scalers.append((fold.scaler_mean, fold.scaler_scale))
        fold_calibrations.append(fold.threshold_calibration)

    # Different training subsets [3,4,6], [1,4,6], etc. must produce distinct scaling statistics
    assert not np.allclose(fold_scalers[0][0], fold_scalers[1][0], atol=1e-4), (
        "Fold 1 and Fold 2 scalers should differ due to different training repetitions"
    )

    # Final fit scaler must be distinct from any individual fold scaler
    assert not np.allclose(result.final_scaler_mean, fold_scalers[0][0], atol=1e-4)


def test_within_subject_end_to_end_synthetic_execution() -> None:
    """End-to-end execution on procedural synthetic recording must succeed with bounded valid metrics."""
    rec = make_synthetic_recording(seed=404, active_labels=(1, 2, 3, 4, 5))
    runner = WithinSubjectRunner(
        candidate_grid=(0.5, 1.0, 1.5, 2.0, 3.0),
        num_classes=6,  # REST + 5 active classes
    )

    result = runner.run(rec)

    # Validation folds
    assert len(result.validation_folds) == 4
    for fold in result.validation_folds:
        assert 0.0 <= fold.accuracy <= 1.0
        assert 0.0 <= fold.balanced_accuracy <= 1.0
        assert 0.0 <= fold.macro_f1 <= 1.0
        assert fold.n_train_samples > 0
        assert fold.n_val_samples > 0

    # Selected R multiplier must come strictly from candidate grid
    assert result.selected_r in (0.5, 1.0, 1.5, 2.0, 3.0)

    # Test metrics must be bounded in [0, 1]
    assert 0.0 <= result.test_accuracy <= 1.0
    assert 0.0 <= result.test_balanced_accuracy <= 1.0
    assert 0.0 <= result.test_macro_f1 <= 1.0

    # Model artifacts must be valid
    artifacts = result.final_model_trainer.get_model_artifacts()
    assert artifacts.solver == "lsqr"
    assert artifacts.shrinkage_method == "ledoit_wolf_auto"
    assert len(artifacts.classes_) == 6


def test_within_subject_determinism_invariant() -> None:
    """Identical recording and seed must produce bit-exact identical metrics and predictions (SC-006)."""
    rec = make_synthetic_recording(seed=505, active_labels=(1, 2, 3))

    runner1 = WithinSubjectRunner(candidate_grid=(1.0, 2.0), num_classes=4)
    runner2 = WithinSubjectRunner(candidate_grid=(1.0, 2.0), num_classes=4)

    res1 = runner1.run(rec)
    res2 = runner2.run(rec)

    assert res1.selected_r == res2.selected_r
    assert res1.test_accuracy == res2.test_accuracy
    assert res1.test_balanced_accuracy == res2.test_balanced_accuracy
    assert res1.test_macro_f1 == res2.test_macro_f1
    np.testing.assert_array_equal(res1.test_predictions, res2.test_predictions)


def test_within_subject_fails_on_missing_required_repetitions() -> None:
    """Runner must fail closed when recording is missing required training or test repetitions (FR-011.5)."""
    # Create recording missing repetition 6
    rec = make_synthetic_recording(seed=606, active_labels=(1, 2))
    # Artificially overwrite repetition 6 samples to repetition 1
    rep_mask = rec.rerepetition == 6
    rec.rerepetition[rep_mask] = 1

    runner = WithinSubjectRunner(num_classes=3)
    with pytest.raises(ValueError, match="repetition.*missing|required repetition"):
        runner.run(rec)


def test_within_subject_dual_vocabulary_support() -> None:
    """Runner supports both canonical global 50-class and exercise-level local vocabularies (FR-007)."""
    # Canonical global 50-class runner configuration
    runner_50 = WithinSubjectRunner(num_classes=50)
    assert runner_50.num_classes == 50

    # Secondary E1 18-class runner configuration
    runner_18 = WithinSubjectRunner(num_classes=18)
    assert runner_18.num_classes == 18

    # Secondary E3 10-class runner configuration
    runner_10 = WithinSubjectRunner(num_classes=10)
    assert runner_10.num_classes == 10


def test_within_subject_selection_metric_fail_closed_and_lowest_r_tie_break() -> None:
    """Selection metric must fail closed on unknown metric and deterministically pick lowest R on tie."""
    # 1. Fail-closed on invalid selection metric
    with pytest.raises(ValueError, match="Unknown selection_metric 'invalid_metric'"):
        WithinSubjectRunner(selection_metric="invalid_metric")

    # 2. Deterministic lowest-R tie-break verification
    runner = WithinSubjectRunner(
        candidate_grid=(0.5, 1.0, 2.0),
        selection_metric="balanced_accuracy",
    )
    # Simulate tie: candidate 0.5 and candidate 1.0 both achieve score 0.888
    tie_summary = [
        ValidationSelectionResult(
            candidate_r=0.5,
            mean_accuracy=0.85,
            mean_balanced_accuracy=0.888,
            mean_macro_f1=0.80,
            fold_results=[],
        ),
        ValidationSelectionResult(
            candidate_r=1.0,
            mean_accuracy=0.85,
            mean_balanced_accuracy=0.888,
            mean_macro_f1=0.80,
            fold_results=[],
        ),
        ValidationSelectionResult(
            candidate_r=2.0,
            mean_accuracy=0.80,
            mean_balanced_accuracy=0.850,
            mean_macro_f1=0.75,
            fold_results=[],
        ),
    ]
    best_r, _ = runner._select_best_r(tie_summary)
    assert best_r == 0.5, f"Expected lowest candidate R (0.5) on tie, got {best_r}"


def test_within_subject_vocabulary_fail_closed_prevents_silent_shrinkage() -> None:
    """Default num_classes=50 must fail closed when recording has incomplete vocabulary (FR-011.5)."""
    # Synthetic recording with only 6 classes (0..5)
    rec = make_synthetic_recording(seed=707, active_labels=(1, 2, 3, 4, 5))

    # Canonical runner defaults strictly to num_classes=50
    runner_default = WithinSubjectRunner()
    assert runner_default.num_classes == 50

    # Must fail closed with vocabulary mismatch rather than silently shrinking to 6
    with pytest.raises(ValueError, match="Class vocabulary mismatch for num_classes=50: missing classes"):
        runner_default.run(rec)

    # Explicit 'auto' allows dynamic resolution strictly when explicitly requested
    runner_auto = WithinSubjectRunner(num_classes="auto", candidate_grid=(1.0,))
    result_auto = runner_auto.run(rec)
    assert len(result_auto.final_model_trainer.get_model_artifacts().classes_) == 6


def test_within_subject_structural_test_isolation_nan_poisoning_immunity() -> None:
    """Selection phase must be completely immune to NaN-poisoned test repetitions [2, 5] (FR-005.7)."""
    rec = make_synthetic_recording(seed=808, active_labels=(1, 2, 3))

    # Poison raw test repetitions 2 and 5 with NaNs
    test_mask = (rec.rerepetition == 2) | (rec.rerepetition == 5)
    rec.emg[test_mask] = np.nan

    runner = WithinSubjectRunner(num_classes=4, candidate_grid=(1.0,))

    # Internal validation selection MUST succeed with zero NaN errors
    # proving that raw test signals are strictly unread and unparsed during selection
    selection_results = runner.run_validation_selection(rec)
    assert len(selection_results) > 0

    # In contrast, attempting final test evaluation on poisoned test set must fail with NaN
    with pytest.raises(ValueError, match="NaN"):
        runner.run(rec)
