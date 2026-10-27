"""Tests for Dual Partitioning Protocols and Positive Controls of Data Leakage.

Scientific-integrity-sensitive:
Ensures Within-Subject (repetition split) and Cross-Subject (5-fold split)
enforce strict zero-leakage guarantees, and verifies that intentional leakage
injections are unequivocally detected and rejected by leakage validators.
"""

import pytest

from semg_dataset.splits import (
    CrossSubjectSplitter,
    DataLeakageError,
    WithinSubjectSplitter,
    validate_cross_subject_leakage,
    validate_within_subject_leakage,
)


def test_within_subject_split_repetitions() -> None:
    """Within-Subject protocol must assign train=[1,3,4,6], test=[2,5], val=[]."""
    splitter = WithinSubjectSplitter()
    manifest = splitter.generate_splits()

    assert manifest.protocol == "within_subject"
    assert manifest.train_repetitions == [1, 3, 4, 6]
    assert manifest.test_repetitions == [2, 5]
    assert manifest.val_repetitions == []

    train_set = set(manifest.train_repetitions)
    test_set = set(manifest.test_repetitions)
    assert train_set.isdisjoint(test_set), "Train and test repetitions must be strictly disjoint"


def test_cross_subject_split_5_folds() -> None:
    """Cross-Subject protocol must generate exactly 5 balanced folds across 40 subjects."""
    all_subjects = [f"S{i}" for i in range(1, 41)]
    splitter = CrossSubjectSplitter()
    manifest = splitter.generate_splits(all_subjects)

    assert manifest.protocol == "cross_subject"
    assert len(manifest.folds) == 5

    for fold in manifest.folds:
        assert len(fold.train_subjects) == 24
        assert len(fold.val_subjects) == 8
        assert len(fold.test_subjects) == 8

        train_set = set(fold.train_subjects)
        val_set = set(fold.val_subjects)
        test_set = set(fold.test_subjects)

        assert train_set.isdisjoint(val_set)
        assert train_set.isdisjoint(test_set)
        assert val_set.isdisjoint(test_set)
        assert (train_set | val_set | test_set) == set(all_subjects)


def test_cross_subject_frozen_assignment_matches_spec() -> None:
    """Verify deterministic rotating fold assignments matching research.md D-003."""
    all_subjects = [f"S{i}" for i in range(1, 41)]
    splitter = CrossSubjectSplitter()
    manifest = splitter.generate_splits(all_subjects)

    g1 = [f"S{i}" for i in range(1, 9)]
    g2 = [f"S{i}" for i in range(9, 17)]
    g3 = [f"S{i}" for i in range(17, 25)]
    g4 = [f"S{i}" for i in range(25, 33)]
    g5 = [f"S{i}" for i in range(33, 41)]

    # Fold 1: Test = G5, Val = G4, Train = G1 + G2 + G3
    f1 = manifest.folds[0]
    assert f1.test_subjects == g5
    assert f1.val_subjects == g4
    assert f1.train_subjects == g1 + g2 + g3

    # Fold 5: Test = G1, Val = G5, Train = G2 + G3 + G4
    f5 = manifest.folds[4]
    assert f5.test_subjects == g1
    assert f5.val_subjects == g5
    assert f5.train_subjects == g2 + g3 + g4


# --- POSITIVE CONTROLS OF LEAKAGE ---


def test_positive_control_repetition_leakage() -> None:
    """POSITIVE CONTROL: Deliberate repetition overlap must raise DataLeakageError."""
    with pytest.raises(DataLeakageError, match="[Ll]eakage.*repetition"):
        validate_within_subject_leakage(
            train_repetitions=[1, 2, 3, 4, 6],
            test_repetitions=[2, 5],  # 2 is duplicated in train!
        )


def test_positive_control_subject_train_test_leakage() -> None:
    """POSITIVE CONTROL: Deliberate subject overlap between train and test must raise DataLeakageError."""
    with pytest.raises(DataLeakageError, match="[Ll]eakage.*train.*test"):
        validate_cross_subject_leakage(
            train_subjects=["S1", "S2", "S3", "S5"],
            val_subjects=["S4"],
            test_subjects=["S5", "S6"],  # S5 is in both train and test!
        )


def test_positive_control_subject_train_val_leakage() -> None:
    """POSITIVE CONTROL: Deliberate subject overlap between train and val must raise DataLeakageError."""
    with pytest.raises(DataLeakageError, match="[Ll]eakage.*train.*val"):
        validate_cross_subject_leakage(
            train_subjects=["S1", "S2", "S3"],
            val_subjects=["S3", "S4"],  # S3 is in both train and val!
            test_subjects=["S5", "S6"],
        )


def test_positive_control_subject_val_test_leakage() -> None:
    """POSITIVE CONTROL: Deliberate subject overlap between val and test must raise DataLeakageError."""
    with pytest.raises(DataLeakageError, match="[Ll]eakage.*val.*test"):
        validate_cross_subject_leakage(
            train_subjects=["S1", "S2"],
            val_subjects=["S3", "S4"],
            test_subjects=["S4", "S5"],  # S4 is in both val and test!
        )


def test_positive_control_duplicate_subject_within_partition() -> None:
    """POSITIVE CONTROL: Duplicate subject within the same partition must raise DataLeakageError."""
    with pytest.raises(DataLeakageError, match="[Dd]uplicate subject"):
        validate_cross_subject_leakage(
            train_subjects=["S1", "S1", "S2"],  # S1 duplicated
            val_subjects=["S3"],
            test_subjects=["S4"],
        )
