"""Dual Partitioning Protocols and Zero-Leakage Split Engines for NinaPro DB2.

Implements Within-Subject (repetition-based) and Cross-Subject (5-fold rotating)
partitioning protocols with strict mathematical zero-leakage enforcement.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


class DataLeakageError(ValueError):
    """Raised when data leakage between partitions is detected."""


def validate_within_subject_leakage(
    train_repetitions: Sequence[int], test_repetitions: Sequence[int]
) -> None:
    """Ensure train and test repetition lists are strictly disjoint.

    Raises:
        DataLeakageError: If any repetition appears in both train and test.
    """
    overlap = set(train_repetitions) & set(test_repetitions)
    if overlap:
        raise DataLeakageError(
            f"Data leakage detected: repetition(s) {sorted(list(overlap))} overlap between train and test"
        )


def validate_cross_subject_leakage(
    train_subjects: Sequence[str],
    val_subjects: Sequence[str],
    test_subjects: Sequence[str],
) -> None:
    """Ensure subject lists across train, validation, and test are strictly disjoint and unique.

    Raises:
        DataLeakageError: If duplicate subjects exist or any subject is shared across partitions.
    """
    # 1. Duplicates within single partition
    if len(train_subjects) != len(set(train_subjects)):
        raise DataLeakageError("Duplicate subject detected within train partition")
    if len(val_subjects) != len(set(val_subjects)):
        raise DataLeakageError("Duplicate subject detected within val partition")
    if len(test_subjects) != len(set(test_subjects)):
        raise DataLeakageError("Duplicate subject detected within test partition")

    train_set = set(train_subjects)
    val_set = set(val_subjects)
    test_set = set(test_subjects)

    # 2. Cross-partition intersections
    train_test_overlap = train_set & test_set
    if train_test_overlap:
        raise DataLeakageError(
            f"Data leakage detected: subject(s) {sorted(list(train_test_overlap))} overlap between train and test"
        )

    train_val_overlap = train_set & val_set
    if train_val_overlap:
        raise DataLeakageError(
            f"Data leakage detected: subject(s) {sorted(list(train_val_overlap))} overlap between train and val"
        )

    val_test_overlap = val_set & test_set
    if val_test_overlap:
        raise DataLeakageError(
            f"Data leakage detected: subject(s) {sorted(list(val_test_overlap))} overlap between val and test"
        )


@dataclass(frozen=True)
class FoldPartition:
    """Single fold partition in cross-subject evaluation."""

    fold_id: int
    train_subjects: List[str]
    val_subjects: List[str]
    test_subjects: List[str]
    sample_counts: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize fold to JSON schema compliant dictionary."""
        return {
            "fold_id": self.fold_id,
            "train_subjects": list(self.train_subjects),
            "val_subjects": list(self.val_subjects),
            "test_subjects": list(self.test_subjects),
        }


@dataclass
class SplitManifest:
    """Deterministic manifest documenting partition configuration."""

    protocol: str
    folds: List[FoldPartition] = field(default_factory=list)
    train_repetitions: List[int] = field(default_factory=list)
    test_repetitions: List[int] = field(default_factory=list)
    val_repetitions: List[int] = field(default_factory=list)
    schema_version: str = "1.0"
    dataset_name: str = "NinaPro DB2"
    git_commit: str = ""
    created_at: str = ""
    manifest_hash: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize manifest to JSON schema compliant format."""
        d: Dict[str, Any] = {
            "schema_version": self.schema_version,
            "dataset_name": self.dataset_name,
            "protocol": self.protocol,
            "git_commit": self.git_commit,
            "created_at": self.created_at,
            "manifest_hash": self.manifest_hash,
            "folds": [f.to_dict() for f in self.folds],
        }
        if self.protocol == "within_subject":
            d["within_subject_parameters"] = {
                "train_repetitions": list(self.train_repetitions),
                "test_repetitions": list(self.test_repetitions),
            }
        return d


class WithinSubjectSplitter:
    """Repetition-based split generator for user-specific prosthetic evaluation."""

    def __init__(
        self,
        train_repetitions: Optional[List[int]] = None,
        test_repetitions: Optional[List[int]] = None,
    ) -> None:
        self.train_repetitions = (
            train_repetitions if train_repetitions is not None else [1, 3, 4, 6]
        )
        self.test_repetitions = test_repetitions if test_repetitions is not None else [2, 5]
        self.val_repetitions: List[int] = []

        validate_within_subject_leakage(self.train_repetitions, self.test_repetitions)

    def generate_splits(self, recordings: Optional[Sequence[Any]] = None) -> SplitManifest:
        """Generate Within-Subject split manifest."""
        return SplitManifest(
            protocol="within_subject",
            train_repetitions=list(self.train_repetitions),
            test_repetitions=list(self.test_repetitions),
            val_repetitions=[],
            folds=[],
        )


class CrossSubjectSplitter:
    """5-fold rotating subject split generator for inter-subject generalizability."""

    def __init__(self, num_folds: int = 5) -> None:
        if num_folds != 5:
            raise ValueError(f"CrossSubjectSplitter requires exactly 5 folds; got {num_folds}")
        self.num_folds = num_folds

    def generate_splits(self, subjects: Optional[Sequence[str]] = None) -> SplitManifest:
        """Generate 5 rotating folds across 40 subjects with 24 train / 8 val / 8 test."""
        if subjects is None:
            subject_list = [f"S{i}" for i in range(1, 41)]
        else:
            subject_list = list(subjects)

        if len(subject_list) != 40:
            raise ValueError(f"Expected exactly 40 subjects for NinaPro DB2; got {len(subject_list)}")

        def _sort_key(s: str) -> int:
            m = re.search(r"\d+", s)
            return int(m.group()) if m else 0

        sorted_subjects = sorted(subject_list, key=_sort_key)

        # 5 balanced groups of 8
        g1 = sorted_subjects[0:8]
        g2 = sorted_subjects[8:16]
        g3 = sorted_subjects[16:24]
        g4 = sorted_subjects[24:32]
        g5 = sorted_subjects[32:40]

        fold_configs = [
            (1, g1 + g2 + g3, g4, g5),  # Fold 1: Test=G5, Val=G4, Train=G1+G2+G3
            (2, g5 + g1 + g2, g3, g4),  # Fold 2: Test=G4, Val=G3, Train=G5+G1+G2
            (3, g4 + g5 + g1, g2, g3),  # Fold 3: Test=G3, Val=G2, Train=G4+G5+G1
            (4, g3 + g4 + g5, g1, g2),  # Fold 4: Test=G2, Val=G1, Train=G3+G4+G5
            (5, g2 + g3 + g4, g5, g1),  # Fold 5: Test=G1, Val=G5, Train=G2+G3+G4
        ]

        folds: List[FoldPartition] = []
        for fold_id, train_subs, val_subs, test_subs in fold_configs:
            validate_cross_subject_leakage(train_subs, val_subs, test_subs)
            folds.append(
                FoldPartition(
                    fold_id=fold_id,
                    train_subjects=train_subs,
                    val_subjects=val_subs,
                    test_subjects=test_subs,
                )
            )

        return SplitManifest(
            protocol="cross_subject",
            folds=folds,
        )
