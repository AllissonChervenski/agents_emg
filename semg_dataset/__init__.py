"""sEMG Dataset Contract, Provenance, Alignment, and Partitioning Package.

Provides audit-grade data contracts, read-only memory streaming loaders,
PROVE-OR-QUARANTINE alignment verification, frozen dataset views, and
leakage-free split partition generators for the NinaPro DB2 dataset.
"""

from __future__ import annotations

__version__ = "0.1.0"

from semg_dataset.alignment import ProveOrQuarantineEngine, anchor_start_truncate_tail
from semg_dataset.contract import (
    AlignmentRecord,
    AlignmentStatus,
    DatasetView,
    FrozenViewManager,
    RecordingData,
    Subject,
)

from semg_dataset.loader import NinaProDB2Loader
from semg_dataset.provenance import NinaProProvenance
from semg_dataset.splits import (
    CrossSubjectSplitter,
    DataLeakageError,
    FoldPartition,
    SplitManifest,
    WithinSubjectSplitter,
    validate_cross_subject_leakage,
    validate_within_subject_leakage,
)

__all__: list[str] = [
    "__version__",
    "AlignmentRecord",
    "AlignmentStatus",
    "CrossSubjectSplitter",
    "DataLeakageError",
    "DatasetView",
    "FoldPartition",
    "FrozenViewManager",
    "NinaProDB2Loader",
    "NinaProProvenance",
    "ProveOrQuarantineEngine",
    "RecordingData",
    "SplitManifest",
    "Subject",
    "WithinSubjectSplitter",
    "anchor_start_truncate_tail",
    "validate_cross_subject_leakage",
    "validate_within_subject_leakage",
]




