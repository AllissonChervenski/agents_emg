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
    RecordingData,
    Subject,
)
from semg_dataset.loader import NinaProDB2Loader
from semg_dataset.provenance import NinaProProvenance

__all__: list[str] = [
    "__version__",
    "AlignmentRecord",
    "AlignmentStatus",
    "NinaProDB2Loader",
    "NinaProProvenance",
    "ProveOrQuarantineEngine",
    "RecordingData",
    "Subject",
    "anchor_start_truncate_tail",
]


