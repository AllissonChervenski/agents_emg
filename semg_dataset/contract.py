"""Domain Contracts and Entities for NinaPro DB2 Dataset.

Defines canonical representations for recordings, exercises, subjects,
alignment status, and derived label mappings with strict validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict

import numpy as np


class AlignmentStatus(str, Enum):
    """Status indicating how temporal alignment was resolved for a recording."""

    IDENTICAL = "IDENTICAL"
    ANCHOR_START_TRUNCATE_TAIL = "ANCHOR_START_TRUNCATE_TAIL"
    QUARANTINE = "QUARANTINE"


@dataclass(frozen=True)
class AlignmentRecord:
    """Audit evidence record documenting temporal alignment for a recording."""

    status: AlignmentStatus
    original_emg_length: int
    original_refined_length: int
    delta_samples: int
    start_anchored_proven: bool
    notes: str = ""


@dataclass(frozen=True)
class Subject:
    """Participant metadata in NinaPro DB2."""

    subject_id: str
    archive_filename: str
    archive_sha256: str
    archive_valid: bool


@dataclass
class RecordingData:
    """Canonical synchronized representation of a NinaPro DB2 recording session.
    
    Guarantees:
      - 12-channel sEMG in float32 without NaN or Inf
      - Nominal sampling rate (2000.0 Hz)
      - Perfectly synchronized stimulus, repetition, restimulus, and rerepetition arrays
    """

    subject_id: str
    exercise_id: str
    emg: np.ndarray
    stimulus: np.ndarray
    repetition: np.ndarray
    restimulus: np.ndarray
    rerepetition: np.ndarray
    alignment: AlignmentRecord
    sampling_rate_hz: float = 2000.0
    provenance: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate all structural and numerical contract invariants."""
        # 1. EMG Array Contract
        if not isinstance(self.emg, np.ndarray):
            raise TypeError("emg must be a numpy ndarray")
        if self.emg.ndim != 2 or self.emg.shape[1] != 12:
            raise ValueError(f"emg array must have shape (N, 12) with exactly 12 channels; got {self.emg.shape}")
        if self.emg.dtype != np.float32:
            raise TypeError(f"emg array must be float32; got {self.emg.dtype}")
        if np.isnan(self.emg).any():
            raise ValueError("emg array contains NaN values; contract requires finite float32")
        if np.isinf(self.emg).any():
            raise ValueError("emg array contains Inf values; contract requires finite float32")

        n_samples = self.emg.shape[0]

        # 2. Synchronized Label and Repetition Arrays
        label_arrays = {
            "stimulus": self.stimulus,
            "repetition": self.repetition,
            "restimulus": self.restimulus,
            "rerepetition": self.rerepetition,
        }
        for name, arr in label_arrays.items():
            if not isinstance(arr, np.ndarray):
                raise TypeError(f"{name} must be a numpy ndarray")
            if arr.ndim != 1 or arr.shape[0] != n_samples:
                raise ValueError(
                    f"{name} length ({arr.shape[0]}) does not match emg sample count ({n_samples})"
                )

        # 3. Sampling Rate
        if self.sampling_rate_hz <= 0:
            raise ValueError(f"sampling_rate_hz must be positive; got {self.sampling_rate_hz}")

    @property
    def num_samples(self) -> int:
        """Return the total number of aligned temporal samples."""
        return int(self.emg.shape[0])

    @property
    def num_channels(self) -> int:
        """Return the number of sEMG channels (strictly 12)."""
        return int(self.emg.shape[1])

    def get_local_labels(self) -> np.ndarray:
        """Return a derived array of exercise-local gesture labels.
        
        Preserves REST=0 across all exercises:
          - E1: 1..17 -> 1..17
          - E2: 18..40 -> 1..23
          - E3: 41..49 -> 1..9
        """
        global_labels = self.stimulus
        local_labels = np.zeros_like(global_labels, dtype=np.int8)

        if self.exercise_id == "E1":
            # E1 native labels are 0..17 (already local)
            mask = global_labels > 0
            local_labels[mask] = global_labels[mask]
        elif self.exercise_id == "E2":
            # E2 native labels are 18..40 -> map to 1..23
            mask = global_labels > 0
            local_labels[mask] = (global_labels[mask] - 17).astype(np.int8)
        elif self.exercise_id == "E3":
            # E3 native labels are 41..49 -> map to 1..9
            mask = global_labels > 0
            local_labels[mask] = (global_labels[mask] - 40).astype(np.int8)
        else:
            raise ValueError(f"Unknown exercise_id for local label derivation: {self.exercise_id}")

        return local_labels
