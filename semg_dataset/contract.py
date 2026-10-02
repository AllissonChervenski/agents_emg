"""Domain Contracts and Entities for NinaPro DB2 Dataset.

Defines canonical representations for recordings, exercises, subjects,
alignment status, derived label mappings, and frozen dataset views with strict validation.
"""

from __future__ import annotations

import hashlib
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

    @property
    def is_accepted(self) -> bool:
        """Return True if recording alignment is accepted into processing pipeline."""
        return self.status != AlignmentStatus.QUARANTINE


@dataclass(frozen=True)
class Subject:
    """Participant metadata in NinaPro DB2."""

    subject_id: str
    archive_filename: str
    archive_sha256: str
    archive_valid: bool


@dataclass(frozen=True)
class DatasetView:
    """Frozen, tamper-evident dataset view binding signals and paired labels."""

    view_id: str
    signals: np.ndarray
    labels: np.ndarray
    repetitions: np.ndarray
    view_hash: str

    def __post_init__(self) -> None:
        if self.view_id not in ("refined", "stimulus") and not self.view_id.startswith("test_"):
            raise ValueError(f"Unknown view_id '{self.view_id}'; must be 'refined' or 'stimulus'")
        if self.signals.ndim != 2 or self.signals.shape[1] != 12:
            raise ValueError(f"signals must be of shape (N, 12); got {self.signals.shape}")
        if self.labels.ndim != 1 or self.repetitions.ndim != 1:
            raise ValueError("labels and repetitions must be 1D arrays")
        if (
            self.signals.shape[0] != self.labels.shape[0]
            or self.signals.shape[0] != self.repetitions.shape[0]
        ):
            raise ValueError("Sample counts between signals, labels, and repetitions do not match")


class FrozenViewManager:
    """Manager and validator ensuring strict label-pair isolation."""

    VALID_PAIRS = {
        "refined": ("restimulus", "rerepetition"),
        "stimulus": ("stimulus", "repetition"),
    }

    @staticmethod
    def compute_view_hash(
        view_id: str, signals: np.ndarray, labels: np.ndarray, repetitions: np.ndarray
    ) -> str:
        """Compute a deterministic SHA-256 digest of view content."""
        hasher = hashlib.sha256()
        hasher.update(view_id.encode("utf-8"))
        hasher.update(signals.tobytes())
        hasher.update(labels.tobytes())
        hasher.update(repetitions.tobytes())
        return hasher.hexdigest()

    @classmethod
    def create_view_explicit(
        cls,
        view_id: str,
        signals: np.ndarray,
        labels: np.ndarray,
        repetitions: np.ndarray,
        label_source: str,
        repetition_source: str,
    ) -> DatasetView:
        """Create a DatasetView after validating strict label-pair isolation.

        Raises:
            ValueError: If an illegal cross-pair combination is attempted.
        """
        is_refined_pair = label_source == "restimulus" and repetition_source == "rerepetition"
        is_stimulus_pair = label_source == "stimulus" and repetition_source == "repetition"
        if not (is_refined_pair or is_stimulus_pair):
            raise ValueError(
                f"Cross-pair mixing between '{label_source}' and '{repetition_source}' is strictly prohibited"
            )

        v_hash = cls.compute_view_hash(view_id, signals, labels, repetitions)
        return DatasetView(
            view_id=view_id,
            signals=signals,
            labels=labels,
            repetitions=repetitions,
            view_hash=v_hash,
        )

    @classmethod
    def get_view_for_recording(cls, recording: RecordingData, view_id: str) -> DatasetView:
        """Construct the requested frozen view for a RecordingData instance."""
        if view_id == "refined":
            return cls.create_view_explicit(
                view_id="refined",
                signals=recording.emg,
                labels=recording.restimulus,
                repetitions=recording.rerepetition,
                label_source="restimulus",
                repetition_source="rerepetition",
            )
        elif view_id == "stimulus":
            return cls.create_view_explicit(
                view_id="stimulus",
                signals=recording.emg,
                labels=recording.stimulus,
                repetitions=recording.repetition,
                label_source="stimulus",
                repetition_source="repetition",
            )
        else:
            raise ValueError(f"Unknown view_id '{view_id}'; must be 'refined' or 'stimulus'")


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
            raise ValueError(
                f"emg array must have shape (N, 12) with exactly 12 channels; got {self.emg.shape}"
            )
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

    def get_view(self, view_id: str = "refined") -> DatasetView:
        """Return a frozen dataset view ('refined' default or 'stimulus' secondary)."""
        return FrozenViewManager.get_view_for_recording(self, view_id)

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
            mask = global_labels > 0
            local_labels[mask] = global_labels[mask]
        elif self.exercise_id == "E2":
            mask = global_labels > 0
            local_labels[mask] = (global_labels[mask] - 17).astype(np.int8)
        elif self.exercise_id == "E3":
            mask = global_labels > 0
            local_labels[mask] = (global_labels[mask] - 40).astype(np.int8)
        else:
            raise ValueError(f"Unknown exercise_id for local label derivation: {self.exercise_id}")

        return local_labels
