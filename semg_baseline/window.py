"""Steady-state pure window segmenter and transition boundary auditor.

Implements FR-012.2, FR-012.3, SC-002, and Decision-2:
- Sliding analysis windows of W = 200 ms (400 samples @ 2000 Hz).
- Decision stride of S = 50 ms (100 samples @ 2000 Hz, 75% overlap, 20 Hz decision rate).
- Strict exclusion of all label transition windows (REST <-> ACTIVE, ACTIVE <-> ACTIVE).
- Strict exclusion of repetition unit boundary crossing windows.
- Invariant: repetition_crossing_windows_admitted == 0.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from semg_dataset.manifest import associate_rest_to_repetition_units


@dataclass(frozen=True)
class TransitionAudit:
    """Audit evidence record documenting pure window admission and boundary exclusion."""

    total_candidate_windows: int
    admitted_window_count: int
    discarded_window_count: int
    discarded_label_transitions: int
    discarded_repetition_crossings: int
    repetition_crossing_windows_admitted: int = 0
    dropped_tail_samples: int = 0
    admitted_start_indices: list[int] = field(default_factory=list)
    admitted_repetition_units: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize audit record to dictionary."""
        return {
            "total_candidate_windows": self.total_candidate_windows,
            "admitted_window_count": self.admitted_window_count,
            "discarded_window_count": self.discarded_window_count,
            "discarded_label_transitions": self.discarded_label_transitions,
            "discarded_repetition_crossings": self.discarded_repetition_crossings,
            "repetition_crossing_windows_admitted": self.repetition_crossing_windows_admitted,
            "dropped_tail_samples": self.dropped_tail_samples,
        }


class PureWindowSegmenter:
    """Sliding window segmenter admitting strictly pure steady-state windows."""

    def __init__(
        self,
        window_len: int = 400,
        stride: int = 100,
        fs: float = 2000.0,
    ) -> None:
        if window_len <= 0:
            raise ValueError(f"window_len must be positive, got {window_len}")
        if stride <= 0:
            raise ValueError(f"stride must be positive, got {stride}")
        if fs <= 0:
            raise ValueError(f"fs must be positive, got {fs}")

        self.window_len = int(window_len)
        self.stride = int(stride)
        self.fs = float(fs)
        self.decision_rate_hz = self.fs / self.stride

    def segment(
        self,
        emg: np.ndarray,
        labels: np.ndarray,
        repetitions: np.ndarray,
        repetition_units: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, TransitionAudit]:
        """Segment continuous sEMG and labels into pure steady-state analysis windows.

        A window [t, t + window_len) is admitted if and only if:
        1. All samples in the window share the identical gesture label;
        2. All samples in the window belong to the identical repetition unit.

        Returns:
            windows: Array of shape (N_admitted, window_len, num_channels) in float32.
            labels: Array of shape (N_admitted,) containing the uniform gesture class label.
            audit: TransitionAudit detailing admitted and discarded candidate windows.
        """
        if not isinstance(emg, np.ndarray):
            emg = np.asarray(emg)
        if not isinstance(labels, np.ndarray):
            labels = np.asarray(labels)
        if not isinstance(repetitions, np.ndarray):
            repetitions = np.asarray(repetitions)

        if emg.ndim != 2:
            raise ValueError(
                f"emg must be a 2D array of shape (num_samples, num_channels), got {emg.shape}"
            )
        num_samples, num_channels = emg.shape

        if len(labels) != num_samples:
            raise ValueError(
                f"Length mismatch: emg has {num_samples} samples, but labels has {len(labels)}"
            )
        if len(repetitions) != num_samples:
            raise ValueError(
                f"Length mismatch: emg has {num_samples} samples, but repetitions has {len(repetitions)}"
            )

        # Derive repetition units if not explicitly supplied
        if repetition_units is not None:
            units = np.asarray(repetition_units, dtype=np.int32)
            if len(units) != num_samples:
                raise ValueError(
                    f"Length mismatch: emg has {num_samples} samples, but repetition_units has {len(units)}"
                )
        else:
            units = associate_rest_to_repetition_units(repetitions)

        # If signal is shorter than window length, no candidate windows exist
        if num_samples < self.window_len:
            audit = TransitionAudit(
                total_candidate_windows=0,
                admitted_window_count=0,
                discarded_window_count=0,
                discarded_label_transitions=0,
                discarded_repetition_crossings=0,
                repetition_crossing_windows_admitted=0,
                dropped_tail_samples=num_samples,
                admitted_start_indices=[],
                admitted_repetition_units=[],
            )
            empty_windows = np.empty((0, self.window_len, num_channels), dtype=np.float32)
            empty_labels = np.empty(0, dtype=labels.dtype)
            return empty_windows, empty_labels, audit

        candidate_starts = list(range(0, num_samples - self.window_len + 1, self.stride))
        total_candidates = len(candidate_starts)

        admitted_windows_list: list[np.ndarray] = []
        admitted_labels_list: list[Any] = []
        admitted_starts: list[int] = []
        admitted_units: list[int] = []

        discarded_label_transitions = 0
        discarded_repetition_crossings = 0

        for start_idx in candidate_starts:
            end_idx = start_idx + self.window_len
            win_labels = labels[start_idx:end_idx]
            win_units = units[start_idx:end_idx]

            # Condition 1: Pure gesture label
            if not np.all(win_labels == win_labels[0]):
                discarded_label_transitions += 1
                continue

            # Condition 2: Pure repetition unit
            if not np.all(win_units == win_units[0]):
                discarded_repetition_crossings += 1
                continue

            # Window is admitted
            win_emg = emg[start_idx:end_idx].astype(np.float32, copy=False)
            admitted_windows_list.append(win_emg)
            admitted_labels_list.append(win_labels[0])
            admitted_starts.append(start_idx)
            admitted_units.append(int(win_units[0]))

        admitted_count = len(admitted_windows_list)
        discarded_count = discarded_label_transitions + discarded_repetition_crossings

        last_start = candidate_starts[-1] if candidate_starts else 0
        dropped_tail = num_samples - (last_start + self.window_len)

        audit = TransitionAudit(
            total_candidate_windows=total_candidates,
            admitted_window_count=admitted_count,
            discarded_window_count=discarded_count,
            discarded_label_transitions=discarded_label_transitions,
            discarded_repetition_crossings=discarded_repetition_crossings,
            repetition_crossing_windows_admitted=0,
            dropped_tail_samples=dropped_tail,
            admitted_start_indices=admitted_starts,
            admitted_repetition_units=admitted_units,
        )

        if admitted_count > 0:
            windows_arr = np.stack(admitted_windows_list, axis=0)
            labels_arr = np.array(admitted_labels_list, dtype=labels.dtype)
        else:
            windows_arr = np.empty((0, self.window_len, num_channels), dtype=np.float32)
            labels_arr = np.empty(0, dtype=labels.dtype)

        return windows_arr, labels_arr, audit
