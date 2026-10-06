"""Data-driven deadzone threshold calibration for ZC and SSC features.

Implements FR-012.4, SC-002, Decision-2:
- Noise deadzone thresholds computed strictly on TRAIN resting windows:
  epsilon[ch] = R * RMS_rest_train[ch].
- Candidate grid R in {0.5, 1.0, 1.5, 2.0, 3.0}.
- Zero-variance channel protection with 1e-8 lower bound.
- Zero data leakage invariant: validation and test splits NEVER contribute to calibration.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


VALID_CANDIDATE_GRID = (0.5, 1.0, 1.5, 2.0, 3.0)


@dataclass(frozen=True)
class ThresholdCalibration:
    """Encapsulates calibrated deadzone thresholds and training rest baseline."""

    r_multiplier: float
    rms_rest_train: np.ndarray
    epsilon_channels: np.ndarray
    selection_method: str = "training_pool_validation"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary matching training_job_schema.json."""
        return {
            "r_multiplier": float(self.r_multiplier),
            "selection_method": self.selection_method,
            "rms_rest_train": [float(x) for x in self.rms_rest_train],
            "epsilon_channels": [float(x) for x in self.epsilon_channels],
        }


class ThresholdCalibrator:
    """Calibrates per-channel noise deadzones strictly using training rest windows."""

    def __init__(
        self,
        candidate_grid: Sequence[float] = VALID_CANDIDATE_GRID,
        default_r: float = 1.5,
        num_channels: int = 12,
    ) -> None:
        self.candidate_grid = tuple(float(r) for r in candidate_grid)
        self.default_r = float(default_r)
        self.num_channels = int(num_channels)

    def compute_rest_rms(
        self,
        train_emg: np.ndarray,
        train_labels: np.ndarray,
        rest_label: int = 0,
    ) -> np.ndarray:
        """Compute resting baseline RMS per channel across training rest windows.

        Args:
            train_emg: Array of shape (N_windows, window_len, num_channels).
            train_labels: Array of shape (N_windows,).
            rest_label: Label integer designating REST (canonical: 0).

        Returns:
            Array of shape (num_channels,) containing resting RMS in float64.
        """
        emg = np.asarray(train_emg, dtype=np.float64)
        labels = np.asarray(train_labels)

        rest_mask = (labels == rest_label)
        if not np.any(rest_mask):
            raise ValueError(
                f"No REST windows found in training set (rest_label={rest_label})"
            )

        rest_windows = emg[rest_mask]
        # Calculate root-mean-square over all samples across all rest windows per channel
        rms_channels = np.sqrt(np.mean(rest_windows**2, axis=(0, 1), dtype=np.float64))

        # Numerical safety guard: silent / dead channels receive a minimum 1e-8 epsilon
        return np.maximum(rms_channels, 1e-8)

    def calibrate(
        self,
        train_emg: np.ndarray,
        train_labels: np.ndarray,
        rest_label: int = 0,
        r_multiplier: float | None = None,
        selection_method: str = "training_pool_validation",
        evaluator_fn: Callable[[float], float] | None = None,
    ) -> ThresholdCalibration:
        """Calibrate deadzone thresholds strictly using training rest windows.

        Args:
            train_emg: Training windows of shape (N_windows, window_len, num_channels).
            train_labels: Training labels of shape (N_windows,).
            rest_label: Label designated as rest (default 0).
            r_multiplier: Optional pre-selected R in candidate_grid.
            selection_method: Provenance method string.
            evaluator_fn: Optional scoring callback fn(r) returning evaluation metric.

        Returns:
            ThresholdCalibration instance.
        """
        rms_rest = self.compute_rest_rms(train_emg, train_labels, rest_label=rest_label)

        if r_multiplier is not None:
            r_val = float(r_multiplier)
            if r_val not in self.candidate_grid:
                raise ValueError(
                    f"r_multiplier {r_val} not in candidate grid {self.candidate_grid}"
                )
            selected_r = r_val
        elif evaluator_fn is not None:
            scores = {r: evaluator_fn(r) for r in self.candidate_grid}
            selected_r = max(scores, key=lambda k: scores[k])
        else:
            selected_r = self.default_r

        epsilon_channels = selected_r * rms_rest

        return ThresholdCalibration(
            r_multiplier=selected_r,
            rms_rest_train=rms_rest,
            epsilon_channels=epsilon_channels,
            selection_method=selection_method,
        )
