"""Integrated baseline feature extraction pipeline.

Implements FR-001, FR-004, FR-012, FR-013, SC-001, SC-002, Decision-2:
- Integrates causal DSP filtering, pure steady-state windowing, noise deadzone calibration,
  time-domain feature extraction, and train-only custom Z-score scaling.
- Consumes frozen Feature 003 RecordingData directly.
- Guarantees zero lookahead, zero NaN/Inf, and strict train-only parameter isolation.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from semg_baseline.filter import CausalBandpassFilter
from semg_baseline.window import PureWindowSegmenter, TransitionAudit
from semg_baseline.thresholds import ThresholdCalibration, ThresholdCalibrator
from semg_baseline.features import TimeDomainFeatureExtractor
from semg_baseline.scaling import FeatureScaler
from semg_dataset.contract import RecordingData


class BaselineFeaturePipeline:
    """End-to-end baseline feature processing pipeline from raw recordings to scaled features."""

    def __init__(
        self,
        filter_low_hz: float = 20.0,
        filter_high_hz: float = 450.0,
        fs: float = 2000.0,
        filter_order: int = 4,
        window_len: int = 400,
        stride: int = 100,
        candidate_grid: Sequence[float] = (0.5, 1.0, 1.5, 2.0, 3.0),
        default_r: float = 1.5,
        num_channels: int = 12,
    ) -> None:
        self.filter_low_hz = float(filter_low_hz)
        self.filter_high_hz = float(filter_high_hz)
        self.fs = float(fs)
        self.filter_order = int(filter_order)
        self.window_len = int(window_len)
        self.stride = int(stride)
        self.num_channels = int(num_channels)

        self.segmenter = PureWindowSegmenter(
            window_len=self.window_len,
            stride=self.stride,
            fs=self.fs,
        )
        self.calibrator = ThresholdCalibrator(
            candidate_grid=candidate_grid,
            default_r=default_r,
            num_channels=self.num_channels,
        )

        self.threshold_calibration: ThresholdCalibration | None = None
        self.feature_extractor: TimeDomainFeatureExtractor | None = None
        self.feature_scaler: FeatureScaler | None = None

    def _process_single_recording(
        self, recording: RecordingData, view_name: str
    ) -> tuple[np.ndarray, np.ndarray, TransitionAudit]:
        """Filter and segment a single RecordingData object statelessly."""
        view = recording.get_view(view_name)

        # Create isolated filter instance per recording to prevent inter-recording state leakage
        bp_filter = CausalBandpassFilter(
            low_hz=self.filter_low_hz,
            high_hz=self.filter_high_hz,
            fs=self.fs,
            order=self.filter_order,
            num_channels=self.num_channels,
        )
        filtered_signals = bp_filter.process(view.signals)

        windows, labels, audit = self.segmenter.segment(
            emg=filtered_signals,
            labels=view.labels,
            repetitions=view.repetitions,
        )
        return windows, labels, audit

    def fit(
        self,
        train_recordings: Sequence[RecordingData],
        view_name: str = "refined",
        rest_label: int = 0,
        r_multiplier: float | None = None,
        evaluator_fn: Callable[[float], float] | None = None,
    ) -> BaselineFeaturePipeline:
        """Fit noise deadzones and feature scaler strictly using training recordings.

        Args:
            train_recordings: Sequence of RecordingData objects for training.
            view_name: Dataset view name ('refined' or 'stimulus').
            rest_label: Label designating REST (canonical: 0).
            r_multiplier: Optional pre-fixed R in candidate grid.
            evaluator_fn: Optional scoring callback fn(r) for tuning R.

        Returns:
            self
        """
        windows_list: list[np.ndarray] = []
        labels_list: list[np.ndarray] = []

        for rec in train_recordings:
            wins, labs, _ = self._process_single_recording(rec, view_name=view_name)
            if len(wins) > 0:
                windows_list.append(wins)
                labels_list.append(labs)

        if not windows_list:
            raise ValueError("No admitted pure windows extracted from training recordings.")

        all_windows = np.concatenate(windows_list, axis=0)
        all_labels = np.concatenate(labels_list, axis=0)

        # 1. Calibrate deadzones strictly on training rest windows
        self.threshold_calibration = self.calibrator.calibrate(
            train_emg=all_windows,
            train_labels=all_labels,
            rest_label=rest_label,
            r_multiplier=r_multiplier,
            evaluator_fn=evaluator_fn,
        )

        # 2. Extract unscaled features on all training windows
        self.feature_extractor = TimeDomainFeatureExtractor(
            epsilon=self.threshold_calibration.epsilon_channels,
            num_channels=self.num_channels,
        )
        X_train_raw = self.feature_extractor.extract(all_windows)

        # 3. Fit scaler strictly on training features
        self.feature_scaler = FeatureScaler()
        self.feature_scaler.fit(X_train_raw)

        return self

    def transform(
        self,
        recordings: Sequence[RecordingData],
        view_name: str = "refined",
    ) -> tuple[np.ndarray, np.ndarray, list[TransitionAudit]]:
        """Extract and normalize features for recordings using fitted parameters.

        Args:
            recordings: Sequence of RecordingData objects to transform.
            view_name: Dataset view name ('refined' or 'stimulus').

        Returns:
            Tuple of (X_scaled, labels, audits).
        """
        if self.feature_extractor is None or self.feature_scaler is None:
            raise RuntimeError(
                "BaselineFeaturePipeline is not fitted yet. Call fit() or fit_transform() first."
            )

        windows_list: list[np.ndarray] = []
        labels_list: list[np.ndarray] = []
        audits_list: list[TransitionAudit] = []

        for rec in recordings:
            wins, labs, audit = self._process_single_recording(rec, view_name=view_name)
            audits_list.append(audit)
            if len(wins) > 0:
                windows_list.append(wins)
                labels_list.append(labs)

        if not windows_list:
            empty_X = np.empty((0, self.num_channels * 5), dtype=np.float64)
            empty_y = np.empty(0, dtype=np.int16)
            return empty_X, empty_y, audits_list

        all_windows = np.concatenate(windows_list, axis=0)
        all_labels = np.concatenate(labels_list, axis=0)

        X_raw = self.feature_extractor.extract(all_windows)
        X_scaled = self.feature_scaler.transform(X_raw)

        return X_scaled, all_labels, audits_list

    def fit_transform(
        self,
        train_recordings: Sequence[RecordingData],
        view_name: str = "refined",
        rest_label: int = 0,
        r_multiplier: float | None = None,
        evaluator_fn: Callable[[float], float] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, list[TransitionAudit]]:
        """Fit pipeline on training recordings and return scaled features."""
        self.fit(
            train_recordings=train_recordings,
            view_name=view_name,
            rest_label=rest_label,
            r_multiplier=r_multiplier,
            evaluator_fn=evaluator_fn,
        )
        return self.transform(train_recordings, view_name=view_name)
