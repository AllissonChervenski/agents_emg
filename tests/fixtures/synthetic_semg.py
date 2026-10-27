"""Deterministic procedural synthetic sEMG generator for NinaPro DB2-like RecordingData (T002).

Provides controlled, reproducible synthetic recording fixtures conforming to
Feature 003 domain contracts (semg_dataset.contract.RecordingData), isolating
unit test suites from raw NinaPro DB2 archive dependencies.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from semg_dataset.contract import (
    AlignmentRecord,
    AlignmentStatus,
    RecordingData,
)


def generate_raw_synthetic_arrays(
    *,
    seed: int,
    active_labels: Tuple[int, ...] = (1, 2, 3, 4, 5),
    repetitions: Tuple[int, ...] = (1, 2, 3, 4, 5, 6),
    sampling_rate_hz: float = 2000.0,
    num_channels: int = 12,
    samples_per_rest: int = 800,
    samples_per_active: int = 1200,
    silent_channel_idx: int | None = None,
    duplicate_channel_pair: Tuple[int, int] | None = None,
    severe_rest_imbalance: bool = False,
    truncated_repetition_samples: int | None = None,
    inject_nan_coords: Tuple[int, int] | None = None,
    inject_inf_coords: Tuple[int, int] | None = None,
) -> Dict[str, np.ndarray]:
    """Generate raw numpy arrays for sEMG signals, labels, and repetitions.

    All signals are generated deterministically using an isolated local RNG
    seeded with `seed`. No global RNG state or external files are accessed.

    Returns:
        Dict with keys: 'emg', 'stimulus', 'repetition', 'restimulus', 'rerepetition'.
    """
    rng = np.random.default_rng(seed)

    actual_rest_samples = samples_per_rest * 10 if severe_rest_imbalance else samples_per_rest

    emg_chunks: list[np.ndarray] = []
    label_chunks: list[np.ndarray] = []
    rep_chunks: list[np.ndarray] = []

    # Initial resting baseline
    rest_emg = rng.normal(loc=0.0, scale=0.01, size=(actual_rest_samples, num_channels)).astype(np.float32)
    emg_chunks.append(rest_emg)
    label_chunks.append(np.zeros(actual_rest_samples, dtype=np.int16))
    rep_chunks.append(np.zeros(actual_rest_samples, dtype=np.int8))

    for rep in repetitions:
        # Check for repetition truncation
        if truncated_repetition_samples is not None and rep == max(repetitions):
            current_active_labels = active_labels[:1]
            active_len = truncated_repetition_samples
        else:
            current_active_labels = active_labels
            active_len = samples_per_active

        for label in current_active_labels:
            t = np.arange(active_len, dtype=np.float32) / sampling_rate_hz
            active_emg = rng.normal(loc=0.0, scale=0.015, size=(active_len, num_channels)).astype(np.float32)

            # Assign distinct spatial and frequency pattern per class
            # Different classes predominantly energize different channels
            primary_ch1 = (label * 2) % num_channels
            primary_ch2 = (label * 2 + 1) % num_channels
            secondary_ch = (label * 2 + 2) % num_channels

            # Distinct carrier frequencies within physiological EMG passband (50-350 Hz)
            f1 = 60.0 + ((label * 37) % 200)
            f2 = 110.0 + ((label * 53) % 180)

            # Add structured MUAP burst modulation to primary and secondary channels
            active_emg[:, primary_ch1] += (0.08 * np.sin(2.0 * np.pi * f1 * t)).astype(np.float32)
            active_emg[:, primary_ch2] += (0.07 * np.sin(2.0 * np.pi * f2 * t)).astype(np.float32)
            active_emg[:, secondary_ch] += (0.04 * np.sin(2.0 * np.pi * ((f1 + f2) / 2.0) * t)).astype(np.float32)

            emg_chunks.append(active_emg)
            label_chunks.append(np.full(active_len, label, dtype=np.int16))
            rep_chunks.append(np.full(active_len, rep, dtype=np.int8))

            # Inter-gesture rest period
            inter_rest = rng.normal(loc=0.0, scale=0.01, size=(actual_rest_samples, num_channels)).astype(np.float32)
            emg_chunks.append(inter_rest)
            label_chunks.append(np.zeros(actual_rest_samples, dtype=np.int16))
            rep_chunks.append(np.zeros(actual_rest_samples, dtype=np.int8))

    full_emg = np.vstack(emg_chunks)
    full_labels = np.concatenate(label_chunks)
    full_reps = np.concatenate(rep_chunks)

    # Apply adversarial modifications if requested
    if silent_channel_idx is not None:
        full_emg[:, silent_channel_idx] = 0.0

    if duplicate_channel_pair is not None:
        ch_src, ch_dst = duplicate_channel_pair
        full_emg[:, ch_dst] = full_emg[:, ch_src]

    if inject_nan_coords is not None:
        r, c = inject_nan_coords
        full_emg[r, c] = np.nan

    if inject_inf_coords is not None:
        r, c = inject_inf_coords
        full_emg[r, c] = np.inf

    return {
        "emg": full_emg,
        "stimulus": full_labels.copy(),
        "repetition": full_reps.copy(),
        "restimulus": full_labels.copy(),
        "rerepetition": full_reps.copy(),
    }


def make_synthetic_recording(
    *,
    seed: int,
    subject_id: str = "SYNTH_S1",
    exercise_id: str = "E1",
    active_labels: Tuple[int, ...] = (1, 2, 3, 4, 5),
    repetitions: Tuple[int, ...] = (1, 2, 3, 4, 5, 6),
    sampling_rate_hz: float = 2000.0,
    num_channels: int = 12,
    samples_per_rest: int = 800,
    samples_per_active: int = 1200,
    silent_channel_idx: int | None = None,
    duplicate_channel_pair: Tuple[int, int] | None = None,
    severe_rest_imbalance: bool = False,
    truncated_repetition_samples: int | None = None,
    provenance: Dict[str, Any] | None = None,
) -> RecordingData:
    """Create a canonical RecordingData instance populated with synthetic signals.

    Complies 100% with Feature 003 RecordingData invariants:
      - 12 channels in float32 without NaN or Inf
      - Nominal sampling rate (2000.0 Hz)
      - Synchronized stimulus, repetition, restimulus, and rerepetition arrays
      - Valid AlignmentRecord with status IDENTICAL

    Args:
        seed: Random seed guaranteeing bit-exact deterministic reproduction.
        subject_id: Synthetic participant identifier (e.g. 'SYNTH_S1').
        exercise_id: Exercise identifier ('E1', 'E2', or 'E3').
        active_labels: Tuple of active gesture labels included in recording.
        repetitions: Tuple of repetition IDs (1..6).
        sampling_rate_hz: Sampling frequency in Hertz (default: 2000.0).
        num_channels: Number of sEMG sensor channels (strictly 12).
        samples_per_rest: Number of temporal samples per rest interval.
        samples_per_active: Number of temporal samples per active contraction.
        silent_channel_idx: Optional channel index to force to zero variance.
        duplicate_channel_pair: Optional (src, dst) pair to create singular covariance.
        severe_rest_imbalance: If True, rest intervals are expanded tenfold.
        truncated_repetition_samples: Optional sample count to truncate final repetition.
        provenance: Optional provenance metadata dictionary.

    Returns:
        RecordingData instance ready for baseline feature extraction and training.
    """
    raw_arrays = generate_raw_synthetic_arrays(
        seed=seed,
        active_labels=active_labels,
        repetitions=repetitions,
        sampling_rate_hz=sampling_rate_hz,
        num_channels=num_channels,
        samples_per_rest=samples_per_rest,
        samples_per_active=samples_per_active,
        silent_channel_idx=silent_channel_idx,
        duplicate_channel_pair=duplicate_channel_pair,
        severe_rest_imbalance=severe_rest_imbalance,
        truncated_repetition_samples=truncated_repetition_samples,
    )

    n_samples = raw_arrays["emg"].shape[0]

    alignment = AlignmentRecord(
        status=AlignmentStatus.IDENTICAL,
        original_emg_length=n_samples,
        original_refined_length=n_samples,
        delta_samples=0,
        start_anchored_proven=True,
        notes="Synthetic procedural alignment: identical timelines by construction.",
    )

    prov = provenance if provenance is not None else {"source": "procedural_synthetic", "seed": seed}

    return RecordingData(
        subject_id=subject_id,
        exercise_id=exercise_id,
        emg=raw_arrays["emg"],
        stimulus=raw_arrays["stimulus"],
        repetition=raw_arrays["repetition"],
        restimulus=raw_arrays["restimulus"],
        rerepetition=raw_arrays["rerepetition"],
        alignment=alignment,
        sampling_rate_hz=sampling_rate_hz,
        provenance=prov,
    )
