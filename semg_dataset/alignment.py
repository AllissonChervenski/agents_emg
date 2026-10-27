"""PROVE-OR-QUARANTINE Temporal Alignment Engine for NinaPro DB2.

Establishes a single canonical timeline per recording. For recordings with
discrepancies between EMG and refined label arrays (such as the 18 E3 recordings,
including the 272-sample outlier S12_E3_A1.mat):
  1. Proves whether t=0 start-anchoring is intact.
  2. Verifies that discarded trailing samples are purely non-active REST.
  3. Applies anchor_start_truncate_tail only when proven.
  4. Quarantines unproven, head-shifted, or ambiguous cases.
  5. Strictly forbids padding, interpolation, silent fallback, and silent min().
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from semg_dataset.contract import AlignmentRecord, AlignmentStatus, RecordingData


def anchor_start_truncate_tail(
    emg: np.ndarray,
    stimulus: np.ndarray,
    repetition: np.ndarray,
    target_length: int,
    restimulus: Optional[np.ndarray] = None,
    rerepetition: Optional[np.ndarray] = None,
) -> Any:
    """Truncate trailing samples from start-anchored arrays to match target_length."""
    if restimulus is not None and rerepetition is not None:
        return (
            emg[:target_length],
            stimulus[:target_length],
            repetition[:target_length],
            restimulus[:target_length],
            rerepetition[:target_length],
        )
    return emg[:target_length], stimulus[:target_length], repetition[:target_length]


class ProveOrQuarantineEngine:
    """Evaluates and enforces PROVE-OR-QUARANTINE temporal alignment policy."""

    def __init__(
        self,
        max_allowed_delta: int = 500,
        max_anticipation_samples: int = 100,
        max_reaction_delay_samples: int = 4000,
    ) -> None:
        self.max_allowed_delta = max_allowed_delta
        self.max_anticipation_samples = max_anticipation_samples
        self.max_reaction_delay_samples = max_reaction_delay_samples

    def evaluate_and_align(
        self,
        emg: np.ndarray,
        stimulus: np.ndarray,
        repetition: np.ndarray,
        restimulus: np.ndarray,
        rerepetition: np.ndarray,
        subject_id: str = "",
        exercise_id: str = "",
        allow_padding: bool = False,
        fallback_to_stimulus: bool = False,
        silent_min: bool = False,
    ) -> Tuple[AlignmentRecord, Dict[str, np.ndarray]]:
        """Evaluate temporal alignment evidence and apply anchor_start_truncate_tail if proven.

        Raises:
            ValueError: If padding, fallback_to_stimulus, or silent_min is requested.
        """
        # Security & Policy Violations
        if allow_padding:
            raise ValueError("Padding is strictly forbidden by PROVE-OR-QUARANTINE policy")
        if fallback_to_stimulus:
            raise ValueError(
                "Silent fallback to stimulus is strictly forbidden by PROVE-OR-QUARANTINE policy"
            )
        if silent_min:
            raise ValueError(
                "Silent min() truncation is strictly forbidden by PROVE-OR-QUARANTINE policy"
            )

        n_emg = emg.shape[0]
        n_stim = stimulus.shape[0]
        n_rep = repetition.shape[0]
        n_ref = restimulus.shape[0]
        n_rerep = rerepetition.shape[0]

        # Invariant checks on primary pairs
        if n_stim != n_emg or n_rep != n_emg or n_rerep != n_ref:
            rec = AlignmentRecord(
                status=AlignmentStatus.QUARANTINE,
                original_emg_length=n_emg,
                original_refined_length=n_ref,
                delta_samples=abs(n_emg - n_ref),
                start_anchored_proven=False,
                notes="Inconsistent component array lengths within paired sets",
            )
            return rec, {
                "emg": emg,
                "stimulus": stimulus,
                "repetition": repetition,
                "restimulus": restimulus,
                "rerepetition": rerepetition,
            }

        # Case 1: Identical length
        if n_emg == n_ref:
            rec = AlignmentRecord(
                status=AlignmentStatus.IDENTICAL,
                original_emg_length=n_emg,
                original_refined_length=n_ref,
                delta_samples=0,
                start_anchored_proven=True,
                notes="Array lengths are identical at source",
            )
            return rec, {
                "emg": emg,
                "stimulus": stimulus,
                "repetition": repetition,
                "restimulus": restimulus,
                "rerepetition": rerepetition,
            }

        delta = abs(n_emg - n_ref)
        target_length = min(n_emg, n_ref)

        # Case 2: Prove-or-Quarantine
        # Check A: Delta threshold
        if delta > self.max_allowed_delta:
            rec = AlignmentRecord(
                status=AlignmentStatus.QUARANTINE,
                original_emg_length=n_emg,
                original_refined_length=n_ref,
                delta_samples=delta,
                start_anchored_proven=False,
                notes=f"Delta ({delta}) exceeds maximum allowable threshold ({self.max_allowed_delta})",
            )
            return rec, {
                "emg": emg,
                "stimulus": stimulus,
                "repetition": repetition,
                "restimulus": restimulus,
                "rerepetition": rerepetition,
            }

        # Check B: Start condition at t=0 (must begin at REST)
        if restimulus[0] != 0 or stimulus[0] != 0:
            rec = AlignmentRecord(
                status=AlignmentStatus.QUARANTINE,
                original_emg_length=n_emg,
                original_refined_length=n_ref,
                delta_samples=delta,
                start_anchored_proven=False,
                notes="Head shift or truncation detected: non-zero label at t=0",
            )
            return rec, {
                "emg": emg,
                "stimulus": stimulus,
                "repetition": repetition,
                "restimulus": restimulus,
                "rerepetition": rerepetition,
            }

        # Check C: First movement onset synchrony
        has_stim_active = np.any(stimulus > 0)
        has_ref_active = np.any(restimulus > 0)

        if has_stim_active and has_ref_active:
            idx_stim = int(np.argmax(stimulus > 0))
            idx_ref = int(np.argmax(restimulus > 0))

            if idx_ref < idx_stim - self.max_anticipation_samples:
                rec = AlignmentRecord(
                    status=AlignmentStatus.QUARANTINE,
                    original_emg_length=n_emg,
                    original_refined_length=n_ref,
                    delta_samples=delta,
                    start_anchored_proven=False,
                    notes=(
                        f"Head shift detected: refined onset ({idx_ref}) precedes stimulus "
                        f"onset ({idx_stim}) by more than {self.max_anticipation_samples} samples"
                    ),
                )
                return rec, {
                    "emg": emg,
                    "stimulus": stimulus,
                    "repetition": repetition,
                    "restimulus": restimulus,
                    "rerepetition": rerepetition,
                }

            if idx_ref - idx_stim > self.max_reaction_delay_samples:
                rec = AlignmentRecord(
                    status=AlignmentStatus.QUARANTINE,
                    original_emg_length=n_emg,
                    original_refined_length=n_ref,
                    delta_samples=delta,
                    start_anchored_proven=False,
                    notes=(
                        f"Ambiguous alignment: onset delay ({idx_ref - idx_stim}) exceeds "
                        f"maximum reaction threshold ({self.max_reaction_delay_samples})"
                    ),
                )
                return rec, {
                    "emg": emg,
                    "stimulus": stimulus,
                    "repetition": repetition,
                    "restimulus": restimulus,
                    "rerepetition": rerepetition,
                }

        # Check D: Trailing boundary verification
        # Refined label must terminate in REST at target_length - 1
        if restimulus[target_length - 1] != 0:
            rec = AlignmentRecord(
                status=AlignmentStatus.QUARANTINE,
                original_emg_length=n_emg,
                original_refined_length=n_ref,
                delta_samples=delta,
                start_anchored_proven=False,
                notes="Refined array ends during active gesture; tail truncation would corrupt gesture boundary",
            )
            return rec, {
                "emg": emg,
                "stimulus": stimulus,
                "repetition": repetition,
                "restimulus": restimulus,
                "rerepetition": rerepetition,
            }

        # If restimulus is longer than emg (n_ref > n_emg): discarded restimulus tail must be rest
        if n_ref > n_emg and np.any(restimulus[target_length:] != 0):
            rec = AlignmentRecord(
                status=AlignmentStatus.QUARANTINE,
                original_emg_length=n_emg,
                original_refined_length=n_ref,
                delta_samples=delta,
                start_anchored_proven=False,
                notes="Trailing samples to be discarded in restimulus contain active gesture",
            )
            return rec, {
                "emg": emg,
                "stimulus": stimulus,
                "repetition": repetition,
                "restimulus": restimulus,
                "rerepetition": rerepetition,
            }

        # All proof checks PASSED -> apply anchor_start_truncate_tail
        emg_trunc = emg[:target_length]
        stim_trunc = stimulus[:target_length]
        rep_trunc = repetition[:target_length]
        restim_trunc = restimulus[:target_length]
        rerep_trunc = rerepetition[:target_length]

        rec = AlignmentRecord(
            status=AlignmentStatus.ANCHOR_START_TRUNCATE_TAIL,
            original_emg_length=n_emg,
            original_refined_length=n_ref,
            delta_samples=delta,
            start_anchored_proven=True,
            notes=f"Start-anchored proof verified; truncated {delta} trailing samples to {target_length}",
        )

        return rec, {
            "emg": emg_trunc,
            "stimulus": stim_trunc,
            "repetition": rep_trunc,
            "restimulus": restim_trunc,
            "rerepetition": rerep_trunc,
        }

    def align_recording(self, recording: RecordingData) -> RecordingData:
        """Evaluate and align a RecordingData instance, returning an updated instance."""
        record, aligned = self.evaluate_and_align(
            emg=recording.emg,
            stimulus=recording.stimulus,
            repetition=recording.repetition,
            restimulus=recording.restimulus,
            rerepetition=recording.rerepetition,
            subject_id=recording.subject_id,
            exercise_id=recording.exercise_id,
        )

        if record.status == AlignmentStatus.QUARANTINE:
            raise RuntimeError(
                f"Recording {recording.subject_id}_{recording.exercise_id} is QUARANTINED: {record.notes}"
            )

        provenance_copy = dict(recording.provenance)
        provenance_copy["alignment_record"] = {
            "status": record.status.value,
            "delta_samples": record.delta_samples,
            "start_anchored_proven": record.start_anchored_proven,
            "notes": record.notes,
        }

        return RecordingData(
            subject_id=recording.subject_id,
            exercise_id=recording.exercise_id,
            emg=aligned["emg"],
            stimulus=aligned["stimulus"],
            repetition=aligned["repetition"],
            restimulus=aligned["restimulus"],
            rerepetition=aligned["rerepetition"],
            alignment=record,
            sampling_rate_hz=recording.sampling_rate_hz,
            provenance=provenance_copy,
        )
