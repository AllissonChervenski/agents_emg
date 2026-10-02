"""Tests for PROVE-OR-QUARANTINE Temporal Alignment Engine using synthetic fixtures.

Scientific-integrity-sensitive:
Ensures that recordings with length discrepancies are evaluated independently,
requiring proof of t=0 start anchoring before applying anchor_start_truncate_tail,
and marking unproven, head-shifted, or ambiguous cases as QUARANTINE.
"""

import numpy as np
import pytest

from semg_dataset.contract import AlignmentStatus


def test_identical_length_recording_is_accepted_as_identical() -> None:
    """Equal length recordings must be classified as IDENTICAL."""
    from semg_dataset.alignment import ProveOrQuarantineEngine

    engine = ProveOrQuarantineEngine()
    n = 500
    emg = np.random.randn(n, 12).astype(np.float32) * 0.001
    stim = np.zeros(n, dtype=np.int8)
    stim[50:100] = 1
    rep = np.zeros(n, dtype=np.int8)
    rep[50:100] = 1

    record, aligned = engine.evaluate_and_align(
        emg=emg,
        stimulus=stim,
        repetition=rep,
        restimulus=stim.copy(),
        rerepetition=rep.copy(),
        subject_id="S1",
        exercise_id="E1",
    )

    assert record.status == AlignmentStatus.IDENTICAL
    assert record.start_anchored_proven is True
    assert record.delta_samples == 0
    assert aligned["emg"].shape == (n, 12)
    assert len(aligned["restimulus"]) == n


def test_tail_short_proven_alignment_applies_anchor_start_truncate_tail() -> None:
    """Tail-short refined labels with proven t=0 start anchor are safely truncated."""
    from semg_dataset.alignment import ProveOrQuarantineEngine

    engine = ProveOrQuarantineEngine()
    n_emg = 1000
    n_ref = 950
    delta = n_emg - n_ref

    emg = np.random.randn(n_emg, 12).astype(np.float32) * 0.001
    stim = np.zeros(n_emg, dtype=np.int8)
    stim[100:200] = 1
    rep = np.zeros(n_emg, dtype=np.int8)
    rep[100:200] = 1

    # restimulus starts at rest, has reaction delay, ends at rest before n_ref
    restim = np.zeros(n_ref, dtype=np.int8)
    restim[120:210] = 1
    rerep = np.zeros(n_ref, dtype=np.int8)
    rerep[120:210] = 1

    record, aligned = engine.evaluate_and_align(
        emg=emg,
        stimulus=stim,
        repetition=rep,
        restimulus=restim,
        rerepetition=rerep,
        subject_id="S1",
        exercise_id="E3",
    )

    assert record.status == AlignmentStatus.ANCHOR_START_TRUNCATE_TAIL
    assert record.start_anchored_proven is True
    assert record.delta_samples == delta
    assert aligned["emg"].shape == (n_ref, 12)
    assert len(aligned["stimulus"]) == n_ref
    assert len(aligned["restimulus"]) == n_ref


def test_head_shifted_case_is_quarantined() -> None:
    """Head-shifted refined labels (label[k] <-> emg[k+Delta]) must be quarantined."""
    from semg_dataset.alignment import ProveOrQuarantineEngine

    engine = ProveOrQuarantineEngine()
    n_emg = 1000
    n_ref = 950

    emg = np.random.randn(n_emg, 12).astype(np.float32) * 0.001
    stim = np.zeros(n_emg, dtype=np.int8)
    stim[100:200] = 1
    rep = np.zeros(n_emg, dtype=np.int8)
    rep[100:200] = 1

    # Head shift: restimulus starts with an active gesture right at t=0
    # because it was shifted left (missing head samples)
    restim = np.zeros(n_ref, dtype=np.int8)
    restim[0:100] = 1
    rerep = np.zeros(n_ref, dtype=np.int8)
    rerep[0:100] = 1

    record, aligned = engine.evaluate_and_align(
        emg=emg,
        stimulus=stim,
        repetition=rep,
        restimulus=restim,
        rerepetition=rerep,
        subject_id="S2",
        exercise_id="E3",
    )

    assert record.status == AlignmentStatus.QUARANTINE
    assert record.start_anchored_proven is False


def test_ambiguous_trailing_gesture_is_quarantined() -> None:
    """Discrepancies where the refined labels end mid-gesture must be quarantined."""
    from semg_dataset.alignment import ProveOrQuarantineEngine

    engine = ProveOrQuarantineEngine()
    n_emg = 1000
    n_ref = 950

    emg = np.random.randn(n_emg, 12).astype(np.float32) * 0.001
    stim = np.zeros(n_emg, dtype=np.int8)
    stim[100:200] = 1
    rep = np.zeros(n_emg, dtype=np.int8)
    rep[100:200] = 1

    # Active gesture right at the cutoff boundary (n_ref - 1)
    restim = np.zeros(n_ref, dtype=np.int8)
    restim[900:950] = 1
    rerep = np.zeros(n_ref, dtype=np.int8)
    rerep[900:950] = 1

    record, aligned = engine.evaluate_and_align(
        emg=emg,
        stimulus=stim,
        repetition=rep,
        restimulus=restim,
        rerepetition=rerep,
        subject_id="S3",
        exercise_id="E3",
    )

    assert record.status == AlignmentStatus.QUARANTINE
    assert record.start_anchored_proven is False


def test_large_delta_without_proof_is_quarantined() -> None:
    """Discrepancies exceeding allowable thresholds without explicit proof must be quarantined."""
    from semg_dataset.alignment import ProveOrQuarantineEngine

    engine = ProveOrQuarantineEngine(max_allowed_delta=500)
    n_emg = 5000
    n_ref = 3000  # delta = 2000 > 500

    emg = np.random.randn(n_emg, 12).astype(np.float32) * 0.001
    stim = np.zeros(n_emg, dtype=np.int8)
    rep = np.zeros(n_emg, dtype=np.int8)
    restim = np.zeros(n_ref, dtype=np.int8)
    rerep = np.zeros(n_ref, dtype=np.int8)

    record, aligned = engine.evaluate_and_align(
        emg=emg,
        stimulus=stim,
        repetition=rep,
        restimulus=restim,
        rerepetition=rerep,
        subject_id="S4",
        exercise_id="E3",
    )

    assert record.status == AlignmentStatus.QUARANTINE
    assert record.start_anchored_proven is False


def test_padding_attempt_is_strictly_forbidden() -> None:
    """Any attempt to enable padding must raise ValueError."""
    from semg_dataset.alignment import ProveOrQuarantineEngine

    engine = ProveOrQuarantineEngine()
    n_emg = 1000
    n_ref = 950
    emg = np.zeros((n_emg, 12), dtype=np.float32)
    arr = np.zeros(n_emg, dtype=np.int8)
    arr_short = np.zeros(n_ref, dtype=np.int8)

    with pytest.raises(ValueError, match="[Pp]adding.*forbidden"):
        engine.evaluate_and_align(
            emg=emg,
            stimulus=arr,
            repetition=arr,
            restimulus=arr_short,
            rerepetition=arr_short,
            allow_padding=True,
        )


def test_fallback_to_stimulus_is_strictly_forbidden() -> None:
    """Any attempt to silently fallback to stimulus must raise ValueError."""
    from semg_dataset.alignment import ProveOrQuarantineEngine

    engine = ProveOrQuarantineEngine()
    n_emg = 1000
    n_ref = 950
    emg = np.zeros((n_emg, 12), dtype=np.float32)
    arr = np.zeros(n_emg, dtype=np.int8)
    arr_short = np.zeros(n_ref, dtype=np.int8)

    with pytest.raises(ValueError, match="[Ff]allback.*forbidden"):
        engine.evaluate_and_align(
            emg=emg,
            stimulus=arr,
            repetition=arr,
            restimulus=arr_short,
            rerepetition=arr_short,
            fallback_to_stimulus=True,
        )


def test_s12_outlier_simulation_proven_and_aligned() -> None:
    """Simulate the S12_E3 outlier (272 samples discrepancy) with start-anchoring proof."""
    from semg_dataset.alignment import ProveOrQuarantineEngine

    engine = ProveOrQuarantineEngine(max_allowed_delta=500)
    n_emg = 875707
    n_ref = 875435
    delta = 272

    # Use memory-efficient representations for 875k samples
    emg = np.zeros((n_emg, 12), dtype=np.float32)
    stim = np.zeros(n_emg, dtype=np.int8)
    rep = np.zeros(n_emg, dtype=np.int8)
    # Add gesture event in valid region
    stim[10000:20000] = 41
    rep[10000:20000] = 1

    restim = np.zeros(n_ref, dtype=np.int8)
    rerep = np.zeros(n_ref, dtype=np.int8)
    restim[10200:20050] = 41
    rerep[10200:20050] = 1

    record, aligned = engine.evaluate_and_align(
        emg=emg,
        stimulus=stim,
        repetition=rep,
        restimulus=restim,
        rerepetition=rerep,
        subject_id="S12",
        exercise_id="E3",
    )

    assert record.status == AlignmentStatus.ANCHOR_START_TRUNCATE_TAIL
    assert record.start_anchored_proven is True
    assert record.delta_samples == delta
    assert aligned["emg"].shape == (n_ref, 12)
    assert len(aligned["restimulus"]) == n_ref
    assert len(aligned["stimulus"]) == n_ref
