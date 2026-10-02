"""In-memory, read-only streaming loader for NinaPro DB2 MATLAB files from ZIP archives.

Provides NinaProDB2Loader that streams and parses .mat recordings directly from
compressed archives into memory without extracting files to disk.
"""

from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.io

from semg_dataset.alignment import ProveOrQuarantineEngine
from semg_dataset.contract import RecordingData
from semg_dataset.provenance import NinaProProvenance




class NinaProDB2Loader:
    """In-memory reader for NinaPro DB2 ZIP archives with fail-closed integrity."""

    def __init__(
        self,
        raw_dir: Path | str,
        inventory_path: Optional[Path | str] = None,
        verify_hashes: bool = False,
    ) -> None:
        self.raw_dir = Path(raw_dir)
        self.inventory_path = Path(inventory_path) if inventory_path else None
        self.verify_hashes = verify_hashes
        self._archive_cache: Dict[str, Path] = {}
        self._scan_archives()

    def _scan_archives(self) -> None:
        """Scan raw_dir for archives matching NinaPro DB2 pattern."""
        if not self.raw_dir.is_dir():
            raise FileNotFoundError(f"Raw directory does not exist: {self.raw_dir}")

        for p in self.raw_dir.glob("*.zip"):
            m = re.match(r"^DB2_s(\d+)\.zip$", p.name, re.IGNORECASE)
            if m:
                s_id = f"S{m.group(1)}"
                self._archive_cache[s_id] = p

    def list_subjects(self) -> List[str]:
        """Return a sorted list of subject IDs found in raw_dir (e.g. ['S1', 'S2', ...])."""

        def _subject_key(sid: str) -> int:
            m = re.search(r"\d+", sid)
            return int(m.group()) if m else 0

        return sorted(self._archive_cache.keys(), key=_subject_key)

    def list_recordings(self) -> List[Tuple[str, str]]:
        """Return a sorted list of available (subject_id, exercise_id) pairs."""
        recordings: List[Tuple[str, str]] = []
        for s_id in self.list_subjects():
            zip_path = self._archive_cache[s_id]
            with zipfile.ZipFile(zip_path, "r") as zf:
                for name in zf.namelist():
                    m = re.search(r"S\d+_(E[1-3])_.*\.mat$", name, re.IGNORECASE)
                    if m:
                        e_id = m.group(1).upper()
                        recordings.append((s_id, e_id))

        def _rec_key(pair: Tuple[str, str]) -> Tuple[int, str]:
            sid, eid = pair
            m = re.search(r"\d+", sid)
            num = int(m.group()) if m is not None else 0
            return (num, eid)

        return sorted(list(set(recordings)), key=_rec_key)

    def load_recording(self, subject_id: str, exercise_id: str) -> RecordingData:
        """Load and return a canonical RecordingData instance from in-memory ZIP member.

        Reads .mat bytes directly into memory without writing uncompressed files to disk.
        """
        if subject_id not in self._archive_cache:
            raise KeyError(f"Subject {subject_id} not found in {self.raw_dir}")

        zip_path = self._archive_cache[subject_id]

        if self.verify_hashes and self.inventory_path:
            baseline = NinaProProvenance.load_baseline_hashes(self.inventory_path)
            expected_hash = baseline.get(zip_path.name)
            if expected_hash and not NinaProProvenance.verify_archive_checksum(zip_path, expected_hash):
                raise ValueError(f"Integrity check failed for {zip_path.name}")

        with zipfile.ZipFile(zip_path, "r") as zf:
            target_name: Optional[str] = None
            pat = re.compile(rf"{subject_id}_{exercise_id}_.*\.mat$", re.IGNORECASE)
            for member in zf.namelist():
                if pat.search(member):
                    target_name = member
                    break

            if not target_name:
                raise FileNotFoundError(
                    f"Recording file for {subject_id} and {exercise_id} not found in {zip_path.name}"
                )

            mat_bytes = zf.read(target_name)
            bio = io.BytesIO(mat_bytes)
            mat_dict = scipy.io.loadmat(bio)

        emg = mat_dict["emg"].astype(np.float32)
        stimulus = mat_dict["stimulus"].squeeze().astype(np.int8)
        repetition = mat_dict["repetition"].squeeze().astype(np.int8)
        restimulus = mat_dict["restimulus"].squeeze().astype(np.int8)
        rerepetition = mat_dict["rerepetition"].squeeze().astype(np.int8)

        if stimulus.ndim == 0:
            stimulus = np.atleast_1d(stimulus)
        if repetition.ndim == 0:
            repetition = np.atleast_1d(repetition)
        if restimulus.ndim == 0:
            restimulus = np.atleast_1d(restimulus)
        if rerepetition.ndim == 0:
            rerepetition = np.atleast_1d(rerepetition)

        # Use ProveOrQuarantineEngine to establish canonical timeline
        engine = ProveOrQuarantineEngine()
        alignment, aligned_dict = engine.evaluate_and_align(
            emg=emg,
            stimulus=stimulus,
            repetition=repetition,
            restimulus=restimulus,
            rerepetition=rerepetition,
            subject_id=subject_id,
            exercise_id=exercise_id,
        )

        provenance: Dict[str, Any] = {
            "dataset": NinaProProvenance.get_metadata().dataset_name,
            "archive_filename": zip_path.name,
            "internal_path": target_name,
            "nominal_sampling_rate_hz": 2000.0,
            "alignment": {
                "status": alignment.status.value,
                "delta_samples": alignment.delta_samples,
                "start_anchored_proven": alignment.start_anchored_proven,
                "notes": alignment.notes,
            },
        }

        return RecordingData(
            subject_id=subject_id,
            exercise_id=exercise_id,
            emg=aligned_dict["emg"],
            stimulus=aligned_dict["stimulus"],
            repetition=aligned_dict["repetition"],
            restimulus=aligned_dict["restimulus"],
            rerepetition=aligned_dict["rerepetition"],
            alignment=alignment,
            sampling_rate_hz=2000.0,
            provenance=provenance,
        )

