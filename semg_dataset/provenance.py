"""Official Provenance, Licensing Terms, and Cryptographic Baseline Verification.

Provides authoritative metadata, official citations, repository redistribution
terms, and SHA-256 integrity verification for the NinaPro DB2 dataset.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class ProvenanceMetadata:
    """Immutable representation of NinaPro DB2 dataset provenance and licensing."""

    dataset_name: str
    origin: str
    citation: str
    official_url: str
    license_terms: str
    redistribution_policy: str
    acquisition_system: str
    nominal_sampling_rate_hz: float
    disallow_real_fixtures: bool


class NinaProProvenance:
    """Authoritative provider for NinaPro DB2 provenance and baseline integrity."""

    _METADATA = ProvenanceMetadata(
        dataset_name="NinaPro DB2",
        origin="NinaPro Database 2 (DB2) — Ninapro project (Atzori et al., 2014)",
        citation=(
            "Atzori, M., Cognolato, M., & Müller, H. (2014). "
            "'Electromyography data for non-invasive classification of hand, wrist and finger movements.' "
            "Scientific Data, 1, 140053. DOI: 10.1038/sdata.2014.53"
        ),
        official_url="https://ninapro.hevs.ch/instructions/DB2.html",
        license_terms=(
            "Open access for academic and non-commercial scientific research with attribution. "
            "Redistribution of raw signals is subject to official dataset licensing terms."
        ),
        redistribution_policy=(
            "No real NinaPro DB2 raw samples may be redistributed or committed as test fixtures "
            "in this repository. All unit test fixtures MUST use synthetic NinaPro-like data."
        ),
        acquisition_system="Delsys Trigno Wireless sEMG System (12 channels, 2000 Hz nominal)",
        nominal_sampling_rate_hz=2000.0,
        disallow_real_fixtures=True,
    )

    @classmethod
    def get_metadata(cls) -> ProvenanceMetadata:
        """Return the authoritative, frozen provenance and licensing metadata."""
        return cls._METADATA

    @staticmethod
    def compute_file_sha256(filepath: Path) -> str:
        """Compute the SHA-256 digest of a file in 8MB chunks without loading it entirely."""
        hasher = hashlib.sha256()
        with open(filepath, "rb") as f:
            while chunk := f.read(8 * 1024 * 1024):
                hasher.update(chunk)
        return hasher.hexdigest()

    @classmethod
    def verify_archive_checksum(cls, archive_path: Path, expected_sha256: str) -> bool:
        """Verify whether an archive file matches the expected SHA-256 digest (fail-closed)."""
        if not archive_path.is_file():
            return False
        observed = cls.compute_file_sha256(archive_path)
        return observed.lower() == expected_sha256.lower()

    @staticmethod
    def load_baseline_hashes(inventory_path: Path) -> Dict[str, str]:
        """Load the baseline SHA-256 hashes from file_inventory.jsonl."""
        hashes: Dict[str, str] = {}
        if not inventory_path.is_file():
            raise FileNotFoundError(f"Baseline inventory not found: {inventory_path}")

        with open(inventory_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                filename = record.get("filename")
                sha256 = record.get("sha256")
                if filename and sha256:
                    hashes[filename] = sha256
        return hashes

    @classmethod
    def verify_all_archives(cls, raw_dir: Path, inventory_path: Path) -> Tuple[bool, List[str]]:
        """Verify all archives in raw_dir against the baseline inventory.
        
        Returns:
            Tuple[bool, List[str]]: (all_valid, list_of_failed_or_missing_filenames)
        """
        baseline = cls.load_baseline_hashes(inventory_path)
        failed: List[str] = []

        for filename, expected_hash in baseline.items():
            archive_path = raw_dir / filename
            if not archive_path.is_file():
                failed.append(f"{filename} (MISSING)")
            elif not cls.verify_archive_checksum(archive_path, expected_hash):
                failed.append(f"{filename} (CORRUPTED_HASH)")

        return (len(failed) == 0, failed)
