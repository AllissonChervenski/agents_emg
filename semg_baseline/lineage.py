"""ScientificLineage tracking and Feature 003 cryptographic verification.

Implements FR-001, FR-002, FR-003, SC-001, and Decision-5:
- Immutable lineage tracking linking TrainingJob to Feature 003 intake, view, alignment, and splits.
- Verification of Feature 003 frozen contracts against tampering.
- Deterministic extraction and generation of dataset view and manifest hashes.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
import subprocess
from typing import Any, Sequence


HEX64_REGEX = re.compile(r"^[a-f0-9]{64}$")


@dataclass(frozen=True)
class ScientificLineage:
    """Scientific lineage metadata sealing cryptographic provenance of training runs."""

    raw_dataset_manifest_hash: str
    dataset_view_hash: str
    alignment_hash: str
    split_manifest_hash: str
    acceptance_package_hash: str
    protocol_id: str = "within_subject"
    partition_id: str = ""
    git_commit: str = ""

    def __post_init__(self) -> None:
        """Validate hash formats according to JSON schema contracts."""
        for field_name in (
            "raw_dataset_manifest_hash",
            "dataset_view_hash",
            "split_manifest_hash",
            "acceptance_package_hash",
        ):
            val = getattr(self, field_name)
            if not isinstance(val, str) or not HEX64_REGEX.match(val):
                raise ValueError(
                    f"{field_name} must be a 64-character hexadecimal SHA-256 digest, got: '{val}'"
                )

        if not isinstance(self.alignment_hash, str) or not self.alignment_hash:
            raise ValueError("alignment_hash must be a non-empty string")

    def to_dict(self) -> dict[str, str]:
        """Serialize strictly to the 5 properties defined in training_job_schema.json."""
        return {
            "raw_dataset_manifest_hash": self.raw_dataset_manifest_hash,
            "dataset_view_hash": self.dataset_view_hash,
            "alignment_hash": self.alignment_hash,
            "split_manifest_hash": self.split_manifest_hash,
            "acceptance_package_hash": self.acceptance_package_hash,
        }

    def to_full_dict(self) -> dict[str, str]:
        """Serialize all lineage properties including protocol, partition, and commit."""
        base = self.to_dict()
        base["protocol_id"] = self.protocol_id
        base["partition_id"] = self.partition_id
        base["git_commit"] = self.git_commit
        return base


def compute_file_sha256(file_path: Path | str) -> str:
    """Compute cryptographic SHA-256 hash of a file reading in binary chunks."""
    path = Path(file_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"File not found for SHA-256 computation: {path}")

    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_feature_003_acceptance_hash(repo_root: Path | str | None = None) -> str:
    """Compute SHA-256 digest of Feature 003 acceptance manifest (reports/.../manifest.sha256)."""
    root = (
        Path(repo_root).resolve()
        if repo_root is not None
        else Path(__file__).resolve().parent.parent
    )
    manifest_path = root / "reports" / "feature_003_acceptance" / "manifest.sha256"
    return compute_file_sha256(manifest_path)


def get_raw_dataset_manifest_hash(repo_root: Path | str | None = None) -> str:
    """Compute SHA-256 digest of Feature 003 raw intake manifest (intake_manifest.json)."""
    root = (
        Path(repo_root).resolve()
        if repo_root is not None
        else Path(__file__).resolve().parent.parent
    )
    manifest_path = root / "data" / "manifests" / "ninapro_db2" / "intake_manifest.json"
    return compute_file_sha256(manifest_path)


def get_alignment_hash(repo_root: Path | str | None = None) -> str:
    """Compute SHA-256 digest of Feature 003 alignment audit evidence."""
    root = (
        Path(repo_root).resolve()
        if repo_root is not None
        else Path(__file__).resolve().parent.parent
    )
    audit_path = (
        root
        / "reports"
        / "feature_003_acceptance"
        / "raw"
        / "alignment_audit.jsonl"
    )
    return compute_file_sha256(audit_path)


def compute_dataset_view_hash(
    recordings: Sequence[Any], view_name: str = "refined"
) -> str:
    """Compute deterministic combined hash across views of a sequence of RecordingData objects."""
    parts: list[str] = []
    for rec in recordings:
        view = rec.get_view(view_name)
        subject_id = getattr(rec, "subject_id", "")
        exercise_id = getattr(rec, "exercise_id", "")
        parts.append(f"{subject_id}_{exercise_id}_{view_name}:{view.view_hash}")

    parts.sort()
    combined_str = "\n".join(parts)
    return hashlib.sha256(combined_str.encode("utf-8")).hexdigest()


def get_current_git_commit(repo_root: Path | str | None = None) -> str:
    """Retrieve current Git commit SHA-1 or fallback identifier."""
    cwd = (
        Path(repo_root).resolve()
        if repo_root is not None
        else Path(__file__).resolve().parent.parent
    )
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "unknown"


def verify_feature_003_frozen_contracts(
    repo_root: Path | str | None = None,
) -> tuple[bool, list[str]]:
    """Cryptographically verify that Feature 003 acceptance and intake contracts are intact."""
    root = (
        Path(repo_root).resolve()
        if repo_root is not None
        else Path(__file__).resolve().parent.parent
    )
    errors: list[str] = []

    acc_dir = root / "reports" / "feature_003_acceptance"
    manifest_file = acc_dir / "manifest.sha256"

    if not manifest_file.is_file():
        errors.append(f"Feature 003 acceptance manifest missing: {manifest_file}")
        return False, errors

    # Verify each entry in manifest.sha256
    content = manifest_file.read_text(encoding="utf-8")
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        expected_sha, rel_path = parts[0], parts[1].strip()
        target_path = acc_dir / rel_path
        if not target_path.is_file():
            errors.append(f"Accepted file referenced in manifest is missing: {target_path}")
            continue
        actual_sha = compute_file_sha256(target_path)
        if actual_sha != expected_sha:
            errors.append(
                f"Checksum mismatch for '{rel_path}': expected {expected_sha}, got {actual_sha}"
            )

    return len(errors) == 0, errors


def build_scientific_lineage(
    repo_root: Path | str | None = None,
    dataset_view_hash: str = "",
    split_manifest_hash: str = "",
    protocol_id: str = "within_subject",
    partition_id: str = "",
    git_commit: str | None = None,
) -> ScientificLineage:
    """Build a validated ScientificLineage instance from active repo contracts and input hashes."""
    root = (
        Path(repo_root).resolve()
        if repo_root is not None
        else Path(__file__).resolve().parent.parent
    )

    raw_hash = get_raw_dataset_manifest_hash(root)
    acc_hash = get_feature_003_acceptance_hash(root)
    align_hash = get_alignment_hash(root)
    commit = git_commit if git_commit is not None else get_current_git_commit(root)

    return ScientificLineage(
        raw_dataset_manifest_hash=raw_hash,
        dataset_view_hash=dataset_view_hash,
        alignment_hash=align_hash,
        split_manifest_hash=split_manifest_hash,
        acceptance_package_hash=acc_hash,
        protocol_id=protocol_id,
        partition_id=partition_id,
        git_commit=commit,
    )
