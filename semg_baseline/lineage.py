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
from typing import Any, Iterable, Sequence


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
    recordings: Iterable[Any], view_name: str = "refined"
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
    *,
    expected_acceptance_sha256: str | None = None,
    required_entries: Sequence[str] = (),
) -> tuple[bool, list[str]]:
    """Check an acceptance manifest fail-closed and optionally pin its digest.

    For scientific/real mode, ``expected_acceptance_sha256`` MUST come from a
    trusted, independently pinned Feature 003 artifact, not from the mutable
    manifest being verified. Otherwise this proves internal consistency only.
    """
    root = (
        Path(repo_root).resolve()
        if repo_root is not None
        else Path(__file__).resolve().parent.parent
    )
    errors: list[str] = []
    acc_dir = (root / "reports" / "feature_003_acceptance").resolve()
    manifest_file = acc_dir / "manifest.sha256"

    if not manifest_file.is_file():
        return False, [f"Feature 003 acceptance manifest missing: {manifest_file}"]

    if expected_acceptance_sha256 is not None:
        if not HEX64_REGEX.fullmatch(expected_acceptance_sha256):
            errors.append("Invalid trusted acceptance SHA-256 fingerprint")
        elif compute_file_sha256(manifest_file) != expected_acceptance_sha256:
            errors.append("Feature 003 acceptance manifest digest differs from trusted fingerprint")

    content = manifest_file.read_text(encoding="utf-8")
    seen: set[str] = set()
    for number, raw_line in enumerate(content.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split(None, 1)
        if len(fields) != 2 or not HEX64_REGEX.fullmatch(fields[0]):
            errors.append(f"Malformed acceptance manifest entry at line {number}")
            continue
        expected_sha, rel_str = fields
        # Support the `sha256sum` text and binary-file markers.
        rel_str = rel_str.removeprefix("*").strip()
        rel = Path(rel_str)
        if not rel_str or rel.is_absolute() or ".." in rel.parts:
            errors.append(f"Unsafe acceptance manifest path at line {number}: {rel_str!r}")
            continue
        if rel.as_posix() in seen:
            errors.append(f"Duplicate acceptance manifest entry: {rel.as_posix()}")
            continue
        seen.add(rel.as_posix())
        path = (acc_dir / rel).resolve()
        if not path.is_relative_to(acc_dir):
            errors.append(f"Acceptance manifest path escapes root: {rel_str!r}")
            continue
        if not path.is_file():
            errors.append(f"Accepted file referenced in manifest is missing: {path}")
            continue
        if compute_file_sha256(path) != expected_sha:
            errors.append(f"Checksum mismatch for '{rel_str}'")

    if not seen:
        errors.append("Feature 003 acceptance manifest has no valid file entries")
    for entry in required_entries:
        if entry not in seen:
            errors.append(f"Required acceptance artifact not sealed by manifest: {entry}")
    return not errors, errors


def build_verified_scientific_lineage(
    repo_root: Path | str,
    *,
    recordings: Iterable[Any],
    split_manifest_path: Path | str,
    expected_dataset_view_sha256: str,
    expected_split_manifest_sha256: str,
    expected_acceptance_sha256: str,
    expected_raw_manifest_sha256: str,
    expected_alignment_sha256: str,
    protocol_id: str,
    partition_id: str,
    git_commit: str | None = None,
) -> ScientificLineage:
    """Construct REAL benchmark lineage exclusively from verified artifacts.

    Every ``expected_*`` SHA-256 must be pinned in an authoritative frozen
    Feature 003 source. An audit-file SHA-256 is *not* interchangeable with a
    split-manifest digest or with the combined dataset-view content digest.
    ``recordings`` must cover the same frozen view population as the trusted
    view fingerprint (normally all accepted E1/E2/E3 recordings).
    """
    root = Path(repo_root).resolve()
    expected = {
        "dataset view": expected_dataset_view_sha256,
        "split manifest": expected_split_manifest_sha256,
        "acceptance": expected_acceptance_sha256,
        "raw manifest": expected_raw_manifest_sha256,
        "alignment": expected_alignment_sha256,
    }
    for name, sha in expected.items():
        if not HEX64_REGEX.fullmatch(sha):
            raise ValueError(f"Invalid trusted {name} SHA-256")

    if protocol_id == "within_subject":
        valid_partition = re.fullmatch(r"S(?:[1-9]|[1-3][0-9]|40)", partition_id)
    elif protocol_id == "cross_subject":
        valid_partition = re.fullmatch(r"fold_[1-5]", partition_id)
    else:
        raise ValueError(f"Unknown protocol_id: {protocol_id!r}")
    if valid_partition is None:
        raise ValueError(f"Invalid {protocol_id} partition: {partition_id!r}")

    valid, errors = verify_feature_003_frozen_contracts(
        root, expected_acceptance_sha256=expected_acceptance_sha256
    )
    if not valid:
        raise RuntimeError("Frozen acceptance verification failed: " + "; ".join(errors))

    split_path = Path(split_manifest_path)
    if not split_path.is_absolute():
        split_path = root / split_path
    split_path = split_path.resolve()
    if not split_path.is_relative_to(root):
        raise ValueError("Split manifest must be located within the repository")

    actual = {
        "dataset view": compute_dataset_view_hash(recordings, "refined"),
        "split manifest": compute_file_sha256(split_path),
        "raw manifest": get_raw_dataset_manifest_hash(root),
        "alignment": get_alignment_hash(root),
    }
    for name in actual:
        if actual[name] != expected[name]:
            raise RuntimeError(f"Verified {name} digest does not match frozen Feature 003 fingerprint")

    commit = git_commit if git_commit is not None else get_current_git_commit(root)
    if not commit or re.fullmatch(r"[a-f0-9]{40}|[a-f0-9]{64}", commit) is None:
        raise RuntimeError("Real benchmark requires a valid Git commit SHA")
    return ScientificLineage(
        raw_dataset_manifest_hash=actual["raw manifest"],
        dataset_view_hash=actual["dataset view"],
        alignment_hash=actual["alignment"],
        split_manifest_hash=actual["split manifest"],
        acceptance_package_hash=expected_acceptance_sha256,
        protocol_id=protocol_id,
        partition_id=partition_id,
        git_commit=commit,
    )


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
