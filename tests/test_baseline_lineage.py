"""Unit tests for ScientificLineage tracking and Feature 003 cryptographic verification (FR-001, FR-002, FR-003, SC-001).

Validates:
- ScientificLineage dataclass contracts conforming to training_job_schema.json.
- Deterministic extraction of raw dataset manifest, alignment, and Feature 003 acceptance hashes.
- Deterministic dataset view hash computation (refined vs stimulus isolation).
- Verification of Feature 003 frozen contracts against tampering.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest
import jsonschema

from semg_baseline.lineage import (
    ScientificLineage,
    build_scientific_lineage,
    compute_dataset_view_hash,
    get_alignment_hash,
    get_feature_003_acceptance_hash,
    get_raw_dataset_manifest_hash,
    verify_feature_003_frozen_contracts,
)
from tests.fixtures.synthetic_semg import make_synthetic_recording


REPO_ROOT = Path(__file__).resolve().parent.parent
TRAINING_JOB_SCHEMA_PATH = (
    REPO_ROOT
    / "specs"
    / "004-baseline-training-and-evaluation"
    / "contracts"
    / "training_job_schema.json"
)


def _load_lineage_schema() -> dict:
    with open(TRAINING_JOB_SCHEMA_PATH, "r", encoding="utf-8") as f:
        schema = json.load(f)
    return schema["$defs"]["ScientificLineage"]


def test_scientific_lineage_dataclass_contract_and_schema_validation() -> None:
    """ScientificLineage.to_dict() must strictly validate against training_job_schema.json."""
    lineage = ScientificLineage(
        raw_dataset_manifest_hash="e3a1fa0b1a40e610c37474bcf1de0aeb705fe1e13a1f037f7fe76b71aeada1bc",
        dataset_view_hash="0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        alignment_hash="434ba6578dca1630764cdd4972d20b2aa37b1b010779dab709b0df64f4e94a83",
        split_manifest_hash="fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210",
        acceptance_package_hash="12fc0a35131545c6f8ea08fea7682a6e211c17b3fea4c1963082241360fcc9e6",
        protocol_id="within_subject",
        partition_id="S1",
        git_commit="79a1880",
    )

    contract_dict = lineage.to_dict()
    assert len(contract_dict) == 5

    lineage_schema = _load_lineage_schema()
    # jsonschema validate raises ValidationError if invalid
    jsonschema.validate(instance=contract_dict, schema=lineage_schema)

    full_dict = lineage.to_full_dict()
    assert full_dict["protocol_id"] == "within_subject"
    assert full_dict["partition_id"] == "S1"
    assert full_dict["git_commit"] == "79a1880"


def test_scientific_lineage_invalid_hex_raises_value_error() -> None:
    """Invalid hash formats must be rejected by ScientificLineage validation."""
    with pytest.raises(ValueError, match="raw_dataset_manifest_hash must be a 64-character"):
        ScientificLineage(
            raw_dataset_manifest_hash="short_hash",
            dataset_view_hash="0" * 64,
            alignment_hash="0" * 64,
            split_manifest_hash="0" * 64,
            acceptance_package_hash="0" * 64,
        )


def test_get_feature_003_acceptance_hash_against_repo() -> None:
    """Feature 003 acceptance package hash must match the frozen manifest.sha256."""
    acc_hash = get_feature_003_acceptance_hash(repo_root=REPO_ROOT)
    assert len(acc_hash) == 64
    assert acc_hash == "12fc0a35131545c6f8ea08fea7682a6e211c17b3fea4c1963082241360fcc9e6"


def test_get_raw_dataset_manifest_hash_against_repo() -> None:
    """Raw dataset manifest hash must match intake_manifest.json."""
    raw_hash = get_raw_dataset_manifest_hash(repo_root=REPO_ROOT)
    assert len(raw_hash) == 64
    assert raw_hash == "e3a1fa0b1a40e610c37474bcf1de0aeb705fe1e13a1f037f7fe76b71aeada1bc"


def test_get_alignment_hash_against_repo() -> None:
    """Alignment hash must match the proven Feature 003 alignment audit record."""
    align_hash = get_alignment_hash(repo_root=REPO_ROOT)
    assert len(align_hash) == 64
    assert align_hash == "434ba6578dca1630764cdd4972d20b2aa37b1b010779dab709b0df64f4e94a83"


def test_compute_dataset_view_hash_refined_vs_stimulus() -> None:
    """Refined and stimulus views must generate distinct, deterministic 64-char hashes."""
    rec1 = make_synthetic_recording(seed=101, subject_id="S1")
    rec2 = make_synthetic_recording(seed=102, subject_id="S2")

    hash_refined = compute_dataset_view_hash([rec1, rec2], view_name="refined")
    hash_refined_repeat = compute_dataset_view_hash([rec1, rec2], view_name="refined")
    assert len(hash_refined) == 64
    assert hash_refined == hash_refined_repeat

    hash_stimulus = compute_dataset_view_hash([rec1, rec2], view_name="stimulus")
    assert len(hash_stimulus) == 64
    assert hash_refined != hash_stimulus


def test_verify_feature_003_frozen_contracts_success() -> None:
    """Verification of active Feature 003 frozen contracts in repo must succeed."""
    is_valid, errors = verify_feature_003_frozen_contracts(repo_root=REPO_ROOT)
    assert is_valid is True
    assert errors == []


def test_verify_feature_003_frozen_contracts_tampering_detection(tmp_path: Path) -> None:
    """Tampering with an acceptance manifest or raw manifest must trigger verification failure."""
    # Create mock repo with tampered manifest
    mock_reports = tmp_path / "reports" / "feature_003_acceptance"
    mock_reports.mkdir(parents=True)
    manifest_sha256 = mock_reports / "manifest.sha256"
    manifest_sha256.write_text("0" * 64 + "  fake_file.txt\n", encoding="utf-8")

    is_valid, errors = verify_feature_003_frozen_contracts(repo_root=tmp_path)
    assert is_valid is False
    assert len(errors) > 0


def test_build_scientific_lineage_integration() -> None:
    """build_scientific_lineage must construct a valid, verified ScientificLineage instance."""
    rec = make_synthetic_recording(seed=42)
    view_hash = compute_dataset_view_hash([rec], view_name="refined")
    split_hash = "a" * 64

    lineage = build_scientific_lineage(
        repo_root=REPO_ROOT,
        dataset_view_hash=view_hash,
        split_manifest_hash=split_hash,
        protocol_id="within_subject",
        partition_id="S1",
    )

    assert lineage.raw_dataset_manifest_hash == "e3a1fa0b1a40e610c37474bcf1de0aeb705fe1e13a1f037f7fe76b71aeada1bc"
    assert lineage.acceptance_package_hash == "12fc0a35131545c6f8ea08fea7682a6e211c17b3fea4c1963082241360fcc9e6"
    assert lineage.alignment_hash == "434ba6578dca1630764cdd4972d20b2aa37b1b010779dab709b0df64f4e94a83"
    assert lineage.dataset_view_hash == view_hash
    assert lineage.split_manifest_hash == split_hash
    assert lineage.protocol_id == "within_subject"
    assert lineage.partition_id == "S1"

    # Schema validation
    jsonschema.validate(instance=lineage.to_dict(), schema=_load_lineage_schema())
