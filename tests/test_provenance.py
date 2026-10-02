"""Tests for NinaPro DB2 provenance, licensing terms, and SHA-256 verification."""

import hashlib
from pathlib import Path


def test_provenance_metadata_completeness_and_license() -> None:
    """Verify official provenance, citation, URL, and license terms are fully documented."""
    from semg_dataset.provenance import NinaProProvenance

    meta = NinaProProvenance.get_metadata()
    assert meta.dataset_name == "NinaPro DB2"
    assert "Atzori" in meta.citation
    assert "10.1038/sdata.2014.53" in meta.citation
    assert "https://ninapro.hevs.ch" in meta.official_url
    assert len(meta.license_terms) > 0
    assert "non-commercial" in meta.license_terms.lower()
    assert "TO_BE_DOCUMENTED" not in str(meta)
    assert meta.disallow_real_fixtures is True


def test_verify_archive_checksum_valid_and_tampered(tmp_path: Path) -> None:
    """Verify checksum verification succeeds on valid file and fails closed on tampering."""
    from semg_dataset.provenance import NinaProProvenance

    test_file = tmp_path / "dummy_archive.zip"
    content = b"Simulated archive content for checksum test"
    test_file.write_bytes(content)
    expected_sha256 = hashlib.sha256(content).hexdigest()

    # Valid check
    assert NinaProProvenance.verify_archive_checksum(test_file, expected_sha256) is True

    # Tampered check (fail-closed)
    tampered_sha256 = "0" * 64
    assert NinaProProvenance.verify_archive_checksum(test_file, tampered_sha256) is False


def test_load_baseline_hashes_from_inventory(tmp_path: Path) -> None:
    """Verify loading baseline hashes from file inventory JSONL."""
    import json
    from semg_dataset.provenance import NinaProProvenance

    inv_file = tmp_path / "test_inventory.jsonl"
    entries = [
        {"filename": "DB2_s1.zip", "sha256": "abc12345" + "0" * 56},
        {"filename": "DB2_s2.zip", "sha256": "def67890" + "0" * 56},
    ]
    inv_file.write_text("\n".join(json.dumps(e) for e in entries) + "\n", encoding="utf-8")

    hashes = NinaProProvenance.load_baseline_hashes(inv_file)
    assert len(hashes) == 2
    assert hashes["DB2_s1.zip"] == "abc12345" + "0" * 56
    assert hashes["DB2_s2.zip"] == "def67890" + "0" * 56
