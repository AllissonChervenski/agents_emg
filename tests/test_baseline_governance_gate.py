"""Unit and integration tests for dependency governance gate (FR-006, SC-001, Decision-1).

Verifies ADR lifecycle (PROPOSED -> CONDITIONAL_PASS, APPROVED -> PASS),
human signature presence, cryptographic SHA-256 fingerprinting, pyproject.toml
dependency containment, and AST import path/API surface allowlists.
"""

from __future__ import annotations

from pathlib import Path
import pytest

from semg_baseline.governance import (
    ADRRecord,
    GovernanceReport,
    GovernanceStatus,
    GovernanceViolation,
    check_dependency_governance,
    compute_adr_fingerprint,
    is_sklearn_allowed,
    assert_sklearn_allowed,
    parse_adr,
    validate_adr,
    verify_imports_in_source,
    verify_pyproject_dependencies,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
ADR_PATH = REPO_ROOT / "docs" / "adr" / "ADR-004-001-scikit-learn-host-only.md"
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"


def test_adr_004_001_parses_current_proposed_state() -> None:
    """ADR-004-001 must parse accurately and show current PROPOSED status."""
    assert ADR_PATH.is_file(), f"ADR file missing at {ADR_PATH}"
    record = parse_adr(ADR_PATH)

    assert record.adr_id == "ADR-004-001"
    assert record.status == "PROPOSED"
    assert record.proposed_date == "2026-10-03"
    assert record.approver == "[PENDING HUMAN APPROVAL]"
    assert record.approval_date == "[PENDING HUMAN APPROVAL]"
    assert record.git_commit == "[PENDING HUMAN COMMIT]"
    assert record.fingerprint == "[PENDING LOCK]"


def test_adr_validation_proposed_yields_conditional_pass() -> None:
    """A PROPOSED ADR validates to CONDITIONAL_PASS with sklearn execution blocked."""
    record = parse_adr(ADR_PATH)
    is_valid, errors, status = validate_adr(record)

    assert is_valid is True
    assert errors == []
    assert status == GovernanceStatus.CONDITIONAL_PASS


def test_approved_adr_validation_success_with_valid_fingerprint(tmp_path: Path) -> None:
    """An APPROVED ADR with valid human signatures and matching SHA-256 passes."""
    adr_content_template = """# Architecture Decision Record (ADR)

## ADR-004-001: Aprovação de Governança para Dependência Host-Only `scikit-learn`

- **ID**: `ADR-004-001`
- **Status**: `APPROVED`
- **Data de Proposição**: 2026-10-03
- **Aprovador Humano**: `Allisson (Lead Scientist)`
- **Data de Aprovação Humana**: 2026-10-06
- **Commit Git de Aprovação**: `79a1880f1234567890abcdef1234567890abcdef`
- **Fingerprint Criptográfico (SHA-256 pós-aprovação)**: [PENDING LOCK]

---

## 1. Contexto & Fundamentação Constitucional
Approved content here.
"""
    expected_fp = compute_adr_fingerprint(adr_content_template)
    final_adr_content = adr_content_template.replace("[PENDING LOCK]", expected_fp)

    test_adr_file = tmp_path / "ADR-004-001-test.md"
    test_adr_file.write_text(final_adr_content, encoding="utf-8")

    record = parse_adr(test_adr_file)
    assert record.status == "APPROVED"
    assert record.fingerprint == expected_fp

    is_valid, errors, status = validate_adr(record)
    assert is_valid is True
    assert errors == []
    assert status == GovernanceStatus.APPROVED


def test_approved_adr_with_pending_approver_fails(tmp_path: Path) -> None:
    """An ADR marked APPROVED but with pending human approver must trigger FAIL."""
    content = """# ADR
- **ID**: `ADR-004-001`
- **Status**: `APPROVED`
- **Data de Proposição**: 2026-10-03
- **Aprovador Humano**: `[PENDING HUMAN APPROVAL]`
- **Data de Aprovação Humana**: 2026-10-06
- **Commit Git de Aprovação**: `abcdef123456`
- **Fingerprint Criptográfico (SHA-256 pós-aprovação)**: 1234567890abcdef
"""
    test_file = tmp_path / "adr_pending_approver.md"
    test_file.write_text(content, encoding="utf-8")

    record = parse_adr(test_file)
    is_valid, errors, status = validate_adr(record)
    assert is_valid is False
    assert status == GovernanceStatus.FAIL
    assert any("pending human approval" in e.lower() for e in errors)


def test_approved_adr_with_fingerprint_mismatch_fails(tmp_path: Path) -> None:
    """An ADR marked APPROVED with tampered content must fail cryptographic lock."""
    content = """# ADR
- **ID**: `ADR-004-001`
- **Status**: `APPROVED`
- **Data de Proposição**: 2026-10-03
- **Aprovador Humano**: `Allisson`
- **Data de Aprovação Humana**: 2026-10-06
- **Commit Git de Aprovação**: `abcdef123456`
- **Fingerprint Criptográfico (SHA-256 pós-aprovação)**: deadbeef00000000000000000000000000000000000000000000000000000000
"""
    test_file = tmp_path / "adr_tampered.md"
    test_file.write_text(content, encoding="utf-8")

    record = parse_adr(test_file)
    is_valid, errors, status = validate_adr(record)
    assert is_valid is False
    assert status == GovernanceStatus.FAIL
    assert any("fingerprint mismatch" in e.lower() for e in errors)


def test_pyproject_clean_baseline_passes_conditional() -> None:
    """Current pyproject.toml without scikit-learn passes under PROPOSED ADR."""
    record = parse_adr(ADR_PATH)
    is_valid, errors = verify_pyproject_dependencies(PYPROJECT_PATH, record)
    assert is_valid is True
    assert errors == []


def test_pyproject_with_unapproved_sklearn_fails_under_proposed_adr(tmp_path: Path) -> None:
    """If pyproject.toml declares scikit-learn while ADR is PROPOSED, gate triggers FAIL."""
    pyproject_tampered = """[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "agents-emg"
version = "0.1.0"
dependencies = [
    "numpy>=1.20",
    "scipy>=1.7",
    "scikit-learn>=1.5.0",
]
"""
    test_pyproject = tmp_path / "pyproject.toml"
    test_pyproject.write_text(pyproject_tampered, encoding="utf-8")

    record = parse_adr(ADR_PATH)  # PROPOSED
    is_valid, errors = verify_pyproject_dependencies(test_pyproject, record)
    assert is_valid is False
    assert any("without approved governance adr" in e.lower() for e in errors)


def test_pyproject_with_unauthorized_third_party_package_fails(tmp_path: Path) -> None:
    """Declaring an unapproved dependency (e.g., torch) triggers immediate FAIL."""
    pyproject_tampered = """[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "agents-emg"
version = "0.1.0"
dependencies = [
    "numpy>=1.20",
    "scipy>=1.7",
    "torch>=2.0",
]
"""
    test_pyproject = tmp_path / "pyproject.toml"
    test_pyproject.write_text(pyproject_tampered, encoding="utf-8")

    record = parse_adr(ADR_PATH)
    is_valid, errors = verify_pyproject_dependencies(test_pyproject, record)
    assert is_valid is False
    assert any("unauthorized dependency" in e.lower() for e in errors)


def test_pyproject_with_approved_sklearn_passes(tmp_path: Path) -> None:
    """If ADR is APPROVED and scikit-learn is declared in valid range, passes."""
    pyproject_content = """[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "agents-emg"
version = "0.1.0"
dependencies = [
    "numpy>=1.20",
    "scipy>=1.7",
]

[project.optional-dependencies]
dev = ["pytest>=7", "PyYAML>=6", "ruff", "mypy"]
baseline = ["scikit-learn>=1.5.0,<2.0.0"]
"""
    test_pyproject = tmp_path / "pyproject.toml"
    test_pyproject.write_text(pyproject_content, encoding="utf-8")

    # Mock approved record
    approved_record = ADRRecord(
        adr_id="ADR-004-001",
        status="APPROVED",
        proposed_date="2026-10-03",
        approver="Allisson",
        approval_date="2026-10-06",
        git_commit="79a1880f",
        fingerprint="dummy_fp",
        raw_content="...",
        file_path=ADR_PATH,
    )

    is_valid, errors = verify_pyproject_dependencies(test_pyproject, approved_record)
    assert is_valid is True
    assert errors == []


def test_import_scanner_detects_scope_violation_in_forbidden_paths() -> None:
    """Importing sklearn inside semg_dsp or semg_dataset must be flagged as SCOPE_VIOLATION."""
    forbidden_dsp_code = """import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
"""
    is_valid, violations = verify_imports_in_source(
        source_code=forbidden_dsp_code,
        file_path="semg_dsp/illegal_filter.py",
    )
    assert is_valid is False
    assert any("SCOPE_VIOLATION" in v for v in violations)


def test_import_scanner_detects_prohibited_submodules() -> None:
    """Importing prohibited submodules (model_selection, pipeline, StandardScaler) fails."""
    code_model_selection = "from sklearn.model_selection import KFold\n"
    is_valid1, violations1 = verify_imports_in_source(
        source_code=code_model_selection,
        file_path="semg_baseline/lda.py",
    )
    assert is_valid1 is False
    assert any("API_VIOLATION" in v for v in violations1)

    code_scaler = "from sklearn.preprocessing import StandardScaler\n"
    is_valid2, violations2 = verify_imports_in_source(
        source_code=code_scaler,
        file_path="semg_baseline/scaling.py",
    )
    assert is_valid2 is False
    assert any("API_VIOLATION" in v for v in violations2)


def test_import_scanner_permits_allowlisted_sklearn_modules() -> None:
    """Importing allowed submodules in semg_baseline is permitted by import rules."""
    allowed_code = """from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.covariance import LedoitWolf
from sklearn.metrics import accuracy_score, confusion_matrix
"""
    is_valid, violations = verify_imports_in_source(
        source_code=allowed_code,
        file_path="semg_baseline/lda.py",
    )
    assert is_valid is True
    assert violations == []


def test_active_repo_dependency_governance_gate() -> None:
    """The full gate on the active repository must return CONDITIONAL_PASS and block sklearn."""
    report: GovernanceReport = check_dependency_governance(repo_root=REPO_ROOT)

    assert report.status == GovernanceStatus.CONDITIONAL_PASS
    assert report.adr_status == "PROPOSED"
    assert report.sklearn_permitted is False
    assert report.errors == []

    assert is_sklearn_allowed(repo_root=REPO_ROOT) is False

    with pytest.raises(GovernanceViolation, match="BLOCKED"):
        assert_sklearn_allowed(repo_root=REPO_ROOT)
