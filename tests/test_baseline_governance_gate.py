"""Unit and integration tests for dependency governance gate (FR-006, SC-001, Decision-1).

Verifies:
- ADR lifecycle across PROPOSED -> CONDITIONAL_PASS, APPROVED Stage A -> APPROVED_PENDING_LOCK,
  and APPROVED Stage B -> APPROVED (PASS).
- Non-circular two-stage approval workflow with cryptographic SHA-256 fingerprinting.
- Positive path allowlists (semg_baseline/**, tests/test_baseline_*.py, tests/feature_004/**)
  with explicit negative scope violation tests for orchestrator, scripts, semg_dsp, semg_dataset, etc.
- Strict symbol-level API surface allowlists and wildcard/bare import prohibitions.
- Exact version range enforcement (>=1.5.0, <2.0.0).
- pyproject.toml containment and active repository gate compliance.
"""

from __future__ import annotations

from pathlib import Path
import pytest

from semg_baseline.governance import (
    ADRRecord,
    GovernanceStatus,
    GovernanceViolation,
    check_dependency_governance,
    compute_adr_fingerprint,
    is_path_allowlisted_for_sklearn,
    is_sklearn_allowed,
    assert_sklearn_allowed,
    parse_adr,
    validate_adr,
    validate_sklearn_version_spec,
    verify_imports_in_source,
    verify_pyproject_dependencies,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
ADR_PATH = REPO_ROOT / "docs" / "adr" / "ADR-004-001-scikit-learn-host-only.md"
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"


# =========================================================================
# 1. Lifecycle Unit Tests (Using Isolated ADR Templates)
# =========================================================================

ADR_TEMPLATE = """# Architecture Decision Record (ADR)
## ADR-004-001: Aprovação de Governança para Dependência Host-Only `scikit-learn`
- **ID**: `ADR-004-001`
- **Status**: {status}
- **Data de Proposição**: 2026-10-03
- **Aprovador Humano**: {approver}
- **Data de Aprovação Humana**: {approval_date}
- **Commit Git de Aprovação**: {git_commit}
- **Fingerprint Criptográfico**: {fingerprint}

## Contexto
Baseline regularized LDA evaluation.
"""


def test_adr_lifecycle_proposed(tmp_path: Path) -> None:
    """A PROPOSED ADR validates to CONDITIONAL_PASS with sklearn execution blocked."""
    adr_file = tmp_path / "ADR.md"
    adr_file.write_text(
        ADR_TEMPLATE.format(
            status="PROPOSED",
            approver="[PENDING HUMAN APPROVAL]",
            approval_date="[PENDING HUMAN APPROVAL]",
            git_commit="[PENDING HUMAN COMMIT]",
            fingerprint="[PENDING LOCK]",
        ),
        encoding="utf-8",
    )

    record = parse_adr(adr_file)
    is_valid, errors, status = validate_adr(record)

    assert is_valid is True
    assert errors == []
    assert status == GovernanceStatus.CONDITIONAL_PASS


def test_adr_lifecycle_approved_pending_commit_stage_a(tmp_path: Path) -> None:
    """An APPROVED ADR with human signature but pending commit (Stage A) yields APPROVED_PENDING_LOCK."""
    adr_file = tmp_path / "ADR.md"
    adr_file.write_text(
        ADR_TEMPLATE.format(
            status="APPROVED",
            approver="Human Evaluator <eval@example.com>",
            approval_date="2026-10-06",
            git_commit="[PENDING COMMIT]",
            fingerprint="[PENDING LOCK]",
        ),
        encoding="utf-8",
    )

    record = parse_adr(adr_file)
    is_valid, errors, status = validate_adr(record)

    assert is_valid is True
    assert errors == []
    assert status == GovernanceStatus.APPROVED_PENDING_LOCK


def test_adr_lifecycle_approved_and_locked_stage_b(tmp_path: Path) -> None:
    """An APPROVED ADR with recorded commit and matching SHA-256 fingerprint (Stage B) passes."""
    content_draft = ADR_TEMPLATE.format(
        status="APPROVED",
        approver="Human Evaluator <eval@example.com>",
        approval_date="2026-10-06",
        git_commit="1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b",
        fingerprint="[PENDING LOCK]",
    )
    fp = compute_adr_fingerprint(content_draft)

    locked_content = ADR_TEMPLATE.format(
        status="APPROVED",
        approver="Human Evaluator <eval@example.com>",
        approval_date="2026-10-06",
        git_commit="1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b",
        fingerprint=fp,
    )
    adr_file = tmp_path / "ADR.md"
    adr_file.write_text(locked_content, encoding="utf-8")

    record = parse_adr(adr_file)
    is_valid, errors, status = validate_adr(record)

    assert is_valid is True
    assert errors == []
    assert status == GovernanceStatus.APPROVED


def test_adr_lifecycle_tampered_fails(tmp_path: Path) -> None:
    """An APPROVED ADR with mismatched fingerprint fails validation."""
    adr_file = tmp_path / "ADR.md"
    adr_file.write_text(
        ADR_TEMPLATE.format(
            status="APPROVED",
            approver="Human Evaluator <eval@example.com>",
            approval_date="2026-10-06",
            git_commit="1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b",
            fingerprint="deadbeef" * 8,
        ),
        encoding="utf-8",
    )

    record = parse_adr(adr_file)
    is_valid, errors, status = validate_adr(record)

    assert is_valid is False
    assert status == GovernanceStatus.FAIL
    assert any("fingerprint mismatch" in err for err in errors)


def test_adr_lifecycle_missing_approver_fails(tmp_path: Path) -> None:
    """An ADR marked APPROVED with pending approver fails validation."""
    adr_file = tmp_path / "ADR.md"
    adr_file.write_text(
        ADR_TEMPLATE.format(
            status="APPROVED",
            approver="[PENDING HUMAN APPROVAL]",
            approval_date="2026-10-06",
            git_commit="1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b",
            fingerprint="abc",
        ),
        encoding="utf-8",
    )

    record = parse_adr(adr_file)
    is_valid, errors, status = validate_adr(record)

    assert is_valid is False
    assert status == GovernanceStatus.FAIL
    assert any("approver is pending" in err for err in errors)


# =========================================================================
# 2. Positive Path Allowlist Tests
# =========================================================================

def test_positive_path_allowlist() -> None:
    """Only semg_baseline/**, tests/test_baseline_*.py, and tests/feature_004/** are allowlisted."""
    # Positive allowlist
    assert is_path_allowlisted_for_sklearn("semg_baseline/lda.py") is True
    assert is_path_allowlisted_for_sklearn("semg_baseline/evaluator.py") is True
    assert is_path_allowlisted_for_sklearn("tests/test_baseline_lda.py") is True
    assert is_path_allowlisted_for_sklearn("tests/test_baseline_governance_gate.py") is True
    assert is_path_allowlisted_for_sklearn("tests/feature_004/test_something.py") is True
    assert is_path_allowlisted_for_sklearn("./semg_baseline/runner.py") is True

    # Negative blacklist / non-allowlisted paths
    assert is_path_allowlisted_for_sklearn("orchestrator/pipeline.py") is False
    assert is_path_allowlisted_for_sklearn("orchestrator/state_machine.py") is False
    assert is_path_allowlisted_for_sklearn("scripts/evaluate.py") is False
    assert is_path_allowlisted_for_sklearn("semg_dsp/filter.py") is False
    assert is_path_allowlisted_for_sklearn("semg_dataset/contract.py") is False
    assert is_path_allowlisted_for_sklearn("tests/test_dataset_contract.py") is False
    assert is_path_allowlisted_for_sklearn("tests/test_dsp_pipeline.py") is False
    assert is_path_allowlisted_for_sklearn("other_module/model.py") is False


def test_import_scanner_detects_scope_violations_outside_allowlist() -> None:
    """Importing sklearn outside positive allowlist triggers SCOPE_VIOLATION."""
    code = "from sklearn.metrics import accuracy_score"

    # Permitted in semg_baseline
    valid, errors = verify_imports_in_source(code, "semg_baseline/evaluator.py")
    assert valid is True
    assert errors == []

    # Prohibited in orchestrator
    valid, errors = verify_imports_in_source(code, "orchestrator/eval.py")
    assert valid is False
    assert any("SCOPE_VIOLATION" in err for err in errors)

    # Prohibited in scripts
    valid, errors = verify_imports_in_source(code, "scripts/run.py")
    assert valid is False
    assert any("SCOPE_VIOLATION" in err for err in errors)

    # Prohibited in semg_dsp
    valid, errors = verify_imports_in_source(code, "semg_dsp/filter.py")
    assert valid is False
    assert any("SCOPE_VIOLATION" in err for err in errors)

    # Prohibited in non-baseline tests
    valid, errors = verify_imports_in_source(code, "tests/test_dsp_filter.py")
    assert valid is False
    assert any("SCOPE_VIOLATION" in err for err in errors)


# =========================================================================
# 3. Strict Symbol Allowlist Tests
# =========================================================================

def test_import_scanner_symbol_allowlist_enforcement() -> None:
    """Only allowlisted symbols from approved submodules are permitted."""
    path = "semg_baseline/lda.py"

    # Valid: LinearDiscriminantAnalysis
    valid, errs = verify_imports_in_source(
        "from sklearn.discriminant_analysis import LinearDiscriminantAnalysis", path
    )
    assert valid is True
    assert errs == []

    # Valid: Metrics functions
    valid, errs = verify_imports_in_source(
        "from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score",
        path,
    )
    assert valid is True
    assert errs == []

    # Valid: Covariance functions
    valid, errs = verify_imports_in_source(
        "from sklearn.covariance import ledoit_wolf, LedoitWolf", path
    )
    assert valid is True
    assert errs == []

    # Prohibited: unauthorized symbol in metrics
    valid, errs = verify_imports_in_source(
        "from sklearn.metrics import make_scorer", path
    )
    assert valid is False
    assert any("API_VIOLATION: Symbol 'make_scorer'" in err for err in errs)

    # Prohibited: unauthorized symbol in discriminant_analysis
    valid, errs = verify_imports_in_source(
        "from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis", path
    )
    assert valid is False
    assert any("API_VIOLATION: Symbol 'QuadraticDiscriminantAnalysis'" in err for err in errs)

    # Prohibited: Wildcard import
    valid, errs = verify_imports_in_source(
        "from sklearn.metrics import *", path
    )
    assert valid is False
    assert any("Wildcard import '*'" in err for err in errs)

    # Prohibited: Bare import
    valid, errs = verify_imports_in_source("import sklearn", path)
    assert valid is False
    assert any("Bare 'import sklearn' prohibited" in err for err in errs)

    # Prohibited: Forbidden submodule
    valid, errs = verify_imports_in_source(
        "from sklearn.model_selection import KFold", path
    )
    assert valid is False
    assert any("Prohibited submodule 'sklearn.model_selection'" in err for err in errs)


# =========================================================================
# 4. Version Range Specification Tests
# =========================================================================

def test_version_range_specifier_validation() -> None:
    """Version validator strictly enforces both lower bound >=1.5.0 and upper bound <2.0.0."""
    # Positive cases
    valid, err = validate_sklearn_version_spec("scikit-learn>=1.5.0,<2.0.0")
    assert valid is True and err == ""

    valid, err = validate_sklearn_version_spec("scikit-learn >= 1.5.0, < 2.0.0")
    assert valid is True and err == ""

    valid, err = validate_sklearn_version_spec("scikit-learn>=1.5,<2.0")
    assert valid is True and err == ""

    # Negative cases: missing upper bound
    valid, err = validate_sklearn_version_spec("scikit-learn>=1.5.0")
    assert valid is False and "upper bound <2.0.0" in err

    # Negative cases: missing lower bound
    valid, err = validate_sklearn_version_spec("scikit-learn<2.0.0")
    assert valid is False and "lower bound >=1.5.0" in err

    # Negative cases: bare package
    valid, err = validate_sklearn_version_spec("scikit-learn")
    assert valid is False and "lacks version constraints" in err

    # Negative cases: incompatible lower bound
    valid, err = validate_sklearn_version_spec("scikit-learn>=1.4.0,<2.0.0")
    assert valid is False

    # Negative cases: incompatible upper bound
    valid, err = validate_sklearn_version_spec("scikit-learn>=1.5.0,<3.0.0")
    assert valid is False


# =========================================================================
# 5. pyproject.toml Dependency Governance Tests
# =========================================================================

def test_pyproject_clean_baseline_passes_conditional(tmp_path: Path) -> None:
    """Current pyproject.toml without scikit-learn passes governance gate."""
    assert PYPROJECT_PATH.is_file()
    record = parse_adr(ADR_PATH)
    is_valid, errors = verify_pyproject_dependencies(PYPROJECT_PATH, record)
    assert is_valid is True
    assert errors == []


def test_pyproject_with_unapproved_sklearn_fails(tmp_path: Path) -> None:
    """pyproject.toml declaring scikit-learn under PROPOSED ADR fails gate."""
    pyproj = tmp_path / "pyproject.toml"
    pyproj.write_text(
        """[project]
name = "test"
dependencies = [
    "numpy>=1.26.0",
    "scikit-learn>=1.5.0,<2.0.0"
]
""",
        encoding="utf-8",
    )
    record = ADRRecord(
        adr_id="ADR-004-001",
        status="PROPOSED",
        proposed_date="2026-10-03",
        approver="[PENDING]",
        approval_date="[PENDING]",
        git_commit="[PENDING]",
        fingerprint="[PENDING]",
        raw_content="",
        file_path=tmp_path / "dummy.md",
    )
    is_valid, errors = verify_pyproject_dependencies(pyproj, record)
    assert is_valid is False
    assert any("without approved governance ADR" in err for err in errors)


def test_pyproject_with_approved_sklearn_valid_spec_passes(tmp_path: Path) -> None:
    """pyproject.toml with scikit-learn matching >=1.5.0,<2.0.0 passes under APPROVED ADR."""
    pyproj = tmp_path / "pyproject.toml"
    pyproj.write_text(
        """[project]
name = "test"
dependencies = [
    "numpy>=1.26.0",
    "scikit-learn>=1.5.0,<2.0.0"
]
""",
        encoding="utf-8",
    )
    record = ADRRecord(
        adr_id="ADR-004-001",
        status="APPROVED",
        proposed_date="2026-10-03",
        approver="Human Evaluator",
        approval_date="2026-10-06",
        git_commit="1a2b3c4d5e",
        fingerprint="valid",
        raw_content="",
        file_path=tmp_path / "dummy.md",
    )
    is_valid, errors = verify_pyproject_dependencies(pyproj, record)
    assert is_valid is True
    assert errors == []


def test_pyproject_with_loose_version_spec_fails_under_approved_adr(tmp_path: Path) -> None:
    """pyproject.toml with loose scikit-learn>=1.5.0 fails even if ADR is APPROVED."""
    pyproj = tmp_path / "pyproject.toml"
    pyproj.write_text(
        """[project]
name = "test"
dependencies = [
    "numpy>=1.26.0",
    "scikit-learn>=1.5.0"
]
""",
        encoding="utf-8",
    )
    record = ADRRecord(
        adr_id="ADR-004-001",
        status="APPROVED",
        proposed_date="2026-10-03",
        approver="Human Evaluator",
        approval_date="2026-10-06",
        git_commit="1a2b3c4d5e",
        fingerprint="valid",
        raw_content="",
        file_path=tmp_path / "dummy.md",
    )
    is_valid, errors = verify_pyproject_dependencies(pyproj, record)
    assert is_valid is False
    assert any("upper bound <2.0.0" in err for err in errors)


# =========================================================================
# 6. Active Repository Governance Invariants (Not Hardcoded to PROPOSED Forever)
# =========================================================================

def test_active_repo_adr_is_valid() -> None:
    """ADR-004-001 in repository must be structurally valid and consistent."""
    assert ADR_PATH.is_file(), f"ADR file missing at {ADR_PATH}"
    record = parse_adr(ADR_PATH)

    assert record.adr_id == "ADR-004-001"
    assert record.proposed_date == "2026-10-03"
    assert record.status in {"PROPOSED", "APPROVED"}


def test_active_repo_dependency_governance_gate() -> None:
    """The active repository must satisfy governance invariants under its current state."""
    report = check_dependency_governance(REPO_ROOT)

    # Repository must never be in broken/tampered FAIL state
    assert report.status != GovernanceStatus.FAIL, f"Governance errors: {report.errors}"
    assert report.errors == []

    if report.status == GovernanceStatus.CONDITIONAL_PASS:
        # Currently proposed
        assert report.adr_status == "PROPOSED"
        assert report.sklearn_permitted is False
        assert is_sklearn_allowed(REPO_ROOT) is False
        with pytest.raises(GovernanceViolation, match="BLOCKED"):
            assert_sklearn_allowed(REPO_ROOT)
    elif report.status == GovernanceStatus.APPROVED_PENDING_LOCK:
        # Human approved (Commit A) but pending lock (Commit B)
        assert report.adr_status == "APPROVED"
        assert report.sklearn_permitted is False
        assert is_sklearn_allowed(REPO_ROOT) is False
        with pytest.raises(GovernanceViolation, match="BLOCKED"):
            assert_sklearn_allowed(REPO_ROOT)
    elif report.status == GovernanceStatus.APPROVED:
        # Fully locked (Commit B)
        assert report.adr_status == "APPROVED"
        assert report.sklearn_permitted is True
        assert is_sklearn_allowed(REPO_ROOT) is True
        assert_sklearn_allowed(REPO_ROOT)  # Should not raise
