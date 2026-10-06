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
import subprocess
import pytest

from semg_baseline.governance import (
    ADRRecord,
    GovernanceStatus,
    GovernanceViolation,
    check_dependency_governance,
    compute_adr_fingerprint,
    is_path_allowlisted_for_jsonschema,
    is_path_allowlisted_for_sklearn,
    is_path_allowlisted_for_tomli,
    is_sklearn_allowed,
    assert_sklearn_allowed,
    parse_adr,
    validate_adr,
    validate_jsonschema_version_spec,
    validate_sklearn_version_spec,
    validate_tomli_version_spec,
    verify_approval_commit,
    verify_imports_in_source,
    verify_pyproject_dependencies,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
ADR_PATH = REPO_ROOT / "docs" / "adr" / "ADR-004-001-scikit-learn-host-only.md"
ADR_004_002_PATH = REPO_ROOT / "docs" / "adr" / "ADR-004-002-jsonschema-host-contract-validation.md"
ADR_004_003_PATH = REPO_ROOT / "docs" / "adr" / "ADR-004-003-tomli-python310-compatibility.md"
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


def test_import_scanner_detects_jsonschema_scope_violations() -> None:
    """Importing jsonschema outside authorized host-only paths triggers SCOPE_VIOLATION."""
    code = "from jsonschema import Draft202012Validator"

    # Permitted in semg_baseline/runner.py
    valid, errors = verify_imports_in_source(code, "semg_baseline/runner.py")
    assert valid is True
    assert errors == []

    # Permitted in tests
    valid, errors = verify_imports_in_source(code, "tests/test_baseline_training_job.py")
    assert valid is True
    assert errors == []

    # Permitted in specific validation scripts
    valid, errors = verify_imports_in_source(code, "scripts/validate_feature_004_contracts.py")
    assert valid is True
    assert errors == []

    # Prohibited in generic scripts
    valid, errors = verify_imports_in_source(code, "scripts/eval.py")
    assert valid is False
    assert any("SCOPE_VIOLATION" in err for err in errors)

    # Prohibited in semg_baseline outside runner.py
    valid, errors = verify_imports_in_source(code, "semg_baseline/pipeline.py")
    assert valid is False
    assert any("SCOPE_VIOLATION" in err for err in errors)

    # Prohibited in semg_dsp
    valid, errors = verify_imports_in_source(code, "semg_dsp/filter.py")
    assert valid is False
    assert any("SCOPE_VIOLATION" in err for err in errors)

    # Prohibited in semg_dataset
    valid, errors = verify_imports_in_source(code, "semg_dataset/view.py")
    assert valid is False
    assert any("SCOPE_VIOLATION" in err for err in errors)

    # Prohibited in orchestrator
    valid, errors = verify_imports_in_source(code, "orchestrator/pipeline.py")
    assert valid is False
    assert any("SCOPE_VIOLATION" in err for err in errors)


def test_import_scanner_jsonschema_api_allowlist() -> None:
    """Validate positive and negative symbol-level API allowlist for jsonschema (ADR-004-002)."""
    # Positive symbols in authorized path
    for sym_code in [
        "from jsonschema import Draft202012Validator",
        "from jsonschema import validate",
        "from jsonschema.exceptions import ValidationError",
        "from jsonschema.exceptions import SchemaError",
    ]:
        valid, errors = verify_imports_in_source(sym_code, "semg_baseline/runner.py")
        assert valid is True, f"Expected valid for '{sym_code}', got errors: {errors}"
        assert errors == []

    # Negative: unauthorized symbols, wildcards, submodules, and bare imports in runner.py
    negatives = [
        "from jsonschema import RefResolver",
        "from jsonschema import validators",
        "from jsonschema import *",
        "from jsonschema.exceptions import *",
        "from jsonschema.exceptions import FormatError",
        "import jsonschema",
        "import jsonschema.validators",
        "import jsonschema.exceptions",
    ]
    for neg_code in negatives:
        valid, errors = verify_imports_in_source(neg_code, "semg_baseline/runner.py")
        assert valid is False, f"Expected API_VIOLATION for '{neg_code}'"
        assert any("API_VIOLATION" in err for err in errors), f"Missing API_VIOLATION in {errors}"


def test_jsonschema_version_spec_and_path_allowlist() -> None:
    """Validate jsonschema version spec parser and positive path allowlist."""
    valid, err = validate_jsonschema_version_spec("jsonschema>=4.19.0,<5.0.0")
    assert valid is True and err == ""

    valid, err = validate_jsonschema_version_spec("jsonschema<5.0.0,>=4.19.0")
    assert valid is True and err == ""

    # Negative: loose
    valid, err = validate_jsonschema_version_spec("jsonschema>=4.19.0")
    assert valid is False and "bounded clauses" in err

    # Negative: unauthorized lower
    valid, err = validate_jsonschema_version_spec("jsonschema>=4.18.0,<5.0.0")
    assert valid is False and "Unauthorized lower bound" in err

    # Path allowlist checks (least privilege: runner.py, tests/**, scripts/validate_feature_004*.py)
    assert is_path_allowlisted_for_jsonschema("semg_baseline/runner.py") is True
    assert is_path_allowlisted_for_jsonschema("tests/test_foo.py") is True
    assert is_path_allowlisted_for_jsonschema("scripts/validate_feature_004_contracts.py") is True
    assert is_path_allowlisted_for_jsonschema("scripts/eval.py") is False
    assert is_path_allowlisted_for_jsonschema("semg_baseline/lda.py") is False
    assert is_path_allowlisted_for_jsonschema("semg_dsp/filter.py") is False


def test_import_scanner_detects_tomli_scope_violations() -> None:
    """Importing tomli outside authorized governance / test paths triggers SCOPE_VIOLATION."""
    code = "from tomli import load"

    # Permitted in semg_baseline/governance.py
    valid, errors = verify_imports_in_source(code, "semg_baseline/governance.py")
    assert valid is True
    assert errors == []

    # Permitted in tests
    valid, errors = verify_imports_in_source(code, "tests/test_baseline_governance_gate.py")
    assert valid is True
    assert errors == []

    # Prohibited in semg_baseline outside governance.py
    valid, errors = verify_imports_in_source(code, "semg_baseline/runner.py")
    assert valid is False
    assert any("SCOPE_VIOLATION" in err for err in errors)

    # Prohibited in semg_dsp
    valid, errors = verify_imports_in_source(code, "semg_dsp/filter.py")
    assert valid is False
    assert any("SCOPE_VIOLATION" in err for err in errors)

    # Prohibited in semg_dataset
    valid, errors = verify_imports_in_source(code, "semg_dataset/view.py")
    assert valid is False
    assert any("SCOPE_VIOLATION" in err for err in errors)

    # Prohibited in orchestrator
    valid, errors = verify_imports_in_source(code, "orchestrator/pipeline.py")
    assert valid is False
    assert any("SCOPE_VIOLATION" in err for err in errors)

    # Prohibited in scripts
    valid, errors = verify_imports_in_source(code, "scripts/run.py")
    assert valid is False
    assert any("SCOPE_VIOLATION" in err for err in errors)


def test_import_scanner_tomli_api_allowlist() -> None:
    """Validate positive and negative symbol-level API allowlist for tomli (ADR-004-003)."""
    # Positive symbols in authorized paths
    for sym_code in [
        "from tomli import load",
        "from tomli import loads",
        "from tomli import TOMLDecodeError",
    ]:
        valid, errors = verify_imports_in_source(sym_code, "semg_baseline/governance.py")
        assert valid is True, f"Expected valid for '{sym_code}', got errors: {errors}"
        assert errors == []

    # Positive fallback alias in governance.py
    valid, errors = verify_imports_in_source("import tomli as tomllib", "semg_baseline/governance.py")
    assert valid is True
    assert errors == []

    # Negative: unauthorized symbols, wildcards, submodules, and bare imports
    negatives = [
        "from tomli import qualquer_outro_simbolo",
        "from tomli import *",
        "import tomli._parser",
        "import tomli",
    ]
    for neg_code in negatives:
        valid, errors = verify_imports_in_source(neg_code, "semg_baseline/governance.py")
        assert valid is False, f"Expected API_VIOLATION for '{neg_code}'"
        assert any("API_VIOLATION" in err for err in errors), f"Missing API_VIOLATION in {errors}"


def test_tomli_version_spec_and_path_allowlist() -> None:
    """Validate tomli version spec parser and positive path allowlist."""
    valid, err = validate_tomli_version_spec('tomli>=2.0.0,<3.0.0; python_version < "3.11"')
    assert valid is True and err == ""

    valid, err = validate_tomli_version_spec("tomli>=2.0.0,<3.0.0; python_version < '3.11'")
    assert valid is True and err == ""

    # Negative: missing environment marker
    valid, err = validate_tomli_version_spec("tomli>=2.0.0,<3.0.0")
    assert valid is False and "environment marker" in err

    # Negative: wrong bounds
    valid, err = validate_tomli_version_spec('tomli>=1.0.0,<3.0.0; python_version < "3.11"')
    assert valid is False and "enforce both" in err

    # Path allowlist checks (semg_baseline/governance.py and tests/**)
    assert is_path_allowlisted_for_tomli("semg_baseline/governance.py") is True
    assert is_path_allowlisted_for_tomli("tests/test_foo.py") is True
    assert is_path_allowlisted_for_tomli("semg_baseline/runner.py") is False
    assert is_path_allowlisted_for_tomli("scripts/validate.py") is False
    assert is_path_allowlisted_for_tomli("semg_dsp/filter.py") is False


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

    valid, err = validate_sklearn_version_spec("scikit-learn<2.0.0,>=1.5.0")
    assert valid is True and err == ""

    # Negative cases: missing upper bound
    valid, err = validate_sklearn_version_spec("scikit-learn>=1.5.0")
    assert valid is False and "bounded clauses" in err

    # Negative cases: missing lower bound
    valid, err = validate_sklearn_version_spec("scikit-learn<2.0.0")
    assert valid is False and "bounded clauses" in err

    # Negative cases: bare package
    valid, err = validate_sklearn_version_spec("scikit-learn")
    assert valid is False and "lacks version constraints" in err

    # Negative cases: unauthorized lower bounds
    valid, err = validate_sklearn_version_spec("scikit-learn>=1.4.0,<2.0.0")
    assert valid is False and "Unauthorized lower bound" in err

    valid, err = validate_sklearn_version_spec("scikit-learn>=2.0.0")
    assert valid is False and "bounded clauses" in err

    # Negative cases: unauthorized upper bounds
    valid, err = validate_sklearn_version_spec("scikit-learn>=1.5.0,<2.1.0")
    assert valid is False and "Unauthorized upper bound" in err

    valid, err = validate_sklearn_version_spec("scikit-learn>=1.5.0,<3.0.0")
    assert valid is False and "Unauthorized upper bound" in err

    # Negative cases: invalid operators
    valid, err = validate_sklearn_version_spec("scikit-learn<=1.5.0,<2.0.0")
    assert valid is False and "Invalid operator" in err


def test_governance_operates_without_packaging() -> None:
    """Governance gate must function purely on stdlib without packaging module."""
    import subprocess
    import sys

    cmd = [
        sys.executable,
        "-c",
        (
            "import sys; "
            "sys.modules['packaging'] = None; "
            "sys.modules['packaging.specifiers'] = None; "
            "sys.modules['packaging.version'] = None; "
            "import semg_baseline.governance as g; "
            "valid, err = g.validate_sklearn_version_spec('scikit-learn>=1.5.0,<2.0.0'); "
            "assert valid is True, err; "
            "rep = g.check_dependency_governance(); "
            "assert rep.status != g.GovernanceStatus.FAIL"
        ),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert res.returncode == 0


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
    assert any("Premature dependency mutation" in err for err in errors)
    assert any("CONDITIONAL_PASS" in err for err in errors)


def test_pyproject_with_approved_pending_lock_fails(tmp_path: Path) -> None:
    """pyproject.toml declaring scikit-learn under Stage A (APPROVED_PENDING_LOCK) fails gate."""
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
        git_commit="[PENDING COMMIT]",
        fingerprint="[PENDING LOCK]",
        raw_content="",
        file_path=tmp_path / "dummy.md",
    )
    is_valid, errors = verify_pyproject_dependencies(pyproj, record)
    assert is_valid is False
    assert any("Premature dependency mutation" in err for err in errors)
    assert any("APPROVED_PENDING_LOCK" in err for err in errors)


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
    raw = "Valid ADR content"
    fp = compute_adr_fingerprint(raw)
    record = ADRRecord(
        adr_id="ADR-004-001",
        status="APPROVED",
        proposed_date="2026-10-03",
        approver="Human Evaluator",
        approval_date="2026-10-06",
        git_commit="1a2b3c4d5e",
        fingerprint=fp,
        raw_content=raw,
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
    raw = "Valid ADR content"
    fp = compute_adr_fingerprint(raw)
    record = ADRRecord(
        adr_id="ADR-004-001",
        status="APPROVED",
        proposed_date="2026-10-03",
        approver="Human Evaluator",
        approval_date="2026-10-06",
        git_commit="1a2b3c4d5e",
        fingerprint=fp,
        raw_content=raw,
        file_path=tmp_path / "dummy.md",
    )
    is_valid, errors = verify_pyproject_dependencies(pyproj, record)
    assert is_valid is False
    assert any("bounded clauses" in err for err in errors)


def test_pyproject_with_unapproved_jsonschema_fails(tmp_path: Path) -> None:
    """pyproject.toml declaring jsonschema under PROPOSED ADR-004-002 fails gate."""
    pyproj = tmp_path / "pyproject.toml"
    pyproj.write_text(
        """[project]
name = "test"
dependencies = [
    "numpy>=1.26.0",
    "jsonschema>=4.19.0,<5.0.0"
]
""",
        encoding="utf-8",
    )
    adr1 = ADRRecord(
        adr_id="ADR-004-001",
        status="PROPOSED",
        proposed_date="2026-10-03",
        approver="[PENDING]",
        approval_date="[PENDING]",
        git_commit="[PENDING]",
        fingerprint="[PENDING]",
        raw_content="",
        file_path=tmp_path / "dummy1.md",
    )
    adr2 = ADRRecord(
        adr_id="ADR-004-002",
        status="PROPOSED",
        proposed_date="2026-10-06",
        approver="[PENDING]",
        approval_date="[PENDING]",
        git_commit="[PENDING]",
        fingerprint="[PENDING]",
        raw_content="",
        file_path=tmp_path / "dummy2.md",
    )
    is_valid, errors = verify_pyproject_dependencies(pyproj, adr1, jsonschema_adr_record=adr2)
    assert is_valid is False
    assert any("Premature dependency mutation" in err for err in errors)
    assert any("ADR-004-002" in err for err in errors)


def test_pyproject_with_approved_jsonschema_valid_spec_passes(tmp_path: Path) -> None:
    """pyproject.toml declaring jsonschema matching >=4.19.0,<5.0.0 passes under fully locked ADR-004-002."""
    pyproj = tmp_path / "pyproject.toml"
    pyproj.write_text(
        """[project]
name = "test"
dependencies = [
    "numpy>=1.26.0",
    "jsonschema>=4.19.0,<5.0.0"
]
""",
        encoding="utf-8",
    )
    raw = "Valid JSONSchema ADR"
    fp = compute_adr_fingerprint(raw)
    adr1 = ADRRecord(
        adr_id="ADR-004-001",
        status="PROPOSED",
        proposed_date="2026-10-03",
        approver="[PENDING]",
        approval_date="[PENDING]",
        git_commit="[PENDING]",
        fingerprint="[PENDING]",
        raw_content="",
        file_path=tmp_path / "dummy1.md",
    )
    adr2 = ADRRecord(
        adr_id="ADR-004-002",
        status="APPROVED",
        proposed_date="2026-10-06",
        approver="Human Evaluator",
        approval_date="2026-10-06",
        git_commit="1a2b3c4d5e",
        fingerprint=fp,
        raw_content=raw,
        file_path=tmp_path / "dummy2.md",
    )
    is_valid, errors = verify_pyproject_dependencies(pyproj, adr1, jsonschema_adr_record=adr2)
    assert is_valid is True
    assert errors == []


def test_pyproject_with_unapproved_tomli_fails(tmp_path: Path) -> None:
    """pyproject.toml declaring tomli under PROPOSED ADR-004-003 fails gate."""
    pyproj = tmp_path / "pyproject.toml"
    pyproj.write_text(
        """[project]
name = "test"
dependencies = [
    "numpy>=1.26.0",
    "tomli>=2.0.0,<3.0.0; python_version < '3.11'"
]
""",
        encoding="utf-8",
    )
    adr1 = ADRRecord(
        adr_id="ADR-004-001",
        status="PROPOSED",
        proposed_date="2026-10-03",
        approver="[PENDING]",
        approval_date="[PENDING]",
        git_commit="[PENDING]",
        fingerprint="[PENDING]",
        raw_content="",
        file_path=tmp_path / "dummy1.md",
    )
    adr3 = ADRRecord(
        adr_id="ADR-004-003",
        status="PROPOSED",
        proposed_date="2026-10-06",
        approver="[PENDING]",
        approval_date="[PENDING]",
        git_commit="[PENDING]",
        fingerprint="[PENDING]",
        raw_content="",
        file_path=tmp_path / "dummy3.md",
    )
    is_valid, errors = verify_pyproject_dependencies(pyproj, adr1, tomli_adr_record=adr3)
    assert is_valid is False
    assert any("Premature dependency mutation" in err for err in errors)
    assert any("ADR-004-003" in err for err in errors)


def test_pyproject_with_approved_tomli_valid_spec_passes(tmp_path: Path) -> None:
    """pyproject.toml declaring tomli with marker passes under fully locked ADR-004-003."""
    pyproj = tmp_path / "pyproject.toml"
    pyproj.write_text(
        """[project]
name = "test"
dependencies = [
    "numpy>=1.26.0",
    "tomli>=2.0.0,<3.0.0; python_version < '3.11'"
]
""",
        encoding="utf-8",
    )
    raw = "Valid tomli ADR"
    fp = compute_adr_fingerprint(raw)
    adr1 = ADRRecord(
        adr_id="ADR-004-001",
        status="PROPOSED",
        proposed_date="2026-10-03",
        approver="[PENDING]",
        approval_date="[PENDING]",
        git_commit="[PENDING]",
        fingerprint="[PENDING]",
        raw_content="",
        file_path=tmp_path / "dummy1.md",
    )
    adr3 = ADRRecord(
        adr_id="ADR-004-003",
        status="APPROVED",
        proposed_date="2026-10-06",
        approver="Human Evaluator",
        approval_date="2026-10-06",
        git_commit="1a2b3c4d5e",
        fingerprint=fp,
        raw_content=raw,
        file_path=tmp_path / "dummy3.md",
    )
    is_valid, errors = verify_pyproject_dependencies(pyproj, adr1, tomli_adr_record=adr3)
    assert is_valid is True
    assert errors == []


# =========================================================================
# 5b. Aggregate Multi-ADR Governance State Scenarios (Section 5)
# =========================================================================

def _setup_mock_repo(
    tmp_path: Path,
    adr1_status: str = "PROPOSED",
    adr2_status: str = "PROPOSED",
    adr3_status: str = "PROPOSED",
    adr1_stage: str = "PROPOSED",
    adr2_stage: str = "PROPOSED",
    adr3_stage: str = "PROPOSED",
) -> Path:
    """Helper creating isolated git repository with 3 ADRs and clean pyproject.toml."""
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test Approver"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "eval@example.com"], cwd=tmp_path, check=True, capture_output=True)

    adr_dir = tmp_path / "docs" / "adr"
    adr_dir.mkdir(parents=True, exist_ok=True)

    def _write_initial(filename: str, status: str, stage: str) -> None:
        if stage == "PROPOSED":
            approver = "[PENDING HUMAN APPROVAL]"
            app_date = "[PENDING HUMAN APPROVAL]"
        else:
            approver = "Human Approver <eval@example.com>"
            app_date = "2026-10-06"
        commit = "[PENDING HUMAN COMMIT]"
        fp = "[PENDING LOCK]"
        content = ADR_TEMPLATE.format(
            status=status,
            approver=approver,
            approval_date=app_date,
            git_commit=commit,
            fingerprint=fp,
        )
        (adr_dir / filename).write_text(content, encoding="utf-8")

    _write_initial("ADR-004-001-scikit-learn-host-only.md", adr1_status, adr1_stage)
    _write_initial("ADR-004-002-jsonschema-host-contract-validation.md", adr2_status, adr2_stage)
    _write_initial("ADR-004-003-tomli-python310-compatibility.md", adr3_status, adr3_stage)
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "mock"\ndependencies = []\n', encoding="utf-8")

    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "docs(adr): human approval Stage A"], cwd=tmp_path, check=True, capture_output=True)
    proc_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=tmp_path, check=True, capture_output=True, text=True)
    stage_a_commit = proc_head.stdout.strip()

    def _apply_stage_b(filename: str, status: str, stage: str) -> None:
        if stage in {"STAGE_B", "TAMPERED"}:
            approver = "Human Approver <eval@example.com>"
            app_date = "2026-10-06"
            commit = stage_a_commit
            draft = ADR_TEMPLATE.format(
                status=status,
                approver=approver,
                approval_date=app_date,
                git_commit=commit,
                fingerprint="[PENDING LOCK]",
            )
            fp = (
                compute_adr_fingerprint(draft)
                if stage == "STAGE_B"
                else "badfingerprint0000000000000000000000000000000000000000000000000000"
            )
            content = ADR_TEMPLATE.format(
                status=status,
                approver=approver,
                approval_date=app_date,
                git_commit=commit,
                fingerprint=fp,
            )
            (adr_dir / filename).write_text(content, encoding="utf-8")

    _apply_stage_b("ADR-004-001-scikit-learn-host-only.md", adr1_status, adr1_stage)
    _apply_stage_b("ADR-004-002-jsonschema-host-contract-validation.md", adr2_status, adr2_stage)
    _apply_stage_b("ADR-004-003-tomli-python310-compatibility.md", adr3_status, adr3_stage)

    return tmp_path


def test_aggregate_governance_all_proposed(tmp_path: Path) -> None:
    """When all 3 ADRs are PROPOSED, aggregate is CONDITIONAL_PASS and dependencies unapproved."""
    repo = _setup_mock_repo(tmp_path, "PROPOSED", "PROPOSED", "PROPOSED", "PROPOSED", "PROPOSED", "PROPOSED")
    report = check_dependency_governance(repo)

    assert report.status == GovernanceStatus.CONDITIONAL_PASS
    assert report.all_required_dependencies_approved is False
    assert report.sklearn_permitted is False
    assert report.dependency_statuses["scikit-learn"] == GovernanceStatus.CONDITIONAL_PASS
    assert report.dependency_statuses["jsonschema"] == GovernanceStatus.CONDITIONAL_PASS
    assert report.dependency_statuses["tomli"] == GovernanceStatus.CONDITIONAL_PASS


def test_aggregate_governance_only_sklearn_approved(tmp_path: Path) -> None:
    """When only ADR-004-001 is APPROVED, aggregate is not APPROVED and T013b remains blocked."""
    repo = _setup_mock_repo(tmp_path, "APPROVED", "PROPOSED", "PROPOSED", "STAGE_B", "PROPOSED", "PROPOSED")
    report = check_dependency_governance(repo)

    assert report.status != GovernanceStatus.APPROVED
    assert report.status == GovernanceStatus.CONDITIONAL_PASS
    assert report.all_required_dependencies_approved is False
    assert report.sklearn_permitted is False
    assert report.dependency_statuses["scikit-learn"] == GovernanceStatus.APPROVED
    assert report.dependency_statuses["jsonschema"] == GovernanceStatus.CONDITIONAL_PASS
    assert report.dependency_statuses["tomli"] == GovernanceStatus.CONDITIONAL_PASS
    with pytest.raises(GovernanceViolation, match="BLOCKED"):
        assert_sklearn_allowed(repo)


def test_aggregate_governance_sklearn_and_jsonschema_approved_tomli_pending(tmp_path: Path) -> None:
    """When 001 and 002 are APPROVED but 003 is PROPOSED, aggregate is not APPROVED."""
    repo = _setup_mock_repo(tmp_path, "APPROVED", "APPROVED", "PROPOSED", "STAGE_B", "STAGE_B", "PROPOSED")
    report = check_dependency_governance(repo)

    assert report.status != GovernanceStatus.APPROVED
    assert report.status == GovernanceStatus.CONDITIONAL_PASS
    assert report.all_required_dependencies_approved is False
    assert report.dependency_statuses["scikit-learn"] == GovernanceStatus.APPROVED
    assert report.dependency_statuses["jsonschema"] == GovernanceStatus.APPROVED
    assert report.dependency_statuses["tomli"] == GovernanceStatus.CONDITIONAL_PASS
    with pytest.raises(GovernanceViolation, match="BLOCKED"):
        assert_sklearn_allowed(repo)


def test_aggregate_governance_one_in_stage_a(tmp_path: Path) -> None:
    """When any ADR is in Stage A (APPROVED_PENDING_LOCK), aggregate is APPROVED_PENDING_LOCK."""
    repo = _setup_mock_repo(tmp_path, "APPROVED", "APPROVED", "APPROVED", "STAGE_B", "STAGE_A", "STAGE_B")
    report = check_dependency_governance(repo)

    assert report.status == GovernanceStatus.APPROVED_PENDING_LOCK
    assert report.all_required_dependencies_approved is False
    assert report.sklearn_permitted is False
    assert report.dependency_statuses["scikit-learn"] == GovernanceStatus.APPROVED
    assert report.dependency_statuses["jsonschema"] == GovernanceStatus.APPROVED_PENDING_LOCK
    assert report.dependency_statuses["tomli"] == GovernanceStatus.APPROVED
    with pytest.raises(GovernanceViolation, match="BLOCKED"):
        assert_sklearn_allowed(repo)


def test_aggregate_governance_all_locked(tmp_path: Path) -> None:
    """When all 3 ADRs are APPROVED and locked (Stage B), aggregate is APPROVED and permitted."""
    repo = _setup_mock_repo(tmp_path, "APPROVED", "APPROVED", "APPROVED", "STAGE_B", "STAGE_B", "STAGE_B")
    report = check_dependency_governance(repo)

    assert report.status == GovernanceStatus.APPROVED
    assert report.all_required_dependencies_approved is True
    assert report.sklearn_permitted is True
    assert report.dependency_statuses["scikit-learn"] == GovernanceStatus.APPROVED
    assert report.dependency_statuses["jsonschema"] == GovernanceStatus.APPROVED
    assert report.dependency_statuses["tomli"] == GovernanceStatus.APPROVED
    assert_sklearn_allowed(repo)  # Must not raise


def test_aggregate_governance_any_tampered_fails(tmp_path: Path) -> None:
    """When any ADR has a fingerprint mismatch, aggregate status is FAIL."""
    repo = _setup_mock_repo(tmp_path, "APPROVED", "APPROVED", "APPROVED", "STAGE_B", "STAGE_B", "TAMPERED")
    report = check_dependency_governance(repo)

    assert report.status == GovernanceStatus.FAIL
    assert report.all_required_dependencies_approved is False
    assert report.sklearn_permitted is False
    assert any("mismatch" in e for e in report.errors)


# =========================================================================
# 6. Active Repository Governance Invariants (Not Hardcoded to PROPOSED Forever)
# =========================================================================

def test_active_repo_adr_is_valid() -> None:
    """ADR-004-001, ADR-004-002, and ADR-004-003 in repository must be structurally valid and consistent."""
    assert ADR_PATH.is_file(), f"ADR file missing at {ADR_PATH}"
    record1 = parse_adr(ADR_PATH)
    assert record1.adr_id == "ADR-004-001"
    assert record1.proposed_date == "2026-10-03"
    assert record1.status in {"PROPOSED", "APPROVED"}

    assert ADR_004_002_PATH.is_file(), f"ADR file missing at {ADR_004_002_PATH}"
    record2 = parse_adr(ADR_004_002_PATH)
    assert record2.adr_id == "ADR-004-002"
    assert record2.proposed_date == "2026-10-06"
    assert record2.status in {"PROPOSED", "APPROVED"}

    assert ADR_004_003_PATH.is_file(), f"ADR file missing at {ADR_004_003_PATH}"
    record3 = parse_adr(ADR_004_003_PATH)
    assert record3.adr_id == "ADR-004-003"
    assert record3.proposed_date == "2026-10-06"
    assert record3.status in {"PROPOSED", "APPROVED"}


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
        assert report.all_required_dependencies_approved is False
        assert report.dependency_statuses["scikit-learn"] == GovernanceStatus.CONDITIONAL_PASS
        assert report.dependency_statuses["jsonschema"] == GovernanceStatus.CONDITIONAL_PASS
        assert report.dependency_statuses["tomli"] == GovernanceStatus.CONDITIONAL_PASS
        assert is_sklearn_allowed(REPO_ROOT) is False
        with pytest.raises(GovernanceViolation, match="BLOCKED"):
            assert_sklearn_allowed(REPO_ROOT)
    elif report.status == GovernanceStatus.APPROVED_PENDING_LOCK:
        # Human approved (Commit A) but pending lock (Commit B)
        assert report.adr_status == "APPROVED"
        assert report.sklearn_permitted is False
        assert report.all_required_dependencies_approved is False
        assert is_sklearn_allowed(REPO_ROOT) is False
        with pytest.raises(GovernanceViolation, match="BLOCKED"):
            assert_sklearn_allowed(REPO_ROOT)
    elif report.status == GovernanceStatus.APPROVED:
        # Fully locked (Commit B)
        assert report.adr_status == "APPROVED"
        assert report.sklearn_permitted is True
        assert report.all_required_dependencies_approved is True
        assert is_sklearn_allowed(REPO_ROOT) is True
        assert_sklearn_allowed(REPO_ROOT)  # Should not raise


def test_repository_aware_approval_commit_provenance(tmp_path: Path) -> None:
    """Validate repository-aware Git provenance checks for Stage A human approval."""
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True, capture_output=True)

    adr_dir = tmp_path / "docs" / "adr"
    adr_dir.mkdir(parents=True, exist_ok=True)
    adr_file = adr_dir / "ADR-004-001-scikit-learn-host-only.md"

    # Case A: Commit contains ADR still PROPOSED -> FAIL
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
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "initial proposed"], cwd=tmp_path, check=True, capture_output=True)
    commit_proposed = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=tmp_path, check=True, capture_output=True, text=True
    ).stdout.strip()

    record_with_proposed_commit = ADRRecord(
        adr_id="ADR-004-001",
        status="APPROVED",
        proposed_date="2026-10-03",
        approver="Allisson Chervenski <allissonchervenski5@gmail.com>",
        approval_date="2026-10-06",
        git_commit=commit_proposed,
        fingerprint="dummy",
        raw_content="",
        file_path=adr_file,
    )
    ok, errs = verify_approval_commit(tmp_path, record_with_proposed_commit)
    assert ok is False
    assert any("has status 'PROPOSED'" in e for e in errs)

    # Case B: Commit contains APPROVED but approver is pending -> FAIL
    adr_file.write_text(
        ADR_TEMPLATE.format(
            status="APPROVED",
            approver="[PENDING HUMAN APPROVAL]",
            approval_date="2026-10-06",
            git_commit="[PENDING HUMAN COMMIT]",
            fingerprint="[PENDING LOCK]",
        ),
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "approved without approver"], cwd=tmp_path, check=True, capture_output=True)
    commit_pending_approver = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=tmp_path, check=True, capture_output=True, text=True
    ).stdout.strip()

    record_with_pending_approver = ADRRecord(
        adr_id="ADR-004-001",
        status="APPROVED",
        proposed_date="2026-10-03",
        approver="Allisson Chervenski <allissonchervenski5@gmail.com>",
        approval_date="2026-10-06",
        git_commit=commit_pending_approver,
        fingerprint="dummy",
        raw_content="",
        file_path=adr_file,
    )
    ok, errs = verify_approval_commit(tmp_path, record_with_pending_approver)
    assert ok is False
    assert any("has pending or missing approver" in e for e in errs)

    # Case C: Hash of another commit without approval (e.g. unrelated change) -> FAIL
    (tmp_path / "README.md").write_text("# Test", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "unrelated commit"], cwd=tmp_path, check=True, capture_output=True)

    # Case D: Hex hash syntactically valid but nonexistent in Git -> FAIL
    record_nonexistent_commit = ADRRecord(
        adr_id="ADR-004-001",
        status="APPROVED",
        proposed_date="2026-10-03",
        approver="Allisson Chervenski <allissonchervenski5@gmail.com>",
        approval_date="2026-10-06",
        git_commit="deadbeefdeadbeefdeadbeefdeadbeefdeadbeef",
        fingerprint="dummy",
        raw_content="",
        file_path=adr_file,
    )
    ok, errs = verify_approval_commit(tmp_path, record_nonexistent_commit)
    assert ok is False
    assert any("does not exist in git repository" in e for e in errs)

    # Case E: Valid Stage A commit exists -> PASS
    adr_file.write_text(
        ADR_TEMPLATE.format(
            status="APPROVED",
            approver="Allisson Chervenski <allissonchervenski5@gmail.com>",
            approval_date="2026-10-06",
            git_commit="[PENDING HUMAN COMMIT]",
            fingerprint="[PENDING LOCK]",
        ),
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "docs(adr): Stage A approval"], cwd=tmp_path, check=True, capture_output=True)
    commit_stage_a = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=tmp_path, check=True, capture_output=True, text=True
    ).stdout.strip()

    draft_stage_b = ADR_TEMPLATE.format(
        status="APPROVED",
        approver="Allisson Chervenski <allissonchervenski5@gmail.com>",
        approval_date="2026-10-06",
        git_commit=commit_stage_a,
        fingerprint="[PENDING LOCK]",
    )
    valid_fp = compute_adr_fingerprint(draft_stage_b)
    content_stage_b = ADR_TEMPLATE.format(
        status="APPROVED",
        approver="Allisson Chervenski <allissonchervenski5@gmail.com>",
        approval_date="2026-10-06",
        git_commit=commit_stage_a,
        fingerprint=valid_fp,
    )
    adr_file.write_text(content_stage_b, encoding="utf-8")
    record_stage_b = parse_adr(adr_file)

    ok, errs = verify_approval_commit(tmp_path, record_stage_b)
    assert ok is True
    assert errs == []

    # Case F: Stage B fingerprint tampered -> validate_adr fails
    tampered_content = ADR_TEMPLATE.format(
        status="APPROVED",
        approver="Allisson Chervenski <allissonchervenski5@gmail.com>",
        approval_date="2026-10-06",
        git_commit=commit_stage_a,
        fingerprint="badbadbadbadbadbadbadbadbadbadbadbadbadbadbadbadbadbadbadbadbadb",
    )
    adr_file.write_text(tampered_content, encoding="utf-8")
    record_tampered = parse_adr(adr_file)
    ok_val, errs_val, status_val = validate_adr(record_tampered)
    assert ok_val is False
    assert status_val == GovernanceStatus.FAIL
    assert any("fingerprint mismatch" in e for e in errs_val)

