"""Automated dependency governance gate for scikit-learn and third-party libraries.

Enforces FR-006, SC-001, Decision-1, and Constitution v1.1.0 (commit ce75be0):
- Human approval lifecycle (PROPOSED -> CONDITIONAL_PASS, APPROVED (Stage A) -> APPROVED_PENDING_LOCK,
  LOCKED (Stage B) -> APPROVED / PASS).
- Cryptographic SHA-256 fingerprinting of the Architecture Decision Record (ADR).
- Strict positive import path allowlist (semg_baseline/**, tests/test_baseline_*.py, tests/feature_004/**).
- Strict symbol-level API surface allowlist (LinearDiscriminantAnalysis, ledoit_wolf, specific metrics;
  prohibited: model_selection, pipeline, preprocessing).
- Strict version range enforcement (>=1.5.0, <2.0.0).
- Deterministic fail-closed gate before pyproject.toml mutation or sklearn execution.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from enum import Enum
import hashlib
from pathlib import Path
import re
import subprocess
from typing import Any

try:
    import tomllib  # Python 3.11+ stdlib
except ImportError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]


class GovernanceStatus(str, Enum):
    """Lifecycle and verification status of dependency governance."""

    PROPOSED = "PROPOSED"
    CONDITIONAL_PASS = "CONDITIONAL_PASS"
    APPROVED_PENDING_LOCK = "APPROVED_PENDING_LOCK"
    APPROVED = "APPROVED"
    FAIL = "FAIL"


class GovernanceViolation(Exception):
    """Raised when a dependency governance policy, scope, or lock is violated."""


@dataclass(frozen=True)
class ADRRecord:
    """Parsed representation of an Architecture Decision Record (ADR)."""

    adr_id: str
    status: str
    proposed_date: str
    approver: str
    approval_date: str
    git_commit: str
    fingerprint: str
    raw_content: str
    file_path: Path


@dataclass(frozen=True)
class GovernanceReport:
    """Outcome report of dependency governance gate evaluation."""

    status: GovernanceStatus
    adr_status: str
    sklearn_permitted: bool
    errors: list[str]
    dependency_statuses: dict[str, GovernanceStatus] = field(default_factory=dict)
    all_required_dependencies_approved: bool = False


# Allowlisted base dependencies (constitutional / pre-approved)
ALLOWED_BASE_PACKAGES = {
    "numpy",
    "scipy",
    "pytest",
    "pyyaml",
    "ruff",
    "mypy",
    "setuptools",
    "anyio",
}

# Allowlisted sklearn submodules
ALLOWED_SKLEARN_SUBMODULES = {
    "discriminant_analysis",
    "covariance",
    "metrics",
}

# Explicitly prohibited sklearn submodules
PROHIBITED_SKLEARN_SUBMODULES = {
    "model_selection",
    "pipeline",
    "preprocessing",
}

# Strict allowlist of permitted symbols per submodule
ALLOWED_SKLEARN_SYMBOLS: dict[str, set[str]] = {
    "discriminant_analysis": {
        "LinearDiscriminantAnalysis",
    },
    "metrics": {
        "confusion_matrix",
        "accuracy_score",
        "balanced_accuracy_score",
        "f1_score",
        "precision_score",
        "recall_score",
    },
    "covariance": {
        "ledoit_wolf",
        "LedoitWolf",
        "empirical_covariance",
        "shrunk_covariance",
    },
}

# Allowlisted jsonschema module-level symbols (ADR-004-002)
ALLOWED_JSONSCHEMA_MODULE_SYMBOLS: set[str] = {
    "Draft202012Validator",
    "validate",
}

# Allowlisted jsonschema.exceptions symbols (ADR-004-002)
ALLOWED_JSONSCHEMA_EXCEPTIONS_SYMBOLS: set[str] = {
    "ValidationError",
    "SchemaError",
}

# Allowlisted tomli symbols (ADR-004-003)
ALLOWED_TOMLI_SYMBOLS: set[str] = {
    "load",
    "loads",
    "TOMLDecodeError",
}


def is_path_allowlisted_for_sklearn(file_path: Path | str) -> bool:
    """Return True only if file_path falls strictly within positive allowlist for sklearn imports.

    Allowed paths (FR-006.2, ADR-004-001):
    1. semg_baseline/**
    2. tests/test_baseline_*.py
    3. tests/feature_004/**
    """
    path_str = str(file_path).replace("\\", "/").strip()
    if path_str.startswith("./"):
        path_str = path_str[2:]

    if path_str.startswith("semg_baseline/"):
        return True
    if path_str.startswith("tests/feature_004/"):
        return True
    if re.match(r"^tests/test_baseline_[^/]+\.py$", path_str):
        return True

    return False


def validate_sklearn_version_spec(dep_spec: str) -> tuple[bool, str]:
    """Validate that scikit-learn dependency specifier strictly satisfies both >=1.5.0 and <2.0.0 using stdlib only.

    Enforces governed contract: scikit-learn>=1.5.0,<2.0.0 (or with whitespace / reversed clause order).
    """
    match = re.match(r"^[a-zA-Z0-9_\-\.]+\s*(.*)$", dep_spec.strip())
    if not match or not match.group(1).strip():
        return False, f"scikit-learn specifier '{dep_spec}' lacks version constraints."

    spec_str = match.group(1).strip()
    clauses = [c.strip().replace(" ", "") for c in spec_str.split(",") if c.strip()]
    if len(clauses) != 2:
        return False, (
            f"scikit-learn specifier '{dep_spec}' must contain exactly two bounded clauses: "
            f"'>=1.5.0' and '<2.0.0' (found {len(clauses)} clauses: {clauses})."
        )

    clause_set = set(clauses)
    has_lower = ">=1.5.0" in clause_set
    has_upper = "<2.0.0" in clause_set

    for c in clauses:
        if c.startswith(">=") and c != ">=1.5.0":
            return False, f"Unauthorized lower bound '{c}' in '{dep_spec}'. Contract strictly requires '>=1.5.0'."
        if c.startswith("<") and not c.startswith("<=") and c != "<2.0.0":
            return False, f"Unauthorized upper bound '{c}' in '{dep_spec}'. Contract strictly requires '<2.0.0'."
        if c.startswith(">") and not c.startswith(">="):
            return False, f"Invalid operator in clause '{c}' in '{dep_spec}'."
        if c.startswith("<="):
            return False, f"Invalid operator '<=' in clause '{c}' in '{dep_spec}'. Contract strictly requires '<2.0.0'."
        if c.startswith("==") or c.startswith("!="):
            return False, f"Invalid operator in clause '{c}' in '{dep_spec}'."

    if not (has_lower and has_upper):
        return False, (
            f"scikit-learn specifier '{dep_spec}' must enforce both lower bound '>=1.5.0' "
            f"and upper bound '<2.0.0' (parsed: {clauses})."
        )

    return True, ""


def is_path_allowlisted_for_jsonschema(file_path: Path | str) -> bool:
    """Return True only if file_path falls strictly within positive allowlist for jsonschema imports.

    Allowed paths (ADR-004-002 least privilege):
    1. semg_baseline/runner.py
    2. tests/**
    3. scripts/validate_feature_004*.py
    """
    path_str = str(file_path).replace("\\", "/").strip()
    if path_str.startswith("./"):
        path_str = path_str[2:]

    if path_str == "semg_baseline/runner.py":
        return True
    if path_str.startswith("tests/"):
        return True
    if re.match(r"^scripts/validate_feature_004[^/]*\.py$", path_str):
        return True

    return False


def validate_jsonschema_version_spec(dep_spec: str) -> tuple[bool, str]:
    """Validate that jsonschema dependency specifier strictly satisfies both >=4.19.0 and <5.0.0 using stdlib only.

    Enforces governed contract: jsonschema>=4.19.0,<5.0.0 (or with whitespace / reversed clause order).
    """
    match = re.match(r"^[a-zA-Z0-9_\-\.]+\s*(.*)$", dep_spec.strip())
    if not match or not match.group(1).strip():
        return False, f"jsonschema specifier '{dep_spec}' lacks version constraints."

    spec_str = match.group(1).strip()
    clauses = [c.strip().replace(" ", "") for c in spec_str.split(",") if c.strip()]
    if len(clauses) != 2:
        return False, (
            f"jsonschema specifier '{dep_spec}' must contain exactly two bounded clauses: "
            f"'>=4.19.0' and '<5.0.0' (found {len(clauses)} clauses: {clauses})."
        )

    clause_set = set(clauses)
    has_lower = ">=4.19.0" in clause_set
    has_upper = "<5.0.0" in clause_set

    for c in clauses:
        if c.startswith(">=") and c != ">=4.19.0":
            return False, f"Unauthorized lower bound '{c}' in '{dep_spec}'. Contract strictly requires '>=4.19.0'."
        if c.startswith("<") and not c.startswith("<=") and c != "<5.0.0":
            return False, f"Unauthorized upper bound '{c}' in '{dep_spec}'. Contract strictly requires '<5.0.0'."
        if c.startswith(">") and not c.startswith(">="):
            return False, f"Invalid operator in clause '{c}' in '{dep_spec}'."
        if c.startswith("<="):
            return False, f"Invalid operator '<=' in clause '{c}' in '{dep_spec}'. Contract strictly requires '<5.0.0'."
        if c.startswith("==") or c.startswith("!="):
            return False, f"Invalid operator in clause '{c}' in '{dep_spec}'."

    if not (has_lower and has_upper):
        return False, (
            f"jsonschema specifier '{dep_spec}' must enforce both lower bound '>=4.19.0' "
            f"and upper bound '<5.0.0' (parsed: {clauses})."
        )

    return True, ""


def is_path_allowlisted_for_tomli(file_path: Path | str) -> bool:
    """Return True only if file_path falls strictly within positive allowlist for tomli imports.

    Allowed paths (ADR-004-003):
    1. semg_baseline/governance.py
    2. tests/**
    """
    path_str = str(file_path).replace("\\", "/").strip()
    if path_str.startswith("./"):
        path_str = path_str[2:]

    if path_str == "semg_baseline/governance.py":
        return True
    if path_str.startswith("tests/"):
        return True

    return False


def validate_tomli_version_spec(dep_spec: str) -> tuple[bool, str]:
    """Validate that tomli dependency specifier strictly satisfies >=2.0.0,<3.0.0 with python_version marker.

    Enforces governed contract: tomli>=2.0.0,<3.0.0; python_version < "3.11".
    """
    spec_clean = dep_spec.strip()
    if ";" not in spec_clean:
        return False, f"tomli specifier '{dep_spec}' must include environment marker '; python_version < \"3.11\"'."

    base_part, marker_part = [p.strip() for p in spec_clean.split(";", 1)]
    marker_clean = marker_part.replace(" ", "").replace('"', "'")
    if marker_clean not in {"python_version<'3.11'", "python_version<='3.10'"}:
        return False, (
            f"tomli environment marker '{marker_part}' must restrict execution to Python < 3.11 "
            f"(e.g. 'python_version < \"3.11\"')."
        )

    match = re.match(r"^[a-zA-Z0-9_\-\.]+\s*(.*)$", base_part)
    if not match or not match.group(1).strip():
        return False, f"tomli specifier '{dep_spec}' lacks version constraints."

    spec_str = match.group(1).strip()
    clauses = [c.strip().replace(" ", "") for c in spec_str.split(",") if c.strip()]
    if len(clauses) != 2:
        return False, (
            f"tomli specifier '{dep_spec}' must contain exactly two bounded clauses: "
            f"'>=2.0.0' and '<3.0.0' (found {len(clauses)} clauses: {clauses})."
        )

    clause_set = set(clauses)
    if ">=2.0.0" not in clause_set or "<3.0.0" not in clause_set:
        return False, f"tomli specifier '{dep_spec}' must enforce both '>=2.0.0' and '<3.0.0'."

    return True, ""


def canonicalize_adr_content(content: str) -> str:
    """Normalize ADR markdown content for cryptographic fingerprinting.

    Replaces the fingerprint line with '[PENDING LOCK]' so that the computed
    hash is self-consistent and independent of the placeholder vs actual hash.
    """
    normalized = content.replace("\r\n", "\n")
    return re.sub(
        r"(\*\*Fingerprint Criptográfico[^\*]*\*\*:\s*).*",
        r"\1[PENDING LOCK]",
        normalized,
    )


def compute_adr_fingerprint(content: str) -> str:
    """Compute deterministic SHA-256 digest of canonicalized ADR content."""
    canonical = canonicalize_adr_content(content)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def parse_adr(adr_path: Path | str) -> ADRRecord:
    """Parse an ADR markdown file and extract governance metadata fields."""
    path = Path(adr_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"ADR file not found at: {path}")

    content = path.read_text(encoding="utf-8")

    def _extract_field(pattern: str, default: str = "") -> str:
        match = re.search(pattern, content)
        if match:
            raw_val = match.group(1).strip()
            return raw_val.strip("`'\"")
        return default

    adr_id = _extract_field(r"-\s*\*\*ID\*\*:\s*(.+)")
    status = _extract_field(r"-\s*\*\*Status\*\*:\s*(.+)")
    proposed_date = _extract_field(r"-\s*\*\*Data de Proposição\*\*:\s*(.+)")
    approver = _extract_field(r"-\s*\*\*Aprovador Humano\*\*:\s*(.+)")
    approval_date = _extract_field(r"-\s*\*\*Data de Aprovação Humana\*\*:\s*(.+)")
    git_commit = _extract_field(r"-\s*\*\*Commit Git de Aprovação\*\*:\s*(.+)")
    fingerprint = _extract_field(
        r"-\s*\*\*Fingerprint Criptográfico[^\*]*\*\*:\s*(.+)"
    )

    return ADRRecord(
        adr_id=adr_id,
        status=status,
        proposed_date=proposed_date,
        approver=approver,
        approval_date=approval_date,
        git_commit=git_commit,
        fingerprint=fingerprint,
        raw_content=content,
        file_path=path,
    )


def validate_adr(record: ADRRecord) -> tuple[bool, list[str], GovernanceStatus]:
    """Validate ADR metadata, human approval signatures, and cryptographic lock.

    Enforces the two-stage approval workflow:
    - Stage A (Human Approval Commit): status is APPROVED, approver and date are filled,
      commit is pending -> returns GovernanceStatus.APPROVED_PENDING_LOCK (sklearn blocked).
    - Stage B (Governance Lock Commit): approval git commit hash is recorded and
      cryptographic fingerprint matches canonical content -> returns GovernanceStatus.APPROVED (sklearn permitted).
    """
    errors: list[str] = []

    if record.status == "PROPOSED":
        return True, [], GovernanceStatus.CONDITIONAL_PASS

    if record.status == "APPROVED":
        # Check human approver
        if not record.approver or "pending" in record.approver.lower():
            errors.append(
                f"ADR marked APPROVED but approver is pending: '{record.approver}'"
            )

        # Check approval date
        if not record.approval_date or "pending" in record.approval_date.lower():
            errors.append(
                f"ADR marked APPROVED but approval date is pending: '{record.approval_date}'"
            )

        # If human approval fields failed, this is an invalid state
        if errors:
            return False, errors, GovernanceStatus.FAIL

        # Check approval git commit (Commit A)
        commit_pending = not record.git_commit or "pending" in record.git_commit.lower()
        fingerprint_pending = not record.fingerprint or "pending" in record.fingerprint.lower()

        # If either commit or fingerprint is pending, ADR is in Stage A (APPROVED_PENDING_LOCK)
        if commit_pending or fingerprint_pending:
            return True, [], GovernanceStatus.APPROVED_PENDING_LOCK

        # Validate git commit hash format (7 to 40 hex chars)
        if not re.match(r"^[a-f0-9]{7,40}$", record.git_commit):
            errors.append(
                f"ADR approval git commit hash format invalid: '{record.git_commit}' (expected 7-40 hex chars)"
            )

        # Validate cryptographic fingerprint against canonicalized content
        expected_fp = compute_adr_fingerprint(record.raw_content)
        if record.fingerprint != expected_fp:
            errors.append(
                f"ADR cryptographic fingerprint mismatch: expected '{expected_fp}', "
                f"found '{record.fingerprint}'"
            )

        if errors:
            return False, errors, GovernanceStatus.FAIL

        return True, [], GovernanceStatus.APPROVED

    errors.append(f"Unrecognized ADR status: '{record.status}'")
    return False, errors, GovernanceStatus.FAIL


def verify_approval_commit(
    repo_root: Path | str,
    adr_record: ADRRecord,
) -> tuple[bool, list[str]]:
    """Verify that an ADR's approval commit exists in Git and contains genuine Stage A human approval.

    Enforces repository-aware cryptographic provenance:
    1. Validates that `adr_record.git_commit` exists via `git cat-file -e <commit>^{commit}`.
    2. Inspects ADR file contents in that commit via `git show <commit>:<relative_path>`.
    3. Confirms that in the approval commit (Stage A):
       - Status is 'APPROVED'
       - Approver is non-empty and not pending (matches adr_record.approver)
       - Approval date is non-empty and not pending (matches adr_record.approval_date)
       - Git commit field contains 'pending' (proves it was recorded at Stage A before commit hash was known)
       - Fingerprint field contains 'pending' (proves it was unsealed at Stage A before Stage B lock)
    """
    root = Path(repo_root).resolve()
    commit = adr_record.git_commit.strip()
    if not commit or "pending" in commit.lower():
        return False, [f"ADR {adr_record.adr_id} approval commit is pending or missing: '{commit}'"]

    # 1. Verify commit exists in repository
    try:
        proc_cat = subprocess.run(
            ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
            cwd=root,
            capture_output=True,
            text=True,
        )
    except Exception as e:
        return False, [f"Failed to execute git cat-file for ADR {adr_record.adr_id}: {e}"]

    if proc_cat.returncode != 0:
        return False, [
            f"ADR {adr_record.adr_id} approval commit '{commit}' does not exist in git repository at '{root}'"
        ]

    # 2. Extract ADR file relative path to repo root
    try:
        rel_path = adr_record.file_path.resolve().relative_to(root).as_posix()
    except ValueError:
        rel_path = f"docs/adr/{adr_record.file_path.name}"

    try:
        proc_show = subprocess.run(
            ["git", "show", f"{commit}:{rel_path}"],
            cwd=root,
            capture_output=True,
            text=True,
        )
    except Exception as e:
        return False, [f"Failed to execute git show for ADR {adr_record.adr_id}: {e}"]

    if proc_show.returncode != 0:
        return False, [
            f"ADR {adr_record.adr_id} file '{rel_path}' not found in approval commit '{commit}'"
        ]

    stage_a_content = proc_show.stdout

    def _extract_field(pattern: str, default: str = "") -> str:
        match = re.search(pattern, stage_a_content)
        if match:
            return match.group(1).strip().strip("`'\"")
        return default

    status = _extract_field(r"-\s*\*\*Status\*\*:\s*(.+)")
    approver = _extract_field(r"-\s*\*\*Aprovador Humano\*\*:\s*(.+)")
    approval_date = _extract_field(r"-\s*\*\*Data de Aprovação Humana\*\*:\s*(.+)")
    git_commit_field = _extract_field(r"-\s*\*\*Commit Git de Aprovação\*\*:\s*(.+)")
    fingerprint_field = _extract_field(r"-\s*\*\*Fingerprint Criptográfico[^\*]*\*\*:\s*(.+)")

    errors: list[str] = []

    if status != "APPROVED":
        errors.append(
            f"ADR {adr_record.adr_id} in approval commit '{commit}' has status '{status}' (expected 'APPROVED')"
        )

    if not approver or "pending" in approver.lower():
        errors.append(
            f"ADR {adr_record.adr_id} in approval commit '{commit}' has pending or missing approver: '{approver}'"
        )
    elif approver != adr_record.approver:
        errors.append(
            f"ADR {adr_record.adr_id} in approval commit '{commit}' approver '{approver}' does not match Stage B approver '{adr_record.approver}'"
        )

    if not approval_date or "pending" in approval_date.lower():
        errors.append(
            f"ADR {adr_record.adr_id} in approval commit '{commit}' has pending or missing approval date: '{approval_date}'"
        )
    elif approval_date != adr_record.approval_date:
        errors.append(
            f"ADR {adr_record.adr_id} in approval commit '{commit}' approval date '{approval_date}' does not match Stage B date '{adr_record.approval_date}'"
        )

    if not git_commit_field or "pending" not in git_commit_field.lower():
        errors.append(
            f"ADR {adr_record.adr_id} in approval commit '{commit}' does not represent Stage A (git_commit is not pending: '{git_commit_field}')"
        )

    if not fingerprint_field or "pending" not in fingerprint_field.lower():
        errors.append(
            f"ADR {adr_record.adr_id} in approval commit '{commit}' does not represent Stage A (fingerprint is not pending: '{fingerprint_field}')"
        )

    return len(errors) == 0, errors


def _extract_package_name(dep_spec: str) -> str:
    """Extract canonical package name from a PEP 508 dependency specification."""
    match = re.match(r"^([a-zA-Z0-9_\-\.]+)", dep_spec.strip())
    if match:
        return match.group(1).lower().replace("_", "-")
    return dep_spec.strip().lower()


def verify_pyproject_dependencies(
    pyproject_path: Path | str,
    adr_record: ADRRecord,
    jsonschema_adr_record: ADRRecord | None = None,
    tomli_adr_record: ADRRecord | None = None,
) -> tuple[bool, list[str]]:
    """Verify that all dependencies declared in pyproject.toml have approved governance."""
    path = Path(pyproject_path).resolve()
    if not path.is_file():
        return False, [f"pyproject.toml not found at: {path}"]

    if jsonschema_adr_record is None and hasattr(adr_record, "file_path") and adr_record.file_path:
        default_json_adr = adr_record.file_path.parent / "ADR-004-002-jsonschema-host-contract-validation.md"
        if default_json_adr.is_file():
            jsonschema_adr_record = parse_adr(default_json_adr)

    if tomli_adr_record is None and hasattr(adr_record, "file_path") and adr_record.file_path:
        default_tomli_adr = adr_record.file_path.parent / "ADR-004-003-tomli-python310-compatibility.md"
        if default_tomli_adr.is_file():
            tomli_adr_record = parse_adr(default_tomli_adr)

    with open(path, "rb") as f:
        data: dict[str, Any] = tomllib.load(f)

    project = data.get("project", {})
    declared_deps: list[str] = list(project.get("dependencies", []))

    optional_deps = project.get("optional-dependencies", {})
    for group_deps in optional_deps.values():
        if isinstance(group_deps, list):
            declared_deps.extend(group_deps)

    errors: list[str] = []

    for dep in declared_deps:
        pkg_name = _extract_package_name(dep)

        if pkg_name in {"scikit-learn", "sklearn"}:
            _, _, adr_gov_status = validate_adr(adr_record)
            if adr_gov_status is not GovernanceStatus.APPROVED:
                errors.append(
                    f"Premature dependency mutation: '{dep}' declared in pyproject.toml before ADR-004-001 is fully locked. "
                    f"Governance status is '{adr_gov_status.value}' (requires 'APPROVED' with valid git_commit and SHA-256 fingerprint)."
                )
            else:
                is_valid_spec, spec_err = validate_sklearn_version_spec(dep)
                if not is_valid_spec:
                    errors.append(spec_err)
        elif pkg_name == "jsonschema":
            if jsonschema_adr_record is None:
                errors.append(
                    f"Dependency '{dep}' declared in pyproject.toml without approved governance ADR (ADR-004-002 missing)."
                )
            else:
                _, _, json_gov_status = validate_adr(jsonschema_adr_record)
                if json_gov_status is not GovernanceStatus.APPROVED:
                    errors.append(
                        f"Premature dependency mutation: '{dep}' declared in pyproject.toml before ADR-004-002 is fully locked. "
                        f"Governance status is '{json_gov_status.value}' (requires 'APPROVED' with valid git_commit and SHA-256 fingerprint)."
                    )
                else:
                    is_valid_spec, spec_err = validate_jsonschema_version_spec(dep)
                    if not is_valid_spec:
                        errors.append(spec_err)
        elif pkg_name == "tomli":
            if tomli_adr_record is None:
                errors.append(
                    f"Dependency '{dep}' declared in pyproject.toml without approved governance ADR (ADR-004-003 missing)."
                )
            else:
                _, _, tomli_gov_status = validate_adr(tomli_adr_record)
                if tomli_gov_status is not GovernanceStatus.APPROVED:
                    errors.append(
                        f"Premature dependency mutation: '{dep}' declared in pyproject.toml before ADR-004-003 is fully locked. "
                        f"Governance status is '{tomli_gov_status.value}' (requires 'APPROVED' with valid git_commit and SHA-256 fingerprint)."
                    )
                else:
                    is_valid_spec, spec_err = validate_tomli_version_spec(dep)
                    if not is_valid_spec:
                        errors.append(spec_err)
        elif pkg_name in ALLOWED_BASE_PACKAGES:
            continue
        else:
            errors.append(
                f"Unauthorized dependency '{dep}' declared in pyproject.toml without prior governance approval."
            )

    return len(errors) == 0, errors


def verify_imports_in_source(
    source_code: str, file_path: Path | str
) -> tuple[bool, list[str]]:
    """Scan Python source code using AST to enforce import path and API surface allowlists."""
    path_str = str(file_path).replace("\\", "/")
    violations: list[str] = []

    try:
        tree = ast.parse(source_code, filename=path_str)
    except SyntaxError as e:
        return False, [f"SyntaxError parsing '{path_str}': {e}"]

    is_sklearn_allowlisted = is_path_allowlisted_for_sklearn(file_path)
    is_jsonschema_allowlisted = is_path_allowlisted_for_jsonschema(file_path)
    is_tomli_allowlisted = is_path_allowlisted_for_tomli(file_path)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.name
                if name == "jsonschema":
                    if not is_jsonschema_allowlisted:
                        violations.append(
                            f"SCOPE_VIOLATION: jsonschema import prohibited outside allowlisted paths in '{path_str}'"
                        )
                    if not path_str.startswith("tests/"):
                        violations.append(
                            f"API_VIOLATION: Bare 'import jsonschema' prohibited in '{path_str}'. "
                            f"Must import specific allowlisted symbols (e.g. 'from jsonschema import Draft202012Validator')."
                        )
                elif name == "jsonschema.exceptions":
                    if not is_jsonschema_allowlisted:
                        violations.append(
                            f"SCOPE_VIOLATION: jsonschema import prohibited outside allowlisted paths in '{path_str}'"
                        )
                    violations.append(
                        f"API_VIOLATION: Bare 'import jsonschema.exceptions' prohibited in '{path_str}'. "
                        f"Must import specific exception symbols (e.g. 'from jsonschema.exceptions import ValidationError')."
                    )
                elif name.startswith("jsonschema."):
                    if not is_jsonschema_allowlisted:
                        violations.append(
                            f"SCOPE_VIOLATION: jsonschema import prohibited outside allowlisted paths in '{path_str}'"
                        )
                    violations.append(
                        f"API_VIOLATION: Submodule '{name}' not in allowlist in '{path_str}'"
                    )
                elif name == "tomli":
                    if not is_tomli_allowlisted:
                        violations.append(
                            f"SCOPE_VIOLATION: tomli import prohibited outside allowlisted paths in '{path_str}'"
                        )
                    if path_str == "semg_baseline/governance.py" and alias.asname == "tomllib":
                        pass  # Approved stdlib-fallback alias
                    else:
                        violations.append(
                            f"API_VIOLATION: Bare 'import tomli' prohibited in '{path_str}'. "
                            f"Must import specific allowlisted symbols or use 'import tomli as tomllib' fallback in governance.py."
                        )
                elif name.startswith("tomli."):
                    if not is_tomli_allowlisted:
                        violations.append(
                            f"SCOPE_VIOLATION: tomli import prohibited outside allowlisted paths in '{path_str}'"
                        )
                    violations.append(
                        f"API_VIOLATION: Submodule '{name}' not in allowlist in '{path_str}'"
                    )
                elif name == "sklearn" or name.startswith("sklearn."):
                    if not is_sklearn_allowlisted:
                        violations.append(
                            f"SCOPE_VIOLATION: scikit-learn import prohibited outside allowlisted paths in '{path_str}'"
                        )
                    if name == "sklearn":
                        violations.append(
                            f"API_VIOLATION: Bare 'import sklearn' prohibited in '{path_str}'. Must import specific allowlisted submodules or symbols."
                        )
                    else:
                        parts = name.split(".")
                        submodule = parts[1]
                        if submodule in PROHIBITED_SKLEARN_SUBMODULES:
                            violations.append(
                                f"API_VIOLATION: Prohibited submodule 'sklearn.{submodule}' imported in '{path_str}'"
                            )
                        elif submodule not in ALLOWED_SKLEARN_SUBMODULES:
                            violations.append(
                                f"API_VIOLATION: Submodule 'sklearn.{submodule}' not in allowlist in '{path_str}'"
                            )
                        elif len(parts) > 2:
                            violations.append(
                                f"API_VIOLATION: Nested submodule '{name}' not in allowlist in '{path_str}'"
                            )

        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod == "jsonschema":
                if not is_jsonschema_allowlisted:
                    violations.append(
                        f"SCOPE_VIOLATION: jsonschema import prohibited outside allowlisted paths in '{path_str}'"
                    )
                for alias in node.names:
                    sym = alias.name
                    if sym == "*":
                        violations.append(
                            f"API_VIOLATION: Wildcard import '*' prohibited for 'jsonschema' in '{path_str}'"
                        )
                    elif sym not in ALLOWED_JSONSCHEMA_MODULE_SYMBOLS:
                        violations.append(
                            f"API_VIOLATION: Symbol '{sym}' from 'jsonschema' not in API allowlist in '{path_str}'"
                        )
            elif mod == "jsonschema.exceptions":
                if not is_jsonschema_allowlisted:
                    violations.append(
                        f"SCOPE_VIOLATION: jsonschema import prohibited outside allowlisted paths in '{path_str}'"
                    )
                for alias in node.names:
                    sym = alias.name
                    if sym == "*":
                        violations.append(
                            f"API_VIOLATION: Wildcard import '*' prohibited for 'jsonschema.exceptions' in '{path_str}'"
                        )
                    elif sym not in ALLOWED_JSONSCHEMA_EXCEPTIONS_SYMBOLS:
                        violations.append(
                            f"API_VIOLATION: Symbol '{sym}' from 'jsonschema.exceptions' not in API allowlist in '{path_str}'"
                        )
            elif mod.startswith("jsonschema."):
                if not is_jsonschema_allowlisted:
                    violations.append(
                        f"SCOPE_VIOLATION: jsonschema import prohibited outside allowlisted paths in '{path_str}'"
                    )
                violations.append(
                    f"API_VIOLATION: Submodule '{mod}' not in allowlist in '{path_str}'"
                )
            elif mod == "tomli":
                if not is_tomli_allowlisted:
                    violations.append(
                        f"SCOPE_VIOLATION: tomli import prohibited outside allowlisted paths in '{path_str}'"
                    )
                for alias in node.names:
                    sym = alias.name
                    if sym == "*":
                        violations.append(
                            f"API_VIOLATION: Wildcard import '*' prohibited for 'tomli' in '{path_str}'"
                        )
                    elif sym not in ALLOWED_TOMLI_SYMBOLS:
                        violations.append(
                            f"API_VIOLATION: Symbol '{sym}' from 'tomli' not in API allowlist in '{path_str}'"
                        )
            elif mod.startswith("tomli."):
                if not is_tomli_allowlisted:
                    violations.append(
                        f"SCOPE_VIOLATION: tomli import prohibited outside allowlisted paths in '{path_str}'"
                    )
                violations.append(
                    f"API_VIOLATION: Submodule '{mod}' not in allowlist in '{path_str}'"
                )
            elif mod == "sklearn" or mod.startswith("sklearn."):
                if not is_sklearn_allowlisted:
                    violations.append(
                        f"SCOPE_VIOLATION: scikit-learn import prohibited outside allowlisted paths in '{path_str}'"
                    )

                if mod == "sklearn":
                    for alias in node.names:
                        submodule = alias.name
                        if submodule == "*":
                            violations.append(
                                f"API_VIOLATION: Wildcard import '*' prohibited for 'sklearn' in '{path_str}'"
                            )
                        elif submodule in PROHIBITED_SKLEARN_SUBMODULES:
                            violations.append(
                                f"API_VIOLATION: Prohibited submodule 'sklearn.{submodule}' imported in '{path_str}'"
                            )
                        elif submodule not in ALLOWED_SKLEARN_SUBMODULES:
                            violations.append(
                                f"API_VIOLATION: Submodule 'sklearn.{submodule}' not in allowlist in '{path_str}'"
                            )
                else:
                    parts = mod.split(".")
                    submodule = parts[1]
                    if submodule in PROHIBITED_SKLEARN_SUBMODULES:
                        violations.append(
                            f"API_VIOLATION: Prohibited submodule 'sklearn.{submodule}' imported in '{path_str}'"
                        )
                    elif submodule not in ALLOWED_SKLEARN_SUBMODULES:
                        violations.append(
                            f"API_VIOLATION: Submodule 'sklearn.{submodule}' not in allowlist in '{path_str}'"
                        )
                    elif len(parts) > 2:
                        violations.append(
                            f"API_VIOLATION: Nested submodule '{mod}' not in allowlist in '{path_str}'"
                        )
                    else:
                        allowed_syms = ALLOWED_SKLEARN_SYMBOLS.get(submodule, set())
                        for alias in node.names:
                            sym_name = alias.name
                            if sym_name == "*":
                                violations.append(
                                    f"API_VIOLATION: Wildcard import '*' prohibited for 'sklearn.{submodule}' in '{path_str}'"
                                )
                            elif sym_name not in allowed_syms:
                                violations.append(
                                    f"API_VIOLATION: Symbol '{sym_name}' from 'sklearn.{submodule}' not in API allowlist in '{path_str}'"
                                )

        elif isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name) and node.value.id == "jsonschema":
                if node.attr not in ALLOWED_JSONSCHEMA_MODULE_SYMBOLS and node.attr != "exceptions":
                    violations.append(
                        f"API_VIOLATION: Attribute '{node.attr}' accessed from 'jsonschema' not in allowlist in '{path_str}'"
                    )
            elif (
                isinstance(node.value, ast.Attribute)
                and isinstance(node.value.value, ast.Name)
                and node.value.value.id == "jsonschema"
                and node.value.attr == "exceptions"
            ):
                if node.attr not in ALLOWED_JSONSCHEMA_EXCEPTIONS_SYMBOLS:
                    violations.append(
                        f"API_VIOLATION: Attribute '{node.attr}' accessed from 'jsonschema.exceptions' not in allowlist in '{path_str}'"
                    )

    return len(violations) == 0, violations


def scan_repository_imports(repo_root: Path | str) -> tuple[bool, list[str]]:
    """Scan all Python files in the repository for sklearn import violations."""
    root = Path(repo_root).resolve()
    all_violations: list[str] = []

    excluded_dirs = {
        ".git",
        ".venv",
        "venv",
        "build",
        "dist",
        "__pycache__",
        ".specify",
        ".agents",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
    }

    for py_file in root.rglob("*.py"):
        if any(part in excluded_dirs for part in py_file.parts):
            continue

        try:
            code = py_file.read_text(encoding="utf-8")
        except Exception:
            continue

        rel_path = py_file.relative_to(root)
        is_valid, violations = verify_imports_in_source(code, rel_path)
        if not is_valid:
            all_violations.extend(violations)

    return len(all_violations) == 0, all_violations


def check_dependency_governance(
    repo_root: Path | str | None = None,
) -> GovernanceReport:
    """Run full deterministic automated dependency governance check across the repository."""
    if repo_root is None:
        repo_path = Path(__file__).resolve().parent.parent
    else:
        repo_path = Path(repo_root).resolve()

    adr_path = repo_path / "docs" / "adr" / "ADR-004-001-scikit-learn-host-only.md"
    jsonschema_adr_path = (
        repo_path / "docs" / "adr" / "ADR-004-002-jsonschema-host-contract-validation.md"
    )
    tomli_adr_path = (
        repo_path / "docs" / "adr" / "ADR-004-003-tomli-python310-compatibility.md"
    )
    pyproject_path = repo_path / "pyproject.toml"

    errors: list[str] = []

    # 1. Parse and validate ADR-004-001
    adr_record: ADRRecord | None = None
    if not adr_path.is_file():
        errors.append(f"ADR-004-001 not found at {adr_path}")
        sklearn_status = GovernanceStatus.FAIL
        adr_status_str = "MISSING"
    else:
        adr_record = parse_adr(adr_path)
        _, adr_errors, sklearn_status = validate_adr(adr_record)
        errors.extend(adr_errors)
        adr_status_str = adr_record.status
        if sklearn_status == GovernanceStatus.APPROVED:
            prov_valid, prov_errors = verify_approval_commit(repo_path, adr_record)
            if not prov_valid:
                sklearn_status = GovernanceStatus.FAIL
                errors.extend(prov_errors)

    # 1b. Parse and validate ADR-004-002
    jsonschema_record: ADRRecord | None = None
    if not jsonschema_adr_path.is_file():
        errors.append(f"ADR-004-002 not found at {jsonschema_adr_path}")
        jsonschema_status = GovernanceStatus.FAIL
    else:
        jsonschema_record = parse_adr(jsonschema_adr_path)
        _, json_errors, jsonschema_status = validate_adr(jsonschema_record)
        errors.extend(json_errors)
        if jsonschema_status == GovernanceStatus.APPROVED:
            prov_valid, prov_errors = verify_approval_commit(repo_path, jsonschema_record)
            if not prov_valid:
                jsonschema_status = GovernanceStatus.FAIL
                errors.extend(prov_errors)

    # 1c. Parse and validate ADR-004-003
    tomli_record: ADRRecord | None = None
    if not tomli_adr_path.is_file():
        errors.append(f"ADR-004-003 not found at {tomli_adr_path}")
        tomli_status = GovernanceStatus.FAIL
    else:
        tomli_record = parse_adr(tomli_adr_path)
        _, tomli_errors, tomli_status = validate_adr(tomli_record)
        errors.extend(tomli_errors)
        if tomli_status == GovernanceStatus.APPROVED:
            prov_valid, prov_errors = verify_approval_commit(repo_path, tomli_record)
            if not prov_valid:
                tomli_status = GovernanceStatus.FAIL
                errors.extend(prov_errors)

    dep_statuses: dict[str, GovernanceStatus] = {
        "scikit-learn": sklearn_status,
        "jsonschema": jsonschema_status,
        "tomli": tomli_status,
    }

    # 2. Verify pyproject.toml
    if pyproject_path.is_file():
        if adr_record is not None:
            pyproject_valid, pyproject_errors = verify_pyproject_dependencies(
                pyproject_path,
                adr_record,
                jsonschema_adr_record=jsonschema_record,
                tomli_adr_record=tomli_record,
            )
            errors.extend(pyproject_errors)
        else:
            errors.append("Cannot verify pyproject.toml because ADR-004-001 is missing.")
    else:
        errors.append(f"pyproject.toml not found at {pyproject_path}")

    # 3. Verify repository imports
    imports_valid, import_errors = scan_repository_imports(repo_path)
    errors.extend(import_errors)

    # 4. Synthesize aggregate status
    statuses = [sklearn_status, jsonschema_status, tomli_status]
    if errors or any(s == GovernanceStatus.FAIL for s in statuses):
        aggregate_status = GovernanceStatus.FAIL
    elif all(s == GovernanceStatus.APPROVED for s in statuses):
        aggregate_status = GovernanceStatus.APPROVED
    elif any(s == GovernanceStatus.APPROVED_PENDING_LOCK for s in statuses):
        aggregate_status = GovernanceStatus.APPROVED_PENDING_LOCK
    else:
        aggregate_status = GovernanceStatus.CONDITIONAL_PASS

    all_approved = (aggregate_status == GovernanceStatus.APPROVED and len(errors) == 0)
    sklearn_ok = (all_approved and sklearn_status == GovernanceStatus.APPROVED)

    return GovernanceReport(
        status=aggregate_status,
        adr_status=adr_status_str,
        sklearn_permitted=sklearn_ok,
        errors=errors,
        dependency_statuses=dep_statuses,
        all_required_dependencies_approved=all_approved,
    )


def is_sklearn_allowed(repo_root: Path | str | None = None) -> bool:
    """Return True if scikit-learn is permitted to execute under approved governance."""
    report = check_dependency_governance(repo_root)
    return report.sklearn_permitted


def assert_sklearn_allowed(repo_root: Path | str | None = None) -> None:
    """Raise GovernanceViolation if scikit-learn execution is blocked by governance."""
    report = check_dependency_governance(repo_root)
    if not (report.sklearn_permitted and report.all_required_dependencies_approved):
        raise GovernanceViolation(
            f"scikit-learn execution is BLOCKED under current governance "
            f"(Status: {report.status.value}, ADR: {report.adr_status}). "
            f"Violations: {report.errors if report.errors else 'ADR remains PROPOSED or PENDING LOCK (requires human approval and lock across all required dependencies)'}."
        )
