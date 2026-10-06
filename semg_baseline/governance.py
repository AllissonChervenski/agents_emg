"""Automated dependency governance gate for scikit-learn and third-party libraries.

Enforces FR-006, SC-001, Decision-1, and Constitution v1.1.0 (commit ce75be0):
- Human approval lifecycle (PROPOSED -> CONDITIONAL_PASS, APPROVED -> PASS).
- Cryptographic SHA-256 fingerprinting of the Architecture Decision Record (ADR).
- Strict import allowlist (semg_baseline/**, tests/**; prohibited in semg_dsp/**, semg_dataset/**).
- Strict API surface allowlist (discriminant_analysis, covariance, metrics;
  prohibited: model_selection, pipeline, preprocessing).
- Deterministic fail-closed gate before pyproject.toml mutation or sklearn execution.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from enum import Enum
import hashlib
from pathlib import Path
import re
from typing import Any

try:
    import tomllib  # Python 3.11+
except ImportError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]


class GovernanceStatus(str, Enum):
    """Lifecycle and verification status of dependency governance."""

    PROPOSED = "PROPOSED"
    CONDITIONAL_PASS = "CONDITIONAL_PASS"
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
            # Strip backticks if present
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

    Returns (is_valid, errors, status).
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

        # Check git commit
        if not record.git_commit or "pending" in record.git_commit.lower():
            errors.append(
                f"ADR marked APPROVED but approval git commit is pending: '{record.git_commit}'"
            )

        # Check cryptographic fingerprint
        if not record.fingerprint or "pending" in record.fingerprint.lower():
            errors.append(
                f"ADR marked APPROVED but fingerprint is pending: '{record.fingerprint}'"
            )
        else:
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


def _extract_package_name(dep_spec: str) -> str:
    """Extract canonical package name from a PEP 508 dependency specification."""
    # Match package name before any version specifier, extra, or environment marker
    match = re.match(r"^([a-zA-Z0-9_\-\.]+)", dep_spec.strip())
    if match:
        return match.group(1).lower().replace("_", "-")
    return dep_spec.strip().lower()


def verify_pyproject_dependencies(
    pyproject_path: Path | str, adr_record: ADRRecord
) -> tuple[bool, list[str]]:
    """Verify that all dependencies declared in pyproject.toml have approved governance."""
    path = Path(pyproject_path).resolve()
    if not path.is_file():
        return False, [f"pyproject.toml not found at: {path}"]

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
            if adr_record.status != "APPROVED":
                errors.append(
                    f"Dependency '{dep}' declared in pyproject.toml without approved governance ADR "
                    f"(ADR status is '{adr_record.status}')."
                )
            else:
                # ADR is APPROVED: verify version specification if present
                if "<2.0" not in dep and "<2" not in dep and "==1." not in dep and ">=1.5" not in dep:
                    errors.append(
                        f"scikit-learn dependency specifier '{dep}' does not match ADR required range >=1.5.0, <2.0.0."
                    )
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

    # Determine if this file path is strictly forbidden from importing sklearn
    is_forbidden_path = (
        "semg_dsp/" in path_str
        or "semg_dataset/" in path_str
        or "data/" in path_str
    )

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.name
                if name == "sklearn" or name.startswith("sklearn."):
                    if is_forbidden_path:
                        violations.append(
                            f"SCOPE_VIOLATION: scikit-learn import prohibited in host-incompatible path '{path_str}'"
                        )
                    # Check submodule allowlist
                    parts = name.split(".")
                    if len(parts) > 1:
                        submodule = parts[1]
                        if submodule in PROHIBITED_SKLEARN_SUBMODULES:
                            violations.append(
                                f"API_VIOLATION: Prohibited submodule 'sklearn.{submodule}' imported in '{path_str}'"
                            )
                        elif submodule not in ALLOWED_SKLEARN_SUBMODULES:
                            violations.append(
                                f"API_VIOLATION: Submodule 'sklearn.{submodule}' not in allowlist in '{path_str}'"
                            )

        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod == "sklearn" or mod.startswith("sklearn."):
                if is_forbidden_path:
                    violations.append(
                        f"SCOPE_VIOLATION: scikit-learn import prohibited in host-incompatible path '{path_str}'"
                    )

                parts = mod.split(".")
                if len(parts) > 1:
                    submodule = parts[1]
                    if submodule in PROHIBITED_SKLEARN_SUBMODULES:
                        violations.append(
                            f"API_VIOLATION: Prohibited submodule 'sklearn.{submodule}' imported in '{path_str}'"
                        )
                    elif submodule not in ALLOWED_SKLEARN_SUBMODULES:
                        violations.append(
                            f"API_VIOLATION: Submodule 'sklearn.{submodule}' not in allowlist in '{path_str}'"
                        )
                elif mod == "sklearn":
                    for alias in node.names:
                        submodule = alias.name
                        if submodule in PROHIBITED_SKLEARN_SUBMODULES:
                            violations.append(
                                f"API_VIOLATION: Prohibited submodule 'sklearn.{submodule}' imported in '{path_str}'"
                            )
                        elif submodule not in ALLOWED_SKLEARN_SUBMODULES:
                            violations.append(
                                f"API_VIOLATION: Submodule 'sklearn.{submodule}' not in allowlist in '{path_str}'"
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
        # Default to repo root relative to this file: semg_baseline/governance.py -> parent parent
        repo_path = Path(__file__).resolve().parent.parent
    else:
        repo_path = Path(repo_root).resolve()

    adr_path = repo_path / "docs" / "adr" / "ADR-004-001-scikit-learn-host-only.md"
    pyproject_path = repo_path / "pyproject.toml"

    errors: list[str] = []

    # 1. Parse and validate ADR
    if not adr_path.is_file():
        errors.append(f"ADR-004-001 not found at {adr_path}")
        return GovernanceReport(
            status=GovernanceStatus.FAIL,
            adr_status="MISSING",
            sklearn_permitted=False,
            errors=errors,
        )

    adr_record = parse_adr(adr_path)
    adr_valid, adr_errors, adr_status = validate_adr(adr_record)
    errors.extend(adr_errors)

    # 2. Verify pyproject.toml
    if pyproject_path.is_file():
        pyproject_valid, pyproject_errors = verify_pyproject_dependencies(
            pyproject_path, adr_record
        )
        errors.extend(pyproject_errors)
    else:
        errors.append(f"pyproject.toml not found at {pyproject_path}")

    # 3. Verify repository imports
    imports_valid, import_errors = scan_repository_imports(repo_path)
    errors.extend(import_errors)

    # 4. Synthesize final status
    if errors:
        return GovernanceReport(
            status=GovernanceStatus.FAIL,
            adr_status=adr_record.status,
            sklearn_permitted=False,
            errors=errors,
        )

    if adr_status == GovernanceStatus.APPROVED:
        return GovernanceReport(
            status=GovernanceStatus.APPROVED,
            adr_status="APPROVED",
            sklearn_permitted=True,
            errors=[],
        )

    if adr_status == GovernanceStatus.CONDITIONAL_PASS:
        return GovernanceReport(
            status=GovernanceStatus.CONDITIONAL_PASS,
            adr_status="PROPOSED",
            sklearn_permitted=False,
            errors=[],
        )

    return GovernanceReport(
        status=GovernanceStatus.FAIL,
        adr_status=adr_record.status,
        sklearn_permitted=False,
        errors=["Governance gate reached unexpected terminal state."],
    )


def is_sklearn_allowed(repo_root: Path | str | None = None) -> bool:
    """Return True if scikit-learn is permitted to execute under approved governance."""
    report = check_dependency_governance(repo_root)
    return report.sklearn_permitted


def assert_sklearn_allowed(repo_root: Path | str | None = None) -> None:
    """Raise GovernanceViolation if scikit-learn execution is blocked by governance."""
    report = check_dependency_governance(repo_root)
    if not report.sklearn_permitted:
        raise GovernanceViolation(
            f"scikit-learn execution is BLOCKED under current governance "
            f"(Status: {report.status.value}, ADR: {report.adr_status}). "
            f"Violations: {report.errors if report.errors else 'ADR remains PROPOSED (PENDING HUMAN SIGN-OFF)'}."
        )
