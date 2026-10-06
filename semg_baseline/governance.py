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
from dataclasses import dataclass
from enum import Enum
import hashlib
from pathlib import Path
import re
from typing import Any

from packaging.specifiers import SpecifierSet
from packaging.version import Version

try:
    import tomllib  # Python 3.11+
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
    "packaging",
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
    """Validate that scikit-learn dependency specifier strictly satisfies both >=1.5.0 and <2.0.0."""
    match = re.match(r"^[a-zA-Z0-9_\-\.]+\s*(.*)$", dep_spec.strip())
    if not match or not match.group(1).strip():
        return False, f"scikit-learn specifier '{dep_spec}' lacks version constraints."

    spec_str = match.group(1).strip()
    try:
        spec_set = SpecifierSet(spec_str)
        has_ge_1_5 = False
        has_lt_2_0 = False

        for spec in spec_set:
            op = spec.operator
            ver = Version(spec.version)
            if op in (">=", "==") and Version("1.5.0") <= ver < Version("2.0.0"):
                has_ge_1_5 = True
            elif op == "<" and Version("1.5.0") < ver <= Version("2.0.0"):
                has_lt_2_0 = True
            elif op == "<=" and Version("1.5.0") < ver < Version("2.0.0"):
                has_lt_2_0 = True

        if not (has_ge_1_5 and has_lt_2_0):
            return False, (
                f"scikit-learn specifier '{dep_spec}' must enforce both lower bound >=1.5.0 "
                f"and upper bound <2.0.0 (parsed: '{spec_str}')."
            )
        return True, ""
    except Exception as e:
        return False, f"Failed to parse scikit-learn version specifier '{spec_str}': {e}"


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


def _extract_package_name(dep_spec: str) -> str:
    """Extract canonical package name from a PEP 508 dependency specification."""
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
                is_valid_spec, spec_err = validate_sklearn_version_spec(dep)
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

    is_allowlisted = is_path_allowlisted_for_sklearn(file_path)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.name
                if name == "sklearn" or name.startswith("sklearn."):
                    if not is_allowlisted:
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
            if mod == "sklearn" or mod.startswith("sklearn."):
                if not is_allowlisted:
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

    if adr_status == GovernanceStatus.APPROVED_PENDING_LOCK:
        return GovernanceReport(
            status=GovernanceStatus.APPROVED_PENDING_LOCK,
            adr_status="APPROVED",
            sklearn_permitted=False,
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
            f"Violations: {report.errors if report.errors else 'ADR remains PROPOSED or PENDING LOCK (requires human approval and lock)'}."
        )
