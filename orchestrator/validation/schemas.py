from dataclasses import asdict
from datetime import datetime, timezone
from orchestrator.config.models import ValidationIssue, ValidationResult


def validation_result(status: str, issues: list[ValidationIssue], summary: str, validator: str, model=None, raw_output="") -> ValidationResult:
    if status not in {"PASS", "REVISE", "BLOCKED"}:
        status = "BLOCKED"
    return ValidationResult(status, issues, summary, validator, model, datetime.now(timezone.utc).isoformat(), raw_output)


def to_dict(result: ValidationResult):
    return asdict(result)
