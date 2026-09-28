"""Deterministic postcondition verification across all SDD and TDD stages.

A provider SUCCESS status or lack of an exception never suffices on its own
to create a durable checkpoint. Every stage must satisfy its concrete,
deterministic artifact and state postcondition before checkpointing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from orchestrator.workflow.constitution import classify_constitution, find_constitution_placeholders
from orchestrator.workflow.task_adapter import parse_speckit_tasks, TaskContractError
from orchestrator.workflow.quality_gates import clarification_questions, analyze_has_critical_findings


class PostconditionError(ValueError):
    """Raised when a stage postcondition is violated."""
    pass


def verify_constitution_created(constitution_path: Path | str) -> None:
    """Verify that constitution artifact exists and is not empty or template."""
    path = Path(constitution_path)
    if not path.is_file():
        raise PostconditionError(f"Constitution artifact missing at {path}")
    if path.stat().st_size == 0:
        raise PostconditionError(f"Constitution artifact at {path} is empty")
    status = classify_constitution(path)
    if status != "VALID":
        placeholders = find_constitution_placeholders(path.read_text(encoding="utf-8", errors="replace"))
        detail = f"placeholders remaining: {placeholders}" if placeholders else f"classified as {status}"
        raise PostconditionError(f"Constitution is not valid: {detail}")


def verify_specification_valid(spec_path: Path | str) -> None:
    """Verify that specification exists and is non-empty."""
    path = Path(spec_path)
    if not path.is_file():
        raise PostconditionError(f"Specification artifact missing at {path}")
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        raise PostconditionError(f"Specification artifact at {path} is empty")


def verify_clarification_complete(spec_path: Path | str) -> None:
    """Verify specification exists and has no unresolved clarification markers."""
    verify_specification_valid(spec_path)
    text = Path(spec_path).read_text(encoding="utf-8", errors="replace")
    questions = clarification_questions(text)
    if questions:
        raise PostconditionError(
            f"Specification at {spec_path} contains unresolved clarification markers: {questions}"
        )


def verify_checklist_complete(checklist_dir: Path | str, allow_legacy_absent: bool = False) -> None:
    """Verify that feature checklists directory exists and contains non-empty markdown checklist."""
    dir_path = Path(checklist_dir)
    if not dir_path.is_dir():
        if allow_legacy_absent:
            return
        raise PostconditionError(f"Checklist directory missing at {dir_path}")
    md_files = [f for f in dir_path.glob("*.md") if f.is_file() and f.stat().st_size > 0]
    if not md_files:
        raise PostconditionError(f"No non-empty checklist markdown files found in {dir_path}")


def verify_plan_valid(plan_path: Path | str) -> None:
    """Verify that plan exists and is non-empty."""
    path = Path(plan_path)
    if not path.is_file():
        raise PostconditionError(f"Plan artifact missing at {path}")
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        raise PostconditionError(f"Plan artifact at {path} is empty")


def verify_tasks_valid(tasks_path: Path | str) -> list[dict[str, Any]]:
    """Verify that tasks.md exists, is non-empty, and parses into executable task contracts."""
    path = Path(tasks_path)
    if not path.is_file():
        raise PostconditionError(f"Tasks artifact missing at {path}")
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        raise PostconditionError("tasks.md must contain valid SpecKit checklist tasks and harness metadata")
    try:
        tasks = parse_speckit_tasks(text)
    except TaskContractError as exc:
        raise PostconditionError(f"tasks.md must contain valid SpecKit checklist tasks and harness metadata: {exc}") from exc
    if not tasks:
        raise PostconditionError("tasks.md must contain valid SpecKit checklist tasks and harness metadata")
    for task in tasks:
        if not task.get("id"):
            raise PostconditionError(f"Task missing required 'id' in {path}")
        if "dependencies" not in task:
            raise PostconditionError(f"Task {task.get('id')} missing 'dependencies' in {path}")
    return tasks


def verify_analysis_complete(report_path: Path | str, allow_legacy_absent: bool = False) -> None:
    """Verify that analysis report exists, is non-empty, and has no critical findings."""
    path = Path(report_path)
    if not path.is_file():
        if allow_legacy_absent:
            return
        raise PostconditionError(f"Analysis report missing at {path}")
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        raise PostconditionError(f"Analysis report at {path} is empty")
    if analyze_has_critical_findings(text):
        raise PostconditionError(f"Analysis report at {path} contains critical cross-artifact findings")


def verify_stage_postcondition(stage: str, **kwargs: Any) -> bool:
    """Verify deterministic stage postcondition. Returns True or raises PostconditionError."""
    if stage in {"CONSTITUTION_CREATE", "CONSTITUTION_CREATED"}:
        verify_constitution_created(kwargs["constitution_path"])
    elif stage in {"SPECIFY", "SPEC_VALIDATED"}:
        verify_specification_valid(kwargs["spec_path"])
    elif stage == "CLARIFICATION_COMPLETE":
        verify_clarification_complete(kwargs["spec_path"])
    elif stage == "CHECKLIST_COMPLETE":
        verify_checklist_complete(kwargs["checklist_dir"], allow_legacy_absent=kwargs.get("allow_legacy_absent", False))
    elif stage in {"PLAN", "PLAN_VALIDATED"}:
        verify_plan_valid(kwargs["plan_path"])
    elif stage in {"TASKS", "TASKS_VALIDATED"}:
        verify_tasks_valid(kwargs["tasks_path"])
    elif stage == "ANALYSIS_COMPLETE":
        verify_analysis_complete(kwargs["report_path"], allow_legacy_absent=kwargs.get("allow_legacy_absent", False))
    return True
