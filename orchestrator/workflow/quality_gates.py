"""Small control-plane parsers for SpecKit quality-stage outcomes.

These helpers recognize stage inputs and outputs; they do not implement the
clarify, checklist, analyze, or converge methodology owned by SpecKit skills.
"""

from __future__ import annotations

import re
from collections.abc import Callable


_CLARIFICATION = re.compile(r"\[NEEDS CLARIFICATION(?::\s*(.*?))?\]", re.IGNORECASE)


def clarification_questions(spec_text: str) -> list[str]:
    """Return the unresolved questions explicitly marked in a SpecKit spec."""
    return [
        (match.group(1) or "").strip() or "Please clarify this requirement."
        for match in _CLARIFICATION.finditer(spec_text)
    ]


def analyze_has_critical_findings(report: str) -> bool:
    """Recognize critical findings in the installed analyze skill's report."""
    if re.search(r"Critical Issues Count\s*[:|]\s*[1-9]\d*", report, re.IGNORECASE):
        return True
    return bool(re.search(r"\|\s*CRITICAL\s*\|", report, re.IGNORECASE))


def convergence_outcome(before: bytes, after: bytes, report: str) -> str:
    """Validate the converge skill's append-only artifact contract."""
    if after != before:
        if not after.startswith(before):
            raise ValueError("speckit-converge changed existing tasks.md content; append-only contract violated")
        return "tasks_appended"
    if re.search(r"\bconverged\b", report, re.IGNORECASE):
        return "converged"
    raise ValueError("speckit-converge returned no append and no explicit converged result")


def run_convergence_loop(
    max_iterations: int,
    verify: Callable[[int], None],
    converge: Callable[[int], str],
    implement_remaining: Callable[[int], None],
) -> int:
    """Run deterministic verification and semantic convergence in a bounded loop.

    The initial implementation pass has already run through the harness's
    task-level TDD engine. ``implement_remaining`` is called only after
    converge appends work, so that same engine remains the sole task executor.
    """
    if max_iterations < 1:
        raise ValueError("max_convergence_iterations must be at least 1")
    for iteration in range(1, max_iterations + 1):
        verify(iteration)
        outcome = converge(iteration)
        if outcome == "converged":
            return iteration
        if outcome != "tasks_appended":
            raise ValueError(f"Unsupported convergence outcome: {outcome}")
        if iteration == max_iterations:
            raise ValueError(
                f"speckit-converge found remaining tasks after max_convergence_iterations={max_iterations}"
            )
        implement_remaining(iteration + 1)
    raise AssertionError("unreachable convergence loop exit")
