from dataclasses import dataclass, field
from pathlib import Path
import json
from .transitions import TDDPhase, transition


@dataclass
class TDDTask:
    task: str
    requirements: list[str] = field(default_factory=list)
    acceptance_criteria: list[str] = field(default_factory=list)
    test_type: str = "UNIT"
    phase: TDDPhase = TDDPhase.ANALYZE
    evidence: dict = field(default_factory=dict)
    attempts: dict = field(default_factory=lambda: {"red": 0, "green": 0, "refactor": 0})

    def advance(self, target, evidence=None):
        merged = {**self.evidence, **(evidence or {})}
        self.phase = transition(self.phase, target, merged)
        self.evidence = merged
        return self.phase

    def save(self, path: str | Path):
        payload = {"task": self.task, "requirement": self.requirements, "acceptance_criteria": self.acceptance_criteria, "test_type": self.test_type, "phase": self.phase.value, "attempts": self.attempts, **self.evidence}
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(payload, indent=2))

    def complete(self):
        return self.phase == TDDPhase.COMPLETE
