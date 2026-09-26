from dataclasses import dataclass


@dataclass(frozen=True)
class AgentRole:
    name: str
    tier: str
    prompt_file: str
    edits_files: bool = False
    validation: bool = False


ROLES = {
    "constitution": AgentRole("constitution", "balanced", "constitution.md", True),
    "constitution_validator": AgentRole("constitution_validator", "balanced", "constitution_validator.md", validation=True),
    "specification": AgentRole("specification", "balanced", "specification.md", True),
    "specification_validator": AgentRole("specification_validator", "balanced", "specification_validator.md", validation=True),
    "planning": AgentRole("planning", "strong", "plan.md", True),
    "plan_validator": AgentRole("plan_validator", "balanced", "plan_validator.md", validation=True),
    "tasks": AgentRole("tasks", "balanced", "tasks.md", True),
    "tasks_validator": AgentRole("tasks_validator", "balanced", "tasks_validator.md", validation=True),
    "cross_artifact_validator": AgentRole("cross_artifact_validator", "strong", "cross_validator.md", validation=True),
    "test_designer": AgentRole("test_designer", "coding_strong", "test_designer.md", True),
    "test_validator": AgentRole("test_validator", "balanced", "test_validator.md", validation=True),
    "coder": AgentRole("coder", "coding_strong", "coder.md", True),
    "refactorer": AgentRole("refactorer", "coding_strong", "refactorer.md", True),
    "code_reviewer": AgentRole("code_reviewer", "strong", "reviewer.md", validation=True),
    "debugger": AgentRole("debugger", "strong", "debugger.md", True),
    "final_reviewer": AgentRole("final_reviewer", "strong", "reviewer.md", validation=True),
}
