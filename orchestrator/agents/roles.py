from dataclasses import dataclass


@dataclass(frozen=True)
class AgentRole:
    name: str
    tier: str
    prompt_file: str
    edits_files: bool = False
    validation: bool = False
    required_capabilities: tuple[str, ...] = ()
    preferred_capabilities: tuple[str, ...] = ()
    preferred_providers: tuple[str, ...] = ()
    prefer_different_provider_from_author: bool = False


ROLES = {
    "constitution": AgentRole("constitution", "balanced", "constitution.md", True, required_capabilities=("DOCUMENT_GENERATION", "REASONING"), preferred_providers=("agy", "opencode", "codex")),
    "constitution_validator": AgentRole("constitution_validator", "balanced", "constitution_validator.md", validation=True, required_capabilities=("VALIDATION", "REASONING"), preferred_providers=("codex", "opencode", "agy"), prefer_different_provider_from_author=True),
    "specification": AgentRole("specification", "balanced", "specification.md", True, required_capabilities=("DOCUMENT_GENERATION", "REASONING"), preferred_providers=("agy", "codex", "opencode")),
    "specification_validator": AgentRole("specification_validator", "balanced", "specification_validator.md", validation=True, required_capabilities=("VALIDATION", "REASONING"), preferred_capabilities=("STRUCTURED_OUTPUT",), preferred_providers=("codex", "agy", "opencode"), prefer_different_provider_from_author=True),
    "planning": AgentRole("planning", "strong", "plan.md", True, required_capabilities=("REASONING", "DOCUMENT_GENERATION"), preferred_providers=("codex", "agy", "opencode")),
    "plan_validator": AgentRole("plan_validator", "balanced", "plan_validator.md", validation=True, required_capabilities=("VALIDATION", "REASONING"), preferred_providers=("codex", "agy", "opencode"), prefer_different_provider_from_author=True),
    "tasks": AgentRole("tasks", "balanced", "tasks.md", True, required_capabilities=("DOCUMENT_GENERATION", "REASONING"), preferred_providers=("agy", "codex", "opencode")),
    "tasks_validator": AgentRole("tasks_validator", "balanced", "tasks_validator.md", validation=True, required_capabilities=("VALIDATION", "REASONING"), preferred_providers=("codex", "agy", "opencode"), prefer_different_provider_from_author=True),
    "cross_artifact_validator": AgentRole("cross_artifact_validator", "strong", "cross_validator.md", validation=True, required_capabilities=("VALIDATION", "REASONING"), preferred_providers=("codex", "agy", "opencode")),
    "test_designer": AgentRole("test_designer", "coding_strong", "test_designer.md", True, required_capabilities=("CODING", "FILE_EDITING"), preferred_capabilities=("AGENTIC_CODING",), preferred_providers=("codex", "agy", "opencode")),
    "test_validator": AgentRole("test_validator", "balanced", "test_validator.md", validation=True, required_capabilities=("VALIDATION", "REASONING"), preferred_providers=("codex", "agy", "opencode"), prefer_different_provider_from_author=True),
    "coder": AgentRole("coder", "coding_strong", "coder.md", True, required_capabilities=("CODING", "FILE_EDITING", "SHELL"), preferred_capabilities=("AGENTIC_CODING",), preferred_providers=("opencode", "codex", "agy")),
    "refactorer": AgentRole("refactorer", "coding_strong", "refactorer.md", True, required_capabilities=("CODING", "FILE_EDITING"), preferred_capabilities=("AGENTIC_CODING",), preferred_providers=("opencode", "codex", "agy")),
    "code_reviewer": AgentRole("code_reviewer", "strong", "reviewer.md", validation=True, required_capabilities=("VALIDATION", "CODING"), preferred_providers=("codex", "agy", "opencode"), prefer_different_provider_from_author=True),
    "debugger": AgentRole("debugger", "strong", "debugger.md", True, required_capabilities=("CODING", "REASONING"), preferred_providers=("codex", "opencode", "agy")),
    "final_reviewer": AgentRole("final_reviewer", "strong", "reviewer.md", validation=True, required_capabilities=("VALIDATION", "REASONING"), preferred_providers=("codex", "agy", "opencode"), prefer_different_provider_from_author=True),
}
