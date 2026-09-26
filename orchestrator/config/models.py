from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ProviderCapabilities:
    provider: str
    cli_available: bool = False
    cli_version: str | None = None
    models: list[str] = field(default_factory=list)
    supports_headless: bool = False
    supports_json: bool = False
    supports_sessions: bool = False
    supports_model_selection: bool = False
    supports_agent_selection: bool = False
    supports_file_editing: bool = False
    supports_shell: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResult:
    provider: str
    model: str | None
    role: str
    success: bool
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    duration: float = 0.0
    structured_output: Any = None
    usage: dict[str, Any] = field(default_factory=dict)
    session_id: str | None = None
    error: str | None = None


@dataclass
class ValidationIssue:
    id: str
    severity: str
    artifact: str
    location: str
    requirement: str
    description: str
    suggested_action: str


@dataclass
class ValidationResult:
    status: str
    issues: list[ValidationIssue]
    summary: str
    validator: str
    model: str | None
    timestamp: str
    raw_output: str = ""


@dataclass
class Config:
    providers: dict[str, Any] = field(default_factory=dict)
    roles: dict[str, Any] = field(default_factory=dict)
    retries: dict[str, int] = field(default_factory=lambda: {"artifact_generation": 3, "implementation": 3, "review": 2})
    verification: dict[str, list[str]] = field(default_factory=lambda: {"build": [], "tests": [], "lint": [], "static": []})
    human_gates: dict[str, Any] = field(default_factory=dict)
    timeouts: dict[str, int] = field(default_factory=lambda: {"provider": 600, "verification": 600})
    path: Path = Path("orchestrator.yaml")
