from abc import ABC, abstractmethod
from pathlib import Path
from orchestrator.config.models import AgentResult, ProviderCapabilities


class AgentProvider(ABC):
    name = "base"

    @abstractmethod
    def discover(self) -> ProviderCapabilities: ...

    @abstractmethod
    def build_command(self, prompt: str, role: str, model: str | None = None, cwd: str | Path | None = None, permissions: str | None = None) -> list[str]: ...

    @abstractmethod
    def run(self, prompt: str, role: str, model: str | None = None, cwd: str | Path | None = None, timeout: int | None = None, permissions: str | None = None) -> AgentResult: ...

    @abstractmethod
    def list_models(self) -> list[str]: ...
