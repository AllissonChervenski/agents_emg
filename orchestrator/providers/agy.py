import re
from pathlib import Path
from orchestrator.config.models import ProviderCapabilities
from .base import AgentProvider
from .common import execute, probe


class AgyProvider(AgentProvider):
    name = "agy"
    def __init__(self): self._models: list[str] = []
    def discover(self):
        ok, version, err = probe("agy", ["--version"])
        help_ok, help_text, help_err = probe("agy", ["--help"])
        model_ok, model_text, model_err = probe("agy", ["models"])
        self._models = sorted(set(re.findall(r"(?m)^\s*([\w./:-]+)\s{2,}", model_text))) if model_ok else []
        return ProviderCapabilities(self.name, ok, version if ok else None, self._models, "--print" in help_text or "-p" in help_text, "json" in help_text, "--continue" in help_text or "--conversation" in help_text, "--model" in help_text, "--agent" in help_text, True, "terminal" in help_text, {"help": help_text, "errors": [x for x in (err, help_err, model_err) if x]})
    def build_command(self, prompt, role, model=None, cwd=None, permissions=None):
        cmd = ["agy", "--print", "--output-format", "json"]
        if model: cmd += ["--model", model]
        if permissions == "read": cmd += ["--mode", "plan"]
        cmd.append(prompt)
        return cmd
    def run(self, prompt, role, model=None, cwd=None, timeout=None, permissions=None):
        return execute(self.name, model, role, self.build_command(prompt, role, model, cwd, permissions), cwd, timeout)
    def list_models(self): return list(self._models)
