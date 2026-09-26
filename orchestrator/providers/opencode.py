import re
from orchestrator.config.models import ProviderCapabilities
from .base import AgentProvider
from .common import execute, probe


class OpenCodeProvider(AgentProvider):
    name = "opencode"
    def __init__(self): self._models = []; self._run_help = ""
    def discover(self):
        ok, version, err = probe("opencode", ["--version"])
        help_ok, help_text, help_err = probe("opencode", ["--help"])
        models_ok, model_text, model_err = probe("opencode", ["models", "--refresh"])
        run_ok, run_text, run_err = probe("opencode", ["run", "--help"])
        self._run_help = run_text if run_ok else ""
        if models_ok: self._models = [line.strip() for line in model_text.splitlines() if line.strip()]
        return ProviderCapabilities(self.name, ok, version.splitlines()[0] if ok else None, self._models, "run" in help_text, "--format" in run_text, "session" in (help_text+run_text).lower(), "--model" in run_text, "--agent" in run_text, True, True, {"help": help_text, "run_help": run_text, "errors": [x for x in (err, help_err, model_err, run_err) if x]})
    def build_command(self, prompt, role, model=None, cwd=None, permissions=None):
        cmd = ["opencode", "run"]
        if model and "--model" in self._run_help: cmd += ["--model", model]
        if "--format" in self._run_help: cmd += ["--format", "json"]
        if role and "--agent" in self._run_help: cmd += ["--agent", role]
        cmd.append(prompt)
        return cmd
    def run(self, prompt, role, model=None, cwd=None, timeout=None, permissions=None): return execute(self.name, model, role, self.build_command(prompt, role, model, cwd, permissions), cwd, timeout)
    def list_models(self): return list(self._models)
