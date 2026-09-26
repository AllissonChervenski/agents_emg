from orchestrator.config.models import ProviderCapabilities
from .base import AgentProvider
from .common import execute, probe


class CodexProvider(AgentProvider):
    name = "codex"
    def __init__(self): self._models = []; self._exec_help = ""
    def discover(self):
        ok, version, err = probe("codex", ["--version"])
        help_ok, help_text, help_err = probe("codex", ["--help"])
        exec_ok, exec_text, exec_err = probe("codex", ["exec", "--help"])
        self._exec_help = exec_text if exec_ok else ""
        return ProviderCapabilities(self.name, ok, version.splitlines()[0] if ok else None, self._models, exec_ok, "json" in exec_text, "resume" in help_text, "--model" in exec_text, False, True, True, {"help": help_text, "exec_help": exec_text, "errors": [x for x in (err, help_err, exec_err) if x]})
    def build_command(self, prompt, role, model=None, cwd=None, permissions=None):
        read_only = permissions == "read" or role.lower().endswith("validator") or role in ("reviewer", "code_reviewer", "final_reviewer", "test_validator")
        cmd = ["codex", "exec"]
        if "--sandbox" in self._exec_help:
            cmd += ["--sandbox", "read-only" if read_only else "workspace-write"]
        if model and "--model" in self._exec_help: cmd += ["--model", model]
        if "--json" in self._exec_help: cmd += ["--json"]
        cmd.append(prompt)
        return cmd
    def run(self, prompt, role, model=None, cwd=None, timeout=None, permissions=None): return execute(self.name, model, role, self.build_command(prompt, role, model, cwd, permissions), cwd, timeout)
    def list_models(self): return list(self._models)
