from orchestrator.providers.agy import AgyProvider
from orchestrator.providers.opencode import OpenCodeProvider
from orchestrator.providers.codex import CodexProvider
from orchestrator.providers import common


def test_agy_command_uses_discovered_print_syntax():
    assert AgyProvider().build_command("hi","coder","m")==["agy","--print","--output-format","json","--model","m","hi"]


def test_opencode_command():
    provider=OpenCodeProvider(); provider._run_help="--model --format --agent"
    assert provider.build_command("hi","coder","m")==["opencode","run","--model","m","--format","json","--agent","coder","hi"]


def test_codex_validator_gets_read_only_sandbox():
    provider=CodexProvider(); provider._exec_help="--sandbox --model --json"
    cmd=provider.build_command("hi","test_validator","m")
    assert "read-only" in cmd and "--model" in cmd


def test_codex_coder_gets_workspace_sandbox():
    provider=CodexProvider(); provider._exec_help="--sandbox"
    assert "workspace-write" in provider.build_command("hi","coder")


def test_mise_binary_resolution_uses_installed_executable(tmp_path, monkeypatch):
    binary=tmp_path/"codex"; binary.write_text("#!/bin/sh\nexit 0\n"); binary.chmod(0o755)
    monkeypatch.setattr(common.shutil,"which",lambda name: "/usr/bin/mise" if name=="mise" else "/home/user/.local/bin/codex")
    class Result:
        returncode=0
        stdout=str(binary)+"\n"
    monkeypatch.setattr(common.subprocess,"run",lambda *a,**kw: Result())
    assert common.resolve_binary("codex")==str(binary)
