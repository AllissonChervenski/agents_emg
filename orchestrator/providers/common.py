import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any


def resolve_binary(binary: str) -> str | None:
    """Resolve mise-managed shims to their real installed executable when possible."""
    mise = shutil.which("mise")
    if mise:
        try:
            cp = subprocess.run([mise, "which", binary], capture_output=True, text=True, timeout=10, check=False)
            resolved = (cp.stdout or "").strip().splitlines()
            if cp.returncode == 0 and resolved:
                candidate = Path(resolved[-1])
                if candidate.is_file() and candidate.stat().st_mode & 0o111:
                    return str(candidate)
        except (OSError, subprocess.TimeoutExpired):
            pass
    return shutil.which(binary)


def execute(provider: str, model: str | None, role: str, command: list[str], cwd: Any, timeout: int | None) -> Any:
    from orchestrator.config.models import AgentResult
    start = time.monotonic()
    try:
        resolved = resolve_binary(command[0]) if command else None
        if not resolved:
            raise FileNotFoundError(f"CLI executable not found: {command[0] if command else '(empty command)'}")
        cp = subprocess.run([resolved, *command[1:]], cwd=cwd, capture_output=True, text=True, timeout=timeout, check=False)
        out = cp.stdout or ""
        structured = None
        try:
            structured = json.loads(out)
        except (ValueError, TypeError):
            pass
        return AgentResult(provider, model, role, cp.returncode == 0, cp.returncode, out, cp.stderr or "", time.monotonic()-start, structured, error=None if cp.returncode == 0 else "PROVIDER_FAILURE")
    except (OSError, subprocess.TimeoutExpired) as exc:
        return AgentResult(provider, model, role, False, None, duration=time.monotonic()-start, error=f"PROVIDER_FAILURE: {exc}")


def probe(binary: str, args: list[str], timeout: int = 20) -> tuple[bool, str, str]:
    path = resolve_binary(binary)
    if not path:
        return False, "", f"{binary} unavailable"
    try:
        cp = subprocess.run([path, *args], capture_output=True, text=True, timeout=timeout, check=False)
        output=(cp.stdout or "")
        if cp.returncode == 0 and not output.strip(): output=cp.stderr or ""
        error=((cp.stderr or "")+(cp.stdout or "")) if cp.returncode else ""
        return cp.returncode == 0, output.strip(), error.strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, "", str(exc)
