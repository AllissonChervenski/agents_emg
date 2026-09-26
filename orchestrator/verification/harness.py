from dataclasses import dataclass, asdict
from pathlib import Path
import json, shlex, subprocess, time
from .policies import CommandPolicy


@dataclass
class VerificationResult:
    command: list[str]
    success: bool
    exit_code: int | None
    stdout: str
    stderr: str
    duration: float
    classification: str = "PASS"


class VerificationHarness:
    def __init__(self, workspace, commands=None, timeout=600):
        self.workspace = Path(workspace).resolve(); self.commands = commands or {}; self.timeout = timeout; self.policy = CommandPolicy(self.workspace)
    def detect(self):
        root = self.workspace
        found = {"build": [], "tests": [], "lint": [], "static": []}
        if (root / "pyproject.toml").exists() or list(root.glob("test*.py")) or (root / "tests").exists():
            found["tests"].append(["python", "-m", "pytest", "-q"])
        if (root / "package.json").exists(): found["tests"].append(["npm", "test"])
        if (root / "Cargo.toml").exists(): found["tests"].append(["cargo", "test"])
        if (root / "go.mod").exists(): found["tests"].append(["go", "test", "./..."])
        if (root / "Makefile").exists(): found["build"].append(["make"])
        if (root / "CMakeLists.txt").exists(): found["build"].append(["cmake", "--build", "build"])
        if (root / "platformio.ini").exists(): found["build"].append(["pio", "run"])
        return found
    def run_command(self, command):
        args = shlex.split(command) if isinstance(command, str) else list(command)
        allowed, reason = self.policy.validate(args)
        if not allowed: return VerificationResult(args, False, None, "", reason, 0, "BLOCKED")
        start = time.monotonic()
        try:
            cp = subprocess.run(args, cwd=self.workspace, text=True, capture_output=True, timeout=self.timeout, check=False)
            return VerificationResult(args, cp.returncode == 0, cp.returncode, cp.stdout, cp.stderr, time.monotonic()-start, "PASS" if cp.returncode == 0 else "FAIL")
        except (OSError, subprocess.TimeoutExpired) as exc:
            return VerificationResult(args, False, None, "", str(exc), time.monotonic()-start, "INFRASTRUCTURE_FAILURE")
    def run(self, categories=None):
        detected = self.detect()
        configured = {key: (value or detected.get(key, [])) for key, value in (self.commands or detected).items()}
        results = []
        for category in categories or ("tests", "build", "lint", "static"):
            commands = configured.get(category, [])
            for command in commands:
                results.append(self.run_command(command))
        return results

    def run_red(self, command, expected_markers=()):
        result = self.run_command(command)
        output = result.stdout + result.stderr
        lowered=output.lower()
        discovered = any(token in lowered for token in ("collected ", "ran ", "=== fail", "failed:", "--- fail", "test result: failed"))
        if result.classification == "INFRASTRUCTURE_FAILURE": result.classification = "INFRASTRUCTURE_FAILURE"
        elif not discovered or any(token in lowered for token in ("syntaxerror", "modulenotfounderror", "importerror", "no tests ran", "no tests collected")): result.classification = "INVALID_TEST"
        elif result.success: result.classification = "UNEXPECTED_FAILURE"
        elif not any(token in lowered for token in ("assertionerror", "assert ", "assertion failed", "expected:", "not equal", "failed: test", "=== fail", "test result: failed")): result.classification = "UNEXPECTED_FAILURE"
        elif expected_markers and not any(marker.lower() in output.lower() for marker in expected_markers): result.classification = "UNEXPECTED_FAILURE"
        else: result.classification = "EXPECTED_FAILURE"
        return result

    def save(self, path, results):
        Path(path).parent.mkdir(parents=True, exist_ok=True); Path(path).write_text(json.dumps([asdict(r) for r in results], indent=2))
