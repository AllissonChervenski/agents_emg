from pathlib import Path
from .models import Config


def load_config(path: str | Path = "orchestrator.yaml") -> Config:
    p = Path(path)
    if not p.exists():
        return Config(path=p)
    try:
        import yaml
        data = yaml.safe_load(p.read_text()) or {}
    except ImportError:
        raise RuntimeError("Install PyYAML to read YAML configuration")
    return Config(
        providers=data.get("providers", {}), roles=data.get("roles", {}),
        retries=data.get("retries", {"artifact_generation": 3, "implementation": 3, "review": 2}),
        verification=data.get("verification", {"build": [], "tests": [], "lint": [], "static": []}),
        human_gates=data.get("human_gates", {}), timeouts=data.get("timeouts", {"provider": 600, "verification": 600}), path=p,
    )
