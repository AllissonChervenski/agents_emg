import json
from pathlib import Path
from orchestrator.providers import PROVIDERS


def discover_all(root: str | Path = "."):
    root = Path(root)
    out = {}
    for name, cls in PROVIDERS.items():
        cap = cls().discover()
        out[name] = cap
    target = root / ".orchestrator" / "capabilities.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({k: v.__dict__ for k, v in out.items()}, indent=2))
    return out


def write_selection_report(root, capabilities, router):
    lines = ["# Model selection report", "", "Initial routing is heuristic and reflects discovered availability, configured preferences and tiers. It does not claim model superiority.", ""]
    from orchestrator.agents.roles import ROLES
    for name, role in ROLES.items():
        route = router.route(name, tier=role.tier)
        lines.append(f"- **{name}**: provider `{route.provider}`, model `{route.model or 'CLI default'}`, tier `{route.tier}` — {route.reason}.")
    path = Path(root) / ".orchestrator" / "model-selection.md"
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text("\n".join(lines)+"\n")
