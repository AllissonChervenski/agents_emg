from dataclasses import dataclass
from typing import Any


TIERS = ("fast", "balanced", "strong", "coding_strong")


@dataclass
class Route:
    provider: str
    model: str | None
    tier: str
    reason: str


class ModelRouter:
    def __init__(self, capabilities: dict[str, Any], config: dict[str, Any] | None = None):
        self.capabilities = capabilities
        self.config = config or {}

    def route(self, role: str, tier: str | None = None, exclude: set[str] | None = None, override_provider: str | None = None, override_model: str | None = None) -> Route:
        role_cfg = self.config.get("roles", {}).get(role, {})
        try:
            from orchestrator.agents.roles import ROLES
            default_tier=ROLES[role].tier
        except (KeyError, ImportError):
            default_tier="balanced"
        tier = (tier or role_cfg.get("tier") or default_tier).lower()
        override_provider = override_provider or role_cfg.get("provider")
        excluded = exclude or set()
        preference = self.config.get("provider_preference", ["agy", "opencode", "codex"])
        if override_provider:
            preference = [override_provider] + [p for p in preference if p != override_provider]
        tier_map = self.config.get("model_tiers", {})
        candidates = [p for p in preference if p not in excluded and self.capabilities.get(p) and self.capabilities[p].cli_available]
        if not candidates:
            return Route(override_provider or "unavailable", override_model, tier, "No discovered provider is available")
        provider = candidates[0]
        configured = tier_map.get(tier, {}).get(provider)
        models = self.capabilities[provider].models
        model = override_model or configured or (models[0] if models else None)
        return Route(provider, model, tier, f"Heuristic tier={tier}; preferred available provider; model availability is discovery/config based")

    def independent_route(self, role: str, author_provider: str, tier: str | None = None) -> tuple[Route, bool]:
        route = self.route(role, tier=tier, exclude={author_provider})
        if route.provider == "unavailable":
            return self.route(role, tier=tier), False
        return route, True
