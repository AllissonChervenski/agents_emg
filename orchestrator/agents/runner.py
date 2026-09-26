from pathlib import Path
from string import Template
from orchestrator.agents.roles import ROLES
from orchestrator.config.models import AgentResult


def load_prompt(role: str, **values: str) -> str:
    spec = ROLES[role]
    path = Path(__file__).resolve().parents[1] / "prompts" / spec.prompt_file
    return Template(path.read_text()).safe_substitute(**values)


class AgentRunner:
    def __init__(self, providers, router, store=None):
        self.providers, self.router, self.store = providers, router, store

    def run(self, role: str, prompt: str, cwd=None, timeout=None, override_provider=None, override_model=None, fallback=True, exclude_providers=None):
        route = self.router.route(role, override_provider=override_provider, override_model=override_model, exclude=exclude_providers)
        independent = True
        if route.provider == "unavailable" and exclude_providers:
            route = self.router.route(role, override_provider=override_provider, override_model=override_model)
            independent = False
        if route.provider == "unavailable":
            return AgentResult("unavailable", None, role, False, error="PROVIDER_FAILURE: no provider available")
        result = self.providers[route.provider].run(prompt, role, route.model, cwd, timeout, "read" if ROLES[role].validation else None)
        result.usage["validation_independence"] = independent
        if not result.success and fallback:
            second = self.router.route(role, exclude=set(exclude_providers or ()) | {route.provider})
            if second.provider != "unavailable":
                result = self.providers[second.provider].run(prompt, role, second.model, cwd, timeout, "read" if ROLES[role].validation else None)
                result.usage["validation_independence"] = independent and second.provider not in set(exclude_providers or ())
        if self.store:
            self.store.record_provider_execution(result, prompt)
        return result
