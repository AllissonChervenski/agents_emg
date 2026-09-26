from types import SimpleNamespace
from orchestrator.agents.router import ModelRouter


def test_router_uses_available_provider_and_tier_config():
    caps={"agy":SimpleNamespace(cli_available=True,models=["a"]),"codex":SimpleNamespace(cli_available=True,models=["c"])}
    router=ModelRouter(caps,{"provider_preference":["codex","agy"],"roles":{"coder":{"tier":"coding_strong"}},"model_tiers":{"coding_strong":{"codex":"c"}}})
    route=router.route("coder")
    assert (route.provider,route.model,route.tier)==("codex","c","coding_strong")


def test_independent_route_uses_alternative():
    caps={n:SimpleNamespace(cli_available=True,models=[]) for n in ("agy","codex")}
    route, independent=ModelRouter(caps).independent_route("code_reviewer","agy")
    assert route.provider=="codex" and independent
