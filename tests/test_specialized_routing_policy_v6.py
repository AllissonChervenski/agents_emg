from argparse import Namespace
import contextlib
import io
from pathlib import Path

from orchestrator.agents.cost import CostAwareRouter
from orchestrator.agents.roles import ROLES
from orchestrator.config.loader import load_config
from orchestrator.config.models import ProviderCapabilities
import orchestrator.cli as cli
import orchestrator.doctor as doctor
from orchestrator.doctor import ProviderSmoke, SmokeCheck


def make_test_router():
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "orchestrator.yaml")
    router, _ = cli._router(root, cfg)
    return router, cfg


def test_red_simple_selects_opencode_qwen_flash():
    router, _ = make_test_router()
    route = router.route("test_designer")
    assert route.provider == "opencode"
    assert route.model == "opencode-go/qwen3.8-flash"


def test_green_simple_selects_opencode_qwen_flash():
    router, _ = make_test_router()
    route = router.route("coder", task={"task_complexity": "LOW"})
    assert route.provider == "opencode"
    assert route.model == "opencode-go/qwen3.8-flash"


def test_green_complex_selects_kimi_before_sol():
    router, _ = make_test_router()
    route = router.route("coder", task={"task_complexity": "HIGH"})
    assert route.provider == "opencode"
    assert route.model == "opencode-go/kimi-k2.7-code"

    models_order = [c["model"] for c in route.candidates if c.get("model")]
    assert "opencode-go/kimi-k2.7-code" in models_order
    assert "gpt-6-sol" in models_order
    assert models_order.index("opencode-go/kimi-k2.7-code") < models_order.index("gpt-6-sol")


def test_refactor_prefers_kimi():
    router, _ = make_test_router()
    route = router.route("refactorer")
    assert route.provider == "opencode"
    assert route.model == "opencode-go/kimi-k2.7-code"


def test_validator_with_agy_author_prefers_codex_luna():
    router, _ = make_test_router()
    route = router.route("specification_validator", author_provider="agy")
    assert route.provider == "codex"
    assert route.model == "gpt-6-luna"
    assert route.independence is True


def test_validator_with_codex_author_prefers_agy_flash():
    router, _ = make_test_router()
    route = router.route("specification_validator", author_provider="codex")
    assert route.provider == "agy"
    assert route.model in {"gemini-3.8-flash-medium", "gemini-3.8-flash-low"}
    assert route.independence is True


def test_validator_with_opencode_author_prefers_codex_luna():
    router, _ = make_test_router()
    route = router.route("specification_validator", author_provider="opencode")
    assert route.provider == "codex"
    assert route.model == "gpt-6-luna"
    assert route.independence is True


def test_validator_never_uses_author_provider_when_independent_alternative_exists():
    router, _ = make_test_router()
    for author in ("agy", "codex", "opencode"):
        route = router.route("specification_validator", author_provider=author)
        assert route.provider != author
        assert route.independence is True


def test_provider_failure_causes_provider_fallback_not_intelligence_escalation():
    router, cfg = make_test_router()
    cost_router = CostAwareRouter(router, cfg.cost_optimization)
    base_route = router.route("coder")

    infra_failures = [
        {"provider": "opencode", "model": "opencode-go/qwen3.8-flash", "category": "PROVIDER_FAILURE"},
        {"provider": "opencode", "model": "opencode-go/qwen3.8-flash", "category": "TIMEOUT"},
        {"provider": "opencode", "model": "opencode-go/qwen3.8-flash", "category": "INFRASTRUCTURE_FAILURE"},
    ]
    assessment = cost_router.assess("coder", base_route, failures=infra_failures)
    assert assessment["previous_failure_reasons"] == []
    # No intelligence escalation occurred because failures were infrastructure/provider-related
    assert assessment["recommended_escalation_level"] == "CODING_ECONOMY"

    # Provider failure causes provider fallback in router
    fallback_route = router.route("coder", exclude={"opencode"})
    assert fallback_route.provider == "codex"
    assert fallback_route.model == "gpt-6-luna"


def test_real_capability_failure_unlocks_escalation_ladder():
    router, cfg = make_test_router()
    cost_router = CostAwareRouter(router, cfg.cost_optimization)
    base_route = router.route("coder")

    cap_failures = [
        {"provider": "opencode", "model": "opencode-go/qwen3.8-flash", "category": "GREEN_IMPLEMENTATION_FAILURE"},
        {"provider": "opencode", "model": "opencode-go/qwen3.8-flash", "category": "MODEL_CAPABILITY_FAILURE"},
    ]
    assessment = cost_router.assess("coder", base_route, failures=cap_failures)
    assert len(assessment["previous_failure_reasons"]) == 2
    for candidate in assessment["candidates"]:
        if candidate["model"] == "opencode-go/qwen3.8-flash":
            assert candidate["eligibility"] == "ineligible"


def test_astra_not_selected_in_normal_routing():
    router, _ = make_test_router()
    for role in ROLES:
        route = router.route(role)
        assert route.model != "gpt-6-astra"
        astra_candidates = [c for c in route.candidates if c.get("model") == "gpt-6-astra"]
        for cand in astra_candidates:
            assert cand["score_breakdown"].get("extreme_fallback") == -100.0


def test_doctor_smoke_check_renders_exactly_once(monkeypatch, tmp_path):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "discover_all", lambda r: {"agy": ProviderCapabilities("agy", cli_available=True)})
    monkeypatch.setattr(cli, "write_selection_report", lambda *args: None)

    check = SmokeCheck("PASS", ["agy", "--print", "AGY_SMOKE_OK"], 0, "PASS (plain stdout)", "PASS (contains AGY_SMOKE_OK)")
    mock_smoke = {"agy": ProviderSmoke("available", "supported", check, check)}
    monkeypatch.setattr(doctor, "live_smoke_tests", lambda *args: mock_smoke)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        cli.doctor(Namespace(config=str(root / "orchestrator.yaml"), live=True, verbose=True))

    output = buf.getvalue()
    assert output.count("Live provider smoke tests (explicitly requested):") == 1


def test_stage_registry_analysis_uses_consistency_agent_and_speckit_analyze():
    from orchestrator.workflow.stages import STAGE_REGISTRY

    stage = STAGE_REGISTRY.get("ANALYSIS")
    assert stage.role == "consistency_agent"
    assert stage.skill_name == "speckit-analyze"
    assert stage.mutability == "read_only"
    assert stage.prerequisites == ("TASKS",)


def test_cross_artifact_validator_is_deprecated_alias_for_consistency_agent():
    from orchestrator.agents.roles import ROLE_ALIASES, ROLES

    assert ROLE_ALIASES.get("cross_artifact_validator") == "consistency_agent"
    legacy_role = ROLES["cross_artifact_validator"]
    assert legacy_role.deprecated is True
    assert legacy_role.alias_for == "consistency_agent"

    router, _ = make_test_router()
    route_canonical = router.route("consistency_agent")
    route_alias = router.route("cross_artifact_validator")

    assert route_alias.provider == route_canonical.provider
    assert route_alias.model == route_canonical.model
    assert route_alias.tier == route_canonical.tier

    explained = router.explain("cross_artifact_validator")
    assert explained["alias_for"] == "consistency_agent"
    assert explained["role"] == "cross_artifact_validator"


def test_cli_and_selection_report_do_not_duplicate_cross_artifact_validator(tmp_path):
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "orchestrator.yaml")
    router, caps = cli._router(root, cfg)

    doctor.write_selection_report(tmp_path, caps, router)
    report_text = (tmp_path / ".orchestrator" / "model-selection.md").read_text()

    assert "`consistency_agent`" in report_text
    assert "`cross_artifact_validator`" not in report_text

    import inspect
    from orchestrator.cli import run
    run_source = inspect.getsource(run)
    assert '"consistency_agent"' in run_source
    assert '"cross_artifact_validator"' not in run_source


def test_workflow_driver_executes_consistency_agent_and_no_legacy_cross_validator(tmp_path):
    from types import SimpleNamespace
    import json
    import pytest
    from orchestrator.workflow.driver import WorkflowBlocked, run_sdd_workflow
    from orchestrator.config.models import Config
    from orchestrator.storage.sqlite import StateStore
    from orchestrator.workflow.resume import WorkspaceFingerprint

    root = tmp_path / "repo"
    root.mkdir()
    (root / ".git").mkdir()
    store = StateStore(root / ".orchestrator" / "state" / "test.sqlite3")
    wid = store.create_workflow("feature")
    folder = root / ".orchestrator" / "runs" / wid
    folder.mkdir(parents=True)

    feature_dir = root / "specs" / "feature"
    feature_dir.mkdir(parents=True)
    memory = root / ".specify" / "memory"
    memory.mkdir(parents=True)
    (root / ".specify" / "feature.json").write_text(json.dumps({"feature_directory": "specs/feature"}))
    (memory / "constitution.md").write_text("approved constitution")
    (feature_dir / "spec.md").write_text("approved spec")
    (feature_dir / "plan.md").write_text("approved plan")
    (feature_dir / "tasks.md").write_text(json.dumps({"tasks": []}))

    fp = WorkspaceFingerprint(root).capture(wid)
    for stage in ("CONSTITUTION_VALIDATED", "SPEC_VALIDATED", "CLARIFICATION_COMPLETE", "CHECKLIST_COMPLETE", "PLAN_VALIDATED", "TASKS_VALIDATED"):
        store.create_checkpoint(wid, f"{stage}:-:1", stage, fp, None)

    executed_roles = []

    class MockRunner:
        def run(self, role, prompt, *args, **kwargs):
            executed_roles.append(role)
            return SimpleNamespace(
                provider="codex",
                model="gpt-6-luna",
                success=True,
                stdout="## Speckit Analyze Report\nNo critical findings.",
                stderr="",
                error=None,
            )

        def run_skill(self, role, provider, model, skill_name, prompt, cwd, **kwargs):
            executed_roles.append(role)
            return SimpleNamespace(
                provider="codex",
                model="gpt-6-luna",
                success=True,
                stdout="## Speckit Analyze Report\nNo critical findings.",
                stderr="",
                error=None,
            )

    class MockHarness:
        def run(self, *args, **kwargs):
            return []
        requirement_results = []

    cfg = Config()
    with pytest.raises(WorkflowBlocked, match="tasks.md must contain valid SpecKit checklist tasks"):
        run_sdd_workflow("feature", root, MockRunner(), MockHarness(), cfg, folder, store=store, resume=True)

    assert executed_roles == ["consistency_agent"]
    assert "cross_artifact_validator" not in executed_roles

    checkpoints = [row["stage"] for row in store.checkpoints(wid)]
    assert "ANALYSIS_COMPLETE" in checkpoints
    assert "CROSS_VALIDATED" not in checkpoints
