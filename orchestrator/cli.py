import argparse, json, os, shutil, subprocess, sys
from pathlib import Path
from orchestrator.config.loader import load_config
from orchestrator.doctor import discover_all, write_selection_report
from orchestrator.agents.router import ModelRouter
from orchestrator.providers import PROVIDERS
from orchestrator.storage.sqlite import StateStore
from orchestrator.verification.harness import VerificationHarness
from orchestrator.workflow.tdd import TDDTask
from orchestrator.workflow.transitions import TDDPhase
from orchestrator.workflow.engine import SDD_STAGES


def _router(root, cfg):
    path = root / ".orchestrator" / "capabilities.json"
    if path.exists():
        try: data=json.loads(path.read_text())
        except ValueError: data={}
    else: data={}
    caps={}
    for name, cls in PROVIDERS.items():
        if name in data:
            from orchestrator.config.models import ProviderCapabilities
            caps[name]=ProviderCapabilities(**data[name])
        else:
            from orchestrator.config.models import ProviderCapabilities
            caps[name]=ProviderCapabilities(provider=name, metadata={"discovery":"not run; use `python -m orchestrator doctor`"})
    return ModelRouter(caps, {"roles":cfg.roles, "model_tiers":cfg.providers.get("model_tiers",{}), "provider_preference":cfg.providers.get("preference", ["agy","opencode","codex"])}), caps


def doctor(args):
    root=Path.cwd(); cfg=load_config(args.config)
    caps=discover_all(root); router=ModelRouter(caps, {"roles":cfg.roles, "model_tiers":cfg.providers.get("model_tiers",{}), "provider_preference":cfg.providers.get("preference", ["agy","opencode","codex"])}); write_selection_report(root,caps,router)
    print(f"Python .......... OK ({sys.version.split()[0]})")
    print(f"Git ............. {'OK' if shutil.which('git') else 'UNAVAILABLE'}")
    for name, cap in caps.items():
        print(f"{name.title():<16} {'OK' if cap.cli_available else 'UNAVAILABLE'} {cap.cli_version or ''}")
    print("\nModels:")
    for name, cap in caps.items(): print(f"{name}: {', '.join(cap.models) or 'not detected'}")
    print("\nVerification:")
    harness=VerificationHarness(root,cfg.verification)
    detected=harness.detect()
    for kind in ("build", "tests", "lint", "static"):
        cmds=cfg.verification.get(kind) or detected.get(kind, [])
        print(f"{kind.title()}: {'configured' if cfg.verification.get(kind) else ('detected' if cmds else 'none detected')}")
    print(f"Capabilities saved to .orchestrator/capabilities.json")


def models(args):
    root=Path.cwd(); cfg=load_config(args.config); router,caps=_router(root,cfg); write_selection_report(root,caps,router)
    for name, cap in caps.items(): print(f"{name}: {', '.join(cap.models) or ('available, CLI default model' if cap.cli_available else 'unavailable')}")
    print("Routing is heuristic; see .orchestrator/model-selection.md")


def run(args):
    root=Path.cwd(); cfg=load_config(args.config); router,caps=_router(root,cfg)
    feature=args.feature or (Path(args.feature_file).read_text() if args.feature_file else "")
    if not feature.strip(): raise SystemExit("Provide --feature or --feature-file")
    harness=VerificationHarness(root,cfg.verification)
    roles=["constitution","constitution_validator","specification","specification_validator","planning","plan_validator","tasks","tasks_validator","cross_artifact_validator","test_designer","test_validator","coder","refactorer","code_reviewer","final_reviewer"]
    print("Workflow: Constitution → Spec → Plan → Tasks → Cross validation → TDD (Analyze/Red/Green/Refactor/Review) → Final verification")
    print("Roles:", ", ".join(roles))
    for role in roles:
        route=router.route(role, override_provider=getattr(args, "coder_provider", None) if role=="coder" else None)
        print(f"  {role}: {route.provider}/{route.model or 'CLI default'} [{route.tier}]")
    print("Validation gates: artifacts; cross-artifact; test validation; RED expected failure; GREEN; regression; review; deterministic final verification")
    detected=harness.detect()
    commands={key:(cfg.verification.get(key) or detected.get(key,[])) for key in ("build","tests","lint","static")}
    print("Verification commands:", json.dumps(commands))
    if args.dry_run: return
    from orchestrator.git.repository import GitRepository
    repository=GitRepository(root)
    if repository.available() and repository.dirty():
        raise SystemExit("Refusing to run agents: Git working tree has uncommitted changes. Commit or stash them before execution.")
    if not repository.available():
        print("Git repository: unavailable; task checkpoints are disabled")
    store=StateStore(root/".orchestrator"/"state"/"orchestrator.sqlite3")
    wid=store.create_workflow(feature, {"stage":"PLANNED","attempts":{},"completed_tasks":[],"blocked_tasks":[],"validation_results":[],"verification_results":[]})
    run_dir=root/".orchestrator"/"runs"/wid
    for d in ("state","logs","runs","reports"): (root/".orchestrator"/d).mkdir(parents=True,exist_ok=True)
    (run_dir).mkdir(parents=True,exist_ok=True)
    from orchestrator.agents.runner import AgentRunner
    from orchestrator.workflow.driver import run_sdd_workflow, WorkflowBlocked
    provider_instances={name:cls() for name,cls in PROVIDERS.items()}
    # Refresh per-adapter discovered flags used to safely construct commands.
    for provider in provider_instances.values(): provider.discover()
    runner=AgentRunner(provider_instances,router,store)
    def approve_constitution():
        if not sys.stdin.isatty(): return False
        return input("Create constitution.md? This is a human-gated project change. Type 'approve': ").strip()=="approve"
    try:
        result=run_sdd_workflow(feature,root,runner,harness,cfg,run_dir,approve_constitution)
        state={"stage":"COMPLETE","completed_tasks":result["completed_tasks"],"traceability":result["traceability"],"verification_results":result["final_verification"]}
        store.update_workflow(wid,"COMPLETE",state)
        print(f"Workflow complete: {wid}")
    except WorkflowBlocked as exc:
        state={"stage":"BLOCKED","reason":str(exc)}
        store.update_workflow(wid,"BLOCKED",state)
        (root/".orchestrator"/"reports"/f"{wid}.json").write_text(json.dumps(state,indent=2))
        print(f"Workflow BLOCKED: {exc}\nWorkflow ID: {wid}")


def status(args):
    store=StateStore(Path.cwd()/".orchestrator"/"state"/"orchestrator.sqlite3")
    print(json.dumps(store.list_workflows(),indent=2))


def resume(args):
    store=StateStore(Path.cwd()/".orchestrator"/"state"/"orchestrator.sqlite3"); item=store.get_workflow(args.workflow_id)
    if not item: raise SystemExit("Workflow not found")
    print(json.dumps(item,indent=2))


def verify(args):
    root=Path.cwd(); cfg=load_config(args.config); harness=VerificationHarness(root,cfg.verification)
    results=harness.run()
    for r in results: print(f"{'PASS' if r.success else r.classification}: {' '.join(r.command)} ({r.duration:.2f}s)")
    if not results: print("No verification commands detected or configured")
    if results and any(not r.success for r in results): raise SystemExit(1)


def validate(args):
    from orchestrator.validation.parser import parse_validation
    path=Path(args.artifact)
    if not path.exists(): raise SystemExit(f"Artifact not found: {path}")
    text=path.read_text(); required=args.requirement or []
    missing=[req for req in required if req not in text]
    status="PASS" if not missing else "REVISE"
    result={"status":status,"issues":[{"id":f"missing-{i}","severity":"major","artifact":str(path),"location":"","requirement":req,"description":"Requirement ID missing","suggested_action":"Add explicit requirement mapping"} for i,req in enumerate(missing)],"summary":"Deterministic textual requirement check","validator":"python","model":None}
    print(json.dumps(result,indent=2))


def main():
    parser=argparse.ArgumentParser(prog="orchestrator",description="Spec-driven development orchestrator")
    sub=parser.add_subparsers(dest="command",required=True)
    p=sub.add_parser("doctor"); p.add_argument("--config",default="orchestrator.yaml"); p.set_defaults(func=doctor)
    p=sub.add_parser("models"); p.add_argument("--config",default="orchestrator.yaml"); p.set_defaults(func=models)
    p=sub.add_parser("run"); p.add_argument("--feature"); p.add_argument("--feature-file"); p.add_argument("--coder-provider"); p.add_argument("--dry-run",action="store_true"); p.add_argument("--config",default="orchestrator.yaml"); p.set_defaults(func=run)
    p=sub.add_parser("resume"); p.add_argument("workflow_id"); p.set_defaults(func=resume)
    p=sub.add_parser("status"); p.set_defaults(func=status)
    p=sub.add_parser("verify"); p.add_argument("--config",default="orchestrator.yaml"); p.set_defaults(func=verify)
    p=sub.add_parser("validate"); p.add_argument("artifact"); p.add_argument("--requirement",action="append"); p.set_defaults(func=validate)
    args=parser.parse_args(); args.func(args)
