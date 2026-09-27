from pathlib import Path
from string import Template
from orchestrator.agents.roles import ROLES
from orchestrator.config.models import AgentResult
from uuid import uuid4
import hashlib
import time
from datetime import datetime, timezone
from orchestrator.agents.history import classify_task


def load_prompt(role: str, **values: str) -> str:
    spec = ROLES[role]
    path = Path(__file__).resolve().parents[1] / "prompts" / spec.prompt_file
    return Template(path.read_text()).safe_substitute(**values)


class AgentRunner:
    def __init__(self, providers, router, store=None, workflow_id=None, safety=None):
        self.providers, self.router, self.store = providers, router, store
        self.on_fallback=None
        self.on_call=None
        self.workflow_id=workflow_id
        self.safety=safety or {}
        self.started_at=time.monotonic()
        self.calls=0
        self.task_calls={}
        self.elapsed_prior=0.0
        if store and workflow_id:
            self.calls,self.task_calls=store.agent_call_counts(workflow_id)
            item=store.get_workflow(workflow_id)
            if item:
                self.elapsed_prior=max(0.0,(datetime.now(timezone.utc)-datetime.fromisoformat(item["created_at"])).total_seconds())

    @staticmethod
    def _snapshot(root):
        if root is None: return {}
        base=Path(root).resolve()
        ignored={".git",".venv","__pycache__","build","dist"}
        def relevant(path):
            parts=path.relative_to(base).parts
            if any(part in ignored for part in parts): return False
            return parts[0]!=".orchestrator" or len(parts)>1 and parts[1]=="runs"
        return {str(p.relative_to(base)):hashlib.sha256(str(p.readlink()).encode() if p.is_symlink() else p.read_bytes()).hexdigest() for p in base.rglob("*") if (p.is_file() or p.is_symlink()) and relevant(p)}

    @staticmethod
    def _in_scope(path, allowed):
        return any(path==item.rstrip("/") or path.startswith(item.rstrip("/")+"/") for item in allowed)

    def _budget_error(self, task_id):
        if self.calls>=int(self.safety.get("max_agent_calls_per_workflow",100)): return "BUDGET_EXCEEDED: workflow agent calls"
        if task_id and self.task_calls.get(task_id,0)>=int(self.safety.get("max_agent_calls_per_task",12)): return "BUDGET_EXCEEDED: task agent calls"
        if time.monotonic()-self.started_at+self.elapsed_prior>=float(self.safety.get("max_wall_time",7200)): return "BUDGET_EXCEEDED: wall time"
        return None

    def record_validation(self, agent_result, validation_result):
        if not self.store: return
        execution_id=agent_result.usage.get("execution_id")
        if execution_id:
            self.store.update_execution_outcome(execution_id,structured_output_valid=validation_result.summary not in ("Could not parse strict validation output","Malformed issue schema"),validator_accepted=validation_result.status=="PASS",blocked=validation_result.status=="BLOCKED")
        if validation_result.summary in ("Could not parse strict validation output","Malformed issue schema"):
            self.store.record_metric(agent_result.provider,agent_result.model,agent_result.role,"structured_output_failure")
        elif validation_result.status!="PASS":
            self.store.record_metric(agent_result.provider,agent_result.model,agent_result.role,"validator_rejection")

    def run(self, role: str, prompt: str, cwd=None, timeout=None, override_provider=None, override_model=None, fallback=True, exclude_providers=None, author_provider=None, task=None, task_id=None, allowed_paths=None, expected_outputs=None):
        invocation_id=str(uuid4())
        route_excludes=set(exclude_providers or ())
        if author_provider is None and exclude_providers and len(exclude_providers)==1:
            author_provider=next(iter(exclude_providers))
            route_excludes.discard(author_provider)
        task_id=task_id or (task or {}).get("id")
        kind,difficulty=classify_task(role,task)
        route = self.router.route(role, override_provider=override_provider, override_model=override_model, exclude=route_excludes, author_provider=author_provider,task=task)
        if route.provider == "unavailable":
            return AgentResult("unavailable", None, role, False, error="PROVIDER_FAILURE: no provider available")
        if self.safety.get("safety_mode")=="strict" and role in {"coder","refactorer"} and not allowed_paths:
            return AgentResult(route.provider,route.model,role,False,error="SCOPE_VIOLATION: production files are not declared for this task")
        configured_scope=self.safety.get("file_scopes",{}).get(role)
        permitted=list(allowed_paths or (configured_scope if isinstance(configured_scope,list) else (["tests/","fixtures/","conftest.py","pytest.ini","tox.ini"] if role=="test_designer" else [])))
        def execute(chosen,attempt):
            budget=self._budget_error(task_id)
            if budget: return AgentResult(chosen.provider,chosen.model,role,False,error=budget)
            outputs=expected_outputs or (["structured validation JSON"] if ROLES[role].validation else (["task tests"] if role=="test_designer" else (["declared production files"] if role in {"coder","refactorer"} else ["SDD artifact or structured task output"])))
            if self.on_call and not self.on_call(role,chosen.provider,chosen.model,task_id,permitted,outputs):
                return AgentResult(chosen.provider,chosen.model,role,False,error="INTERACTIVE_ABORT")
            before=self._snapshot(cwd) if self.safety.get("safety_mode")=="strict" else {}
            self.calls+=1
            if task_id: self.task_calls[task_id]=self.task_calls.get(task_id,0)+1
            if self.store and self.workflow_id:
                item=self.store.get_workflow(self.workflow_id)
                if item: self.store.update_workflow(self.workflow_id,"RUNNING_AGENT",{**item["state"],"running_role":role,"running_task":task_id,"running_provider":chosen.provider,"running_model":chosen.model},task_id)
            result=self.providers[chosen.provider].run(prompt,role,chosen.model,cwd,timeout,"read" if ROLES[role].validation else None)
            if self.safety.get("safety_mode")=="strict":
                after=self._snapshot(cwd)
                changed=sorted(path for path in set(before)|set(after) if before.get(path)!=after.get(path))
                violations=[path for path in changed if not self._in_scope(path,permitted)]
                if violations:
                    result.success=False; result.error="SCOPE_VIOLATION: "+", ".join(violations)
                    result.usage["scope_violations"]=violations
                if changed and not result.success and self.store and self.workflow_id:
                    item=self.store.get_workflow(self.workflow_id)
                    if item: self.store.update_workflow(self.workflow_id,"PARTIAL_WRITE",{**item["state"],"interrupted_stage":"PARTIAL_WRITE","partial_files":changed},task_id)
                if changed and not result.success and not violations:
                    result.error="PARTIAL_WRITE: provider failed after modifying "+", ".join(changed)
                    result.usage["partial_files"]=changed
            result.usage["validation_independence"]=chosen.independence
            result.usage["validation_independence_reason"]=chosen.reason if author_provider else None
            result.usage["routing_selection_mode"]=chosen.selection_mode
            result.usage["task_type"]=kind; result.usage["task_complexity"]=difficulty
            if self.store:
                eid=self.store.record_provider_execution(result,prompt,attempt=attempt,invocation_id=invocation_id,workflow_id=self.workflow_id,task_id=task_id,task_type=kind,task_complexity=difficulty,
                    outcome={"success":result.success,"first_pass_success":result.success and attempt==1,"attempts":attempt,"blocked":bool(result.error and result.error.startswith(("BUDGET_EXCEEDED","SCOPE_VIOLATION","PARTIAL_WRITE"))),"provider_failure":bool(result.error and result.error.startswith("PROVIDER_FAILURE"))})
                result.usage["execution_id"]=eid
            return result
        result=execute(route,1)
        provider_failure=(result.error or "").startswith("PROVIDER_FAILURE") or (result.error is None and result.exit_code not in {None,0})
        if not result.success and fallback and provider_failure:
            second = self.router.route(role, exclude=route_excludes | {route.provider}, author_provider=author_provider)
            if second.provider != "unavailable":
                if self.on_fallback and not self.on_fallback(role,second.provider,second.model):
                    result.error="PROVIDER_FAILURE: interactive fallback aborted"
                    return result
                result=execute(second,2)
        return result
