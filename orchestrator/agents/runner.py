from pathlib import Path
from string import Template
from orchestrator.agents.roles import ROLES
from orchestrator.config.models import AgentResult
from uuid import uuid4
import hashlib
from orchestrator.agents.history import classify_task
from orchestrator.agents.cost import TaskProfile
from orchestrator.agents.skills import SkillDispatcher


def load_prompt(role: str, **values: str) -> str:
    spec = ROLES[role]
    path = Path(__file__).resolve().parents[1] / "prompts" / spec.prompt_file
    return Template(path.read_text()).safe_substitute(**values)


class AgentRunner:
    def __init__(self, providers, router, store=None, workflow_id=None, safety=None, cost_router=None, execution_policy_router=None, skill_dispatcher=None):
        self.providers, self.router, self.store = providers, router, store
        self.on_fallback=None
        self.on_call=None
        self.on_call_complete=None
        self.workflow_id=workflow_id
        self.safety=safety or {}
        self.cost_router=cost_router
        self.execution_policy_router=execution_policy_router
        self.skill_dispatcher=skill_dispatcher or SkillDispatcher(providers)
        self.model_feedback=[]
        self.calls=0
        self.task_calls={}
        self.active_elapsed=0.0
        if store and workflow_id:
            self.calls,self.task_calls=store.agent_call_counts(workflow_id)
            self.active_elapsed=store.active_execution_seconds(workflow_id)

    @staticmethod
    def _snapshot(root):
        if root is None: return {}
        base=Path(root).resolve()
        ignored={".git",".venv","__pycache__","build","dist"}
        def relevant(path):
            parts=path.relative_to(base).parts
            if any(part in ignored for part in parts): return False
            # Operational state and reports are written by the Python harness,
            # not by SpecKit workers.  They must not silently expand a worker's
            # write scope or become duplicate SDD artifacts.
            return parts[0] != ".orchestrator"
        return {str(p.relative_to(base)):hashlib.sha256(str(p.readlink()).encode() if p.is_symlink() else p.read_bytes()).hexdigest() for p in base.rglob("*") if (p.is_file() or p.is_symlink()) and relevant(p)}

    @staticmethod
    def _in_scope(path, allowed):
        return any(path==item.rstrip("/") or path.startswith(item.rstrip("/")+"/") for item in allowed)

    def _budget_error(self, task_id):
        if self.calls>=int(self.safety.get("max_agent_calls_per_workflow",100)): return "BUDGET_EXCEEDED: workflow agent calls"
        if task_id and self.task_calls.get(task_id,0)>=int(self.safety.get("max_agent_calls_per_task",12)): return "BUDGET_EXCEEDED: task agent calls"
        active=self.store.active_execution_seconds(self.workflow_id) if self.store and self.workflow_id else self.active_elapsed
        if active>=float(self.safety.get("max_wall_time",7200)): return "BUDGET_EXCEEDED: active execution time"
        return None

    def record_validation(self, agent_result, validation_result):
        malformed=validation_result.summary in ("Could not parse strict validation output","Malformed issue schema")
        if malformed:
            self.record_model_feedback(agent_result,"STRUCTURED_OUTPUT_FAILURE",validation_result.summary)
        if not self.store: return
        execution_id=agent_result.usage.get("execution_id")
        if execution_id:
            self.store.update_execution_outcome(execution_id,structured_output_valid=not malformed,validator_accepted=validation_result.status=="PASS",blocked=validation_result.status=="BLOCKED")
        if malformed:
            self.store.record_metric(agent_result.provider,agent_result.model,agent_result.role,"structured_output_failure")
        elif validation_result.status!="PASS":
            self.store.record_metric(agent_result.provider,agent_result.model,agent_result.role,"validator_rejection")

    def record_model_feedback(self, agent_result, category, reason=""):
        """Only attributable failures can unlock the next cost escalation level."""
        event={"provider":agent_result.provider,"model":agent_result.model,"category":category,"reason":reason,
               "role":agent_result.role,"task_id":agent_result.usage.get("task_id")}
        self.model_feedback.append(event)
        if self.store and agent_result.usage.get("execution_id"):
            self.store.update_execution_outcome(agent_result.usage["execution_id"],model_failure_category=category,model_failure_reason=reason)

    def run_skill(self, role, provider, model, skill_name, arguments, cwd, execution_policy=None,
                  timeout=None, fallback=True, task=None, task_id=None, allowed_paths=None,
                  expected_outputs=None, exclude_providers=None):
        """Run a skill through the normal router, safety, retry, and telemetry path."""
        return self.run(role, arguments, cwd=cwd, timeout=timeout, override_provider=provider,
                        override_model=model, fallback=fallback, task=task, task_id=task_id,
                        allowed_paths=allowed_paths, expected_outputs=expected_outputs,
                        skill_name=skill_name, execution_policy=execution_policy,
                        exclude_providers=exclude_providers)

    def run(self, role: str, prompt: str, cwd=None, timeout=None, override_provider=None, override_model=None, fallback=True, exclude_providers=None, author_provider=None, task=None, task_id=None, allowed_paths=None, expected_outputs=None, skill_name=None, execution_policy=None):
        invocation_id=str(uuid4())
        route_excludes=set(exclude_providers or ())
        if author_provider is None and exclude_providers and len(exclude_providers)==1:
            author_provider=next(iter(exclude_providers))
            route_excludes.discard(author_provider)
        task_id=task_id or (task or {}).get("id")
        kind,difficulty=classify_task(role,task)
        profile=TaskProfile.derive(role,task)
        route = self.router.route(role, override_provider=override_provider, override_model=override_model, exclude=route_excludes, author_provider=author_provider,task=task)
        decision=None
        if self.cost_router and route.provider!="unavailable":
            failures=(self.store.model_failure_events(self.workflow_id,task_id,role) if self.store and self.workflow_id else self.model_feedback)
            decision=self.cost_router.assess(role,route,task,author_provider,failures,override_model,(task or {}).get("astra_escalation_reason"))
            route=decision["actual_route"]
        if route.provider == "unavailable":
            return AgentResult("unavailable", None, role, False, error="PROVIDER_FAILURE: no provider available")
        def cost_safety_error(chosen, assessment):
            if not assessment: return None
            row=next((item for item in assessment["candidates"] if item["provider"]==chosen.provider and item["model"]==chosen.model),None)
            if row and row["level"]=="ASTRA" and not assessment["astra_escalation_reason"]:
                return "ASTRA_ESCALATION_REQUIRED: explicit escalation reason is missing"
            if row and any("budget exhausted" in reason for reason in row["eligibility_reasons"]):
                return "BUDGET_EXCEEDED: strong model call limit"
            if row and row["level"]=="OPUS" and "extraordinary_fallback_requires_override" in row["eligibility_reasons"]:
                return "OPUS_OVERRIDE_REQUIRED: extraordinary model is outside the normal ladder"
            return None
        route_error=cost_safety_error(route,decision)
        if route_error: return AgentResult(route.provider,route.model,role,False,error=route_error)
        configured_scope=self.safety.get("file_scopes",{}).get(role)
        permitted=list(allowed_paths or (configured_scope if isinstance(configured_scope,list) else (["tests/","fixtures/","conftest.py","pytest.ini","tox.ini"] if role=="test_designer" else [])))
        if self.safety.get("safety_mode")=="strict" and ROLES[role].edits_files and not permitted:
            return AgentResult(route.provider,route.model,role,False,
                               error="SCOPE_VIOLATION: writable paths are not declared for this role")
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
            policy=execution_policy if execution_policy is not None else (self.execution_policy_router.select(role,chosen.provider,profile) if self.execution_policy_router else None)
            effective_prompt=(policy.prefix+"\n\n"+prompt) if policy and policy.prefix else prompt
            permissions="read" if ROLES[role].validation else None
            if skill_name:
                result=self.skill_dispatcher.run_skill(role,chosen.provider,chosen.model,skill_name,prompt,cwd,policy,timeout,permissions)
                effective_prompt=f"skill:{skill_name}\n{effective_prompt}"
            else:
                result=self.providers[chosen.provider].run(effective_prompt,role,chosen.model,cwd,timeout,permissions)
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
            self.active_elapsed+=max(0.0,result.duration)
            if self.on_call_complete: self.on_call_complete(result)
            result.usage["validation_independence"]=chosen.independence
            result.usage["validation_independence_reason"]=chosen.reason if author_provider else None
            result.usage["routing_selection_mode"]=chosen.selection_mode
            result.usage["task_type"]=kind; result.usage["task_complexity"]=difficulty
            result.usage["task_id"]=task_id
            if policy:
                result.usage.update({"ponytail_enabled":policy.ponytail_enabled,"caveman_enabled":policy.caveman_enabled,
                    "policy_source":policy.policy_source,"policy_overhead_estimate":policy.policy_overhead_estimate,
                    "policy_enabled_reason":policy.policy_enabled_reason})
            if decision:
                chosen_cost=next((item for item in decision["candidates"] if item["provider"]==chosen.provider and item["model"]==chosen.model),None)
                result.usage.update({"cost_aware_recommendation":decision["cost_aware_recommendation"],"cost_aware_mode":decision["mode"],
                    "escalation_level":next((item["level"] for item in decision["candidates"] if item["provider"]==chosen.provider and item["model"]==chosen.model),None),
                    "escalation_reason":decision["astra_escalation_reason"],
                    "estimated_cost":chosen_cost["expected_cost"] if chosen_cost else None})
            if self.store:
                eid=self.store.record_provider_execution(result,effective_prompt,attempt=attempt,invocation_id=invocation_id,workflow_id=self.workflow_id,task_id=task_id,task_type=kind,task_complexity=difficulty,
                    outcome={"success":result.success,"first_pass_success":result.success and attempt==1,"attempts":attempt,"blocked":bool(result.error and result.error.startswith(("BUDGET_EXCEEDED","SCOPE_VIOLATION","PARTIAL_WRITE"))),"provider_failure":bool(result.error and result.error.startswith("PROVIDER_FAILURE")),
                        "task_risk":profile.risk,"task_scope":profile.scope,"input_tokens":result.usage.get("input_tokens"),"output_tokens":result.usage.get("output_tokens"),
                        "total_tokens":result.usage.get("total_tokens"),"reported_cost":result.usage.get("reported_cost"),"estimated_cost":result.usage.get("estimated_cost"),
                        "ponytail_enabled":result.usage.get("ponytail_enabled"),"caveman_enabled":result.usage.get("caveman_enabled"),"policy_source":result.usage.get("policy_source"),
                        "escalation_level":result.usage.get("escalation_level"),"escalation_reason":result.usage.get("escalation_reason")})
                result.usage["execution_id"]=eid
            return result
        result=execute(route,1)
        provider_failure=(result.error or "").startswith("PROVIDER_FAILURE") or (result.error is None and result.exit_code not in {None,0})
        if not result.success and fallback and provider_failure:
            second = self.router.route(role, exclude=route_excludes | {route.provider}, author_provider=author_provider)
            if self.cost_router and second.provider!="unavailable":
                second_decision=self.cost_router.assess(role,second,task,author_provider,(),override_model=None)
                second=second_decision["actual_route"]
                decision=second_decision
            fallback_error=cost_safety_error(second,decision)
            if fallback_error: return AgentResult(second.provider,second.model,role,False,error=fallback_error)
            if second.provider != "unavailable":
                if self.on_fallback and not self.on_fallback(role,second.provider,second.model):
                    result.error="PROVIDER_FAILURE: interactive fallback aborted"
                    return result
                result=execute(second,2)
        return result
