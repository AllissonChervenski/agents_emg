"""Sequential deterministic SDD driver; semantic work stays with provider agents."""
import json
import re
from dataclasses import asdict
from pathlib import Path
from orchestrator.agents.runner import load_prompt
from orchestrator.validation.parser import parse_validation
from orchestrator.workflow.tdd import TDDTask
from orchestrator.workflow.transitions import TDDPhase
from orchestrator.tdd import execute_tdd_task
from orchestrator.verification.harness import final_verification_pass
from orchestrator.traceability import records_from_task
from orchestrator.workflow.resume import WorkspaceFingerprint


ARTIFACTS = [
    ("specification", "specification_validator", "spec.md"),
    ("planning", "plan_validator", "plan.md"),
    ("tasks", "tasks_validator", "tasks.md"),
]


class WorkflowBlocked(RuntimeError): pass


def _generate(runner, role, prompt, cwd, timeout, exclude=None):
    result=runner.run(role,prompt,cwd=cwd,timeout=timeout,exclude_providers=exclude)
    if not result.success: raise WorkflowBlocked(f"{role} provider failed: {result.error or result.stderr}")
    return result


def _create_and_validate(runner, author_role, validator_role, artifact_path, prompt, cwd, timeout, retries=3, gate_callback=None):
    path=Path(artifact_path); last=None
    for attempt in range(1,retries+1):
        if gate_callback and not gate_callback(author_role.upper(),author_role,[str(path)],[]): raise WorkflowBlocked(f"Interactive gate aborted before {author_role}")
        author=_generate(runner,author_role,prompt,cwd,timeout)
        if author.stdout.strip(): path.write_text(author.stdout)
        if not path.exists(): raise WorkflowBlocked(f"{author_role} produced no artifact at {path}")
        if gate_callback and not gate_callback(validator_role.upper(),validator_role,[],[]): raise WorkflowBlocked(f"Interactive gate aborted before {validator_role}")
        reviewer=_generate(runner,validator_role,load_prompt(validator_role,artifact=path.read_text(),feature=prompt),cwd,timeout,{author.provider})
        validation=parse_validation(reviewer.stdout,validator_role,reviewer.model)
        runner.record_validation(reviewer,validation)
        if getattr(runner,"store",None) and author.usage.get("execution_id"):
            runner.store.update_execution_outcome(author.usage["execution_id"],validator_accepted=validation.status=="PASS",blocked=validation.status=="BLOCKED")
        if validation.status!="PASS" and hasattr(runner,"record_model_feedback"):
            runner.record_model_feedback(author,"VALIDATOR_REJECTION",validation.summary)
        last=validation
        if validation.status=="PASS": return author,validation
        if validation.status=="BLOCKED":
            if getattr(runner,"store",None) and author.usage.get("execution_id"):
                runner.store.update_execution_outcome(author.usage["execution_id"],blocked=True,success=False)
            raise WorkflowBlocked(f"{validator_role} BLOCKED: {validation.summary}; raw={validation.raw_output}")
        prompt += "\nRevise based on these issues: "+json.dumps([issue.__dict__ for issue in validation.issues])
    if getattr(runner,"store",None) and author.usage.get("execution_id"):
        runner.store.update_execution_outcome(author.usage["execution_id"],blocked=True,success=False)
    raise WorkflowBlocked(f"{validator_role} retry limit exceeded: {last.summary if last else 'no result'}")


def _parse_tasks(text):
    match=re.search(r"\{\s*\"tasks\"\s*:.*\}",text,re.S)
    if not match: return None
    try: obj=json.loads(match.group(0))
    except ValueError: return None
    tasks=obj.get("tasks")
    return tasks if isinstance(tasks,list) and all(isinstance(x,dict) and x.get("id") for x in tasks) else None


def run_sdd_workflow(feature, workspace, runner, harness, config, workflow_dir, approve_constitution=None, store=None, gate_callback=None, resume=False):
    """Run SDD artifacts and TDD tasks sequentially, stopping on any failed gate."""
    root=Path(workspace).resolve(); out=Path(workflow_dir); out.mkdir(parents=True,exist_ok=True)
    workflow_id=out.name
    prior={row["transition_id"] for row in store.checkpoints(workflow_id)} if store and resume else set()
    last_checkpoint=store.latest_checkpoint(workflow_id) if store and resume else None
    def checkpoint(stage,task_id=None,attempt=1):
        if not store: return
        transition_id=f"{stage}:{task_id or '-'}:{attempt}"
        if transition_id in prior: return
        fp=WorkspaceFingerprint(root).capture(workflow_id,task_id)
        store.create_checkpoint(workflow_id,transition_id,stage,fp,task_id,attempt)
        prior.add(transition_id)
        item=store.get_workflow(workflow_id)
        if item: store.update_workflow(workflow_id,stage,{**item["state"],"last_checkpoint":transition_id},task_id)
    def validated(stage): return any(key.startswith(stage+":") for key in prior)
    constitution=root/"constitution.md"
    if resume and validated("CONSTITUTION_VALIDATED"):
        if not constitution.is_file(): raise WorkflowBlocked("Missing validated constitution")
    elif constitution.exists():
        if gate_callback and not gate_callback("CONSTITUTION_VALIDATE","constitution_validator",[],[]): raise WorkflowBlocked("Interactive gate aborted before constitution validation")
        cap=runner.run("constitution_validator",load_prompt("constitution_validator",feature=feature,artifact=constitution.read_text()),cwd=root,timeout=config.timeouts.get("provider"))
        parsed=parse_validation(cap.stdout,"constitution_validator",cap.model) if cap.success else None
        if parsed: runner.record_validation(cap,parsed)
        if not parsed or parsed.status!="PASS": raise WorkflowBlocked("Existing constitution did not pass independent validation")
    else:
        if not approve_constitution or not approve_constitution(): raise WorkflowBlocked("Constitution generation requires human approval")
        generated=_generate(runner,"constitution",load_prompt("constitution",feature=feature),root,config.timeouts.get("provider"))
        constitution.write_text(generated.stdout)
        if gate_callback and not gate_callback("CONSTITUTION_VALIDATE","constitution_validator",[],[]): raise WorkflowBlocked("Interactive gate aborted before constitution validation")
        result=_generate(runner,"constitution_validator",load_prompt("constitution_validator",feature=feature,artifact=generated.stdout),root,config.timeouts.get("provider"),{generated.provider})
        verdict=parse_validation(result.stdout,"constitution_validator",result.model)
        runner.record_validation(result,verdict)
        if verdict.status!="PASS": raise WorkflowBlocked(f"Constitution validation: {verdict.status}")
    (out/"constitution.md").write_text(constitution.read_text())
    checkpoint("CONSTITUTION_VALIDATED")
    authors={}
    for author,validator,filename in ARTIFACTS:
        prompt=load_prompt(author,feature=feature,artifact=str(out))
        path=out/filename
        stage={"specification":"SPEC_VALIDATED","planning":"PLAN_VALIDATED","tasks":"TASKS_VALIDATED"}[author]
        if resume and validated(stage):
            if not path.is_file(): raise WorkflowBlocked(f"Missing validated artifact {path}")
            continue
        if author=="tasks": prompt += '\nReturn exactly a JSON object with tasks array; each task has id, requirements, acceptance_criteria, plan_decisions, dependencies, test_type, allowed_files. Strict mode requires narrow production paths in allowed_files. TestDesigner will declare task-specific test IDs and commands. Use NOT_AUTOMATABLE only with justification and alternative_verification.'
        author_result,_=_create_and_validate(runner,author,validator,path,prompt,root,config.timeouts.get("provider"),max(1,min(config.retries.get("artifact_generation",3),config.real_run.get("max_retries",3))),gate_callback)
        authors[author]=author_result.provider
        checkpoint(stage)
    cross_prompt=load_prompt("cross_artifact_validator",artifact="\n".join(f"{p.name}:\n{p.read_text()}" for p in (out/"constitution.md",out/"spec.md",out/"plan.md",out/"tasks.md")))
    if not (resume and validated("CROSS_VALIDATED")):
        if gate_callback and not gate_callback("CROSS_ARTIFACT_VALIDATE","cross_artifact_validator",[],[]): raise WorkflowBlocked("Interactive gate aborted before cross-artifact validation")
        cross=_generate(runner,"cross_artifact_validator",cross_prompt,root,config.timeouts.get("provider"),{authors["tasks"]} if authors.get("tasks") else None)
        cross_result=parse_validation(cross.stdout,"cross_artifact_validator",cross.model)
        runner.record_validation(cross,cross_result)
        if cross_result.status!="PASS": raise WorkflowBlocked(f"Cross-artifact validation: {cross_result.status}")
        checkpoint("CROSS_VALIDATED")
    tasks=_parse_tasks((out/"tasks.md").read_text())
    if not tasks: raise WorkflowBlocked("tasks.md must contain the validated JSON tasks array")
    completed={row["task_id"] for row in store.checkpoints(workflow_id) if row["stage"]=="TASK_COMPLETE"} if store and resume else set()
    traceability=json.loads((out/"traceability.json").read_text()) if resume and (out/"traceability.json").is_file() else []
    last_coder_provider=None; tdd_tasks=[]
    for task_data in tasks:
        if task_data["id"] in completed: continue
        dependencies=set(task_data.get("dependencies",[]))
        if not dependencies.issubset(completed): raise WorkflowBlocked(f"Task {task_data['id']} dependencies are not satisfied")
        if task_data.get("test_type")=="NOT_AUTOMATABLE":
            if not task_data.get("justification") or not task_data.get("alternative_verification"):
                raise WorkflowBlocked(f"Task {task_data['id']} has an unapproved non-automatable exception")
            raise WorkflowBlocked(f"Task {task_data['id']} requires human verification: {task_data['alternative_verification']}")
        tdd=TDDTask(str(task_data["id"]),task_data.get("requirements",[]),task_data.get("acceptance_criteria",[]),task_data.get("test_type","UNIT"))
        regression=config.verification.get("regression_tests") or config.verification.get("tests",[])
        resume_stage=last_checkpoint["stage"] if last_checkpoint and last_checkpoint.get("task_id")==tdd.task and last_checkpoint["stage"] in {"RED_VALIDATED","GREEN_VALIDATED","REFACTOR_VALIDATED"} else None
        execute_tdd_task(tdd,runner,harness,root,task_data.get("test_command"),regression,max(1,min(config.retries.get("implementation",3),config.real_run.get("max_retries",3))),out,gate_callback,checkpoint,task_data,resume_stage)
        if store:
            store.record_tdd_metrics(out.name,tdd.task,{"red_valid":tdd.evidence.get("red_expected_failure_confirmed",False),"first_pass_green":tdd.evidence.get("green_attempts")==1,
                "green_attempts":tdd.evidence.get("green_attempts",0),"regression_failed":tdd.evidence.get("green_regression_failed",False) or tdd.evidence.get("regression",{}).get("status")=="FAIL", "test_tampering":tdd.evidence.get("test_tampering_detected",False)})
        traceability=[record for record in traceability if tdd.task not in record.get("task_ids",[])]
        records=records_from_task(task_data,tdd.evidence)
        traceability.extend([asdict(record) for record in records])
        if store:
            for record in records: store.upsert_traceability(out.name,record)
        (out/"traceability.json").write_text(json.dumps(traceability,indent=2))
        if not tdd.complete(): raise WorkflowBlocked(f"TDD task {tdd.task} blocked in phase {tdd.phase.value}")
        tdd_tasks.append(tdd)
        last_coder_provider=tdd.evidence.get("coder_provider") or last_coder_provider
        completed.add(tdd.task)
        checkpoint("TASK_COMPLETE",tdd.task,tdd.attempts.get("green",0)+1)
    for task_id in completed:
        if any(task.task==task_id for task in tdd_tasks): continue
        path=out/task_id/"tdd.json"
        if not path.is_file(): raise WorkflowBlocked(f"Missing completed task evidence: {task_id}")
        payload=json.loads(path.read_text())
        task=TDDTask(task_id,payload.get("requirement",[]),payload.get("acceptance_criteria",[]),payload.get("test_type","UNIT"))
        task.phase=TDDPhase.COMPLETE
        task.evidence={key:value for key,value in payload.items() if key not in {"task","requirement","acceptance_criteria","test_type","phase","attempts"}}
        tdd_tasks.append(task)
        last_coder_provider=task.evidence.get("coder_provider") or last_coder_provider
    from orchestrator.traceability import TraceabilityRecord
    if resume and validated("FINAL_VERIFIED"):
        path=out/"final-verification.json"
        if not path.is_file(): raise WorkflowBlocked("Missing final verification evidence")
        final_results=json.loads(path.read_text()); deterministic_ok=True
    else:
        if gate_callback and not gate_callback("FINAL_VERIFY","python",[],[item for key in ("build","tests","lint","static","requirements") for item in config.verification.get(key,[])]): raise WorkflowBlocked("Interactive gate aborted before final verification")
        final=harness.run()
        deterministic_ok=final_verification_pass(final,harness.requirement_results)
        final_results=[{"name":result.name,"command":result.command,"status":result.status,"exit_code":result.exit_code} for result in final]
        for task in tdd_tasks:
            task.evidence["final_verify"]={"status":"PASS" if deterministic_ok else "FAIL","results":final_results}
            task.save(out/task.task/"tdd.json")
            if store:
                for execution in task.evidence.get("executions",[]):
                    store.update_execution_outcome(execution["id"],final_verification_passed=deterministic_ok)
        for record in traceability:
            record["verification_results"].extend(final_results)
            record["final_status"]="PENDING_REVIEW" if deterministic_ok else "FAIL"
            if store: store.upsert_traceability(out.name,TraceabilityRecord(**record))
        (out/"requirement-verification.json").write_text(json.dumps([asdict(item) for item in harness.requirement_results],indent=2))
        (out/"traceability.json").write_text(json.dumps(traceability,indent=2))
        (out/"final-verification.json").write_text(json.dumps(final_results,indent=2))
        if not deterministic_ok: raise WorkflowBlocked("Final deterministic verification failed or no commands were configured")
        checkpoint("FINAL_VERIFIED")
    if not (resume and validated("FINAL_REVIEWED")):
        if gate_callback and not gate_callback("FINAL_REVIEW","final_reviewer",[],[]): raise WorkflowBlocked("Interactive gate aborted before final review")
        final_review=_generate(runner,"final_reviewer",load_prompt("final_reviewer",task="whole feature",artifact=json.dumps({"traceability":traceability,"verification":final_results})),root,config.timeouts.get("provider"),{last_coder_provider} if last_coder_provider else None)
        final_verdict=parse_validation(final_review.stdout,"final_reviewer",final_review.model)
        runner.record_validation(final_review,final_verdict)
        for record in traceability:
            record["final_status"]="PASS" if final_verdict.status=="PASS" else "BLOCKED"
            if store: store.upsert_traceability(out.name,TraceabilityRecord(**record))
        (out/"traceability.json").write_text(json.dumps(traceability,indent=2))
        (out/"final-review.json").write_text(json.dumps({"status":final_verdict.status,"provider":final_review.provider,"model":final_review.model},indent=2))
        if final_verdict.status!="PASS": raise WorkflowBlocked(f"Final reviewer: {final_verdict.status}; deterministic checks had passed")
        checkpoint("FINAL_REVIEWED")
    return {"completed_tasks":sorted(completed),"traceability":traceability,"final_verification":[r["status"] for r in final_results],"requirement_verification":[asdict(item) for item in harness.requirement_results]}
