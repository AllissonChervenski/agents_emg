"""Sequential deterministic SDD driver; semantic work stays with provider agents."""
import json, re
from pathlib import Path
from orchestrator.agents.runner import load_prompt
from orchestrator.validation.parser import parse_validation
from orchestrator.workflow.tdd import TDDTask
from orchestrator.tdd import execute_tdd_task


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


def _create_and_validate(runner, author_role, validator_role, artifact_path, prompt, cwd, timeout, retries=3):
    path=Path(artifact_path); last=None
    for attempt in range(1,retries+1):
        author=_generate(runner,author_role,prompt,cwd,timeout)
        if author.stdout.strip(): path.write_text(author.stdout)
        if not path.exists(): raise WorkflowBlocked(f"{author_role} produced no artifact at {path}")
        reviewer=_generate(runner,validator_role,load_prompt(validator_role,artifact=path.read_text(),feature=prompt),cwd,timeout,{author.provider})
        validation=parse_validation(reviewer.stdout,validator_role,reviewer.model)
        last=validation
        if validation.status=="PASS": return author,validation
        if validation.status=="BLOCKED": raise WorkflowBlocked(f"{validator_role} BLOCKED: {validation.summary}; raw={validation.raw_output}")
        prompt += "\nRevise based on these issues: "+json.dumps([issue.__dict__ for issue in validation.issues])
    raise WorkflowBlocked(f"{validator_role} retry limit exceeded: {last.summary if last else 'no result'}")


def _parse_tasks(text):
    match=re.search(r"\{\s*\"tasks\"\s*:.*\}",text,re.S)
    if not match: return None
    try: obj=json.loads(match.group(0))
    except ValueError: return None
    tasks=obj.get("tasks")
    return tasks if isinstance(tasks,list) and all(isinstance(x,dict) and x.get("id") for x in tasks) else None


def run_sdd_workflow(feature, workspace, runner, harness, config, workflow_dir, approve_constitution=None):
    """Run SDD artifacts and TDD tasks sequentially, stopping on any failed gate."""
    root=Path(workspace).resolve(); out=Path(workflow_dir); out.mkdir(parents=True,exist_ok=True)
    constitution=root/"constitution.md"
    if constitution.exists():
        cap=runner.run("constitution_validator",load_prompt("constitution_validator",feature=feature,artifact=constitution.read_text()),cwd=root,timeout=config.timeouts.get("provider"))
        parsed=parse_validation(cap.stdout,"constitution_validator",cap.model) if cap.success else None
        if not parsed or parsed.status!="PASS": raise WorkflowBlocked("Existing constitution did not pass independent validation")
    else:
        if not approve_constitution or not approve_constitution(): raise WorkflowBlocked("Constitution generation requires human approval")
        generated=_generate(runner,"constitution",load_prompt("constitution",feature=feature),root,config.timeouts.get("provider"))
        constitution.write_text(generated.stdout)
        result=_generate(runner,"constitution_validator",load_prompt("constitution_validator",feature=feature,artifact=generated.stdout),root,config.timeouts.get("provider"),{generated.provider})
        verdict=parse_validation(result.stdout,"constitution_validator",result.model)
        if verdict.status!="PASS": raise WorkflowBlocked(f"Constitution validation: {verdict.status}")
    (out/"constitution.md").write_text(constitution.read_text())
    for author,validator,filename in ARTIFACTS:
        prompt=load_prompt(author,feature=feature,artifact=str(out))
        path=out/filename
        if author=="tasks": prompt += '\nReturn exactly a JSON object with tasks array; each task has id, requirements, acceptance_criteria, dependencies, test_type, test_command. Use NOT_AUTOMATABLE only with justification and alternative_verification.'
        _create_and_validate(runner,author,validator,path,prompt,root,config.timeouts.get("provider"),config.retries.get("artifact_generation",3))
    cross_prompt=load_prompt("cross_artifact_validator",artifact="\n".join(f"{p.name}:\n{p.read_text()}" for p in (out/"constitution.md",out/"spec.md",out/"plan.md",out/"tasks.md")))
    cross=_generate(runner,"cross_artifact_validator",cross_prompt,root,config.timeouts.get("provider"))
    cross_result=parse_validation(cross.stdout,"cross_artifact_validator",cross.model)
    if cross_result.status!="PASS": raise WorkflowBlocked(f"Cross-artifact validation: {cross_result.status}")
    tasks=_parse_tasks((out/"tasks.md").read_text())
    if not tasks: raise WorkflowBlocked("tasks.md must contain the validated JSON tasks array")
    completed=set(); traceability=[]
    for task_data in tasks:
        dependencies=set(task_data.get("dependencies",[]))
        if not dependencies.issubset(completed): raise WorkflowBlocked(f"Task {task_data['id']} dependencies are not satisfied")
        if task_data.get("test_type")=="NOT_AUTOMATABLE":
            if not task_data.get("justification") or not task_data.get("alternative_verification"):
                raise WorkflowBlocked(f"Task {task_data['id']} has an unapproved non-automatable exception")
            raise WorkflowBlocked(f"Task {task_data['id']} requires human verification: {task_data['alternative_verification']}")
        command=task_data.get("test_command")
        if not command: raise WorkflowBlocked(f"Task {task_data['id']} has no task-specific test command")
        tdd=TDDTask(str(task_data["id"]),task_data.get("requirements",[]),task_data.get("acceptance_criteria",[]),task_data.get("test_type","UNIT"))
        execute_tdd_task(tdd,runner,harness,root,command,config.verification.get("tests",[]),config.retries.get("implementation",3),out)
        if not tdd.complete(): raise WorkflowBlocked(f"TDD task {tdd.task} blocked in phase {tdd.phase.value}")
        completed.add(tdd.task)
        traceability.append({"requirement":tdd.requirements,"acceptance_criteria":tdd.acceptance_criteria,"task":tdd.task,"test_files":tdd.evidence.get("red_test_files",[]),"verification":tdd.evidence.get("verification_results",[])})
    final=harness.run()
    if any(not result.success for result in final): raise WorkflowBlocked("Final deterministic verification failed")
    (out/"traceability.json").write_text(json.dumps(traceability,indent=2))
    final_review=_generate(runner,"final_reviewer",load_prompt("final_reviewer",task="whole feature",artifact=json.dumps({"traceability":traceability,"verification":[r.__dict__ for r in final]})),root,config.timeouts.get("provider"))
    final_verdict=parse_validation(final_review.stdout,"final_reviewer",final_review.model)
    if final_verdict.status!="PASS": raise WorkflowBlocked(f"Final reviewer: {final_verdict.status}; deterministic checks had passed")
    return {"completed_tasks":sorted(completed),"traceability":traceability,"final_verification":[r.classification for r in final]}
