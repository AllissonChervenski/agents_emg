"""Policy-level TDD enforcement helpers used by workflow drivers."""
from pathlib import Path
from orchestrator.workflow.tdd import TDDTask
from orchestrator.workflow.transitions import TDDPhase


class TDDGate:
    def __init__(self, task: TDDTask, max_attempts=3): self.task=task; self.max_attempts=max_attempts
    def red(self, test_validation_status: str, red_result: str):
        if test_validation_status != "PASS":
            self.task.attempts["red"] += 1
            if self.task.attempts["red"] >= self.max_attempts: self.task.advance(TDDPhase.BLOCKED)
            return False
        if self.task.phase == TDDPhase.ANALYZE: self.task.advance(TDDPhase.RED_GENERATE)
        if self.task.phase == TDDPhase.RED_GENERATE: self.task.advance(TDDPhase.RED_VERIFY)
        self.task.evidence["test_validated"] = True
        self.task.evidence["red_result"] = red_result
        if red_result != "EXPECTED_FAILURE": return False
        self.task.advance(TDDPhase.GREEN_IMPLEMENT, {"red_result":"EXPECTED_FAILURE"})
        return True
    def green(self, passed: bool, tampered=False):
        if tampered:
            self.task.evidence["test_tampering_detected"] = True
            self.task.attempts["green"] += 1
            return "TEST_TAMPERING"
        if self.task.phase != TDDPhase.GREEN_IMPLEMENT: raise ValueError("Cannot verify GREEN before valid RED")
        if passed:
            self.task.evidence["green_pass"] = True
            self.task.advance(TDDPhase.GREEN_VERIFY, {"green_pass":True})
            return "PASS"
        self.task.attempts["green"] += 1
        if self.task.attempts["green"] >= self.max_attempts: self.task.advance(TDDPhase.BLOCKED)
        return "GREEN_FAIL"
    def refactor(self, performed: bool, regression_pass: bool):
        if self.task.phase == TDDPhase.GREEN_VERIFY:
            self.task.advance(TDDPhase.REFACTOR, {"refactor_accounted": True, "refactor_performed":performed})
        if self.task.phase != TDDPhase.REFACTOR: raise ValueError("Refactor requires GREEN")
        self.task.advance(TDDPhase.REGRESSION_VERIFY)
        self.task.evidence["regression_pass"] = regression_pass
        if not regression_pass: return False
        self.task.advance(TDDPhase.REVIEW, {"regression_pass":True})
        return True
    def review(self, verification_pass: bool, traceability_recorded: bool, review_status="PASS"):
        if self.task.phase != TDDPhase.REVIEW: raise ValueError("Review requires regression verification")
        if review_status != "PASS" or not verification_pass: return False
        self.task.advance(TDDPhase.COMPLETE, {"verification_pass":True,"traceability_recorded":traceability_recorded,"refactor_accounted":self.task.evidence.get("refactor_accounted",False),"test_validated":self.task.evidence.get("test_validated",False)})
        return self.task.complete()


def test_files_snapshot(root):
    root=Path(root)
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file() and ("test" in p.parts or p.name.startswith("test_")) and ".git" not in p.parts and ".orchestrator" not in p.parts}


def test_files_changed(before, root):
    after=test_files_snapshot(root)
    return sorted(k for k in set(before)|set(after) if before.get(k)!=after.get(k))


def workspace_snapshot(root):
    root=Path(root)
    ignored={".git", ".orchestrator", "__pycache__", ".venv"}
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file() and not any(part in ignored for part in p.parts)}


def _test_or_fixture(path):
    p=Path(path)
    return "tests" in p.parts or p.name.startswith("test_") or p.name in {"conftest.py", "pytest.ini", "tox.ini"}


def execute_tdd_task(task: TDDTask, runner, harness, workspace, task_test_command, regression_commands=None, max_attempts=3, workflow_dir=None):
    """Run a bounded TDD cycle. Python alone advances phases and evaluates checks."""
    import json, time
    from pathlib import Path
    from orchestrator.validation.parser import parse_validation
    from orchestrator.workflow.transitions import TDDPhase
    gate=TDDGate(task,max_attempts)
    root=Path(workspace)
    ctx=f"Task: {task.task}\nRequirements: {task.requirements}\nAcceptance criteria: {task.acceptance_criteria}"
    def invoke(role, extra=""):
        result=runner.run(role, ctx+"\n"+extra, cwd=root)
        if not result.success: return result, None
        return result, result.stdout
    def save():
        if workflow_dir: task.save(Path(workflow_dir)/task.task/"tdd.json")
    production_before=workspace_snapshot(root)
    before_red=test_files_snapshot(root)
    analysis,_=invoke("test_designer", "ANALYZE only: inspect linked requirements, acceptance criteria, plan, task, and existing code. Propose observable deterministic tests. Do not edit any file.")
    if not analysis.success:
        task.advance(TDDPhase.BLOCKED); save(); return task
    after_analysis=workspace_snapshot(root)
    if production_before != after_analysis:
        task.evidence["analyze_files_changed"]=sorted(k for k in set(production_before)|set(after_analysis) if production_before.get(k)!=after_analysis.get(k))
        task.advance(TDDPhase.BLOCKED); save(); return task
    task.advance(TDDPhase.RED_GENERATE)
    designer,_=invoke("test_designer", "RED: create only the tests from the approved analysis. Production files must not change.")
    if not designer.success:
        task.advance(TDDPhase.BLOCKED); save(); return task
    after_design=workspace_snapshot(root)
    changed_design={k for k in set(production_before)|set(after_design) if production_before.get(k)!=after_design.get(k)}
    if any(not _test_or_fixture(path) for path in changed_design):
        task.evidence["red_production_files_changed"]=sorted(k for k in changed_design if not _test_or_fixture(k))
        task.advance(TDDPhase.BLOCKED); save(); return task
    task.evidence["red_test_created"]=bool(test_files_changed(before_red,root))
    task.evidence["test_files_changed"]=test_files_changed(before_red,root)
    validator, raw=invoke("test_validator", f"Review tests just created.\n{designer.stdout}")
    vr=parse_validation(raw or "", "test_validator", validator.model) if validator.success else None
    task.evidence["test_validated"] = bool(vr and vr.status=="PASS")
    if not task.evidence["test_validated"]:
        task.evidence["test_validation"] = "BLOCKED" if not vr else vr.status
        task.advance(TDDPhase.BLOCKED); save(); return task
    red_start=time.monotonic(); red=harness.run_red(task_test_command)
    task.evidence["task_test_duration"]=time.monotonic()-red_start
    task.evidence.update({"red_result":red.classification,"red_test_files":test_files_changed(before_red,root)})
    task.evidence["red_expected_failure_confirmed"]=red.classification=="EXPECTED_FAILURE"
    task.evidence["red_attempts"]=1
    if not gate.red("PASS",red.classification):
        if task.phase != TDDPhase.BLOCKED: task.advance(TDDPhase.BLOCKED)
        save(); return task
    while task.phase==TDDPhase.GREEN_IMPLEMENT and task.attempts["green"]<max_attempts:
        protected=test_files_snapshot(root)
        code_before=workspace_snapshot(root)
        coder,_=invoke("coder", f"Validated RED test output:\n{red.stdout}\n{red.stderr}")
        if not coder.success:
            task.attempts["green"]+=1
            continue
        changed=test_files_changed(protected,root)
        if changed:
            task.evidence["test_tampering_detected"]=True; gate.green(False,tampered=True)
            task.advance(TDDPhase.BLOCKED); save(); return task
        code_after=workspace_snapshot(root)
        task.evidence["production_files_changed"]=sorted(k for k in set(code_before)|set(code_after) if code_before.get(k)!=code_after.get(k) and not _test_or_fixture(k))
        green=harness.run_command(task_test_command)
        if green.success:
            gate.green(True); break
        task.attempts["green"]+=1
        if task.attempts["green"]>=max_attempts:
            task.advance(TDDPhase.BLOCKED); save(); return task
    if task.phase != TDDPhase.GREEN_VERIFY:
        save(); return task
    task.evidence["green_attempts"]=task.attempts["green"]+1
    task.evidence["refactor_attempts"]=1
    tests_before_refactor=test_files_snapshot(root)
    refactor,_=invoke("refactorer", "Refactor only; preserve validated test files and behavior.")
    if not refactor.success:
        task.advance(TDDPhase.BLOCKED); save(); return task
    refactor_test_changes=test_files_changed(tests_before_refactor,root)
    if refactor_test_changes:
        task.evidence["test_tampering_detected"]=True
        task.evidence["refactor_test_files_changed"]=refactor_test_changes
        task.advance(TDDPhase.BLOCKED); save(); return task
    regression_start=time.monotonic(); regression=[harness.run_command(c) for c in (regression_commands or [task_test_command])]
    task.evidence["regression_duration"]=time.monotonic()-regression_start
    regression_ok=all(r.success for r in regression)
    task.evidence["verification_results"]=[{"command":r.command,"status":r.classification,"exit_code":r.exit_code} for r in regression]
    if not gate.refactor(True,regression_ok):
        task.advance(TDDPhase.BLOCKED); save(); return task
    review,raw=invoke("code_reviewer", "Review code and TDD evidence.\n"+json.dumps(task.evidence))
    review_result=parse_validation(raw or "", "code_reviewer", review.model) if review.success else None
    task.evidence["review_status"]=review_result.status if review_result else "BLOCKED"
    if review_result and review_result.status=="PASS":
        gate.review(True, bool(task.requirements and task.evidence.get("red_test_files")), "PASS")
    else:
        task.advance(TDDPhase.BLOCKED)
    save()
    task.evidence["final_status"]=task.phase.value
    save()
    return task
