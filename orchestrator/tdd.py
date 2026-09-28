"""Policy-level TDD enforcement helpers used by workflow drivers."""
from pathlib import Path
from orchestrator.agents.roles import ROLES
from orchestrator.workflow.tdd import TDDTask
from orchestrator.workflow.transitions import TDDPhase
from orchestrator.tdd_contract import parse_test_design, hash_test_files, changed_test_hashes


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
    return hash_test_files(root)


def test_files_changed(before, root):
    return changed_test_hashes(before,root)


def workspace_snapshot(root):
    root=Path(root)
    ignored={".git", ".orchestrator", "__pycache__", ".venv"}
    return {str(p.relative_to(root)): str(p.readlink()).encode() if p.is_symlink() else p.read_bytes() for p in root.rglob("*") if (p.is_file() or p.is_symlink()) and not any(part in ignored for part in p.parts)}


def _test_or_fixture(path):
    p=Path(path)
    return "tests" in p.parts or p.name.startswith("test_") or p.name in {"conftest.py", "pytest.ini", "tox.ini"}


def execute_tdd_task(task: TDDTask, runner, harness, workspace, task_test_command=None, regression_commands=None, max_attempts=3, workflow_dir=None, gate_callback=None, checkpoint_callback=None, task_data=None, resume_stage=None, artifact_paths=None):
    """Run a bounded TDD cycle. Python alone advances phases and evaluates checks."""
    import json
    import time
    from pathlib import Path
    from orchestrator.validation.parser import parse_validation
    from orchestrator.workflow.transitions import TDDPhase
    gate=TDDGate(task,max_attempts)
    root=Path(workspace)
    # TDD reports are operational data, while the protected SDD artifacts are
    # the canonical SpecKit files.  Keep the old locations only for direct
    # callers that have not supplied a layout yet.
    artifact_root=Path(workflow_dir) if workflow_dir else root
    artifact_paths=artifact_paths or {"constitution":str(root/"constitution.md"),"specification":str(artifact_root/"spec.md"),"plan":str(artifact_root/"plan.md"),"tasks":str(artifact_root/"tasks.md")}
    def artifact_hashes():
        import hashlib
        return {name:hashlib.sha256(str(Path(path).readlink()).encode() if Path(path).is_symlink() else Path(path).read_bytes()).hexdigest() for name,path in artifact_paths.items() if Path(path).is_file() or Path(path).is_symlink()}
    protected_artifacts=artifact_hashes()
    ctx=f"Task: {task.task}\nRequirements: {task.requirements}\nAcceptance criteria: {task.acceptance_criteria}\nArtifact paths: {artifact_paths}"
    def invoke(role, extra="", author_provider=None):
        options={"task":task_data,"task_id":task.task} if task_data else {}
        if task_data and role in {"coder","refactorer"}: options["allowed_paths"]=task_data.get("allowed_files") or task_data.get("production_files") or []
        if role == "coder":
            # Python's task-level TDD machine is the only implementation
            # authority. SpecKit implement is dispatched as the worker skill
            # for this GREEN task, with an explicit one-task boundary.
            skill_name=ROLES[role].skill_name
            options["skill_name"]=skill_name
            extra=(f"Use the installed {skill_name} skill for task {task.task} only. "
                   "Do not process any other task, edit tasks.md, mark checklist items, "
                   "or advance the TDD phase; Python owns task selection and phase gates.\n"+extra)
        result=runner.run(role, ctx+"\n"+extra, cwd=root, author_provider=author_provider,**options)
        if result.usage.get("execution_id"):
            task.evidence.setdefault("executions",[]).append({"role":role,"id":result.usage["execution_id"]})
        if not result.success: return result, None
        return result, result.stdout
    def save():
        if workflow_dir: task.save(Path(workflow_dir)/task.task/"tdd.json")
        if getattr(runner,"store",None):
            values={"red_valid":task.evidence.get("red_expected_failure_confirmed"),"red_attempts":task.evidence.get("red_attempts",0),
                "green_attempts":task.evidence.get("green_attempts",0),"first_pass_green":task.evidence.get("green_attempts")==1,
                "attempts":max(1,task.evidence.get("green_attempts",0)),
                "test_tampering":task.evidence.get("test_tampering_detected",False),"review_accepted":task.evidence.get("review_status")=="PASS" if "review_status" in task.evidence else None,
                "regression_passed":task.evidence.get("regression",{}).get("status")=="PASS" if task.evidence.get("regression") else None,
                "blocked":task.phase==TDDPhase.BLOCKED}
            for execution in task.evidence.get("executions",[]):
                specific=dict(values)
                if task.phase in {TDDPhase.COMPLETE,TDDPhase.BLOCKED}:
                    if execution["role"] in {"coder","refactorer"}:
                        specific.update(success=task.complete(),first_pass_success=task.complete() and task.evidence.get("green_attempts")==1)
                    elif execution["role"]=="test_designer":
                        specific.update(success=bool(task.evidence.get("red_expected_failure_confirmed")),first_pass_success=bool(task.evidence.get("red_expected_failure_confirmed")) and task.evidence.get("red_attempts")==1)
                runner.store.update_execution_outcome(execution["id"],**specific)
    def checkpoint(stage):
        save()
        if checkpoint_callback: checkpoint_callback(stage,task.task,task.attempts.get("green",0)+1)
    def gate_phase(phase, role, files=(), commands=()):
        if gate_callback and not gate_callback(phase,role,list(files),list(commands)):
            task.evidence["gate_abort"]=phase
            task.advance(TDDPhase.BLOCKED); save(); return False
        return True
    if resume_stage in {"RED_VALIDATED","GREEN_VALIDATED","REFACTOR_VALIDATED"}:
        from types import SimpleNamespace
        report=Path(workflow_dir)/task.task/"tdd.json"
        payload=json.loads(report.read_text())
        task.evidence={k:v for k,v in payload.items() if k not in {"task","requirement","acceptance_criteria","test_type","phase","attempts"}}
        task.attempts=payload.get("attempts",task.attempts)
        design=SimpleNamespace(**task.evidence["test_design"])
        task.phase={"RED_VALIDATED":TDDPhase.GREEN_IMPLEMENT,"GREEN_VALIDATED":TDDPhase.GREEN_VERIFY,"REFACTOR_VALIDATED":TDDPhase.REVIEW}[resume_stage]
    else:
        production_before=workspace_snapshot(root)
        before_red=test_files_snapshot(root)
        if not gate_phase("ANALYZE","test_designer"): return task
        analysis,_=invoke("test_designer", "ANALYZE only: inspect linked requirements, acceptance criteria, plan, task, and existing code. Propose observable deterministic tests. Do not edit any file.")
        task.evidence["analysis"]={"status":"PASS" if analysis.success else "BLOCKED","provider":analysis.provider,"model":analysis.model,"summary":analysis.stdout[:2000]}
        if not analysis.success:
            task.advance(TDDPhase.BLOCKED); save(); return task
        after_analysis=workspace_snapshot(root)
        if production_before != after_analysis or artifact_hashes()!=protected_artifacts:
            task.evidence["analyze_files_changed"]=sorted(k for k in set(production_before)|set(after_analysis) if production_before.get(k)!=after_analysis.get(k))
            task.advance(TDDPhase.BLOCKED); save(); return task
        task.advance(TDDPhase.RED_GENERATE)
        if not gate_phase("RED_GENERATE","test_designer",["tests/"]): return task
        designer,_=invoke("test_designer", "RED: create only task-specific tests from the approved analysis. Production files must not change. Return the required TestDesign JSON contract.")
        if not designer.success:
            task.advance(TDDPhase.BLOCKED); save(); return task
        after_design=workspace_snapshot(root)
        changed_design={k for k in set(production_before)|set(after_design) if production_before.get(k)!=after_design.get(k)}
        if any(not _test_or_fixture(path) for path in changed_design) or artifact_hashes()!=protected_artifacts:
            task.evidence["red_production_files_changed"]=sorted(k for k in changed_design if not _test_or_fixture(k))
            task.advance(TDDPhase.BLOCKED); save(); return task
        task.evidence["test_designer_provider"]=designer.provider
        task.evidence["red_test_created"]=bool(test_files_changed(before_red,root))
        task.evidence["test_files_changed"]=test_files_changed(before_red,root)
        try:
            design=parse_test_design(designer.structured_output if isinstance(designer.structured_output,dict) else designer.stdout,
                task.task,task.requirements,task.acceptance_criteria,root,task.evidence["test_files_changed"])
        except ValueError as exc:
            task.evidence["test_design_error"]=str(exc)
            if getattr(runner,"store",None):
                runner.store.record_metric(designer.provider,designer.model,designer.role,"structured_output_failure")
                if designer.usage.get("execution_id"): runner.store.update_execution_outcome(designer.usage["execution_id"],structured_output_valid=False)
            task.advance(TDDPhase.BLOCKED); save(); return task
        task.evidence["test_design"]={"task_id":design.task_id,"requirement_ids":design.requirement_ids,"acceptance_criteria_ids":design.acceptance_criteria_ids,"created_tests":design.created_tests,"test_commands":design.test_commands}
        test_sources={path:(root/path).read_text(errors="replace")[:12000] for path in task.evidence["test_files_changed"] if (root/path).is_file()}
        if not gate_phase("RED_VALIDATE","test_validator",task.evidence["test_files_changed"]): return task
        validator, raw=invoke("test_validator", f"Review declared task tests, sources and commands before RED.\n{json.dumps({'design':task.evidence['test_design'],'sources':test_sources})}", author_provider=designer.provider)
        vr=parse_validation(raw or "", "test_validator", validator.model) if validator.success else None
        if vr and hasattr(runner,"record_validation"): runner.record_validation(validator,vr)
        task.evidence["test_validated"] = bool(vr and vr.status=="PASS")
        if not task.evidence["test_validated"]:
            task.evidence["test_validation"] = "BLOCKED" if not vr else vr.status
            task.advance(TDDPhase.BLOCKED); save(); return task
        if not gate_phase("RED_VERIFY","python",task.evidence["test_files_changed"],design.test_commands): return task
        red_start=time.monotonic()
        red_results=[harness.run_red(command,expected_test_ids=[test_id for test_id in design.created_tests if test_id in command]) for command in design.test_commands]
        task.evidence["task_test_duration"]=time.monotonic()-red_start
        red_status="EXPECTED_FAILURE" if red_results and all(result.classification=="EXPECTED_FAILURE" for result in red_results) else next((result.classification for result in red_results if result.classification!="EXPECTED_FAILURE"),"INVALID_TEST")
        task.evidence.update({"red_result":red_status,"red_test_files":test_files_changed(before_red,root),"red":{"classification":red_status,"results":[{"command":result.command,"exit_code":result.exit_code,"stdout":result.stdout,"stderr":result.stderr,"cause":result.cause,"classification":result.classification} for result in red_results]}})
        task.evidence["red_expected_failure_confirmed"]=red_status=="EXPECTED_FAILURE"
        task.evidence["red_attempts"]=1
        if red_status=="EXPECTED_FAILURE":
            red_validator,red_raw=invoke("test_validator", "Validate the observed RED failures semantically against the linked acceptance criteria. Return PASS only if each failure demonstrates missing behavior.\n"+json.dumps(task.evidence["red"]),author_provider=designer.provider)
            red_validation=parse_validation(red_raw or "","test_validator",red_validator.model) if red_validator.success else None
            if red_validation and hasattr(runner,"record_validation"): runner.record_validation(red_validator,red_validation)
            task.evidence["red"]["semantic_validation"]=red_validation.status if red_validation else "BLOCKED"
            if not red_validation or red_validation.status!="PASS":
                red_status="INVALID_TEST"; task.evidence["red_result"]=red_status; task.evidence["red"]["classification"]=red_status; task.evidence["red_expected_failure_confirmed"]=False
        if not gate.red("PASS" if red_status=="EXPECTED_FAILURE" else "BLOCKED",red_status):
            if task.phase != TDDPhase.BLOCKED: task.advance(TDDPhase.BLOCKED)
            save(); return task
        checkpoint("RED_VALIDATED")
    while task.phase==TDDPhase.GREEN_IMPLEMENT and task.attempts["green"]<max_attempts:
        protected=test_files_snapshot(root)
        task.evidence["green_test_hashes_before"]=protected
        code_before=workspace_snapshot(root)
        if not gate_phase("GREEN_IMPLEMENT","coder",["production files"],design.test_commands): return task
        coder,_=invoke("coder", f"Validated task-specific tests: {json.dumps(task.evidence['test_design'])}\nValidated RED results: {json.dumps(task.evidence['red'])}\nDo not modify tests or fixtures.", author_provider=task.evidence.get("test_designer_provider"))
        if not coder.success:
            task.attempts["green"]+=1
            task.evidence["green_attempts"]=task.attempts["green"]
            continue
        task.evidence["coder_provider"]=coder.provider
        task.evidence["coder_independent_from_test_designer"]=coder.usage.get("validation_independence")
        changed=test_files_changed(protected,root)
        if changed:
            task.evidence["test_tampering_detected"]=True; task.evidence["test_tampering_files"]=changed; gate.green(False,tampered=True)
            task.evidence["green_attempts"]=task.attempts["green"]
            tamper_validator,tamper_raw=invoke("test_validator", "TEST_TAMPERING: inspect unauthorized GREEN changes. Return REVISE with suggested_action RETURN_TO_RED only if a legitimate test correction is needed; otherwise BLOCKED. Files: "+json.dumps(changed),author_provider=coder.provider)
            tamper_review=parse_validation(tamper_raw or "","test_validator",tamper_validator.model) if tamper_validator.success else None
            if tamper_review and hasattr(runner,"record_validation"): runner.record_validation(tamper_validator,tamper_review)
            if tamper_review and tamper_review.status=="REVISE" and any(issue.suggested_action=="RETURN_TO_RED" for issue in tamper_review.issues):
                task.evidence["test_change_approved"]=True
                task.evidence["red_restart_required"]=True
                task.advance(TDDPhase.RED_GENERATE,{"test_change_approved":True})
                save(); return task
            task.advance(TDDPhase.BLOCKED); save(); return task
        if artifact_hashes()!=protected_artifacts:
            task.evidence["artifact_tampering_detected"]=True
            invoke("code_reviewer","ARTIFACT_TAMPERING: Coder modified a protected SDD artifact. Review and require restoration.",author_provider=coder.provider)
            task.advance(TDDPhase.BLOCKED); save(); return task
        code_after=workspace_snapshot(root)
        changed_production={k for k in set(code_before)|set(code_after) if code_before.get(k)!=code_after.get(k) and not _test_or_fixture(k)}
        task.evidence["production_files_changed"]=sorted(set(task.evidence.get("production_files_changed",[]))|changed_production)
        if not gate_phase("GREEN_VERIFY","python",commands=design.test_commands): return task
        green_results=[harness.run_command(command,category="task_tests") for command in design.test_commands]
        task.evidence.setdefault("green",{}).setdefault("attempts",[]).append({"task_tests":[{"command":result.command,"status":result.status,"exit_code":result.exit_code} for result in green_results]})
        green_pass=all(result.success for result in green_results)
        if green_pass:
            early_regression=[harness.run_command(c["command"],c.get("name","regression"),"regression_tests") if isinstance(c,dict) else harness.run_command(c,category="regression_tests") for c in (regression_commands or [])]
            task.evidence["green"]["attempts"][-1]["regression"]=[{"command":result.command,"status":result.status} for result in early_regression]
            if early_regression and not all(result.success for result in early_regression): task.evidence["green_regression_failed"]=True
            green_pass=bool(early_regression) and all(result.success for result in early_regression)
        if not green_pass and any(result.status=="FAIL" for result in green_results) and hasattr(runner,"record_model_feedback"):
            runner.record_model_feedback(coder,"GREEN_IMPLEMENTATION_FAILURE","Validated task tests still fail after implementation")
        if green_pass:
            gate.green(True); task.evidence["green_attempts"]=task.attempts["green"]+1; checkpoint("GREEN_VALIDATED"); break
        task.attempts["green"]+=1
        task.evidence["green_attempts"]=task.attempts["green"]
        if task.attempts["green"]>=max_attempts:
            task.advance(TDDPhase.BLOCKED); save(); return task
    if task.phase != TDDPhase.GREEN_VERIFY and resume_stage!="REFACTOR_VALIDATED":
        if task.phase != TDDPhase.BLOCKED: task.advance(TDDPhase.BLOCKED)
        save(); return task
    if resume_stage!="REFACTOR_VALIDATED":
        task.evidence["green_attempts"]=task.attempts["green"]+1
        task.evidence["refactor_attempts"]=1
        tests_before_refactor=test_files_snapshot(root)
        if not gate_phase("REFACTOR","refactorer",["production files"]): return task
        refactor,_=invoke("refactorer", "Refactor only; preserve validated test files and behavior.")
        if not refactor.success:
            task.advance(TDDPhase.BLOCKED); save(); return task
        refactor_test_changes=test_files_changed(tests_before_refactor,root)
        if refactor_test_changes:
            task.evidence["test_tampering_detected"]=True
            task.evidence["refactor_test_files_changed"]=refactor_test_changes
            task.advance(TDDPhase.BLOCKED); save(); return task
        if artifact_hashes()!=protected_artifacts:
            task.evidence["artifact_tampering_detected"]=True
            task.advance(TDDPhase.BLOCKED); save(); return task
        if not gate_phase("REGRESSION_VERIFY","python",commands=design.test_commands+list(regression_commands or [])): return task
        regression_start=time.monotonic()
        task_results=[harness.run_command(command,category="task_tests") for command in design.test_commands]
        regression=[harness.run_command(c["command"],c.get("name","regression"),"regression_tests") if isinstance(c,dict) else harness.run_command(c,category="regression_tests") for c in (regression_commands or [])]
        regression=task_results+regression
        task.evidence["regression_duration"]=time.monotonic()-regression_start
        regression_ok=bool(regression_commands) and all(r.success for r in regression)
        task.evidence["verification_results"]=[{"command":r.command,"status":r.classification,"exit_code":r.exit_code} for r in regression]
        task.evidence["regression"]={"status":"PASS" if regression_ok else "FAIL","results":task.evidence["verification_results"]}
        if not gate.refactor(True,regression_ok):
            task.advance(TDDPhase.BLOCKED); save(); return task
        checkpoint("REFACTOR_VALIDATED")
    if not gate_phase("REVIEW","code_reviewer",task.evidence.get("production_files_changed",[])): return task
    review,raw=invoke("code_reviewer", "Review code and TDD evidence.\n"+json.dumps(task.evidence), author_provider=task.evidence.get("coder_provider"))
    review_result=parse_validation(raw or "", "code_reviewer", review.model) if review.success else None
    if review_result and hasattr(runner,"record_validation"): runner.record_validation(review,review_result)
    task.evidence["review_status"]=review_result.status if review_result else "BLOCKED"
    task.evidence["reviewer_provider"]=review.provider
    task.evidence["review_independent_from_coder"]=review.usage.get("validation_independence")
    if review_result and review_result.status=="PASS":
        gate.review(True, bool(task.requirements and task.evidence.get("red_test_files")), "PASS")
    else:
        if review_result and "coder" in locals() and hasattr(runner,"record_model_feedback"):
            runner.record_model_feedback(coder,"REVIEW_REJECTION",review_result.summary)
        task.advance(TDDPhase.BLOCKED)
    save()
    task.evidence["final_status"]=task.phase.value
    if getattr(runner,"store",None):
        outcomes={"red_valid":task.evidence.get("red_expected_failure_confirmed",False),"red_attempts":task.evidence.get("red_attempts",0),
            "green_attempts":task.evidence.get("green_attempts",0),"first_pass_green":task.evidence.get("green_attempts")==1,
            "refactor_regression":task.evidence.get("regression",{}).get("status")=="PASS","test_tampering":task.evidence.get("test_tampering_detected",False),
            "review_accepted":task.evidence.get("review_status")=="PASS","regression_passed":task.evidence.get("regression",{}).get("status")=="PASS","blocked":not task.complete()}
        for execution in task.evidence.get("executions",[]): runner.store.update_execution_outcome(execution["id"],**outcomes)
    save()
    return task
