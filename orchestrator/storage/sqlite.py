import hashlib
import json
import sqlite3
import uuid
import os
import socket
import re
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path


class StateStore:
    SCHEMA_VERSION = 2
    def __init__(self, db_path):
        self.db_path = Path(db_path); self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            if db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='schema_meta'").fetchone():
                newer=db.execute("SELECT value FROM schema_meta WHERE key='state_schema_version'").fetchone()
                if newer and int(newer[0])>self.SCHEMA_VERSION:
                    raise RuntimeError(f"State schema {newer[0]} is newer than supported {self.SCHEMA_VERSION}")
            db.executescript("""
            CREATE TABLE IF NOT EXISTS workflows(id TEXT PRIMARY KEY, feature TEXT, stage TEXT, current_task TEXT, state_json TEXT, created_at TEXT, updated_at TEXT);
            CREATE TABLE IF NOT EXISTS provider_executions(id TEXT PRIMARY KEY, workflow_id TEXT, provider TEXT, model TEXT, role TEXT, prompt_hash TEXT, started_at TEXT, duration REAL, exit_code INTEGER, status TEXT, result_json TEXT);
            CREATE TABLE IF NOT EXISTS validations(id TEXT PRIMARY KEY, workflow_id TEXT, result_json TEXT, created_at TEXT);
            CREATE TABLE IF NOT EXISTS verifications(id TEXT PRIMARY KEY, workflow_id TEXT, result_json TEXT, created_at TEXT);
            CREATE TABLE IF NOT EXISTS metrics(id TEXT PRIMARY KEY, provider TEXT, model TEXT, role TEXT, metric TEXT, value REAL, created_at TEXT);
            CREATE TABLE IF NOT EXISTS traceability(workflow_id TEXT, requirement_id TEXT, task_id TEXT, record_json TEXT, updated_at TEXT, PRIMARY KEY(workflow_id, requirement_id, task_id));
            CREATE TABLE IF NOT EXISTS tdd_metrics(workflow_id TEXT, task_id TEXT, metrics_json TEXT, created_at TEXT, PRIMARY KEY(workflow_id, task_id));
            """)
            columns={row[1] for row in db.execute("PRAGMA table_info(provider_executions)")}
            if "attempt" not in columns: db.execute("ALTER TABLE provider_executions ADD COLUMN attempt INTEGER DEFAULT 1")
            if "invocation_id" not in columns: db.execute("ALTER TABLE provider_executions ADD COLUMN invocation_id TEXT")
            for column,definition in {
                "task_type":"TEXT", "task_complexity":"TEXT", "task_id":"TEXT", "outcome_json":"TEXT"
            }.items():
                if column not in columns: db.execute(f"ALTER TABLE provider_executions ADD COLUMN {column} {definition}")
            db.executescript("""
            CREATE TABLE IF NOT EXISTS schema_meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS checkpoints(
                id TEXT PRIMARY KEY, workflow_id TEXT NOT NULL, transition_id TEXT NOT NULL,
                stage TEXT NOT NULL, substage TEXT, task_id TEXT, attempt INTEGER NOT NULL,
                created_at TEXT NOT NULL, git_commit_base TEXT, git_branch TEXT, git_head TEXT,
                fingerprint_json TEXT NOT NULL, artifact_hashes_json TEXT NOT NULL,
                test_hashes_json TEXT NOT NULL, state_version INTEGER NOT NULL,
                UNIQUE(workflow_id, transition_id));
            CREATE TABLE IF NOT EXISTS workflow_locks(workflow_id TEXT PRIMARY KEY, pid INTEGER NOT NULL,
                hostname TEXT NOT NULL, owner TEXT NOT NULL, acquired_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS resume_reports(id TEXT PRIMARY KEY, workflow_id TEXT NOT NULL,
                report_json TEXT NOT NULL, created_at TEXT NOT NULL);
            """)
            version=db.execute("SELECT value FROM schema_meta WHERE key='state_schema_version'").fetchone()
            if version and int(version[0]) > self.SCHEMA_VERSION:
                raise RuntimeError(f"State schema {version[0]} is newer than supported {self.SCHEMA_VERSION}")
            db.execute("INSERT OR REPLACE INTO schema_meta VALUES('state_schema_version',?)",(str(self.SCHEMA_VERSION),))
            for wid,payload in db.execute("SELECT id,state_json FROM workflows").fetchall():
                state=json.loads(payload or "{}")
                if "state_schema_version" not in state:
                    state["state_schema_version"]=self.SCHEMA_VERSION
                    state["migrated_from_state_version"]=1
                    state["resume_integrity"]="LEGACY_NO_FINGERPRINT"
                    db.execute("UPDATE workflows SET state_json=? WHERE id=?",(json.dumps(state),wid))
    def connect(self): return sqlite3.connect(self.db_path)
    def create_workflow(self, feature, state=None):
        wid = str(uuid.uuid4()); now = datetime.now(timezone.utc).isoformat()
        payload={"state_schema_version":self.SCHEMA_VERSION,**(state or {})}
        with self.connect() as db: db.execute("INSERT INTO workflows VALUES(?,?,?,?,?,?,?)", (wid, feature, "PLANNED", None, json.dumps(payload), now, now))
        return wid
    def update_workflow(self, workflow_id, stage, state, current_task=None):
        with self.connect() as db: db.execute("UPDATE workflows SET stage=?,current_task=?,state_json=?,updated_at=? WHERE id=?", (stage,current_task,json.dumps({"state_schema_version":self.SCHEMA_VERSION,**state}),datetime.now(timezone.utc).isoformat(),workflow_id))
    def get_workflow(self, workflow_id):
        with self.connect() as db:
            row = db.execute("SELECT id,feature,stage,current_task,state_json,created_at,updated_at FROM workflows WHERE id=?", (workflow_id,)).fetchone()
        if not row: return None
        return dict(zip(("workflow_id","feature","stage","current_task","state","created_at","updated_at"), (*row[:4], json.loads(row[4]), *row[5:])))
    def list_workflows(self):
        with self.connect() as db: rows=db.execute("SELECT id,feature,stage,current_task,updated_at FROM workflows ORDER BY updated_at DESC").fetchall()
        return [dict(zip(("workflow_id","feature","stage","current_task","updated_at"),r)) for r in rows]
    def record_provider_execution(self, result, prompt, attempt=1, invocation_id=None, workflow_id=None, task_id=None, task_type=None, task_complexity=None, outcome=None):
        rid = str(uuid.uuid4()); now = datetime.now(timezone.utc).isoformat(); digest=hashlib.sha256(prompt.encode()).hexdigest()
        fields=("success","first_pass_success","attempts","latency","structured_output_valid","validator_accepted","regression_passed","final_verification_passed","blocked","provider_failure","red_valid","red_attempts","green_attempts","first_pass_green","refactor_regression","test_tampering","review_accepted")
        outcome={**{key:None for key in fields},**(outcome or {})}
        outcome["latency"]=result.duration
        safe_result={"provider":result.provider,"model":result.model,"role":result.role,"success":result.success,"exit_code":result.exit_code,"duration":result.duration,
            "error_code":(result.error or "").split(":",1)[0] or None,"structured_output_present":result.structured_output is not None,
            "usage":{key:result.usage[key] for key in ("input_tokens","output_tokens","total_tokens","reported_cost") if isinstance(result.usage.get(key),(int,float))}}
        with self.connect() as db:
            db.execute("INSERT INTO provider_executions(id,workflow_id,provider,model,role,prompt_hash,started_at,duration,exit_code,status,result_json,attempt,invocation_id,task_type,task_complexity,task_id,outcome_json) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (rid,workflow_id,result.provider,result.model,result.role,digest,now,result.duration,result.exit_code,"PASS" if result.success else "FAIL",json.dumps(safe_result),attempt,invocation_id,task_type,task_complexity,task_id,json.dumps(outcome)))
            db.execute("INSERT INTO metrics VALUES(?,?,?,?,?,?,?)", (str(uuid.uuid4()),result.provider,result.model,result.role,"success",1.0 if result.success else 0.0,now))
        return rid

    def update_execution_outcome(self, execution_id, **fields):
        with self.connect() as db:
            row=db.execute("SELECT outcome_json FROM provider_executions WHERE id=?",(execution_id,)).fetchone()
            if row:
                values=json.loads(row[0] or "{}"); values.update(fields)
                db.execute("UPDATE provider_executions SET outcome_json=? WHERE id=?",(json.dumps(values),execution_id))

    def routing_history(self, role=None, provider=None, model=None):
        with self.connect() as db:
            rows=db.execute("SELECT provider,model,role,task_type,task_complexity,workflow_id,task_id,started_at,duration,status,attempt,outcome_json FROM provider_executions ORDER BY started_at").fetchall()
        results=[]
        for p,m,r,kind,difficulty,wid,tid,started,latency,status,attempt,payload in rows:
            if role and r!=role or provider and p!=provider or model and m!=model: continue
            outcome=json.loads(payload or "{}")
            results.append({"provider":p,"model":m,"role":r,"task_type":kind,"task_complexity":difficulty,
                "workflow_id":wid,"task_id":tid,"timestamp":started,"latency":latency or 0,
                "success":outcome.get("success",status=="PASS"),"first_pass_success":outcome.get("first_pass_success",status=="PASS" and attempt==1),
                "attempts":outcome.get("attempts",attempt or 1),"structured_output_valid":outcome.get("structured_output_valid",True),
                "validator_accepted":outcome.get("validator_accepted",True),"regression_passed":outcome.get("regression_passed",True),
                "final_verification_passed":outcome.get("final_verification_passed",True),"blocked":outcome.get("blocked",False),
                "provider_failure":outcome.get("provider_failure",False),"red_valid":outcome.get("red_valid",False),
                "red_attempts":outcome.get("red_attempts",0),"green_attempts":outcome.get("green_attempts",0),
                "first_pass_green":outcome.get("first_pass_green",False),"refactor_regression":outcome.get("refactor_regression",False),
                "test_tampering":outcome.get("test_tampering",False),"review_accepted":outcome.get("review_accepted",True)})
            results[-1].update({key:outcome.get(key) for key in ("task_risk","task_scope","input_tokens","output_tokens","total_tokens","reported_cost","estimated_cost","ponytail_enabled","caveman_enabled","policy_source","escalation_level","escalation_reason","model_failure_category","model_failure_reason")})
        return results

    def count_provider_model_calls(self, workflow_id, provider, model, task_id=None):
        query="SELECT COUNT(*) FROM provider_executions WHERE workflow_id=? AND provider=? AND model=?"
        params=[workflow_id,provider,model]
        if task_id is not None: query+=" AND task_id=?"; params.append(task_id)
        with self.connect() as db: return db.execute(query,params).fetchone()[0]

    def model_failure_events(self, workflow_id, task_id=None, role=None):
        with self.connect() as db:
            rows=db.execute("SELECT provider,model,role,task_id,outcome_json FROM provider_executions WHERE workflow_id=? ORDER BY started_at",(workflow_id,)).fetchall()
        events=[]
        for provider,model,recorded_role,recorded_task,payload in rows:
            if task_id and recorded_task!=task_id or role and recorded_role!=role: continue
            data=json.loads(payload or "{}")
            if data.get("model_failure_category"):
                events.append({"provider":provider,"model":model,"role":recorded_role,"task_id":recorded_task,
                               "category":data["model_failure_category"],"reason":data.get("model_failure_reason")})
        return events

    def cost_roi(self, minimum_samples=10, role=None):
        groups={}
        for row in self.routing_history(role=role):
            key=(row["provider"],row["model"],row["role"],row["task_type"],row["task_complexity"],row.get("task_risk"))
            groups.setdefault(key,[]).append(row)
        results=[]
        for key,rows in groups.items():
            if len(rows)<minimum_samples: continue
            successful=sum(bool(row.get("success")) for row in rows)
            first_green=sum(bool(row.get("first_pass_green")) for row in rows)
            costs=[row.get("reported_cost") for row in rows]
            tokens=[row.get("total_tokens") for row in rows]
            costs_complete=all(isinstance(value,(int,float)) for value in costs)
            tokens_complete=all(isinstance(value,(int,float)) for value in tokens)
            results.append({"provider":key[0],"model":key[1],"role":key[2],"task_type":key[3],"complexity":key[4],"risk":key[5],"runs":len(rows),
                "cost_per_success":sum(costs)/successful if costs_complete and successful else None,
                "tokens_per_success":sum(tokens)/successful if tokens_complete and successful else None,
                "cost_per_first_pass_green":sum(costs)/first_green if costs_complete and first_green else None})
        return results

    def agent_call_counts(self, workflow_id):
        with self.connect() as db:
            total=db.execute("SELECT COUNT(*) FROM provider_executions WHERE workflow_id=?",(workflow_id,)).fetchone()[0]
            rows=db.execute("SELECT task_id,COUNT(*) FROM provider_executions WHERE workflow_id=? AND task_id IS NOT NULL GROUP BY task_id",(workflow_id,)).fetchall()
        return total,{task:count for task,count in rows}

    def create_checkpoint(self, workflow_id, transition_id, stage, fingerprint, task_id=None, attempt=1, substage=None):
        now=datetime.now(timezone.utc).isoformat(); fp=fingerprint
        with self.connect() as db:
            db.execute("INSERT OR IGNORE INTO checkpoints VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",(
                str(uuid.uuid4()),workflow_id,transition_id,stage,substage,task_id,attempt,now,
                fp.get("git_commit_base"),fp.get("branch"),fp.get("head"),json.dumps(fp),
                json.dumps(fp.get("artifact_hashes",{})),json.dumps(fp.get("test_hashes",{})),self.SCHEMA_VERSION))
            row=db.execute("SELECT id FROM checkpoints WHERE workflow_id=? AND transition_id=?",(workflow_id,transition_id)).fetchone()
        return row[0]

    def checkpoints(self, workflow_id):
        with self.connect() as db:
            rows=db.execute("SELECT id,transition_id,stage,substage,task_id,attempt,created_at,git_commit_base,git_branch,git_head,fingerprint_json,state_version FROM checkpoints WHERE workflow_id=? ORDER BY rowid",(workflow_id,)).fetchall()
        return [{"id":r[0],"workflow_id":workflow_id,"transition_id":r[1],"stage":r[2],"substage":r[3],"task_id":r[4],"attempt":r[5],"timestamp":r[6],"git_commit_base":r[7],"git_branch":r[8],"git_head":r[9],"workspace_fingerprint":json.loads(r[10]),"state_version":r[11]} for r in rows]

    def latest_checkpoint(self, workflow_id):
        rows=self.checkpoints(workflow_id)
        return rows[-1] if rows else None

    def record_resume(self, workflow_id, report):
        rid=report.get("resume_id") or str(uuid.uuid4()); now=datetime.now(timezone.utc).isoformat()
        report={**report,"resume_id":rid,"workflow_id":workflow_id,"timestamp":now}
        with self.connect() as db:
            db.execute("INSERT INTO resume_reports VALUES(?,?,?,?)",(rid,workflow_id,json.dumps(report),now))
        return report

    def update_resume_report(self, report):
        with self.connect() as db:
            db.execute("UPDATE resume_reports SET report_json=? WHERE id=?",(json.dumps(report),report["resume_id"]))

    def record_verification(self, workflow_id, result):
        command=[]; hide_next=False
        for part in result.command:
            if hide_next:
                command.append("[REDACTED]"); hide_next=False; continue
            if part in {"--token","--api-key","--password","--secret"}:
                command.append(part); hide_next=True; continue
            command.append(re.sub(r"(?i)(token|api[_-]?key|password|secret)=.+",r"\1=[REDACTED]",part))
        with self.connect() as db:
            db.execute("INSERT INTO verifications VALUES(?,?,?,?)",(str(uuid.uuid4()),workflow_id,json.dumps({"command":command,"category":result.category,"name":result.name,"status":result.status,"exit_code":result.exit_code,"duration":result.duration}),datetime.now(timezone.utc).isoformat()))

    def list_resume_reports(self, workflow_id):
        with self.connect() as db: rows=db.execute("SELECT report_json FROM resume_reports WHERE workflow_id=? ORDER BY created_at",(workflow_id,)).fetchall()
        return [json.loads(row[0]) for row in rows]

    @contextmanager
    def workflow_lock(self, workflow_id):
        owner=str(uuid.uuid4()); host=socket.gethostname(); pid=os.getpid()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            existing=db.execute("SELECT pid,hostname FROM workflow_locks WHERE workflow_id=?",(workflow_id,)).fetchone()
            if existing:
                other_pid,other_host=existing
                orphan=other_host==host and not _pid_alive(other_pid)
                if orphan: db.execute("DELETE FROM workflow_locks WHERE workflow_id=?",(workflow_id,))
                else: raise RuntimeError("WORKFLOW_ALREADY_RUNNING")
            db.execute("INSERT INTO workflow_locks VALUES(?,?,?,?,?)",(workflow_id,pid,host,owner,datetime.now(timezone.utc).isoformat()))
        try: yield
        finally:
            with self.connect() as db: db.execute("DELETE FROM workflow_locks WHERE workflow_id=? AND owner=?",(workflow_id,owner))

    def abort_workflow(self, workflow_id):
        item=self.get_workflow(workflow_id)
        if not item: raise ValueError("Workflow not found")
        self.update_workflow(workflow_id,"ABORTED",{**item["state"],"status":"ABORTED"},item["current_task"])


    def record_metric(self, provider, model, role, metric, value=1.0):
        with self.connect() as db:
            db.execute("INSERT INTO metrics VALUES(?,?,?,?,?,?,?)",(str(uuid.uuid4()),provider,model,role,metric,float(value),datetime.now(timezone.utc).isoformat()))

    def record_tdd_metrics(self, workflow_id, task_id, values):
        with self.connect() as db:
            db.execute("INSERT OR REPLACE INTO tdd_metrics VALUES(?,?,?,?)",(workflow_id,task_id,json.dumps(values),datetime.now(timezone.utc).isoformat()))

    def upsert_traceability(self, workflow_id, record):
        now=datetime.now(timezone.utc).isoformat()
        data=asdict(record)
        for task_id in record.task_ids:
            with self.connect() as db:
                db.execute("INSERT OR REPLACE INTO traceability VALUES(?,?,?,?,?)",(workflow_id,record.requirement_id,task_id,json.dumps(data),now))

    def list_traceability(self, requirement_id=None):
        with self.connect() as db:
            if requirement_id:
                rows=db.execute("SELECT workflow_id,requirement_id,task_id,record_json,updated_at FROM traceability WHERE requirement_id=? ORDER BY updated_at DESC",(requirement_id,)).fetchall()
            else: rows=db.execute("SELECT workflow_id,requirement_id,task_id,record_json,updated_at FROM traceability ORDER BY updated_at DESC").fetchall()
        return [{"workflow_id":workflow,"requirement_id":requirement,"task_id":task,"record":json.loads(data),"updated_at":updated} for workflow,requirement,task,data,updated in rows]

    def summarize_metrics(self):
        with self.connect() as db:
            rows=db.execute("""SELECT provider,model,role,COUNT(*),AVG(CASE WHEN status='PASS' THEN 1.0 ELSE 0.0 END),
                AVG(CASE WHEN attempt=1 AND status='PASS' THEN 1.0 ELSE 0.0 END),AVG(attempt-1),AVG(duration)
                FROM provider_executions GROUP BY provider,model,role ORDER BY provider,model,role""").fetchall()
            event_rows=db.execute("SELECT provider,model,role,metric,SUM(value) FROM metrics WHERE metric IN ('structured_output_failure','validator_rejection') GROUP BY provider,model,role,metric").fetchall()
            tdd_rows=[json.loads(row[0]) for row in db.execute("SELECT metrics_json FROM tdd_metrics").fetchall()]
        events={(provider,model,role,metric):value for provider,model,role,metric,value in event_rows}
        providers=[{"provider":provider,"model":model,"role":role,"runs":runs,"success_rate":success,
            "first_pass_success":first,"average_retries":retries,"average_latency":latency,
            "structured_output_failures":events.get((provider,model,role,"structured_output_failure"),0),
            "validator_rejections":events.get((provider,model,role,"validator_rejection"),0)}
            for provider,model,role,runs,success,first,retries,latency in rows]
        def rate(key): return sum(bool(row.get(key)) for row in tdd_rows)/len(tdd_rows) if tdd_rows else None
        green_rows=[row for row in tdd_rows if row.get("green_attempts",0)>0]
        green_first=sum(bool(row.get("first_pass_green")) for row in green_rows)/len(green_rows) if green_rows else None
        avg_green=sum(row["green_attempts"] for row in green_rows)/len(green_rows) if green_rows else None
        tdd={"tasks":len(tdd_rows),"red_valid_rate":rate("red_valid"),"first_pass_green_rate":green_first,
             "average_green_attempts":avg_green,"regression_failure_rate":rate("regression_failed"),"test_tampering_rate":rate("test_tampering")}
        return {"providers":providers,"tdd":tdd}

    @staticmethod
    def load_provider_metrics(db_path):
        path=Path(db_path)
        if not path.exists(): return {}
        try:
            with sqlite3.connect(path.as_uri()+"?mode=ro",uri=True) as db:
                rows=db.execute("SELECT provider,model,role,AVG(value),COUNT(*),SUM(CASE WHEN value < 0.5 THEN 1 ELSE 0 END) FROM metrics WHERE metric='success' GROUP BY provider,model,role").fetchall()
                aggregate=db.execute("SELECT provider,model,AVG(value),COUNT(*) FROM metrics WHERE metric='success' GROUP BY provider,model").fetchall()
                latencies=db.execute("SELECT provider,model,AVG(duration) FROM provider_executions GROUP BY provider,model").fetchall()
                recent=db.execute("SELECT provider,model,status FROM provider_executions ORDER BY started_at DESC LIMIT 1000").fetchall()
            failures={}
            for provider,model,status in recent:
                key=(provider,model)
                if key not in failures: failures[key]=[]
                if len(failures[key])<10: failures[key].append(status!="PASS")
            out={(provider,model,role):{"success_rate":success,"samples":count,"recent_failures":0} for provider,model,role,success,count,_ in rows}
            out.update({(provider,model):{"success_rate":success,"samples":count,"recent_failures":sum(failures.get((provider,model),[]))} for provider,model,success,count in aggregate})
            for provider,model,latency in latencies:
                out.setdefault((provider,model),{})["average_latency"]=latency
            return out
        except sqlite3.Error:
            return {}

    @staticmethod
    def load_model_metrics(db_path):
        path=Path(db_path)
        if not path.exists(): return {}
        try:
            with sqlite3.connect(path.as_uri()+"?mode=ro",uri=True) as db:
                columns={row[1] for row in db.execute("PRAGMA table_info(provider_executions)")}
                retry_expr="AVG(attempt-1)" if "attempt" in columns else "NULL"
                rows=db.execute(f"SELECT provider,model,AVG(CASE WHEN status='PASS' THEN 1.0 ELSE 0.0 END),AVG(duration),COUNT(*),{retry_expr} FROM provider_executions GROUP BY provider,model").fetchall()
            return {(provider,model):{"success_rate":success,"average_latency":latency,"samples":count,"average_retries":retries} for provider,model,success,latency,count,retries in rows}
        except sqlite3.Error: return {}


def _pid_alive(pid):
    try: os.kill(pid,0); return True
    except ProcessLookupError: return False
    except PermissionError: return True
