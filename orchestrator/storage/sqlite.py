import hashlib, json, sqlite3, uuid
from datetime import datetime, timezone
from pathlib import Path


class StateStore:
    def __init__(self, db_path):
        self.db_path = Path(db_path); self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript("""
            CREATE TABLE IF NOT EXISTS workflows(id TEXT PRIMARY KEY, feature TEXT, stage TEXT, current_task TEXT, state_json TEXT, created_at TEXT, updated_at TEXT);
            CREATE TABLE IF NOT EXISTS provider_executions(id TEXT PRIMARY KEY, workflow_id TEXT, provider TEXT, model TEXT, role TEXT, prompt_hash TEXT, started_at TEXT, duration REAL, exit_code INTEGER, status TEXT, result_json TEXT);
            CREATE TABLE IF NOT EXISTS validations(id TEXT PRIMARY KEY, workflow_id TEXT, result_json TEXT, created_at TEXT);
            CREATE TABLE IF NOT EXISTS verifications(id TEXT PRIMARY KEY, workflow_id TEXT, result_json TEXT, created_at TEXT);
            CREATE TABLE IF NOT EXISTS metrics(id TEXT PRIMARY KEY, provider TEXT, model TEXT, role TEXT, metric TEXT, value REAL, created_at TEXT);
            """)
    def connect(self): return sqlite3.connect(self.db_path)
    def create_workflow(self, feature, state=None):
        wid = str(uuid.uuid4()); now = datetime.now(timezone.utc).isoformat()
        with self.connect() as db: db.execute("INSERT INTO workflows VALUES(?,?,?,?,?,?,?)", (wid, feature, "PLANNED", None, json.dumps(state or {}), now, now))
        return wid
    def update_workflow(self, workflow_id, stage, state, current_task=None):
        with self.connect() as db: db.execute("UPDATE workflows SET stage=?,current_task=?,state_json=?,updated_at=? WHERE id=?", (stage,current_task,json.dumps(state),datetime.now(timezone.utc).isoformat(),workflow_id))
    def get_workflow(self, workflow_id):
        with self.connect() as db:
            row = db.execute("SELECT id,feature,stage,current_task,state_json,created_at,updated_at FROM workflows WHERE id=?", (workflow_id,)).fetchone()
        if not row: return None
        return dict(zip(("workflow_id","feature","stage","current_task","state","created_at","updated_at"), (*row[:4], json.loads(row[4]), *row[5:])))
    def list_workflows(self):
        with self.connect() as db: rows=db.execute("SELECT id,feature,stage,current_task,updated_at FROM workflows ORDER BY updated_at DESC").fetchall()
        return [dict(zip(("workflow_id","feature","stage","current_task","updated_at"),r)) for r in rows]
    def record_provider_execution(self, result, prompt):
        rid = str(uuid.uuid4()); now = datetime.now(timezone.utc).isoformat(); digest=hashlib.sha256(prompt.encode()).hexdigest()
        with self.connect() as db:
            db.execute("INSERT INTO provider_executions VALUES(?,?,?,?,?,?,?,?,?,?,?)", (rid,None,result.provider,result.model,result.role,digest,now,result.duration,result.exit_code,"PASS" if result.success else "FAIL",json.dumps(result.__dict__)))
            db.execute("INSERT INTO metrics VALUES(?,?,?,?,?,?,?)", (str(uuid.uuid4()),result.provider,result.model,result.role,"success",1.0 if result.success else 0.0,now))
        return rid
