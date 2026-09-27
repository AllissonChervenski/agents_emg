"""Human gate for small, explicitly interactive feature runs."""
import json
import subprocess
from pathlib import Path


class InteractiveGate:
    def __init__(self, input_fn=input, workspace=None): self.input_fn=input_fn; self.workspace=Path(workspace or Path.cwd()).resolve()

    def confirm(self, stage, role, provider, model, files, commands, task_id=None) -> bool:
        summary={"stage":stage,"role":role,"provider":provider,"model":model or "CLI default",
                 "task":task_id,"files_expected_to_change":list(files),"commands_or_expected_outputs":list(commands)}
        print(json.dumps(summary,indent=2))
        while True:
            try: decision=self.input_fn("Gate [continue/inspect/abort]: ").strip().lower()
            except (EOFError,KeyboardInterrupt): return False
            if decision=="continue": return True
            if decision=="abort": return False
            if decision=="inspect":
                print("Review the stage plan above. No command has run for this gate yet.")
                print(json.dumps(summary,indent=2))
                paths=[path for path in files if isinstance(path,str) and (self.workspace/path).exists()]
                if paths:
                    try:
                        diff=subprocess.run(["git","diff","--",*paths],cwd=self.workspace,capture_output=True,text=True,timeout=10,check=False)
                        if diff.stdout: print("Current diff:\n"+diff.stdout[:4000])
                    except (OSError,subprocess.TimeoutExpired): pass
                continue
            print("Choose continue, inspect, or abort.")
