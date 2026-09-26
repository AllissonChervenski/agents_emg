import subprocess
from pathlib import Path


class GitRepository:
    def __init__(self, cwd): self.cwd=Path(cwd).resolve()
    def available(self): return subprocess.run(["git","rev-parse","--is-inside-work-tree"],cwd=self.cwd,capture_output=True,text=True).returncode == 0
    def dirty(self):
        if not self.available(): return False
        return bool(subprocess.run(["git","status","--porcelain"],cwd=self.cwd,capture_output=True,text=True).stdout.strip())
    def checkpoint(self, message):
        if not self.available() or self.dirty(): return False
        # Empty checkpoint commits are deliberately avoided; task changes must be reviewed first.
        return False
