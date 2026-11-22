"""Isolated git working trees for the improvement agent (candidate) and for the incumbent (baseline arm), both
checked out at the same commit. Only committed code is used: uncommitted changes in the source tree are not seen."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


def git(*args, cwd, check=True) -> str:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=check).stdout.strip()


@dataclass
class Sandbox:
    repo: Path
    path: Path
    base: str

    mode: str = "clone"

    @classmethod
    def create(cls, repo: str | Path, path: str | Path, base: str = "HEAD", mode: str | None = None) -> "Sandbox":
        """mode "clone" (default): a shared local clone inside `path`, so nothing is written to the source repo's
        .git; mode "worktree": `git worktree add --detach` (registers the sandbox in the source repo)."""
        mode = mode or os.environ.get("SKILLS_SANDBOX_MODE", "clone")
        repo, path = Path(repo).resolve(), Path(path).resolve()
        sha = git("rev-parse", base, cwd=repo)
        path.parent.mkdir(parents=True, exist_ok=True)
        if mode == "worktree":
            git("worktree", "add", "--detach", str(path), sha, cwd=repo)
        else:
            git("clone", "--quiet", "--shared", "--no-checkout", str(repo), str(path), cwd=path.parent)
            git("checkout", "--quiet", "--detach", sha, cwd=path)
        return cls(repo, path, sha, mode)

    def diff(self) -> str:
        git("add", "-A", "--intent-to-add", ".", cwd=self.path, check=False)     # untracked files show in the diff
        return subprocess.run(["git", "diff", self.base], cwd=self.path, capture_output=True, text=True).stdout

    def run(self, args: list[str], timeout: float = 600, env: dict | None = None) -> dict:
        """Run a command inside the sandbox with only the sandbox on the Python path."""
        e = {**os.environ, "PYTHONPATH": str(self.path), "SKILLS_LOOP_INNER": "1", "PYTHONDONTWRITEBYTECODE": "1",
             **(env or {})}
        try:
            r = subprocess.run(args, cwd=self.path, capture_output=True, text=True, timeout=timeout, env=e)
            return {"cmd": args, "returncode": r.returncode, "stdout": r.stdout[-6000:], "stderr": r.stderr[-4000:]}
        except subprocess.TimeoutExpired as ex:
            return {"cmd": args, "returncode": None, "stdout": (ex.stdout or "")[-6000:] if isinstance(ex.stdout, str) else "",
                    "stderr": f"timeout after {timeout}s"}

    def python(self, *args, **kw) -> dict:
        return self.run([sys.executable, *args], **kw)

    def remove(self):
        if self.mode == "worktree":
            git("worktree", "remove", "--force", str(self.path), cwd=self.repo, check=False)
            git("worktree", "prune", cwd=self.repo, check=False)
        elif (self.path / ".git").is_dir():
            shutil.rmtree(self.path, ignore_errors=True)
