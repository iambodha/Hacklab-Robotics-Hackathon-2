"""Copy exactly the git shipping set, without .git, caches or local checkouts.

Run from the repository root: python3 scripts/release_fresh.py [destination]
The default is $TMPDIR/r5-release-fresh. Refuses to overwrite an existing path.
"""
import os
from pathlib import Path
import shutil
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
destination = Path(sys.argv[1] if len(sys.argv) > 1 else Path(os.environ.get("TMPDIR", "/tmp")) / "r5-release-fresh").resolve()
if destination.exists():
    raise SystemExit(f"Refusing to overwrite {destination}; use a fresh destination")
names = subprocess.check_output(["git", "ls-files", "-co", "--exclude-standard", "-z"], cwd=root).split(b"\0")
destination.mkdir(parents=True)
count = 0
for name in sorted(set(n for n in names if n)):
    relative = Path(os.fsdecode(name))
    source = root / relative
    if not source.exists():
        raise SystemExit(f"Shipping path is missing: {relative}")
    if source.is_symlink():
        raise SystemExit(f"Review symlink before shipping: {relative}")
    target = destination / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    count += 1
print(f"Copied {count} shipping files to {destination}; no .git or ignored data")
