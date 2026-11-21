"""Deterministic control of the agent process. No LLM involved; this is what you (or Claude in chat) call.

  python3 scripts/evg/ctl.py start --task "..." --demo NAME[,NAME2] [--provider-preset claude-haiku] [--model M] [--dry-run] [--step-mode] [--replay NAME]
  python3 scripts/evg/ctl.py stop [--no-hold]     stop file -> SIGTERM -> SIGKILL, then hold the arm (torque on)
  python3 scripts/evg/ctl.py pause | resume       agent finishes the current action, then waits
  python3 scripts/evg/ctl.py status               process, last status.json, last log lines
  python3 scripts/evg/ctl.py demos                list demos (samples, duration, filtered or not)
  python3 scripts/evg/ctl.py demo-filter NAME [--eps N]
  python3 scripts/evg/ctl.py demo-delete NAME     moves to demos/.trash (not deleted)

Contract with the agent loop (scripts/evg/agent/run.py): every tick it must
  * exit cleanly if evg-run/STOP exists, and wait while evg-run/PAUSE exists,
  * rewrite evg-run/status.json (tick, last action, phase, ts),
  * log to evg-run/agent.log.
"""
import argparse, json, os, shutil, signal, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "evg-run"
PID, STOP, PAUSE, STATUS, LOG = (RUN / n for n in ("agent.pid", "STOP", "PAUSE", "status.json", "agent.log"))
DEMOS = ROOT / "demos"
AGENT = Path(__file__).resolve().parent / "agent" / "run.py"


def alive():
    try:
        pid = int(PID.read_text())
        os.kill(pid, 0)
        return pid
    except (FileNotFoundError, ValueError, ProcessLookupError):
        return None
    except PermissionError:
        return int(PID.read_text())


def wait_dead(pid, secs):
    end = time.time() + secs
    while time.time() < end:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.1)
    return False


def cmd_start(a):
    if alive():
        sys.exit(f"already running (pid {alive()}); stop it first")
    if not AGENT.exists():
        sys.exit(f"{AGENT} does not exist yet; the agent loop is not built")
    for d in a.demo.split(","):
        if not (DEMOS / d).is_dir():
            sys.exit(f"no such demo: {d}  (see: ctl.py demos)")
    RUN.mkdir(exist_ok=True)
    for f in (STOP, PAUSE, STATUS):
        f.unlink(missing_ok=True)
    args = [sys.executable, str(AGENT), "--task", a.task, "--demo", a.demo]
    for k in ("model", "provider", "base_url", "provider_preset", "replay", "max_ticks", "chunk", "effort"):
        if getattr(a, k):
            args += [f"--{k.replace('_', '-')}", str(getattr(a, k))]
    if a.step_mode:
        args.append("--step-mode")
    if a.dry_run:
        args.append("--dry-run")
    p = subprocess.Popen(args, stdout=open(LOG, "ab"), stderr=subprocess.STDOUT, start_new_session=True)
    PID.write_text(str(p.pid))
    print(f"started pid {p.pid}{' (dry-run)' if a.dry_run else ''}; log {LOG}")


def cmd_stop(a):
    RUN.mkdir(exist_ok=True)
    STOP.touch()
    pid = alive()
    if pid:
        print(f"STOP file set, waiting for pid {pid} to exit...")
        if not wait_dead(pid, 5):
            os.kill(pid, signal.SIGTERM); print("SIGTERM")
            if not wait_dead(pid, 3):
                os.kill(pid, signal.SIGKILL); print("SIGKILL"); wait_dead(pid, 2)
        print("agent stopped")
    else:
        print("agent was not running")
    PID.unlink(missing_ok=True); PAUSE.unlink(missing_ok=True)
    sys.path.insert(0, str(Path(__file__).parent))
    from agent import mcp_client
    if not a.no_hold and mcp_client.backend() == "mcp":
        try:
            print("arm hold (MCP):", mcp_client.hold_arm()[:120])
        except Exception as e:
            print("WARNING hold failed:", e)
    elif not a.no_hold:
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / "arm.py"), "hold"], capture_output=True, text=True, timeout=20)
        print("arm hold:", (r.stdout.strip() or r.stderr.strip() or "ok").splitlines()[-1])


def cmd_pause(a):
    RUN.mkdir(exist_ok=True); PAUSE.touch(); print("PAUSE set")


def cmd_resume(a):
    PAUSE.unlink(missing_ok=True); print("PAUSE cleared")


def cmd_status(a):
    pid = alive()
    print(f"process: {'running pid ' + str(pid) if pid else 'not running'}"
          f"{'  [PAUSED]' if PAUSE.exists() else ''}{'  [STOP set]' if STOP.exists() else ''}")
    if STATUS.exists():
        age = time.time() - STATUS.stat().st_mtime
        print(f"status ({age:.0f}s old):", STATUS.read_text().strip())
        if pid and age > 15:
            print("WARNING: status.json is stale while the process runs; the loop may be hung")
    if LOG.exists():
        print("--- log tail ---"); print("\n".join(LOG.read_text(errors="replace").splitlines()[-8:]))


def cmd_demos(a):
    if not DEMOS.is_dir():
        return print("no demos/ directory")
    for d in sorted(p for p in DEMOS.iterdir() if p.is_dir() and not p.name.startswith(".")):
        s = d / "states.jsonl"
        n = sum(1 for _ in open(s)) if s.exists() else 0
        dur = ""
        if n:
            last = json.loads(s.read_text().splitlines()[-1]); dur = f"{last['t']:.0f}s"
        f = d / "filtered.jsonl"
        fk = f"filtered {sum(1 for _ in open(f))}" if f.exists() else "not filtered"
        print(f"{d.name:20} {n:6} samples {dur:>6}  {len(list(d.glob('*.jpg'))):4} frames  {fk}")


def cmd_filter(a):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import demo_filter
    n, k, path = demo_filter.run(a.name, a.eps, a.grip_eps)
    print(f"{a.name}: {n} -> {k} samples  {path}")


def cmd_delete(a):
    src = DEMOS / a.name
    if not src.is_dir():
        sys.exit(f"no such demo: {a.name}")
    trash = DEMOS / ".trash"; trash.mkdir(exist_ok=True)
    dst = trash / f"{a.name}-{int(time.time())}"
    shutil.move(str(src), str(dst)); print(f"moved to {dst}")


def main():
    import common; common.apply_robot_env()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("start"); s.set_defaults(f=cmd_start)
    s.add_argument("--task", required=True); s.add_argument("--demo", required=True)
    s.add_argument("--model"); s.add_argument("--provider"); s.add_argument("--base-url", dest="base_url")
    s.add_argument("--dry-run", action="store_true"); s.add_argument("--step-mode", action="store_true")
    s.add_argument("--provider-preset"); s.add_argument("--replay"); s.add_argument("--max-ticks", type=int); s.add_argument("--chunk", type=int); s.add_argument("--effort")
    s = sp.add_parser("stop"); s.set_defaults(f=cmd_stop); s.add_argument("--no-hold", action="store_true")
    sp.add_parser("pause").set_defaults(f=cmd_pause)
    sp.add_parser("resume").set_defaults(f=cmd_resume)
    sp.add_parser("status").set_defaults(f=cmd_status)
    sp.add_parser("demos").set_defaults(f=cmd_demos)
    s = sp.add_parser("demo-filter"); s.set_defaults(f=cmd_filter)
    s.add_argument("name"); s.add_argument("--eps", type=float, default=40); s.add_argument("--grip-eps", type=float, default=60)
    s = sp.add_parser("demo-delete"); s.set_defaults(f=cmd_delete); s.add_argument("name")
    a = ap.parse_args(); a.f(a)


if __name__ == "__main__":
    main()
