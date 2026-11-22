"""python -m executor <command>: the operator console server and a command-line client for it.

  serve                      run the executor + operator console on http://127.0.0.1:8910 (owns all arm motion)
  task "<prompt>"            start a task with an advisor on a backend and follow its logic until it ends
  status [execution_id]      console, backend, advisor and status pills; or one execution's result
  cancel [execution_id]      cancel the running motion (the arm holds); --stop-task also ends the task
  observe                    take an observation and print the objects in world coordinates
  skills                     the skills the executor accepts, with their parameters
  runs                       recordings (executor + recorder service) and recent executions with outcomes

Every command except `serve` talks to a running console (EXECUTOR_URL, default http://127.0.0.1:8910).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

URL = os.environ.get("EXECUTOR_URL", "http://127.0.0.1:8910")
ADVISORS = ("scripted", "claude", "gemini", "none")
BACKENDS = ("fake", "sim", "real")


class ConsoleDown(SystemExit):
    pass


def api(path: str, body: dict | None = None, timeout: float = 30.0, method: str | None = None):
    req = urllib.request.Request(URL + path, method=method or ("POST" if body is not None else "GET"),
                                 data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        try:
            msg = json.loads(e.read()).get("error", "")
        except ValueError:
            msg = ""
        sys.exit(f"console refused ({e.code}): {msg or e.reason}")
    except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
        raise ConsoleDown(f"no executor console at {URL} ({getattr(e, 'reason', e)}). Start it: python -m executor serve "
                          f"(or scripts/stack.sh up)")


def v3(a) -> str:
    return "[" + ", ".join(f"{float(x):.3f}" for x in a) + "]" if a else "-"


# --- task: print the robot's logic as it happens -------------------------------------------------
def event_line(e: dict) -> str | None:
    k = e["kind"]
    t = time.strftime("%H:%M:%S", time.localtime(e["t"]))
    if k == "advisor_message":
        return f"{t}  advisor   {e.get('text', '')}"
    if k == "advisor_tool_call":
        inp = json.dumps(e.get("input") or {})
        return f"{t}  tool      {e.get('tool')} {inp if inp != '{}' else ''}".rstrip()
    if k == "advisor_tool_result":
        return f"{t}  result    " + (e.get("text") or "").replace("\n", "\n" + " " * 20)[:900]
    if k == "advisor_error":
        return f"{t}  ERROR     {e.get('text')}"
    if k == "advisor_result":
        return f"{t}  advisor   {e.get('text')}  ({e.get('turns')} turns, {e.get('seconds')} s)"
    if k == "proposal":
        p = e["proposal"]
        return f"{t}  proposal  {p['skill']} {json.dumps(p['params'])} from {p['source']}: {p['rationale']}"
    if k == "validation":
        if e["ok"]:
            n = len([c for c in e["checks"] if c != "validation_s"])
            tw = e["checks"].get("twin_preflight", {}).get("result", "?")
            return f"{t}  validate  passed {n} checks (twin preflight: {tw})"
        return f"{t}  validate  REJECTED: {'; '.join(e['reasons'])}"
    if k == "phase" and e.get("status") != "running":
        return f"{t}  phase     {e['name']} {e['status'].lower()}" + (f": {e['termination']}" if e.get("termination") else "")
    if k == "result":
        res = f"  residuals {json.dumps(e['residuals'])}" if e.get("residuals") else ""
        return f"{t}  {e['status']:<9} {e['skill']}: {e['termination']}{res}"
    if k == "outcome":
        ok = "unknown" if e["success"] is None else "success" if e["success"] else "failure"
        return f"{t}  outcome   {e['metric']} = {e['value']} {e['unit']} -> {ok}"
    if k == "cancel":
        return f"{t}  cancel    {e.get('execution_id')}"
    if k == "task":
        return f"{t}  task      {e['state']}" + (f": {e['why']}" if e.get("why") else "")
    return None


def follow(timeout: float, since: int) -> dict:
    t_end = time.monotonic() + timeout
    seq = since
    while time.monotonic() < t_end:
        r = api(f"/api/timeline?since={seq}")
        for e in r["events"]:
            line = event_line(e)
            if line:
                print(line, flush=True)
        seq = r["seq"]
        st = api("/api/state")
        if st["task"].get("state") != "running":
            return st["task"]
        time.sleep(0.4)
    print(f"still running after {timeout:.0f} s; stopping the task")
    return api("/api/task/stop", {})


def cmd_task(a):
    seq = api("/api/timeline?since=0&limit=1")["seq"]
    if a.backend:
        st = api("/api/state")
        if st["backend"]["name"] != a.backend:
            print(f"switching backend to {a.backend}" + (" (loading the twin)" if a.backend == "sim" else ""), flush=True)
            api("/api/backend", {"name": a.backend}, timeout=120)
    if a.arm_enabled:
        r = api("/api/arm", {"enabled": True})
        print(r["message"])
        if not r["can_move"]:
            sys.exit("the arm is not enabled; not starting the task")
    t = api("/api/task/start", {"prompt": a.prompt, "advisor": a.advisor, "model": a.model})
    print(f"task started: advisor {t['advisor']} on {t['backend']}; recording {t['recording_id']}\n  viewer {t['viewer']}", flush=True)
    if a.no_follow:
        return
    try:
        end = follow(a.timeout, seq)
    except KeyboardInterrupt:
        print("\ninterrupted: stopping the task (motion cancelled, the arm holds)")
        end = api("/api/task/stop", {})
    print(f"task ended: {end.get('why')}")
    sys.exit(0 if str(end.get("why", "")).startswith("DONE") else 1)


def cmd_status(a):
    if a.execution_id:
        r = api(f"/api/status/{a.execution_id}?wait={a.wait}", timeout=a.wait + 10)
        print(f"{r['execution_id']}  {r['skill']}  {r['status']}: {r['termination']}")
        for ph in r.get("phases", []):
            print(f"  {ph['name']:<9} {ph.get('status')}")
        if r.get("residuals"):
            print("  residuals", json.dumps(r["residuals"]))
        return
    st, h = api("/api/state"), api("/api/health")
    t = st["task"]
    cams = ", ".join(f"{r} {c['fps'] if c['fps'] is not None else '-'} fps" for r, c in h["cameras"].items()) or "no frames yet"
    arm = h["arm"]
    rows = [("console", URL),
            ("station", f"{'up' if h['station']['ok'] else 'DOWN'} ({h['station']['addr']})"),
            ("recorder", f"{'up' if h['recorder']['ok'] else 'down'} ({h['recorder']['url']}); this backend records {h['recorder']['mode']}"),
            ("cameras", cams),
            ("arm", "error " + arm["error"] if "error" in arm else
             f"torque {arm['torque_on']}/{arm['motors']}, {'calibrated' if arm['calibrated'] else 'NOT calibrated'}, "
             f"{'in range' if arm['in_range'] else 'OUT OF RANGE'}" + (f", faults {arm['faults']}" if arm["faults"] else "")),
            ("backend", st["backend"]["name"] + ("" if not h["backend"]["needs_enable"] else " ARMED" if h["backend"]["can_move"] else " read-only")),
            ("advisor", f"{h['advisor']['name'] or 'none'}{' (running)' if h['advisor']['running'] else ''}"),
            ("setup", h["setup_id"]), ("perception", h["locator"]),
            ("motion", f"executing {st['current']}" if st["busy"] else "idle"),
            ("task", t.get("state", "idle") + (f": {t.get('prompt')!r} via {t.get('advisor')}" if t.get("prompt") else "")
             + (f" -> {t['why']}" if t.get("why") else "")),
            ("twin mirror", f"{'up' if h['mirror']['ok'] else 'down'} ({h['mirror']['url']})")]
    for k, v in rows:
        print(f"{k:<12} {v}")


def cmd_cancel(a):
    r = api(f"/api/cancel/{a.execution_id or 'current'}", {})
    print("cancel sent: no further commands, the arm holds" if r["cancelled"] else "nothing is moving")
    if a.stop_task:
        print("task:", api("/api/task/stop", {}).get("why"))


def cmd_observe(a):
    o = api("/api/observe", {})
    if a.json:
        return print(json.dumps(o, indent=1))
    j = o["joints"]
    print(f"{o['obs_id']}  setup {o['setup_id']}  freshness {o['freshness_s']} s")
    for role, f in o["frames"].items():
        print(f"  {role:<9} frame #{f['seq']} age {f['age_s']} s" + ("  STALE" if f["stale"] else ""))
    print(f"  TCP world {v3(j['tcp_world'])} m   gripper {j['gripper_open']}")
    print(f"\n  {'object':<22} {'x':>7} {'y':>7} {'z':>7}  {'±mm':>5}  visible  source")
    for ob in o["objects"]:
        x, y, z = ob["xyz_world"]
        unc = f"{ob['uncertainty_mm']:.0f}" if ob.get("uncertainty_mm") is not None else "-"
        print(f"  {ob['name'][:22]:<22} {x:7.3f} {y:7.3f} {z:7.3f}  {unc:>5}  {'yes' if ob['visible'] else 'no ':<7}  {ob.get('source', '')}")
    if not o["objects"]:
        print("  (no objects estimated)")
    for n in o.get("notes", []):
        print("  note:", n)
    print("\n  world frame: metres, origin = arm base on the table top, +x forward, +y left, +z up")


def cmd_skills(a):
    for name, s in api("/api/skills").items():
        print(f"{name} (v{s['version']}, observation <= {s['freshness_s']} s old)\n  {s['doc']}")
        for k, v in s["params"].items():
            print(f"    {k}: {v}")


def cmd_runs(a):
    try:
        r = api("/api/recordings")
        ex = api("/api/executions")["executions"]
    except ConsoleDown:
        from recorder.store import list_recordings
        from .recording import LOCAL_ROOT
        print(f"(console not running: local recordings in {LOCAL_ROOT})")
        r, ex = {"local": [{"id": m["id"], "duration": m.get("duration"), "markers": m.get("counts", {}).get("markers")}
                           for m in list_recordings(LOCAL_ROOT)[:a.n]], "service": []}, []
    for title, rows in (("executor recordings (fake / sim)", r["local"]), ("recorder service (real arm)", r["service"])):
        print(title)
        for s in rows[:a.n]:
            d = f"{float(s['duration']):.0f} s" if s.get("duration") is not None else "active"
            print(f"  {s['id']:<44} {d:>7}  {s.get('markers') or 0:>3} markers  {s.get('viewer', '')}")
        if not rows:
            print("  none")
    if ex:
        outcomes = {}
        for e in api("/api/timeline?since=0&limit=3000")["events"]:
            if e["kind"] == "outcome":
                outcomes[e["execution_id"]] = e
        print("executions (this console, newest first)")
        for x in ex[:a.n]:
            o = outcomes.get(x["execution_id"])
            oc = "" if not o else f"  outcome {o['metric']}={o['value']} " + ("unknown" if o["success"] is None else "success" if o["success"] else "failure")
            print(f"  {x['execution_id']}  {x['skill']:<11} {x['status']:<9} {x['termination'][:60]}{oc}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m executor", formatter_class=argparse.RawDescriptionHelpFormatter,
                                 description=__doc__.split("\n\n")[0], epilog="Console URL: EXECUTOR_URL (default %s)." % URL)
    sub = ap.add_subparsers(dest="cmd", required=True, metavar="command")
    s = sub.add_parser("serve", help="run the executor + operator console (loopback only)")
    s.add_argument("--backend", default="fake", help="fake | sim | real | replay:<recording> (default fake)")
    s.add_argument("--port", type=int, default=8910)
    s.add_argument("--host", default="127.0.0.1", help="loopback only: the console has no login")

    t = sub.add_parser("task", help="start a task and follow the robot's logic until it ends",
                       description="Start a task on the console and print the advisor's messages, tool calls, proposals, "
                                   "validation, phases and outcomes as they happen. Ctrl-C stops the task (the arm holds). "
                                   "Exit code 0 when the advisor reports DONE.")
    t.add_argument("prompt", help='e.g. "Put the blue object in the bin"')
    t.add_argument("--advisor", choices=ADVISORS, default="scripted", help="default scripted (deterministic planner)")
    t.add_argument("--backend", choices=BACKENDS, help="switch the console's backend first (default: keep the current one)")
    t.add_argument("--model", default="sonnet", help="Claude model (claude advisor only)")
    t.add_argument("--arm-enabled", action="store_true",
                   help="real backend: enable real motor commands first (the console must run with EXECUTOR_REAL_MOTION=1)")
    t.add_argument("--timeout", type=float, default=600, help="stop the task after this many seconds (default 600)")
    t.add_argument("--no-follow", action="store_true", help="start it and return")

    st = sub.add_parser("status", help="console and robot status, or one execution's result")
    st.add_argument("execution_id", nargs="?")
    st.add_argument("--wait", type=float, default=0, help="with an execution id: wait up to S seconds for it to finish")
    c = sub.add_parser("cancel", help="cancel the running motion (the arm holds position)")
    c.add_argument("execution_id", nargs="?", help="default: whatever is running")
    c.add_argument("--stop-task", action="store_true", help="also stop the task and its advisor")
    o = sub.add_parser("observe", help="take an observation; print objects in world coordinates")
    o.add_argument("--json", action="store_true", help="print the whole Observation record")
    sub.add_parser("skills", help="list the executor's skills and parameters")
    r = sub.add_parser("runs", help="list recordings and recent executions with their outcomes")
    r.add_argument("-n", type=int, default=12, help="rows per list (default 12)")

    a = ap.parse_args(argv)
    if a.cmd == "serve":
        from .console.server import serve
        return serve(a.backend, a.port, a.host)
    try:
        {"task": cmd_task, "status": cmd_status, "cancel": cmd_cancel, "observe": cmd_observe, "skills": cmd_skills,
         "runs": cmd_runs}[a.cmd](a)
    except ConsoleDown as e:
        sys.exit(str(e))


if __name__ == "__main__":
    main()
