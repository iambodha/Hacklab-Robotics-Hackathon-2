"""Web page to enhance a rough instruction, then start, watch and stop the autonomous operator on the REAL arm: http://127.0.0.1:8894

  python3 sim/operator_ui.py            # this machine only
  python3 sim/operator_ui.py --port 8894

It runs sim/run_agent_real.sh (Claude with only the elrobot-real tools) with the task typed into the page, shows both
cameras, every tool call as it happens (sim/real_sessions/actions.jsonl), the agent's short notes and its final
report. Stop kills the agent; the servos then hold their last goal. Park sends move_tcp(0.0, 0.19, 0.06).
Resume starts a new run that is told what the last run did. Open gripper, Relax arm (torque off: it goes limp) and
Hold arm (torque on again) act on the arm directly while no run is in progress.
Needs Station (:8888) and the camera relay (:8890) running, like run_agent_real.sh. Stdlib only; `uv` for Park.
Transcripts go to sim/real_sessions/ui_run_<time>.jsonl. The page has no login: keep it on loopback.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
SESS = HERE / "real_sessions"
CAM_URL = os.environ.get("ROBOT_CAM_URL", "http://127.0.0.1:8890")
def _cams() -> dict:
    sys.path.insert(0, str(HERE))
    import operator_prompt
    c = operator_prompt.profile()["cameras"]
    return {"wrist": os.environ.get("ROBOT_WRIST_CAM", c["wrist"]), "overview": os.environ.get("ROBOT_OVERVIEW_CAM", c["overview"])}


CAMS = _cams()
RUN = {"proc": None, "file": None, "t0": None, "task": "", "model": "", "stopped": False}
LOCK = threading.Lock()

DEFAULT_TASK = """Sort the loose items on the white table into the two boxes.
- GARBAGE (wrappers, foil, paper) goes into the WHITE cardboard box.
- Everything else goes into the GREY box.
Nothing about positions is given: start with survey() and find the items and both boxes yourself.
Ignore the printed black-and-white marker sheets on the table.
When finished, park and report DONE with what went where, or BLOCKED."""


def env():
    e = dict(os.environ)
    if "NORMA_CORE_REPO" not in e:
        for c in (REPO / "norma-core", REPO.parent / "norma-core"):
            if (c / "hardware/elrobot/simulation").exists():
                e["NORMA_CORE_REPO"] = str(c.resolve())
                break
    return e


def other_agent_running():
    # any live real-arm server means some session is driving the arm (run_agent_real.sh execs claude, so its own
    # name never shows up in the process list)
    out = subprocess.run(["pgrep", "-f", "real_arm_mcp.py"], capture_output=True, text=True).stdout.split()
    mine = RUN["proc"].pid if RUN["proc"] and RUN["proc"].poll() is None else None
    return bool(out) and mine is None


def running():
    return RUN["proc"] is not None and RUN["proc"].poll() is None


def start(task, model, effort):
    with LOCK:
        if running() or other_agent_running():
            return "an operator run is already in progress"
        SESS.mkdir(exist_ok=True)
        f = SESS / f"ui_run_{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
        if model == "endpoint":       # an OpenAI-compatible endpoint (Gemma etc.): sim/model_endpoint.env or MODEL_* env
            cmd = ["bash", str(HERE / "run_agent_model.sh"), "--log", str(f), "-p", task]
            out, label = open(str(f) + ".out", "w"), "model endpoint"
        else:
            cmd = ["bash", str(HERE / "run_agent_real.sh"), "--model", model, "--max-turns", "90",
                   "--output-format", "stream-json", "--verbose", "-p", "Task for this run (it replaces the default task): " + task]
            if effort:
                cmd[4:4] = ["--effort", effort]
            out, label = open(f, "w"), f"{model} {effort}".strip()
        RUN.update(file=f, t0=time.time(), task=task, model=label, stopped=False)
        RUN["proc"] = subprocess.Popen(cmd, cwd=REPO, env=env(), stdin=subprocess.DEVNULL, stdout=out,
                                       stderr=open(str(f) + ".err", "w"), start_new_session=True)
    return None


def stop():
    if not running():
        return "no run in progress"
    RUN["stopped"] = True
    try:
        os.killpg(RUN["proc"].pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    return None


ENHANCER = """You turn a short, casual instruction from a person standing at a robot table into a clear task for the
robot's operator agent. You do not control the robot and must not try to.

First read the two image files in the current folder with the Read tool: overview.jpg (the whole table seen from the
front; the arm is at the back) and wrist.jpg (the view from the camera on the gripper). They show the table as it
is right now. Use them to work out what the person means ("this box", "the cap", "the garbage").

Then write the task. Rules:
- Keep the person's intent exactly. Do not add goals they did not ask for.
- Name each object the way it looks in the images (colour, kind, rough size) so it can be recognised, and say what
  it is NOT to be confused with if something similar is on the table.
- Say where each object must end up and what "done" looks like in a camera image.
- Never give coordinates or pixel positions: the operator finds everything itself with survey().
- Add practical hints that follow from the operator's abilities below: which part of an object to grip (the gripped
  part must be narrower than 4 cm: a thin edge, a brim, a strap, a corner of fabric), object_height below 0.01 for
  flat or soft things, placing something ON TOP of a box means centring over the box top and releasing, and so on.
- If the request cannot be done with these abilities, or the images do not show the object or the target, say so
  in one line starting with "PROBLEM:" and still write the best task you can.
- Plain text, at most 12 short lines, no headings, no markdown. End with: "When finished, park and report DONE with
  what went where, or BLOCKED."

What the operator agent can do (its own instructions):
----
%s
----
Reply with the task text only."""


def enhance(rough: str) -> str:
    """Rewrite a casual instruction into an operator task, grounded in the current camera images."""
    import tempfile
    work = tempfile.mkdtemp(prefix="elrobot-enhance-")
    seen = []
    for name in ("overview", "wrist"):
        try:
            open(os.path.join(work, f"{name}.jpg"), "wb").write(snapshot(name))
            seen.append(name)
        except Exception:
            pass
    import importlib
    import operator_prompt
    importlib.reload(operator_prompt)        # the template and profile can change while this page keeps running
    system = ENHANCER % operator_prompt.render()
    if len(seen) < 2:
        system += f"\n(Only these images could be captured: {seen or 'none'}; work from the text for the rest.)"
    cmd = ["claude", "-p", "The person's instruction: " + rough, "--model", "sonnet", "--effort", "low",
           "--tools", "Read", "--allowedTools", "Read", "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}',
           "--append-system-prompt", system]
    r = subprocess.run(cmd, cwd=work, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=120)
    out = r.stdout.strip()
    if r.returncode != 0 or not out:
        raise RuntimeError((r.stderr.strip() or "the enhancer returned nothing")[-300:])
    return out


def _profile() -> dict:
    sys.path.insert(0, str(HERE))
    import importlib
    import operator_prompt
    importlib.reload(operator_prompt)
    return operator_prompt.profile()


CALL = """
import asyncio, json, os, sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
async def main():
    p = StdioServerParameters(command="bash", args=[sys.argv[1]], env=dict(os.environ))
    async with stdio_client(p) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            res = await s.call_tool(sys.argv[2], json.loads(sys.argv[3]))
            print(res.content[0].text.splitlines()[0])
asyncio.run(main())
"""


def arm_call(tool: str, args: dict) -> str:
    """One tool call on the real arm, outside a run (Park, Open gripper, Relax, Hold)."""
    with LOCK:
        if running() or other_agent_running():
            return "stop the run first"
        r = subprocess.run(["uv", "run", "-q", "--no-project", "--with", "mcp>=1.12,<2", "python", "-c", CALL,
                            str(HERE / "run_real_server.sh"), tool, json.dumps(args)],
                           cwd=REPO, env=env(), capture_output=True, text=True, timeout=120)
    lines = [l for l in r.stdout.splitlines() if l.strip()]
    return lines[-1] if lines else (f"{tool} failed: " + r.stderr.strip()[-200:])


def park():
    return arm_call("move_tcp", dict(zip("xyz", _profile()["poses"]["park"])))


def last_transcript():
    """The transcript to resume from: this page's last run, else the newest run file in real_sessions."""
    if RUN["file"] and Path(RUN["file"]).exists():
        return Path(RUN["file"])
    files = [f for pat in ("ui_run_*.jsonl", "agent_*.jsonl") for f in SESS.glob(pat) if f.stat().st_size]
    return max(files, key=lambda f: f.stat().st_mtime) if files else None


def run_history(f, limit=40):
    """What a run did, as text: its tool calls with the first line of each result, and its closing words."""
    calls, notes, report = [], [], ""
    for line in open(f):
        try:
            d = json.loads(line)
        except ValueError:
            continue
        if d.get("type") == "assistant":
            for b in d["message"].get("content", []):
                if b.get("type") == "tool_use":
                    calls.append([b["name"].split("__")[-1] + "(" + json.dumps(b.get("input", {}))[1:-1][:110] + ")", ""])
                elif b.get("type") == "text" and b.get("text", "").strip():
                    notes.append(b["text"].strip()[:300])
        elif d.get("type") == "user" and calls and isinstance(d["message"].get("content"), list):
            for b in d["message"]["content"]:
                if b.get("type") == "tool_result" and isinstance(b.get("content"), list):
                    text = next((c.get("text", "") for c in b["content"] if c.get("type") == "text"), "")
                    calls[-1][1] = (text.strip().splitlines() or ["done"])[0][:170]
        elif d.get("type") == "result":
            report = (d.get("result") or "")[:1500]
    skip = ("look", "survey", "pixel_to_table", "tcp_pose", "joint_state", "camera_info")
    acts = [f"- {c} -> {r or '(no reply: the run was stopped during this call)'}" for c, r in calls if c.split("(")[0] not in skip]
    return acts[-limit:], (report or "\n".join(notes[-2:]))


def resume(task, model, effort):
    f = last_transcript()
    if f is None:
        return "there is no earlier run to resume"
    acts, closing = run_history(f)
    text = ("CONTINUE a task that an earlier run left unfinished. The task:\n" + task + "\n\n"
            "What the earlier run did, in order (positions may have changed since, and people may have moved "
            "things or the arm by hand: start with survey() and check in the images what is already done):\n"
            + ("\n".join(acts) or "- nothing yet") + "\n\n"
            + ("How the earlier run ended:\n" + closing + "\n\n" if closing else "The earlier run was stopped by the operator.\n\n")
            + "Do only what is still missing. Do not repeat an approach that failed above in the same way. If a "
            "move is refused because torque is off, call set_torque(true) once and go on. If the gripper holds "
            "something you do not need, put it down first.")
    return start(text, model, effort)


def read_transcript():
    notes, result = [], None
    try:
        for line in open(RUN["file"]):
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if d.get("type") == "assistant":
                for b in d["message"].get("content", []):
                    if b.get("type") == "text" and b.get("text", "").strip():
                        notes.append(b["text"].strip()[:400])
            elif d.get("type") == "result":
                result = {"text": d.get("result", ""), "seconds": round(d.get("duration_ms", 0) / 1000),
                          "turns": d.get("num_turns"), "cost": round(d.get("total_cost_usd", 0), 2)}
    except (OSError, TypeError):
        pass
    return notes, result


def read_actions():
    if not RUN["t0"]:
        return []
    since = time.strftime("%H:%M:%S", time.localtime(RUN["t0"]))
    rows = []
    try:
        for line in open(SESS / "actions.jsonl").read().splitlines()[-400:]:
            d = json.loads(line)
            if "tool" not in d or d["t"] < since:
                continue
            args = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in d["args"].items() if k not in ("step", "dt", "top_down")}
            res = d["error"] or d["result"]
            rows.append({"t": d["t"], "tool": d["tool"], "args": json.dumps(args)[:90], "s": d["s"],
                         "result": (res.splitlines()[0] if res else "")[:160] if d["tool"] != "look" else "(images)", "error": bool(d["error"])})
    except (OSError, ValueError, KeyError):
        pass
    return rows[-60:]


def status():
    notes, result = read_transcript()
    return {"running": running(), "other": other_agent_running(), "task": RUN["task"], "model": RUN["model"],
            "elapsed": round(time.time() - RUN["t0"]) if RUN["t0"] and running() else None, "stopped": RUN["stopped"],
            "actions": read_actions(), "notes": notes[-6:], "result": result, "default_task": DEFAULT_TASK}


_hash = {}


def snapshot(name):
    prefix = CAMS[name]
    if prefix not in _hash:
        cams = json.loads(urllib.request.urlopen(CAM_URL + "/api/cameras", timeout=2).read())
        _hash[prefix] = next(c["hash"] for c in cams if c["hash"].startswith(prefix))
    return urllib.request.urlopen(f"{CAM_URL}/snapshot/{_hash[prefix]}", timeout=2).read()


PAGE = rb"""<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Arm operator</title>
<style>
:root{--bg:#f3f5f8;--card:#fff;--fg:#18202b;--mut:#657082;--ok:#0e8a5f;--okbg:#e3f6ee;--bad:#c5372c;--badbg:#fdeceb;--warn:#a86400;--warnbg:#fdf3df;
 --line:#e2e6ec;--acc:#2f5bea;--accbg:#e9eeff;--chip:#eef1f5;--shadow:0 1px 2px rgba(16,24,40,.05),0 4px 14px rgba(16,24,40,.05)}
@media (prefers-color-scheme:dark){:root{--bg:#0e1116;--card:#171c23;--fg:#e7eaf0;--mut:#93a0b4;--ok:#42d39b;--okbg:#12322a;--bad:#ff7468;--badbg:#3a1a18;
 --warn:#f2b54a;--warnbg:#35280f;--line:#262d38;--acc:#8aa7ff;--accbg:#1c2748;--chip:#222a35;--shadow:none}}
*{box-sizing:border-box}html,body{margin:0}body{background:var(--bg);color:var(--fg);font:14px/1.5 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}
header{position:sticky;top:0;z-index:5;display:flex;gap:12px;align-items:center;flex-wrap:wrap;padding:10px 20px;background:var(--card);border-bottom:1px solid var(--line)}
header h1{font-size:16px;margin:0;letter-spacing:-.01em;display:flex;gap:9px;align-items:center}
.logo{width:22px;height:22px;border-radius:6px;background:var(--acc);display:inline-grid;place-items:center;color:#fff;font-size:12px;font-weight:700}
.pill{display:inline-flex;gap:7px;align-items:center;padding:3px 11px;border-radius:999px;font-weight:600;font-size:13px;background:var(--chip);color:var(--mut)}
.pill i{width:8px;height:8px;border-radius:50%;background:currentColor}
.pill.run{background:var(--okbg);color:var(--ok)}.pill.run i{animation:pulse 1.2s infinite}.pill.warn{background:var(--warnbg);color:var(--warn)}.pill.bad{background:var(--badbg);color:var(--bad)}
@keyframes pulse{50%{opacity:.25}}
.stats{margin-left:auto;display:flex;gap:18px;color:var(--mut);font-size:13px;font-variant-numeric:tabular-nums}.stats b{color:var(--fg);font-weight:600}
main{padding:16px 20px 32px;max-width:1500px;margin:0 auto;display:grid;gap:14px;grid-template-columns:minmax(0,5fr) minmax(0,6fr)}
@media (max-width:980px){main{grid-template-columns:minmax(0,1fr);padding:12px}.stats{margin-left:0;width:100%}}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:14px 16px;box-shadow:var(--shadow);min-width:0}
.wide{grid-column:1/-1}
h2{font-size:11px;margin:0 0 10px;color:var(--mut);font-weight:700;letter-spacing:.07em;text-transform:uppercase;display:flex;justify-content:space-between;align-items:center}
h2 span{font-weight:500;letter-spacing:0;text-transform:none}
textarea{width:100%;height:220px;resize:vertical;font:13px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;background:var(--bg);color:var(--fg);border:1px solid var(--line);border-radius:10px;padding:10px 12px}
textarea:focus,select:focus,button:focus-visible{outline:2px solid var(--acc);outline-offset:1px}
select,button{font:inherit;padding:8px 14px;border-radius:10px;border:1px solid var(--line);background:var(--card);color:var(--fg)}
button{cursor:pointer;font-weight:600;transition:transform .05s,background .15s}button:hover:not(:disabled){background:var(--chip)}button:active:not(:disabled){transform:translateY(1px)}
button:disabled{opacity:.4;cursor:default}
#start{background:var(--acc);color:#fff;border-color:var(--acc)}#start:hover:not(:disabled){filter:brightness(1.08);background:var(--acc)}
@media (prefers-color-scheme:dark){#start{color:#0e1116}}
#resume{border-color:var(--acc);color:var(--acc)}#stop,#relax{color:var(--bad);border-color:var(--bad)}#stop:hover:not(:disabled),#relax:hover:not(:disabled){background:var(--badbg)}
.row{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-top:10px}#model{max-width:190px}.grow{flex:1}
.arm{margin-top:12px;padding-top:12px;border-top:1px dashed var(--line)}.arm .lab{color:var(--mut);font-size:12px;font-weight:700;letter-spacing:.07em;text-transform:uppercase;margin-right:4px}
#msg{color:var(--mut);font-size:13px;min-height:20px;margin-top:8px}#orig{color:var(--mut);font-size:12px;margin-top:4px}
.cams{display:grid;gap:10px;grid-template-columns:1fr 1fr}@media (max-width:560px){.cams{grid-template-columns:1fr}}
.cam{position:relative;border-radius:10px;overflow:hidden;background:#000}.cam img{display:block;width:100%;aspect-ratio:299/224;object-fit:contain}
.cam b{position:absolute;left:8px;top:8px;background:rgba(0,0,0,.6);color:#fff;font-size:11px;font-weight:600;padding:2px 8px;border-radius:999px;letter-spacing:.03em}
.cam em{position:absolute;right:8px;top:8px;font-style:normal;background:rgba(197,55,44,.92);color:#fff;font-size:10px;font-weight:700;padding:2px 7px;border-radius:999px;letter-spacing:.06em}
.cam.off em{background:rgba(90,90,90,.9)}
pre{white-space:pre-wrap;margin:0;font:14px/1.55 ui-sans-serif,system-ui,sans-serif;overflow-wrap:anywhere}
#resultcard{border-left:4px solid var(--line)}#resultcard.done{border-left-color:var(--ok)}#resultcard.blocked{border-left-color:var(--bad)}#resultcard.running{border-left-color:var(--acc)}
.meta{display:inline-flex;gap:10px;margin-bottom:8px;font-size:12px;color:var(--mut);font-variant-numeric:tabular-nums}.meta span{background:var(--chip);padding:2px 9px;border-radius:999px}
.note{padding:8px 0;border-bottom:1px solid var(--line);color:var(--fg)}.note:last-child{border:0}.note:first-child{padding-top:0}
.scroll{max-height:460px;overflow:auto;border-radius:8px}
table{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}
th{position:sticky;top:0;background:var(--card);color:var(--mut);font-size:11px;letter-spacing:.06em;text-transform:uppercase;font-weight:700}
td,th{padding:6px 8px;text-align:left;border-bottom:1px solid var(--line);vertical-align:top}td.t,td.s{color:var(--mut);white-space:nowrap}
td.a{font:12px/1.5 ui-monospace,Menlo,monospace;color:var(--mut);overflow-wrap:anywhere}tr.fail td.r{color:var(--bad)}tr.good td.r{color:var(--ok)}
.chip{display:inline-block;padding:1px 9px;border-radius:999px;font-weight:600;font-size:12px;background:var(--chip);white-space:nowrap}
.chip.move{background:var(--accbg);color:var(--acc)}.chip.grip{background:var(--warnbg);color:var(--warn)}.chip.see{background:var(--chip);color:var(--mut)}
.empty{color:var(--mut)}
</style>
<header><h1><span class=logo>A</span>Arm operator</h1><span id=state class=pill><i></i>connecting</span>
 <div class=stats><span>model <b id=smodel>-</b></span><span>elapsed <b id=selapsed>-</b></span><span>tool calls <b id=scalls>0</b></span><span>failed <b id=sfail>0</b></span></div></header>
<main>
<section class=card><h2>Task</h2><textarea id=task spellcheck=false placeholder="Say what the arm should do, in plain words. Enhance rewrites it using the camera images."></textarea>
 <div class=row><select id=model title="Model that operates the arm"><option value=sonnet>Sonnet</option><option value=haiku>Haiku</option><option value=opus>Opus</option><option value=claude-fable-5-2>Fable</option><option value=endpoint>Model endpoint (Gemma, see model_endpoint.env)</option></select>
 <select id=effort title="Reasoning effort"><option value=low>effort: low</option><option value="">effort: default</option><option value=medium>effort: medium</option></select>
 <span class=grow></span><button id=enh title="Rewrite the instruction into a clear task, using the cameras">Enhance</button></div>
 <div class=row><button id=start>Start</button><button id=resume title="Start a new run that is told what the last run did">Resume task</button><span class=grow></span><button id=stop>Stop</button></div>
 <div class="row arm"><span class=lab>Arm</span><button id=open>Open gripper</button><button id=park>Park</button><button id=hold title="Torque on again, holding where it is">Hold</button><span class=grow></span><button id=relax title="Torque off: the arm goes limp">Relax</button></div>
 <div id=msg></div><div id=orig></div></section>
<section class=card><h2>Cameras <span id=camnote></span></h2><div class=cams>
 <div class=cam id=cwb><img id=cw alt="wrist camera"><b>Wrist</b><em>LIVE</em></div>
 <div class=cam id=cob><img id=co alt="overview camera"><b>Overview</b><em>LIVE</em></div></div></section>
<section class="card wide" id=resultcard><h2>Result</h2><div id=rmeta></div><pre id=result class=empty>No run yet.</pre></section>
<section class="card wide"><h2>Agent notes <span>latest first</span></h2><div id=notes class=empty>-</div></section>
<section class="card wide"><h2>Tool calls <span id=actcount></span></h2><div class=scroll><table id=acts></table></div></section>
</main>
<script>
const $=id=>document.getElementById(id), esc=s=>String(s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
const KIND={move_tcp:'move',move_to:'move',center_over:'move',drop_at:'move',release_here:'move',wipe:'move',stir:'move',touch_table:'move',
 pick_at:'grip',pick_thin:'grip',grab_tip:'grip',grab_upright:'grip',gripper:'grip',grip_state:'grip',set_torque:'grip'};
const FAIL=/ABORT|STOPPED|NOT holding|UNREACHABLE|slipped out|refused|failed/, GOOD=/^HOLDING|^RELEASED|^WIPED|already above/;
let filled=false, rough='';
function fmt(s){if(s==null)return '-';const m=Math.floor(s/60);return m?m+' min '+String(s%60).padStart(2,'0')+' s':s+' s';}
async function post(p,b){try{const r=await fetch(p,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b||{})});const j=await r.json();$('msg').textContent=j.message&&j.message!=='ok'?j.message:'';}catch(e){$('msg').textContent='request failed';}tick();}
const run=()=>({task:$('task').value,model:$('model').value,effort:$('effort').value});
$('start').onclick=()=>post('/api/start',run());
$('resume').onclick=()=>post('/api/resume',run());
$('stop').onclick=()=>post('/api/stop');
$('open').onclick=()=>{$('msg').textContent='Opening the gripper...';post('/api/open');};
$('park').onclick=()=>{$('msg').textContent='Parking...';post('/api/park');};
$('hold').onclick=()=>{$('msg').textContent='Torque on...';post('/api/hold');};
$('relax').onclick=()=>{if(confirm('Torque off: the arm goes limp and DROPS if it is raised. Hold it or make sure it is near the table. Relax now?')){$('msg').textContent='Relaxing...';post('/api/relax');}};
$('enh').onclick=async()=>{rough=$('task').value;$('enh').disabled=true;$('msg').textContent='Enhancing (looks at the cameras, about 15 s)...';
 try{const r=await fetch('/api/enhance',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({task:rough})});const j=await r.json();
  if(j.task){$('task').value=j.task;$('orig').textContent='Your original instruction: '+rough;}$('msg').textContent=j.message||'';}
 catch(e){$('msg').textContent='Enhance failed';}$('enh').disabled=false;};
function pill(cls,text){const p=$('state');p.className='pill '+cls;p.innerHTML='<i></i>'+esc(text);}
async function tick(){let s;try{s=await (await fetch('/api/status')).json();}catch(e){pill('bad','page lost contact');return;}
 if(!filled){$('task').value=s.task||s.default_task;filled=true;}
 const busy=s.running||s.other;
 if(s.running)pill('run','running');else if(s.other)pill('warn','another run is in progress (started outside this page)');else pill('','idle');
 $('smodel').textContent=s.model||'-';$('selapsed').textContent=s.running?fmt(s.elapsed):(s.result?fmt(s.result.seconds):'-');
 $('start').disabled=busy;$('stop').disabled=!s.running;for(const b of ['park','resume','open','relax','hold'])$(b).disabled=busy;
 const rc=$('resultcard'),res=$('result');
 if(s.result){const t=s.result.text||'';rc.className='card wide '+(/BLOCKED|not finished|could not|couldn\'t/i.test(t.slice(0,400))?'blocked':'done');
  $('rmeta').innerHTML=`<div class=meta><span>${fmt(s.result.seconds)}</span><span>${s.result.turns} turns</span><span>$${s.result.cost}</span></div>`;res.textContent=t;res.className='';}
 else{$('rmeta').innerHTML='';rc.className='card wide'+(s.running?' running':'');res.className='empty';
  res.textContent=s.running?'Running...':s.stopped?'Stopped by the operator. The servos hold their last goal.':'No run yet.';}
 const n=$('notes');if(s.notes.length){n.className='';n.innerHTML=s.notes.slice().reverse().map(x=>`<div class=note><pre>${esc(x)}</pre></div>`).join('');}else{n.className='empty';n.textContent='-';}
 const acts=s.actions.slice().reverse(),fails=acts.filter(a=>a.error||FAIL.test(a.result)).length;
 $('scalls').textContent=acts.length;$('sfail').textContent=fails;$('actcount').textContent=acts.length?'latest first':'';
 $('acts').innerHTML=acts.length?'<tr><th>time</th><th>tool</th><th>arguments</th><th>s</th><th>result</th></tr>'+acts.map(a=>{
  const bad=a.error||FAIL.test(a.result),good=!bad&&GOOD.test(a.result);
  return `<tr class="${bad?'fail':good?'good':''}"><td class=t>${a.t}</td><td><span class="chip ${KIND[a.tool]||'see'}">${esc(a.tool)}</span></td><td class=a>${esc(a.args)}</td><td class=s>${a.s}</td><td class=r>${esc(a.result)}</td></tr>`;}).join('')
  :'<tr><td class=empty>No tool calls in this run yet.</td></tr>';}
function cam(img,box,path){const i=new Image();i.onload=()=>{$(img).src=i.src;$(box).classList.remove('off');$(box).querySelector('em').textContent='LIVE';};
 i.onerror=()=>{$(box).classList.add('off');$(box).querySelector('em').textContent='NO SIGNAL';};i.src=path+'?'+Date.now();}
function cams(){cam('cw','cwb','/cam/wrist');cam('co','cob','/cam/overview');}
setInterval(tick,1500);setInterval(cams,700);tick();cams();
</script>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, code, body, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/":
            self.send(200, PAGE, "text/html; charset=utf-8")
        elif path == "/api/status":
            self.send(200, json.dumps(status()).encode())
        elif path in ("/cam/wrist", "/cam/overview"):
            try:
                self.send(200, snapshot(path.rsplit("/", 1)[1]), "image/jpeg")
            except Exception:
                self.send(503, b"")
        else:
            self.send(404, b"{}")

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except ValueError:
            body = {}
        if self.path == "/api/start":
            task = (body.get("task") or "").strip()
            model = body.get("model") if body.get("model") in ("sonnet", "haiku", "opus", "claude-fable-5-2", "endpoint") else "sonnet"
            effort = body.get("effort") if body.get("effort") in ("low", "medium", "high", "") else "low"
            msg = "the task is empty" if not task else start(task, model, effort)
        elif self.path == "/api/enhance":
            rough = (body.get("task") or "").strip()
            if not rough:
                return self.send(200, json.dumps({"message": "type an instruction first"}).encode())
            try:
                return self.send(200, json.dumps({"message": "enhanced: read it, edit if needed, then Start", "task": enhance(rough)}).encode())
            except Exception as e:
                return self.send(200, json.dumps({"message": f"enhance failed: {e}"}).encode())
        elif self.path == "/api/resume":
            task = (body.get("task") or "").strip()
            model = body.get("model") if body.get("model") in ("sonnet", "haiku", "opus", "claude-fable-5-2", "endpoint") else "sonnet"
            effort = body.get("effort") if body.get("effort") in ("low", "medium", "high", "") else "low"
            msg = "the task is empty" if not task else resume(task, model, effort)
        elif self.path == "/api/stop":
            msg = stop()
        elif self.path in ("/api/park", "/api/open", "/api/relax", "/api/hold"):
            name = self.path.rsplit("/", 1)[1]
            try:
                msg = {"park": park, "open": lambda: arm_call("gripper", {"open_fraction": 1.0}),
                       "relax": lambda: arm_call("set_torque", {"enabled": False}),
                       "hold": lambda: arm_call("set_torque", {"enabled": True})}[name]()
            except Exception as e:
                msg = f"{name} failed: {e}"
        else:
            return self.send(404, b"{}")
        self.send(200, json.dumps({"message": msg or "ok"}).encode())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Web page to run the real-arm operator")
    ap.add_argument("--port", type=int, default=8894)
    a = ap.parse_args()
    print(f"operator page -> http://127.0.0.1:{a.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()
