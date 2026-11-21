"""The closed-loop agent. Normally started via ctl.py / evg.py; can be run directly.

  python3 scripts/evg/agent/run.py --task "..." --demo take-1[,take-2] --provider-preset claude-haiku [--dry-run] [--step-mode]
  python3 scripts/evg/agent/run.py --task "..." --demo take-1 --replay take-1 --provider-preset claude-haiku   # no hardware

Each tick (no memory between ticks; the scene is the memory): observe -> prompt(task + demo + current images/state) -> one JSON
action -> validate/clamp -> execute -> repeat. Control files in evg-run/: STOP ends the run, PAUSE waits, APPROVE/REJECT answer --step-mode.
"""
import argparse, asyncio, json, shutil, signal, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as C
from agent import demo_pkg, providers, robot as robot_mod, safety

SYSTEM_HEAD = """You control a robot arm (7 joints + gripper) through a closed loop. Each turn you get: the TASK, a DEMONSTRATION of how a human did a similar job (with keyframe photos and joint positions), the CURRENT camera images (overview and gripper camera) and CURRENT joint positions. You have no memory of earlier turns except the short note about your previous plan, if one is shown: judge what is already done and what is left from the images.
"""

FORMAT_ONE = """Reply with ONLY one JSON object: {"thought": "<=30 words: what you see and why this action", "action": {...}}
Give exactly one action per turn."""

FORMAT_MANY = """Reply with ONLY one JSON object: {"thought": "<=30 words: what you see and why", "actions": [{...}, {...}]}
Give a short plan of at most %d actions, executed in order WITHOUT new camera images in between (joint positions are re-checked). Put only actions you are confident about; the plan stops at the first action that fails, hits a limit, or touches something, and the rest is discarded. Use one action when you need to look again (e.g. while aiming)."""

ACTIONS = """Actions:
  {"action":"step","joints":{"1":deg,"3":deg}}   small joint rotations in degrees, joints 1-7, each |deg| <= 6
  {"action":"goto","waypoint":"A_pregrasp"}       move all joints to a waypoint from the demonstration (interpolated in small steps)
  {"action":"grip","opening":0.0-1.0}             0 closed, 1 open
  {"action":"wait","seconds":0}                   look again without moving (the next turn shows fresh images)
  {"action":"done","reason":"..."}                only when the whole TASK is visibly complete (must be the only action in a turn)

Rules: move in small steps and check the next image; never push into the table or objects; if a result says a limit or contact stopped the move, choose a different, gentler action. Positions are servo steps (4096 per turn). Use the demonstration for waypoints, heights (the floor) and the grasp/release sequence, but adapt to where objects are NOW.
"""


def system_prompt(chunk, hints):
    fmt = FORMAT_ONE if chunk <= 1 else FORMAT_MANY % chunk
    return f"{SYSTEM_HEAD}\n{fmt}\n{ACTIONS}\n{hints}"


def short(act):
    return json.dumps(act, separators=(",", ":"))


def history_text(h):
    """Note about the previous tick's plan, placed AFTER the cache break so it never disturbs the cached prefix."""
    out = ["Your previous plan:"]
    for act, res in h["executed"]:
        out.append(f"  executed {short(act)} -> {res}")
    for act in h["pending"]:
        out.append(f"  NOT executed {short(act)} (stopped early or not reached; check it still makes sense)")
    return "\n".join(out)


def log(*a):
    print(C.now(), *a, flush=True)


def stop_requested():
    return C.STOP.exists()


async def wait_while_paused(status):
    while C.PAUSE.exists() and not stop_requested():
        status(phase="paused"); await asyncio.sleep(0.3)


def fmt_state(st, unit="mA"):
    rows = []
    for j in range(1, 8):
        m = st[j]; rows.append(f"j{j}: pos {m['pos']} (range {m['mn']}..{m['mx']}) {'current' if unit == 'mA' else unit} {m['cur']}{' mA' if unit == 'mA' else ''}")
    g = st[8]; rows.append(f"gripper: opening {(g['pos'] - g['mn']) / max(g['mx'] - g['mn'], 1):.2f} {'current' if unit == 'mA' else unit} {g['cur']}{' mA' if unit == 'mA' else ''}")
    return "\n".join(rows)


async def main(a):
    C.ensure_run(); C.apply_robot_env()
    names = [n for n in a.demo.split(",") if n]
    pkg = demo_pkg.build(names, C.DEMOS) if names else dict(text="(no demonstration)", images=[], waypoints={}, floor_j2=0, envelope={})
    prov = providers.make(dict(preset=a.provider_preset, provider=a.provider, model=a.model, base_url=a.base_url, api_key_env=a.api_key_env, effort=a.effort))
    hints = (Path(__file__).parent / "hints.md").read_text()
    system = system_prompt(a.chunk, hints)
    schema = safety.reply_schema(a.chunk, list(pkg["waypoints"]))
    from agent import mcp_client
    be = "replay" if a.replay else (a.backend if a.backend != "auto" else mcp_client.backend())
    rb = (robot_mod.ReplayRobot(C.DEMOS / a.replay) if a.replay else robot_mod.McpRobot(dry_run=a.dry_run) if be == "mcp"
          else robot_mod.Robot(dry_run=a.dry_run))
    ticks_dir = C.RUN / "ticks"; shutil.rmtree(ticks_dir, ignore_errors=True); ticks_dir.mkdir()
    st = dict(task=a.task[:300], provider=prov.name, backend=be, mode="replay" if a.replay else "dry-run" if a.dry_run else "live", demos=names, tick=0)

    def status(**kw):
        st.update(kw, ts=time.time()); C.write_json(C.STATUS, st)

    status(phase="connecting"); log(f"provider {prov.name}; robot {be}; mode {st['mode']}; demos {names}; {len(pkg['waypoints'])} waypoints, floor j2={pkg['floor_j2']}")
    await rb.connect()
    if isinstance(prov, providers.OpenAICompat) and "api.openai.com" not in prov.base:
        # local model: load it and fill its prefix cache now, so tick 1 is not a cold start
        t0 = time.time(); status(phase="warming up model")
        try: await asyncio.to_thread(prov.complete, system, [f"=== TASK ===\n{a.task}"] + demo_pkg.parts(pkg) + ["Reply with OK."], None)
        except providers.ProviderError as e: log(f"warm-up failed: {e}")
        log(f"model warm-up {time.time() - t0:.1f}s")
    margin = rb.mod.MARGIN
    signal.signal(signal.SIGTERM, lambda *_: C.STOP.touch())
    warned = False
    t_start, last, obs_fail = time.time(), None, 0
    try:
        for tick in range(1, a.max_ticks + 1):
            if stop_requested(): log("STOP requested"); break
            if time.time() - t_start > a.max_seconds: log("time limit reached"); break
            await wait_while_paused(status)
            if stop_requested(): break
            status(phase="observing", tick=tick)
            try:
                obs = await rb.observe(); s = obs["state"]; obs_fail = 0
            except robot_mod.RobotError as e:
                obs_fail += 1
                log(f"observe failed ({obs_fail}/3): {e}"); status(phase="error", error=str(e))
                if obs_fail >= 3: break
                continue
            td = ticks_dir / f"{tick:04d}"; td.mkdir()
            for k in ("over", "grip"):
                if obs[k]:
                    (td / f"{k}.jpg").write_bytes(obs[k]); (C.RUN / f"latest_{k}.jpg").write_bytes(obs[k])
            if not obs["over"] and not obs["grip"]:
                log("no camera frames yet"); status(phase="no_frames"); await asyncio.sleep(0.1); continue
            # stable prefix (system + task + demo) first, CACHE_BREAK, then everything that changes per tick
            prefix = [f"=== TASK ===\n{a.task}"] + demo_pkg.parts(pkg)
            parts = prefix + [providers.CACHE_BREAK, "=== CURRENT STATE (tick %d) ===\n%s" % (tick, fmt_state(s, getattr(rb, "cur_unit", "mA")))]
            if last and (a.last_result or a.chunk > 1): parts.append(history_text(last))
            for k, label in (("over", "Overview camera now:"), ("grip", "Gripper camera now:")):
                parts += [label, obs[k]] if obs[k] else [f"{label} (no frame available)"]
            parts.append("Reply with the JSON object only.")
            if not warned:
                warned = True
                est = sum(len(x) / 3.5 if isinstance(x, str) else 90 for x in prefix) + len(system) / 3.5
                log(f"stable prefix ~{est:.0f} tokens (cached across ticks)")
                if "haiku" in prov.name and prov.name.startswith("anthropic") and est < 4096:
                    log("NOTE: Haiku 4.5 only caches prefixes of 4096+ tokens; this one is shorter, so it will not be cached (use Sonnet, or add more demo frames)")
            status(phase="thinking")
            action, raw, err, acts = None, "", None, []
            for attempt in range(2):
                try:
                    raw = await asyncio.to_thread(prov.complete, system, parts, schema)
                    action = providers.extract_json(raw)
                    acts = action["actions"] if isinstance(action.get("actions"), list) else [action.get("action")]
                    acts = acts[:max(a.chunk, 1)]
                    if not acts: raise ValueError("no action given")
                    safety.plan(acts[0], s, pkg, margin)          # the first action must be valid now; the rest are checked when reached
                    err = None; break
                except (ValueError, safety.Invalid) as e:
                    err = f"invalid reply: {e}"; parts.append(f"Your previous reply was rejected: {e}. Reply with valid JSON only.")
                except providers.ProviderError as e:
                    err = f"provider error: {e}"; await asyncio.sleep(0.3)
            (td / "reply.json").write_text(json.dumps(dict(raw=raw, error=err, usage=prov.usage), indent=1))
            u = prov.usage
            log(f"    tokens: new {u.get('input', '?')}, cache read {u.get('cache_read', 0)}, cache write {u.get('cache_write', 0)}, out {u.get('output', '?')}, {u.get('ms', '?')} ms")
            if err:
                log(err); status(phase="error", error=err); await asyncio.sleep(0.2)
                if tick > 2 and err.startswith("provider"): break
                continue
            thought = (action.get("thought") or "")[:200]
            log(f"[{tick}] {thought} -> {' ; '.join(short(x) for x in acts)}")
            status(phase="planned", action=acts[0] if len(acts) == 1 else acts, thought=thought, error=None, usage=u)
            if a.step_mode and any(isinstance(x, dict) and x.get("action") not in ("wait",) for x in acts):
                C.APPROVE.unlink(missing_ok=True); C.REJECT.unlink(missing_ok=True)
                status(phase="awaiting_approval", pending=acts[0] if len(acts) == 1 else acts)
                while not (C.APPROVE.exists() or C.REJECT.exists() or stop_requested()): await asyncio.sleep(0.2)
                if stop_requested(): break
                if C.REJECT.exists():
                    C.REJECT.unlink(); last = dict(executed=[], pending=[]); last["executed"].append((acts[0], "rejected by the human operator")); last["pending"] = acts[1:]
                    log("rejected"); continue
                C.APPROVE.unlink(missing_ok=True); status(pending=None)
            status(phase="executing")
            executed, pending, finished, s_cur, lost = [], [], False, s, False
            for i, act in enumerate(acts):
                if stop_requested(): pending = acts[i:]; break
                try:
                    kind, payload, notes = safety.plan(act, s_cur, pkg, margin)     # re-validated against the latest joint state
                except safety.Invalid as e:
                    executed.append((act, f"REJECTED by safety layer: {e}")); pending = acts[i + 1:]; break
                try:
                    if kind == "done":
                        log(f"DONE: {payload['reason']}"); status(phase="done", result=payload["reason"]); finished = True; break
                    elif kind == "wait":
                        res = "looked again (no movement)"                    # realtime: no sleeping, the next tick re-observes
                    elif kind == "step":
                        _, res = await rb.drive(payload["targets"], stop_requested)
                    elif kind == "goto":
                        res = await rb.goto_path(payload["targets"], s_cur, stop_requested)
                    elif kind == "grip":
                        res = await rb.grip(payload["opening"], stop_requested)
                    if a.replay: rb.advance()
                except robot_mod.RobotError as e:
                    res = f"FAILED: {e}"; log(res); lost = lost or e.fatal
                except Exception as e:      # anything unexpected mid-move: we cannot trust what the arm is doing -> stop the run
                    res = f"FAILED: unexpected {type(e).__name__}: {e}"; log(res); lost = True
                res += f" ({'; '.join(notes)})" if notes else ""
                executed.append((act, res)); log(f"    {short(act)} -> {res}")
                pending = acts[i + 1:]
                if lost: break
                if res.startswith("FAILED") or "did not fully arrive" in res or notes:
                    break                                  # something unexpected: drop the rest of the plan and look again
                if pending:
                    try: s_cur = (await rb.observe())["state"]
                    except robot_mod.RobotError: break
            if finished: break
            if lost:
                msg = "LOST CONTACT with the arm during a move. Run stopped. The servos keep holding their last goal; check the arm before resuming."
                log(msg); status(phase="error", error=msg, result=msg); break
            last = dict(executed=executed, pending=pending)
            if pending: log(f"    not executed: {' ; '.join(short(x) for x in pending)}")
            status(phase="idle", result="; ".join(f"{short(x)} -> {r}" for x, r in executed))
            (td / "action.json").write_text(json.dumps(dict(plan=acts, executed=[[x, r] for x, r in executed], pending=pending), indent=1))
        else:
            log("max ticks reached")
    finally:
        if not (a.dry_run or a.replay):
            try: await asyncio.wait_for(rb.hold(), 8); log("arm holding position")
            except Exception as e: log(f"WARNING could not hold arm ({type(e).__name__}: {e}); the servos keep their last goal")
        if hasattr(prov, "close"): prov.close()
        status(phase="stopped")
        C.PID.unlink(missing_ok=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True); ap.add_argument("--demo", default="")
    ap.add_argument("--provider-preset", default=None, help="claude-haiku|claude-sonnet|api-haiku|api-sonnet|openai|ollama|lmstudio")
    ap.add_argument("--provider"); ap.add_argument("--model"); ap.add_argument("--base-url"); ap.add_argument("--api-key-env")
    ap.add_argument("--backend", default="auto", choices=["auto", "station", "mcp"], help="auto = robot.json 'backend' (or EVG_BACKEND)")
    ap.add_argument("--dry-run", action="store_true", help="real observations, never moves")
    ap.add_argument("--replay", metavar="DEMO", help="use a recorded demo as the fake robot (no hardware)")
    ap.add_argument("--step-mode", action="store_true", help="every action needs APPROVE from the operator")
    ap.add_argument("--last-result", action="store_true", help="tell the model the result of its previous action (off = fully stateless)")
    ap.add_argument("--effort", default="low", help="thinking effort for Claude models: low (fast, default) | medium | high | xhigh | max; '' = model default")
    ap.add_argument("--chunk", type=int, default=1, help="max actions per model call (>1 = plan several, executed without new images)")
    ap.add_argument("--max-ticks", type=int, default=80); ap.add_argument("--max-seconds", type=int, default=900)
    args = ap.parse_args()
    if not (args.provider_preset or args.provider):
        args.provider_preset = "gemma-e4b"
    try:
        asyncio.run(main(args))
    except KeyboardInterrupt:
        pass
