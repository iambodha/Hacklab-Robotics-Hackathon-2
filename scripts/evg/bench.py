"""Latency / validity benchmark of models on the agent's REAL prompt (system + task + demo with keyframes + current frames + state).

  python3 scripts/evg/bench.py --demo take-20261121-151415 \
      --models ollama:gemma4:e4b-mlx,ollama:gemma4:12b-mlx,ollama:gemma4:31b-mlx,claude-haiku,claude-sonnet --n 4

Per model: cold call (includes loading), N warm calls (same prefix, new frames = a normal run), one call with the prefix changed
(no prefix cache), and whether each reply is valid JSON and a valid action for the safety layer. Nothing is sent to the arm.
Ollama models are run one at a time (others unloaded first) so they don't compete for memory.
"""
import argparse, asyncio, json, statistics, sys, time, urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
from agent import demo_pkg, providers, robot as robot_mod, safety
from agent.run import fmt_state, system_prompt

OLLAMA = "http://localhost:11434"


def ollama_unload_all():
    try:
        ps = json.loads(urllib.request.urlopen(f"{OLLAMA}/api/ps", timeout=5).read())
        for m in ps.get("models", []):
            body = json.dumps({"model": m["name"], "keep_alive": 0}).encode()
            urllib.request.urlopen(urllib.request.Request(f"{OLLAMA}/api/generate", body, {"content-type": "application/json"}), timeout=60).read()
    except Exception as e:
        print(f"  (could not unload Ollama models: {e})")


def make(spec, effort):
    if spec.startswith("ollama:"):
        return providers.make(dict(provider="openai", model=spec[7:], base_url=f"{OLLAMA}/v1", api_key_env="NO_KEY_NEEDED", effort=effort))
    return providers.make(dict(preset=spec, effort=effort))


async def observations(demo, k):
    rb = robot_mod.ReplayRobot(C.DEMOS / demo)
    out = []
    step = max(1, (len(rb.rows) - 1) // max(k, 1))
    for i in range(k):
        rb.i = min(i * step, len(rb.rows) - 1)
        out.append(await rb.observe())
    return out, rb.mod.MARGIN


def build_parts(task, pkg, obs, tick, nonce=""):
    parts = [f"=== TASK ===\n{nonce}{task}"] + demo_pkg.parts(pkg)
    parts += [providers.CACHE_BREAK, "=== CURRENT STATE (tick %d) ===\n%s" % (tick, fmt_state(obs["state"]))]
    for k, label in (("over", "Overview camera now:"), ("grip", "Gripper camera now:")):
        parts += [label, obs[k]] if obs[k] else [f"{label} (no frame available)"]
    parts.append("Reply with the JSON object only.")
    return parts


def judge(raw, st, pkg, margin):
    try:
        a = providers.extract_json(raw)
    except ValueError:
        return False, False, "no JSON"
    act = a["actions"][0] if isinstance(a.get("actions"), list) and a["actions"] else a.get("action")
    try:
        safety.plan(act, st, pkg, margin)
        return True, True, json.dumps(act, separators=(",", ":"))[:70]
    except safety.Invalid as e:
        return True, False, f"invalid: {e}"[:70]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demo", required=True)
    ap.add_argument("--models", required=True, help="comma list: presets (claude-haiku, ...) or ollama:<model>")
    ap.add_argument("--n", type=int, default=4, help="warm calls per model")
    ap.add_argument("--task", default=None)
    ap.add_argument("--effort", default="low")
    ap.add_argument("--no-schema", action="store_true", help="do not constrain local models to the reply schema")
    a = ap.parse_args()
    task = a.task or (C.TASK.read_text() if C.TASK.exists() else "Sort the desk: garbage into the black box, parts into the white box.")
    pkg = demo_pkg.build([a.demo], C.DEMOS)
    obs, margin = asyncio.run(observations(a.demo, a.n + 2))
    system = system_prompt(1, (Path(__file__).parent / "agent" / "hints.md").read_text())
    schema = safety.reply_schema(1, list(pkg["waypoints"]))
    rows = []
    for spec in a.models.split(","):
        spec = spec.strip()
        print(f"\n=== {spec}")
        if spec.startswith("ollama:"):
            ollama_unload_all()
        prov = make(spec, a.effort)
        r = dict(model=spec, warm=[], ok_json=0, ok_act=0, calls=0, samples=[])

        def call(o, tick, nonce=""):
            t = time.time()
            try:
                raw = prov.complete(system, build_parts(task, pkg, o, tick, nonce), None if a.no_schema else schema)
            except providers.ProviderError as e:
                print(f"  ERROR {e}"); return None
            dt = time.time() - t
            j, v, desc = judge(raw, o["state"], pkg, margin)
            r["calls"] += 1; r["ok_json"] += j; r["ok_act"] += v
            u = prov.usage
            r["in"], r["out"] = u.get("input", 0) + (u.get("cache_read") or 0) + (u.get("cache_write") or 0), u.get("output")
            print(f"  {dt:6.2f}s  in {r['in']} out {u.get('output')}  {desc}")
            r["samples"].append(desc)
            return dt

        print("  cold (includes loading the model):"); r["cold"] = call(obs[0], 1)
        print(f"  warm x{a.n} (same demo prefix, new frames):")
        for i in range(a.n):
            dt = call(obs[i + 1], i + 2)
            if dt is not None: r["warm"].append(dt)
        print("  prefix changed (no prefix cache):"); r["nocache"] = call(obs[-1], 99, nonce=f"[run {time.time()}]\n")
        if hasattr(prov, "close"): prov.close()
        rows.append(r)
    if any(s.startswith("ollama:") for s in a.models.split(",")):
        ollama_unload_all()
    print("\n| model | cold | warm median (min-max) | prefix changed | prompt tok | out tok | valid JSON | valid action |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        w = r["warm"]
        warm = f"{statistics.median(w):.2f}s ({min(w):.2f}-{max(w):.2f})" if w else "-"
        f = lambda x: f"{x:.2f}s" if x else "-"
        print(f"| {r['model']} | {f(r.get('cold'))} | {warm} | {f(r.get('nocache'))} | {r.get('in', '-')} | {r.get('out', '-')} | {r['ok_json']}/{r['calls']} | {r['ok_act']}/{r['calls']} |")


if __name__ == "__main__":
    main()
