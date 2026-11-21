"""Run the real-arm operator with ANY OpenAI-compatible chat endpoint (Gemma on Ollama / llama.cpp / LM Studio /
vLLM, also behind a Cloudflare tunnel) instead of the Claude CLI. Same robot tools (the elrobot-real MCP server),
same instructions (OPERATOR_REAL.md).

  export MODEL_BASE_URL=https://<name>.trycloudflare.com/v1     # the /v1 root of the endpoint
  export MODEL_NAME=gemma3:12b                                  # as the server names it
  export MODEL_API_KEY=...                                      # only if the endpoint wants one
  bash sim/run_agent_model.sh --check                           # reachability, model list, a vision + JSON test
  bash sim/run_agent_model.sh -p "Sort the items ..."           # run the operator

Tool calling: Gemma has no reliable native tool calls, so the default protocol is JSON in plain text: the model
replies with one JSON object per turn, {"thought": .., "tool": .., "args": {..}} or {"thought": .., "done": ".."}.
`--tool-mode native` uses the endpoint's own tools API instead (for models that support it).
Images from tool results are sent as image_url data URIs; only the newest --keep-images stay in the context
(older ones are replaced by a note), because small models have small context windows.

The transcript goes to sim/real_sessions/model_run_<time>.jsonl in the same line format the operator page reads.
Needs the `mcp` package (run_agent_model.sh uses uv), Station and the camera relay, like run_agent_real.sh.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
SESS = HERE / "real_sessions"

PROTOCOL = """
## How you act (strict)
You act by replying with EXACTLY ONE JSON object and nothing else: no prose before or after, no code fences.
- To use a tool:   {"thought": "<one short sentence>", "tool": "<tool name>", "args": {<arguments>}}
- When finished:   {"thought": "<one short sentence>", "done": "DONE: what went where"}   or   "done": "BLOCKED: why"
One tool per reply. After each tool you receive its result text and, for camera tools, the images. Use only the
tools listed below, with exactly these argument names. Numbers are plain JSON numbers (metres, pixels).

## Tools
%s
"""


def post(url: str, payload: dict, key: str, timeout: float) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST", headers={
        "Content-Type": "application/json", "User-Agent": "elrobot-operator/1.0",
        **({"Authorization": f"Bearer {key}"} if key else {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code} from the model endpoint: {e.read().decode('utf-8', 'replace')[:400]}") from None
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise RuntimeError(f"cannot reach the model endpoint ({url}): {e}") from None


def get(url: str, key: str, timeout: float = 20) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "elrobot-operator/1.0", **({"Authorization": f"Bearer {key}"} if key else {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def extract_json(text: str):
    """First balanced {...} object in the reply (models wrap JSON in prose or code fences)."""
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            c = text[i]
            if in_str:
                esc = (c == "\\") and not esc
                if c == '"' and not esc:
                    in_str = False
                continue
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except ValueError:
                        break
        start = text.find("{", start + 1)
    return None


def tool_doc(tools) -> str:
    out = []
    for t in tools:
        props = (t.inputSchema or {}).get("properties", {})
        req = set((t.inputSchema or {}).get("required", []))
        args = ", ".join(f"{k}: {v.get('type', 'any')}{'' if k in req else ' (optional)'}" for k, v in props.items()) or "no arguments"
        out.append(f"- {t.name}({args})\n  {' '.join((t.description or '').split())}")
    return "\n".join(out)


def trim_images(messages: list, keep: int) -> None:
    """Keep only the newest `keep` images in the context."""
    seen = 0
    for m in reversed(messages):
        if isinstance(m.get("content"), list):
            for part in reversed(m["content"]):
                if part.get("type") == "image_url":
                    seen += 1
                    if seen > keep:
                        part.clear()
                        part.update(type="text", text="[an older camera image was removed to save context]")


class Log:
    def __init__(self, path: Path):
        self.f = open(path, "w")
        self.path = path

    def write(self, obj: dict) -> None:
        self.f.write(json.dumps(obj) + "\n")
        self.f.flush()

    def assistant(self, text: str) -> None:
        self.write({"type": "assistant", "message": {"content": [{"type": "text", "text": text}]}})


async def run(a) -> int:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    SESS.mkdir(exist_ok=True)
    log = Log(Path(a.log) if a.log else SESS / f"model_run_{time.strftime('%Y%m%d-%H%M%S')}.jsonl")
    url, t0, turns, usage = a.base_url.rstrip("/") + "/chat/completions", time.time(), 0, {"in": 0, "out": 0}
    params = StdioServerParameters(command="bash", args=[str(HERE / "run_real_server.sh")], cwd=str(HERE.parent), env=dict(os.environ))
    final = "BLOCKED: the run ended without a report"
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as mcp:
            await mcp.initialize()
            tools = (await mcp.list_tools()).tools
            names = {t.name for t in tools}
            system = (HERE / "OPERATOR_REAL.md").read_text()
            if a.tool_mode == "json":
                system += PROTOCOL % tool_doc(tools)
            messages = [{"role": "system", "content": system},
                        {"role": "user", "content": "Task for this run (it replaces the default task): " + a.prompt}]
            log.write({"type": "system", "subtype": "init", "model": a.model, "base_url": a.base_url, "tool_mode": a.tool_mode})
            native = [{"type": "function", "function": {"name": t.name, "description": t.description or "", "parameters": t.inputSchema or {"type": "object"}}}
                      for t in tools] if a.tool_mode == "native" else None
            bad = 0
            while turns < a.max_turns:
                turns += 1
                trim_images(messages, a.keep_images)
                payload = {"model": a.model, "messages": messages, "temperature": a.temperature, "max_tokens": a.max_tokens}
                if native:
                    payload["tools"] = native
                t = time.time()
                try:
                    resp = await asyncio.to_thread(post, url, payload, a.api_key, a.timeout)
                except RuntimeError as e:
                    final = f"BLOCKED: {e}"
                    break
                ms = round((time.time() - t) * 1000)
                u = resp.get("usage") or {}
                usage["in"] += u.get("prompt_tokens", 0); usage["out"] += u.get("completion_tokens", 0)
                msg = (resp.get("choices") or [{}])[0].get("message") or {}
                text = msg.get("content") or ""
                call = None
                if native and msg.get("tool_calls"):
                    tc = msg["tool_calls"][0]
                    try:
                        call = {"tool": tc["function"]["name"], "args": json.loads(tc["function"].get("arguments") or "{}"), "id": tc.get("id")}
                    except ValueError:
                        call = None
                    messages.append({"role": "assistant", "content": text or None, "tool_calls": [tc]})
                else:
                    obj = extract_json(text) if text else None
                    messages.append({"role": "assistant", "content": text})
                    if isinstance(obj, dict) and obj.get("done"):
                        final = str(obj["done"])
                        log.assistant(f"{obj.get('thought', '')} -> {final}")
                        print(f"[{turns}] {ms} ms  done: {final}", flush=True)
                        break
                    if isinstance(obj, dict) and obj.get("tool") in names:
                        call = {"tool": obj["tool"], "args": obj.get("args") if isinstance(obj.get("args"), dict) else {}, "thought": obj.get("thought", "")}
                if call is None:
                    if native and text and not msg.get("tool_calls"):       # a native-mode model that just talks = its report
                        final = text.strip()
                        log.assistant(final)
                        break
                    bad += 1
                    print(f"[{turns}] {ms} ms  invalid reply ({bad}/3): {text[:120]!r}", flush=True)
                    log.write({"type": "invalid", "text": text[:2000]})
                    if bad >= 3:
                        final = "BLOCKED: the model did not produce a valid action three times in a row"
                        break
                    messages.append({"role": "user", "content": "That was not a valid action. Reply with exactly one JSON object: "
                                     '{"thought": "...", "tool": "<one of: ' + ", ".join(sorted(names)) + '>", "args": {...}} or {"thought": "...", "done": "..."}'})
                    continue
                bad = 0
                log.assistant(f"{call.get('thought', '')} -> {call['tool']}({json.dumps(call['args'])})".strip())
                try:
                    res = await mcp.call_tool(call["tool"], call["args"])
                    rtext = "\n".join(c.text for c in res.content if c.type == "text")
                    images = [c.data for c in res.content if c.type == "image"]
                    if res.isError:
                        rtext = "TOOL ERROR: " + rtext
                except Exception as e:
                    rtext, images = f"TOOL ERROR: {type(e).__name__}: {e}", []
                print(f"[{turns}] {ms} ms  {call['tool']}({json.dumps(call['args'])[:80]}) -> {rtext.splitlines()[0][:90] if rtext else ''}"
                      f"{f' (+{len(images)} images)' if images else ''}", flush=True)
                log.write({"type": "tool", "tool": call["tool"], "args": call["args"], "model_ms": ms, "result": rtext[:3000], "images": len(images)})
                content = [{"type": "text", "text": f"Result of {call['tool']}:\n{rtext}"}]
                content += [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + d}} for d in images]
                if native and call.get("id"):
                    messages.append({"role": "tool", "tool_call_id": call["id"], "content": rtext})
                    if images:
                        messages.append({"role": "user", "content": [{"type": "text", "text": "Images from that tool:"}] + content[1:]})
                else:
                    messages.append({"role": "user", "content": content if images else content[0]["text"]})
            else:
                final = f"BLOCKED: reached the limit of {a.max_turns} turns"
    log.write({"type": "result", "result": final, "duration_ms": round((time.time() - t0) * 1000), "num_turns": turns,
               "total_cost_usd": 0, "usage": usage})
    print(f"\n{final}\n({turns} turns, {time.time() - t0:.0f} s, tokens in {usage['in']} / out {usage['out']}; transcript {log.path})", flush=True)
    return 0


def check(a) -> int:
    base = a.base_url.rstrip("/")
    print(f"endpoint {base}   model {a.model}")
    try:
        models = [m.get("id") for m in get(base + "/models", a.api_key).get("data", [])]
        print(f"  reachable; models: {models[:12]}")
        if models and a.model not in models:
            print(f"  WARNING: '{a.model}' is not in that list: set MODEL_NAME to one of them")
    except Exception as e:
        print(f"  /models failed: {e}")
    # 8x8 red JPEG: a vision model answers 'red'
    red = ("/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/"
           "2wBDAQkJCQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wAARCAAIAAgDASIAAhEBAxEB/8QAFQAB"
           "AQAAAAAAAAAAAAAAAAAAAAf/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAABgf/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIR"
           "AxEAPwCdAAYqf//Z")
    payload = {"model": a.model, "temperature": 0, "max_tokens": 80, "messages": [
        {"role": "system", "content": 'Reply with exactly one JSON object and nothing else: {"colour": "<the main colour of the image>"}'},
        {"role": "user", "content": [{"type": "text", "text": "What colour is this image?"},
                                     {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + red}}]}]}
    try:
        t = time.time()
        r = post(base + "/chat/completions", payload, a.api_key, a.timeout)
        text = r["choices"][0]["message"].get("content") or ""
        obj = extract_json(text)
        print(f"  chat with an image: {round((time.time() - t) * 1000)} ms; reply {text[:100]!r}")
        print(f"  valid JSON: {'yes' if isinstance(obj, dict) else 'NO'};  saw the image: "
              f"{'yes' if isinstance(obj, dict) and 'red' in str(obj.get('colour', '')).lower() else 'NOT CONFIRMED (the model may not accept images)'}")
        return 0
    except Exception as e:
        print(f"  chat failed: {e}")
        return 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-p", "--prompt", help="the task")
    ap.add_argument("--check", action="store_true", help="test the endpoint and exit (does not touch the arm)")
    ap.add_argument("--base-url", default=os.environ.get("MODEL_BASE_URL", ""))
    ap.add_argument("--model", default=os.environ.get("MODEL_NAME", "gemma3:12b"))
    ap.add_argument("--api-key", default=os.environ.get("MODEL_API_KEY", ""))
    ap.add_argument("--tool-mode", choices=["json", "native"], default=os.environ.get("MODEL_TOOL_MODE", "json"))
    ap.add_argument("--max-turns", type=int, default=70)
    ap.add_argument("--keep-images", type=int, default=int(os.environ.get("MODEL_KEEP_IMAGES", "4")))
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--timeout", type=float, default=180)
    ap.add_argument("--log", help="transcript file (default sim/real_sessions/model_run_<time>.jsonl)")
    a = ap.parse_args()
    if not a.base_url:
        sys.exit("set MODEL_BASE_URL (or --base-url) to the endpoint's /v1 root, e.g. https://<name>.trycloudflare.com/v1")
    if a.check:
        sys.exit(check(a))
    if not a.prompt:
        sys.exit("give the task with -p")
    sys.exit(asyncio.run(run(a)))


if __name__ == "__main__":
    main()
