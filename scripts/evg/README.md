# evg: closed-loop VLM agent for the arm

    python3 scripts/evg/evg.py          # console flow: task -> demo -> model -> run
    python3 scripts/evg/evg.py --web    # same flow in the browser (http://127.0.0.1:8891)
    python3 scripts/evg/ctl.py status|stop|pause|resume|demos   # deterministic control from any shell / chat

Flow: describe the task (text, or `/path/to/file.txt`) -> pick existing demos or record (SPACE start/stop, ENTER keep,
SPACE re-record, A keep + another, D discard) -> choose model -> ENTER start. SPACE/X stops, P pauses, Y/N approve.

Models: `claude-haiku` / `claude-sonnet` use your Claude subscription through the local `claude` CLI (no key).
`api-*` need ANTHROPIC_API_KEY. `openai`, `ollama`, `lmstudio` are OpenAI-compatible (OPENAI_API_KEY or the env var you name).

Test without the arm: `R` in the console (or "Replay test" in the web UI) plays a recorded demo as a fake robot and only logs actions.
`D` / "Dry run" reads the real cameras and arm state but never moves.

Layout: `agent/run.py` loop · `agent/providers.py` · `agent/demo_pkg.py` demo -> prompt · `agent/safety.py` clamps ·
`agent/robot.py` Station access + replay · `agent/hints.md` arm facts given to the model (edit it) · `demo_filter.py` drops pauses ·
`session.py` shared logic · runtime files in `evg-run/`.

Safety: model output is validated in `agent/safety.py` (6 deg per step, calibrated range, demo envelope, j2 floor); contact current
aborts moves and holds position; STOP file / SIGTERM / SIGKILL ladder in `ctl.py stop`; web UI is loopback-only with a per-launch token.

## Speed and caching

Measured on this machine (subscription, replay demo, ~1.3k-token prefix; your numbers will differ with real demos):

| Setup | per tick |
|---|---|
| Sonnet 5.5, default effort | 12-24 s (mostly thinking tokens) |
| **Sonnet 5.5, `--effort low`, cached prefix** | **~2-3 s** (3 new tokens billed, ~1.6k read from cache) |
| Haiku 4.5, thinking off (`--effort low`) | ~1.4 s, not cached (prefix is under Haiku's 4096-token minimum) |

- Every tick sends `system + task + demo` first (stable), then a cache break, then the changing part (joint state, two frames, last plan).
- **Subscription (`claude-*`)**: the CLI only caches whole identical prompts, so the provider *primes* a stored session once with the prefix and
  *forks* it every tick (`--resume --fork-session`). Forks are discarded, so nothing accumulates; session files are deleted when the run ends.
- **Anthropic API (`api-*`)**: `cache_control` on the end of the prefix. Minimums: Sonnet 5.5 512 tokens, Haiku 4.5 4096. TTL 5 min (each hit refreshes it).
- **OpenAI-compatible**: prefix caching is automatic on OpenAI; llama.cpp gets `cache_prompt: true` (KV reuse of the common prefix). Ollama/vLLM reuse a common prefix if their prefix cache is on.
- The log prints `tokens: new / cache read / cache write` for every tick, so you can see whether caching works. If `cache read` stays 0, something in the prefix changes between ticks.
- `--effort` (low default) and `--chunk N` (plan up to N actions per call; executed without new images, each re-validated against fresh joint
  state, stops at the first failure/limit/contact; the next prompt gets "executed / not executed" so the model sees what was left of its plan).
