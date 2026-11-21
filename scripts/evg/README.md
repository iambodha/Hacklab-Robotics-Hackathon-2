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

## Local models (Ollama)

Ollama serves an OpenAI-compatible API on `http://localhost:11434/v1` (the Ollama app starts it; otherwise `ollama serve`).
Pick `gemma-e4b`, `gemma-12b` or `gemma-31b` in the model list (or `--provider-preset gemma-12b`). No key needed.

What makes local models usable here (all automatic for non-OpenAI endpoints):
- **thinking off**: Gemma 4 thinks by default with image prompts (200-850 hidden tokens per tick). `effort: low` sends `reasoning_effort: "none"`.
- **schema-constrained output**: the reply must match the action schema (no joint 8, only real waypoints) and generation stops when the
  JSON object is complete. Without it 12B repeated its answer until the token limit and e4b produced invalid actions.
- **prefix cache**: Ollama reuses the KV cache of the unchanged prefix (system + task + demo); a changed prefix costs 2-3x.

Benchmark on the real agent prompt (M3 Max 64 GB, one model at a time, 100% GPU, MLX runner; `scripts/evg/bench.py`):

| model | warm tick (median) | prefix changed | cold (load) | valid actions |
|---|---|---|---|---|
| gemma4:e4b-mlx | 0.8 s | 1.8 s | 3.2 s | 6/6 |
| gemma4:12b-mlx | 1.7 s | 5.1 s | 6.0 s | 6/6 |
| gemma4:31b-mlx | 4.3 s | 13.8 s | 16.4 s | 6/6 |
| claude-haiku (subscription) | 2.3 s | 3.7 s | 4.1 s | 6/6 |
| claude-sonnet (subscription) | 2.6 s | 4.4 s | 3.9 s | 6/6 |

"Valid" means the action passes the safety layer; it says nothing about whether it is the *right* action. Compare that on dry runs.
The first tick after loading is slower (the cache is still being filled).

## Realtime motion (Station backend)

- One coordinated command per move: every joint gets its own speed so all arrive together (no stopping at sub-steps).
- The arm is checked 20x per second: contact current over `CUR_LIMIT` stops and holds; STOP is honoured within ~0.05 s.
- A joint that stopped short (P-only servos) gets its goal nudged once it has stalled for 0.15 s.
- The gripper returns as soon as it arrives or stalls on an object (a grip), instead of a fixed wait.
- No sleeps in the loop: `wait` means "look again"; error back-offs are 0.2-0.3 s.
- Speed limit is the project's `SPEED, ACCEL = 350, 15` in `scripts/arm.py` (about 31 deg/s, "deliberately slow"). Raise it only with
  the area clear.
- Default model: `gemma-e4b` (Ollama, MLX). It is loaded and its prefix cache filled before tick 1; ~0.8 s per decision after that.

## Robot plugged into this laptop

`robot.json` has `"station_host": "localhost"`. Camera ids depend on the USB port, so after plugging in run
`python3 scripts/evg/ctl.py cameras` (lists the cameras Station has seen and saves `frames/cam-<id>.jpg` for each) and, if needed,
`python3 scripts/evg/ctl.py cameras --overview <id> --gripper <id>` (a prefix is enough).
