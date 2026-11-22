# models: Gemini Robotics ER advisor, benchmark, and the data/policy pipeline

The higher-level model side of the robot. Nothing in here moves the arm: the executor is the only motion
owner. `models` reads camera frames and recordings, asks Gemini Robotics ER, and returns `ActionProposal`s
and `Outcome`s (records from `executor/contracts.py`; `models/_contracts.py` is a copy until merge).

| File | What |
|---|---|
| `gemini_er.py` | Client: `GeminiER().point / boxes / judge / propose`, prompt templates, parsers for both box formats, per-call log, cost. Transports: `GenAITransport` (google-genai, v1beta, optional 429 back-off), `FakeTransport` (tests). |
| `advisor_gemini.py` | `GeminiAdvisor`: Observation + goal → `ActionProposal`s from a fixed vocabulary; `judge()` / `residual()` → `Outcome`. Pixels → world metres via `perception.locate`. |
| `perception_bridge.py` | Read-only import of `perception.locate` (path `$PERCEPTION_ROOT`, default `ws/perception`). |
| `claude_cli.py` | Claude Code CLI (`claude -p`) as a transport with the same prompts and parsers, for the benchmark. |
| `_secrets.py` | Key loading (`$GEMINI_API_KEY`, `$GOOGLE_API_KEY`, `secrets.env`) and redaction. The key is never logged. |
| `autolabel.py` | Recorder sessions → sub-task segments, success, task string, keep/drop (a sensor pass, plus an optional Gemini keyframe pass). |
| `lerobot_export.py` | Recorder sessions → LeRobot v3.0 dataset, no lerobot or torch install needed. |
| `dataset_audit.py`, `AUDIT.md` | Whether the recordings are good enough for ACT / SmolVLA (rates, sync, coverage). |
| `POLICY_PLAN.md` | ACT vs SmolVLA plan and admission gate. |
| `RESEARCH.md` | Which Gemini Robotics models exist and what this arm can use. |
| `bench/` | 29-item labelled set (`build.py`), runner (`run.py`), scoring (`score.py`), live blue-cap/bin items (`live.py`), smoke test (`smoke.py`), **results in `bench/REPORT.md`**. |

## Gemini setup

- Model `gemini-robotics-er-2-preview` (env `GEMINI_ER_MODEL`). ER 1.5 and 1.6 are shut down. The streaming
  variant only works on the Live API.
- SDK `google-genai==3.3.0`. It must use API version **v1beta**: on v1 the preview models answer
  `404 NOT_FOUND`. Override with `GEMINI_API_VERSION`.
- Price: $1 per M input tokens and $5 per M output tokens (thinking billed as output), until 31 Dec 2026.
  That's about $0.005 per call with one 299x224 frame and `thinking_level=low`.
- **Quota:** the current key is on the Free Tier, with **5 requests/min and 20 requests/day**. Enable
  billing on the AI Studio project before a real run. `GenAITransport(wait_429_s=N)` rides out the
  per-minute limit. The live advisor keeps the default of 0, so it fails at once and the executor carries on
  without advice.
- Smoke test: `uv run --no-project --with google-genai==3.3.0 python models/bench/smoke.py [--list] [--fallbacks]`.

## Usage

```python
from models.gemini_er import GeminiER, Image
er = GeminiER()                                     # key from env / secrets.env
pts = er.point(Image.from_path("frame.jpg", role="overview"), ["blue cap", "bin"])   # pixels in the image
bxs = er.boxes(Image.from_path("frame.jpg", role="overview"), ["bin"])
j = er.judge([wrist_img, overview_img], "put the blue cap in the bin")               # done / failure / residual

from models.advisor_gemini import GeminiAdvisor, to_api
adv = GeminiAdvisor()
props = adv.propose(obs, "take the blue object and put it in the bin")               # [ActionProposal]
```

Benchmark: `python -m models.bench.run --advisor gemini|claude|cv|mock [--retry-errors] [--wait-429 S]` and
`python -m models.bench.live [--no-gemini] [--grab N]`. Run them with
`uv run --no-project --with google-genai==3.3.0 --with opencv-contrib-python --with numpy --with scipy --with pillow`.

## NormaCore integration

```
NormaCore Station (TCP :8888)  ── motors + cameras ──>  recorder (:8900)  ── recordings on DATA_ROOT
        ^                                                   │ GET /live, /live/frame?cam=overview|wrist
        │ commands (only the executor)                      │ recording:<id>@<t> frames
   executor (:8770) <── POST /api/submit (ActionProposal) ──┤
        │  POST /api/observe → Observation + /api/frame/<obs>/<role>.jpg
        └──────────────>  models.GeminiAdvisor  ── perception.locate (pixels → world m)
                                │
                                └─ Gemini Robotics ER (HTTPS, v1beta)
```

- **Frames in.** `models` never opens a Station connection, so it adds no `follow()` subscriptions (Station
  has no unfollow). Frames come from the executor's observation (`/api/frame/<obs_id>/<role>.jpg`), from
  the recorder (`/live/frame`, or `recording:<id>@<t>` for recorded sessions), or from files. All of these
  are 299x224 JPEGs as Station delivers them.
- **Proposals out.** `GeminiAdvisor.propose()` picks a skill from a small vocabulary: `pick_at`,
  `drop_at`, `wipe_region`, `inspect`, `stop`. It attaches world coordinates from perception and sets
  `expires_at = obs.t_mono + TTL`. `to_api(p, now)` turns the proposal into the executor's
  `POST /api/submit` body. The executor validates the proposal (workspace, preconditions, TTL), then runs
  it through Station, or rejects it. Gemini output never reaches the motors directly.
- **Outcomes.** `judge()` and `residual()` return `Outcome`s. These are a second opinion next to the
  executor's own gripper and load checks, never instead of them (see `bench/REPORT.md`: 0 false successes,
  but many missed holds).
- **Joint mapping.** For wrist-camera pixels, perception needs the joint readings
  `{"j1": {"raw", "lo", "hi"}, ...}`. Build them from Station's calibrated ranges with
  `perception_bridge.joints_for_locate`. The mapping is `range_fraction`, the same one the executor uses.
- **Data loop.** Recorder sessions (Station clock `station_ns`) → `autolabel` (segments, keep/drop) →
  `lerobot_export` (LeRobot v3.0) → ACT / SmolVLA training per `POLICY_PLAN.md`. Teleop demos need the
  leader arm connected so an action stream is recorded.
- **Logs.** `models/logs/gemini-<date>.jsonl` holds one line per call with the call id, source tag
  (`advisor`, `bench`, `autolabel`, `smoke`), prompt, image refs (sha1, not bytes), reply, latency, tokens
  and cost. The key is redacted.
