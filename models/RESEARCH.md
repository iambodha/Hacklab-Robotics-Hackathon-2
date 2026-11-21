# Gemini Robotics: what exists today and what this arm can use

Checked against the live docs on 22 Nov 2026. Model names have changed twice this year; re-check
before trusting a model id from memory.

## Gemini Robotics ER (embodied reasoning): a VLM you call over the Gemini API

| | |
|---|---|
| Current model ids | **`gemini-robotics-er-2-preview`** (standard, `generateContent` / Interactions API) and **`gemini-robotics-er-2-streaming-preview`** (Live API only). Both say "preview" and were last updated July 2026. |
| Deprecated | `gemini-robotics-er-1.6-preview`, shut down 31 Aug 2026 ([deprecations](https://ai.google.dev/gemini-api/docs/deprecations#robotics-models)). Older `gemini-robotics-er-1.5-preview` code samples won't work. |
| Base | ER 2 "builds on Gemini 3.5 Flash" ([overview](https://aistudio.google.com/docs/robotics-overview)) |
| Inputs / outputs | text, image, video, audio in; **text out only**. 131,072 input tokens, 65,536 output tokens ([model page](https://aistudio.google.com/docs/models/gemini-robotics-er-2-preview)) |
| Supported on the standard id | function calling, structured outputs (JSON schema), thinking (`thinking_level` low/medium/high), code execution ("agentic vision": crop/zoom with code), context caching, Batch API, Search grounding. **No Live API.** |
| Streaming id | Live API (WebSocket), function calling, thinking, Search. **No** structured output, caching, code execution or batch. Images ≤ 1 FPS JPEG, 16 kHz PCM audio ([streaming](https://aistudio.google.com/docs/robotics-streaming)). Physical actions declared as tools with `"behavior": "BLOCKING"`. |
| Access | Public preview with a Gemini API key from AI Studio. **The API rejects unrestricted keys with 403**: the key must have API restrictions set in AI Studio ([limitations](https://aistudio.google.com/docs/robotics-overview)). Not on Vertex as public preview (Gemini Enterprise Agent Platform: private preview). |
| Price (paid tier) | $1.00 / 1M input tokens (text, image, video), $5.00 / 1M output tokens **including thinking tokens**, cache $0.10 / 1M, through 31 Dec 2026. **Doubles on 1 Jan 2027** ($2 / $10). Batch: half price. Free tier exists (data used to improve products). Same token prices on the streaming id ([pricing](https://ai.google.dev/gemini-api/docs/pricing#gemini-robotics-er-2)). |
| Image cost | Gemini 3-family `media_resolution`: up to 1120 tokens per image at `high` (the recommended default), 280 at `low`/`medium` for video frames ([media resolution](https://ai.google.dev/gemini-api/docs/media-resolution)). Our 299x224 frames cost at most ~$0.001 each at `high`. |
| Rate limits | Per project, by usage tier, shown only in AI Studio (no published table for ER 2). Preview models have tighter limits. Tier 1 also has a $10 per 10 minutes spend limit; 429 `RESOURCE_EXHAUSTED` when hit ([rate limits](https://ai.google.dev/gemini-api/docs/rate-limits)). |
| SDK | `google-genai` (Python). 3.6.1 is the newest (18 Nov 2026); this package pins **3.3.0** (12 Nov, > 7 days old). `client.models.generate_content(model=..., contents=[Part.from_bytes(jpeg), prompt], config=GenerateContentConfig(thinking_config=ThinkingConfig(thinking_level="low"), response_mime_type="application/json"))`. The newer Interactions API (`client.interactions.create`) is used in some doc samples; `generate_content` is still documented on every capability page. |

### Capabilities (each with the documented output convention)

| Capability | Documented convention | Guide |
|---|---|---|
| Pointing | `[{"point": [y, x], "label": ...}]`, **y first**, normalised 0-1000 | [spatial](https://aistudio.google.com/docs/robotics-spatial) |
| Boxes | current page: `{"label", "y", "x", "y2", "x2"}` 0-1000; older samples and Gemini detection use `"box_2d": [ymin, xmin, ymax, xmax]`. The client here accepts both. | same |
| Trajectories | `[{"step", "point": [y, x], "action"}]` 0-1000 | same |
| Tracking in video | one point per frame | same |
| Success detection / progress (ER 2 only) | **moment finding** `{"completion_time_seconds": float|null}` on a video; **progress classification** into five brackets `"0-20" … "80-100"` | [video progress](https://aistudio.google.com/docs/robotics-video-progress) |
| Orchestration | function calling against your own robot API in an agent loop, step-limited | [orchestration](https://aistudio.google.com/docs/robotics-orchestration) |
| Agentic vision | code execution to crop / zoom / measure in the image before answering | [agentic](https://aistudio.google.com/docs/robotics-agentic) |

Google's own advice: `thinking_level` **medium** balances latency and quality; low for detection-type
prompts; high for planning. Known limits: hallucinated objects on ambiguous prompts, latency grows
with resolution and thinking. Pixel coordinates are 2D only: no depth, no metric output. Metric
positions come from our calibration (`perception.locate.pixel_to_table`).

## Gemini Robotics On-Device (the VLA): not accessible to us

- **Gemini Robotics On-Device 2** ([page](https://deepmind.google/models/gemini-robotics/on-device/),
  [model card, 30 Jul 2026](https://deepmind.google/models/model-cards/gemini-robotics-on-device-2/)): a
  vision-language-action model on Gemma, image + text + proprioception in, **actions out**, runs
  locally, adapts to a new embodiment with "fewer than 200 examples".
- Distribution: **trusted testers only**, through the Safari SDK
  ([github.com/google-deepmind/gemini-robotics-sdk](https://github.com/google-deepmind/gemini-robotics-sdk)),
  "most of the functionality requires you to join Gemini Robotics Trusted Tester Program". No public
  checkpoint, no API.
- The model card lists "limited … controlling high-degree-of-freedom robots" and the base is trained
  for bi-arm robots. Our ElRobot has 7 joints + gripper and a single arm: even with access it would be a
  fine-tuning project. For this hackathon the learned-policy route is LeRobot (ACT / SmolVLA, see
  `POLICY_PLAN.md`), with Gemini ER as the high-level advisor and data labeller.
- The cloud **Gemini Robotics** VLA (non on-device) has no public API either.

## What matters for this robot

The arm has scripted, sensor-checked skills (`sim/real_arm_mcp.py`: `move_tcp`, `gripper`, `pick_at`,
`drop_at`, `wipe`, `grab_tip`, `stir`) and a calibrated overview camera, so Gemini isn't needed to output
motions. Where it helps:

1. **Pointing and boxes on the 299x224 frames** -> pixels -> world through the calibration. This is the
   input to `pick_at(x, y)` and to wipe-region planning. Measured against ArUco ground truth in
   `bench/`.
2. **Success and failure checks** after a skill: "is the stain gone, where is the residual", "is
   something between the jaws". The executor's own sensors (gripper load, contact) stay the primary
   signal; Gemini is an independent second opinion that feeds `Outcome` (never `ExecutionResult`).
3. **Progress classification / moment finding on recorder clips**: segmenting demos into sub-tasks,
   marking success, and filtering demos for policy training (`models/autolabel.py`).
4. **Next step from a small vocabulary** (`inspect`, `retry_grasp`, `wipe_region`, `pick`, `place`,
   `stop`) as `ActionProposal`s that the executor validates; Gemini never drives the arm directly.
5. Streaming (Live API) is the right tool for continuous monitoring (e.g. "tell me when the stain is
   gone"), but it has no structured output, so it's left for later. The executor's polling advisor uses
   the standard id.

Not useful here: trajectories in pixel space (our skills plan in 3D with IK and contact sensing),
multi-robot orchestration, audio.
