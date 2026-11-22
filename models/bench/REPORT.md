# Benchmark: Gemini Robotics ER 2 on our scene (22 Nov 2026)

Model `gemini-robotics-er-2-preview`, `thinking_level=low`, Gemini API **v1beta** (`google-genai==3.3.0`).
Every call is logged in `models/logs/gemini-<date>.jsonl` (prompt, image refs, reply, latency, tokens, cost;
no key). Raw rows: `results/gemini.jsonl`, `results/live-colour.jsonl`; summaries: `results/*.summary.json`.

**This is partial.** 18 of the 29 bench items are scored. The live items have only the colour-locate side.
The key is on the **Free Tier**, which allows **5 requests/min and 20 requests/day** per model. The
per-minute limit is handled with back-off (`--wait-429`). The daily limit ran out after 24 successful calls:

```
429 RESOURCE_EXHAUSTED ... Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests,
limit: 20, model: gemini-robotics-er-2-preview, description: Generate content requests per day limit exceeded on Free Tier.
Please retry in 7h26m56s.
```

**Claude: not compared.** The Claude CLI is not logged in on this machine, so the integrator decided to
skip it. `models/claude_cli.py` sends Claude the same prompts and parses its replies with the same parsers,
so `--advisor claude` gives directly comparable numbers once the CLI is logged in.

## Access: what was wrong and what works now

| Symptom | Cause | Fix |
|---|---|---|
| `404 NOT_FOUND ... models/gemini-robotics-er-2-preview is not found for API version v1` (same for er-1.6, er-1.5, gemini-3-flash-preview) | google-genai 3.3.0 calls `v1` by default; the preview ER models are only served on `v1beta` | `genai.Client(http_options={"api_version": "v1beta"})` (`gemini_er.API_VERSION`, env `GEMINI_API_VERSION`) |
| `404 ... gemini-2.5-flash is no longer available to new users` | model retired for new keys | not used |
| `429 ... requests per minute ... limit: 5` | Free Tier | `GenAITransport(wait_429_s=...)` sleeps the server's `retryDelay`; bench `--wait-429` |
| `429 ... requests per day ... limit: 20` | Free Tier | **user: enable billing on the key's AI Studio project** (paid Tier 1 has no 20/day cap). The client fails fast on this one instead of waiting. |

No 403 happened, so the key's API restrictions are fine. ER 1.5 and 1.6 are shut down; there is no ER fallback.

## Results (Gemini ER 2, 18 of 29 items)

Pixel errors are on a 299-px-wide frame (Station resolution). mm errors are table-plane projections through
perception's overview calibration. Converting a perfect marker pixel already gives 4 to 8 mm, so the mm
figures are mostly calibration error, not model error.

| Kind | Scored | Result |
|---|---|---|
| ArUco markers (pointing) | 4/4 frames, 10 points | all 10 found; error p50 **0.98 px**, p90 3.1 px; **6.3 mm** p50, 15.9 mm max (6 points with world truth); 2 false points (one extra on each wrist frame) |
| Objects (point + box) | 2/4 frames, 7 objects | 7/7 points inside their box; error p50 **1.07 px**, p90 3.5 px; box IoU mean **0.80**; **1 hallucination** (pointed at the "black object" after it had been removed, run4/018) |
| Grasp judged held or not | 10/12 | accuracy **0.60**; **false success 0/4**; false failure 4/6: it said "the object is still on the table" for 4 lifts the gripper's hold test counted as held |
| Stain residual | 2/9 | 2/2 correct; residual box IoU 0.83, centre error 3.8 mm |

Baselines on the same items: the HSV stain detector scores 0.667 accuracy on 9/9, 0 false successes and
residual IoU 0.46. The noisy-oracle mock only checks the pipeline; it says nothing about any model.

**Latency** (bench calls, back-off excluded): p50 **3.3 s**, p95 **6.3 s**, max 10.3 s (an object-box call).
**Tokens** per call (median): 1230 prompt, 109 output, 329 thinking.
**Cost**: $0.0047 per call at paid prices ($1 / $5 per M tokens, thinking billed as output); **$0.115 for
all 24 successful calls**. Actually billed: $0, because it's the Free Tier. Finishing the run is about 21
more calls, about $0.10.

## Live demo items: blue cap and bin

There are 4 frames from the live cameras (`data/live/`): 3 overview frames and 1 wrist frame with its joint
readings. All are 299x224. The boxes were hand-labelled on a zoomed pixel grid, about ±3 px. **In this scene
the cap is already inside the clear bin** (the task's end state), so its box sits inside the bin's box.

| Method | Blue cap | Bin |
|---|---|---|
| colour, whole image (`segment(color="blue")`, as `locate_object(color=...)` does with no bbox or pixel) | **wrong on all 3 overview frames**: it picks a blue patch in the background at (269, 11), off the table (124 px / ~1.9 m off). On the wrist frame: 2.0 px, 3.7 mm | no answer (clear plastic: no colour class) |
| colour, table pixels only (calibrated table extent + 2 cm mask) | 4/4 inside the box; error p50 **2.9 px**, **12.7 mm** (max 22.7 mm); footprint centre p50 14.7 mm | no answer |
| Gemini point / box | **pending** (daily cap) | **pending** |
| hybrid: Gemini box → `locate_object(bbox=...)` | pending | pending |

Two of the overview masks reach into the bin's blue-tinted wall, which moves the centroid by 4 px (~22 mm).
The wrist view is clean.

## Recommendation

1. **Enable billing before the real run.** With 20 requests a day, one demo with a few retries uses up the
   quota. That's a setting in AI Studio; no code change is needed.
2. **Gemini recognises, perception measures.** Gemini pointing is about 1 px on our frames and finds objects
   by name. The world-coordinate error comes from the calibration, so take Gemini's box and give it to
   `perception.locate_object(bbox=...)`. For the bin there is no alternative: colour can't see clear plastic.
3. **Blue cap: colour is fine, but only inside the table.** Without a bbox or pixel, `locate_object(color=...)`
   picks the largest blob anywhere in the image. On the overview camera that blob is in the background. The
   perception owner should mask off-table pixels when no ROI is given (see the r2-gemini report).
4. **Don't use Gemini's "done" as the success signal.** It had 0 false successes but missed 4 of 6 real
   holds: thin objects between the jaws look like "still on the table". Keep the gripper's hold test as the
   primary signal and use Gemini as an extra veto, which is how `GeminiAdvisor.judge()` is used.
5. **Call it once per step, never in a control loop.** It takes 3 to 6 s per call.

## Finish the run (after the reset at about 09:00 local, or once billing is on)

```bash
cd ws/r2-gemini
uv run --no-project --with google-genai==3.3.0 --with opencv-contrib-python --with numpy --with scipy --with pillow \
    python -m models.bench.run --advisor gemini --retry-errors      # re-runs only the 11 failed items
uv run --no-project --with google-genai==3.3.0 --with opencv-contrib-python --with numpy --with scipy --with pillow \
    python -m models.bench.live                                     # 8 calls: Gemini + hybrid on the live frames
```
