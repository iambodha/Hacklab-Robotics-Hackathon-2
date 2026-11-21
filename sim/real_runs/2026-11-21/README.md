# Real-arm operator runs, 21 Nov 2026

Autonomous runs of `sim/run_agent_real.sh` on the physical follower arm (bus `5B61033301`), plus the manual tool tests before them.

| Run | Model | Task | Tool calls | Time | Cost | Outcome |
|---|---|---|---|---|---|---|
| 1 | claude-sonnet-5-5 | purple item: pick, lift, set down 8 cm to the left, park | 22 | 92 s | $0.16 | done, verified by camera |
| 2 | claude-sonnet-5-5 | purple item into the white box (first try) | 2 | 7 s | $0.06 | stopped at once: person's hands beside the box; nothing moved |
| 3 | claude-sonnet-5-5 | purple item into the white box | 26 | 107 s | $0.19 | done; item seen inside the box afterwards |
| 4 | claude-haiku-5-5 | sort: wrappers into the white box, black object into the grey box | 31 | 188 s | $0.02 | black object into the grey box; orange wrapper failed twice; purple not tried; then stopped: person in view |

## Files
- `actions.jsonl`: every tool call on the real arm since 15:53 (manual tests and all four runs): time, tool, arguments, duration, result text with the joint table. Calls made inside `pick_at`/`drop_at` are logged too.
- `agent_runN.jsonl`: the full Claude Code transcript of run N (stream-json): every model message, tool call and tool result, including the camera images the agent was shown (base64).
- `runN_frames/`: the same images as JPEG files, in the order the agent saw them (598x448 with the pixel grid).

Runs 1-3 used the tools as of commit `0cd5133`; run 4 used `pick_at`/`drop_at` and the one-call gripper (`a0a1481`).

**Personal data:** several frames (mainly runs 2 and 4) show identifiable people at the table. Keep this folder inside the team's private repo; remove those frames before sharing it outside the team.

Not in git: Station's continuous recording of both cameras and the motor bus (about 7 GB, `station-run/station_data` on the operator's laptop).
