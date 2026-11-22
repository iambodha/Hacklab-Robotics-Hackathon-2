# Validation record — 22 November 2026

No live Station connections, real serving, service restarts or live-test opt-in were used. Python ran through `uv run --no-project` with the common brief's dependency set, `STATION_HOST=station-disabled-in-tests.invalid`, the main NormaCore checkout and `MUJOCO_GL=cgl`.

## Focused suite

Final isolated run: **32 passed in 27.84 seconds**. Inherited WIP tests initially gave 27 passed; five regression tests were added during review. Coverage includes connector metadata and skills for both robots, FK/IK round trips, actual stdio tool parity for ElRobot sim, real import parity, fake-Station torque and movement, Station guard/host policy, optional comms absent/present/broken cases, legacy connector isolation, SO-101 layout cache and physics-clock accounting.

The ElRobot twin advertises the same **18 tools**, descriptions and input schemas as the original server. Real import parity covers the original real tools without running real serving. Fake-bus tests cover both connectors and assert motor goal/torque writes.

## Scripted MCP client, physics-only cameras

| Check | Independent truth result | Outcome |
|---|---|---|
| ElRobot pick | cube centre 9.9 → 110.0 mm | Passed |
| SO-101 pick, after final clock fix | cube centre 9.9 → 115.6 mm; HOLDING | Passed |
| ElRobot bottle `twist_cap`, ten strokes | cap turned 652.7°, threads free; lift -0.4 mm | Thread release passed; cap removal unverified |

These commands launched sim servers and called tools through MCP stdio, rather than calling twin internals. The truth log judges outcomes separately. The cap's motor rotation was 886.9°. `check_cap` returned `unclear: holding after lifting only 0 mm`; the episode's sensor-derived `cap_free` was false. The scripted client's `opened=true` is based on the simulator's released-thread state, **not** evidence of lifting off the cap.

Sandbox CoreGraphics access fails with `invalid CoreGraphics connection`. The explicit `GENERALOID_CHECK_HEADLESS=1` test launcher replaced rendering with labeled diagnostic images. Physics, IK and MCP were exercised; camera rendering and image-based perception were not. The two inherited SO-101 images under `img/` predate the takeover and were not regenerated or visually revalidated here.

## Baseline and environment limits

The takeover baseline common suite had **278 passed, 6 failed, 1 skipped, 3 errors in 106.63 s**. Eight failure/error cases were CoreGraphics-dependent; the other was the known marker-name assertion in `executor/tests/test_wiring.py::test_preflight_names_the_marker_under_a_table_hit`. Devin previously recorded 285 passed, 2 failed and 1 skipped in a different execution context.

The final combined common + Generaloid run had **310 passed, 6 failed, 1 skipped, 3 errors in 145.74 s**. The exact six failing and three error test identifiers match the takeover baseline; all 32 added Generaloid tests pass in the combined process too. There are no new failures relative to that baseline. Rendering-dependent tests remain unvalidated in this sandbox.

`nice -n 10` was attempted for heavy runs, but sandbox `setpriority` is denied; the commands still execute. `ps` is denied too, so session completion and logs were used. Long checks were serialized and artifacts stayed on the external disk. No permission escalation was requested.

## Evidence locations

Under `/Volumes/Dell HDD/Robotics-Hackathon/r5-generaloid/`:

- `takeover-baseline.txt`, `connector-tests-initial.txt`, `connector-tests-final.txt`, `combined-tests-final.txt`.
- `takeover-pick-elrobot.json`, `takeover-pick-so101-final.json`, `takeover-twist-elrobot.json`.
- `takeover_pick_elrobot/`, `takeover_pick_so101_final/`, `takeover_twist_elrobot/`: server stderr, truth logs and isolated learning artifacts.

The command-output JSON files also contain the sandbox's `nice` warning before the JSON line. Parse their final line as the check result.

## Integration limits

SO-101 hardware calibration and twin/real frame alignment remain unqualified. The existing ElRobot simulator shares the overload-clock pattern corrected in the new generic twin; its owner should review that separately. FastMCP ToolManager adoption and script-global legacy imports are migration constraints, covered by parity tests and the one-connector-per-process rule. No files under `generaloid/comms/` or the shared implementation packages were edited.
