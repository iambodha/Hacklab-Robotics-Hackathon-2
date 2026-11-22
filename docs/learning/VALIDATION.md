# Validation and integration notes

No real-arm server was started, no Station client opened, and no live services restarted. Commands run with
`STATION_HOST=station-disabled-in-tests.invalid`; Python runs through `uv run --no-project`.

The WIP had no rule module, no extraction MCP endpoint, an unsafe failed-release retreat, and resume omitted the
previous stroke’s backoff. Those are corrected. The tests cover profile provenance, safety-class rejection,
nonfinite inputs, round 1/2 replay, proposal persistence and inheritance, guarded/resisted lifts, failed release,
hard load stops, deadline cancellation, completed-stroke counts, and a deterministic kinematic bottle twin.
The deterministic model opens within the profile ceiling and extracts 60 mm; it verifies shared skill/config
behavior without claiming physical contact or IK accuracy.

Full-suite counts and final verification are recorded in [JOURNAL.md](JOURNAL.md). The exact required suite was run
at takeover and after the corrections. The remaining environment failures include `invalid CoreGraphics connection`
in MuJoCo renderer tests. One known pre-existing executor preflight assertion expects a marker name while the actual
collision message names the table. These are retained as failures, not silently skipped.

Physical twin probes (headless, frames disabled) exposed two limitations independently of camera rendering:

1. Learned real-height bottle (cap top 205 mm, x/y -7/190 mm): park-to-hover aborted on tracking; starting the fixture
   at the reachable hover instead reached 45° modeled lean and contacted at z=198 mm, about 9 mm above the intended
   grasp. The real arm’s recorded lean was 32–35°. The grasp is refused, rather than widening the contact allowance.
2. Historical short fixture with the same learned stroke settings and a 350° modeled thread: 11 strokes in two calls
   (100.1 s and 31.0 s) achieved 339.4° gripper rotation but only about 270° cap rotation. The 60 mm extraction stopped
   after five guarded steps on a 410-unit load rise; gripper rise was 4 mm. That is safe behavior, not a successful
   sim-to-real acceptance. This probe used the inherited WIP’s 11-stroke ceiling, before raising it to 12 for signal
   confirmation. Additional strokes alone do not establish correct grip/friction geometry.

Reproduce a physical acceptance probe, with the dependency prefix from `r5-common.md`:

```sh
STATION_HOST=station-disabled-in-tests.invalid \
NORMA_CORE_REPO=/Users/bodha/Documents/b0dha/Robotics-Hackathon/repo/norma-core \
MUJOCO_GL=cgl uv run --no-project --with numpy --with scipy --with pillow \
  --with mujoco --with opencv-contrib-python --with 'mcp>=1.12,<2' --with httpx \
  python -m learning.validate_bottle_twin --preposition
```

Add `--small` for the historical short fixture. It disables image rendering, uses temporary layouts/config, prints
full JSON measurements, and exits nonzero when physical opening/extraction fails. `--preposition` explicitly places
an offline fixture at hover to isolate grasp from park travel. It never relaxes tracking, contact or torque guards.
The observed physical twin gap remains unresolved; this brief’s full MuJoCo success criterion is not met. Calibrate
jaw contact geometry, motor dynamics and screw friction against the real recordings before treating that model as
validation of real extraction. Do not change hardware safety settings to make a twin test pass.

Sandbox restrictions: CGL cannot connect to CoreGraphics, `nice` cannot set priority, and process listing via `ps`
is denied. Worked around rendering only for the isolated headless physics probe; ran one physical probe at a time
without a priority adjustment. Normal full-suite renderer failures remain visible. No attempt was made to bypass
sandbox permissions or commit changes.

Merge notes: changes are uncommitted on `r5-skills` as requested. Preserve public server names and the shared
`learning.tools` calls when integrating the parallel hardware-agnostic package. `extract_cap` and `Rig.halt` are
additive; both provided rigs implement halt. A downstream custom rig should implement halt too, so asynchronous
budgets can stop outstanding actuator goals. Restart MCP servers only when the operator is finished; no live
validation is authorized by this report. `sim/stir_skill.py` still needs an audit outside this owned-file scope.
