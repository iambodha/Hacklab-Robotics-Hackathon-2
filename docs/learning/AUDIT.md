# Tunable audit

The audit distinguishes strategy defaults, immutable safety/geometry, numerical operations, and offline fixtures.
The learned bottle values are in `learning/profiles/bottle_cap.json`. Shared remaining strategies are registered from
`skills/strategy_defaults.json`: each value has an origin, units in its key/description and an allowed envelope.
Readers use `skillcfg.get` at call time; wipe-only improvement sandboxes read the identical local defaults through
`skills/config.py` when the learning package is absent.

| Area | Source/read path after this change |
|---|---|
| Bottle grasp, stroke, yaw, force, approach | `params.default.*` and `twist.*` registry + learned bottle profile |
| Bottle height if omitted | `bottle.cap_height_m`: learned 22 mm; old baseline 20 mm |
| Free signature, budget, resume | `twist.free_*`, `twist.time_budget_s`, `twist.cycle_s.*`, `twist.resume_tol_m` |
| Check/extraction and resistance | `check.lift_m`, `extract.lift_m`, `lift.*`; immutable load stops still in code |
| Twist waypoint strategy and settling | `twist.step_*`, approach/tilt/hover keys, opening/backoff/settle keys |
| Real MCP wipe, stir, pick/place | Existing robot profile `P.wipe`, `P.touch`, `P.stir`, `P.pick`, `P.place`, `P.gripper`, `P.poses`; already named, preserved |
| Twin bottle rig descent, grip, hold | `rig.*` defaults preserve old descent increments, lag, force and settle behavior |
| Twin stir tool defaults | `sim.stir.*`; explicit caller values still win; old clamp bounds remain safety envelopes |
| Executor pick/place and perception | Existing robot profile plus `executor.pick.*`, `executor.drop.*` for formerly inline geometry assumptions/tolerances |
| Executor stir/wipe | Existing robot profile plus `executor.stir.*`, `executor.wipe.*` |
| Executor observation freshness | `executor.freshness.*`, dynamic descriptors; relocalization age in `executor.pick.relocalise_after_s` |
| Wipe learning | Existing `skills/wipe/defaults.json` + protected `bounds.json`; new named `wipe.adapt.alpha`, `wipe.plan.overshoot_m`, `wipe.proposal.ttl_s` |
| Executor kinematics | r4 already reads `ik.*`, `executor.ik.search` and robot IK profile. Transform matrix, zero offsets, trigonometry and limit checks remain physical/math constants |

Compatibility: public tool/function names remain, including `check_cap`; `extract_cap` is additive. Optional default
arguments that must read live config now accept `None`, with the same old baseline behavior under
`SKILL_PROFILES=none`. Existing explicit caller arguments retain their meaning. Safety-class entries cannot be
changed by profiles or runtime proposals.

Numbers intentionally retained: degrees/radians/mm conversions, normalized jaw endpoints, vector axes, list indices,
numerical epsilons and interpolation algebra; measured fingertip geometry and IK continuity limit; physical screw
fixtures, colors, rendering resolution and offline synthetic trials. `skills/eval` already exposes an `EvalConfig`
for evaluator thresholds; `skills/experiment.PromotionRule` is an explicit immutable preregistered benchmark input,
not a robot strategy. Benchmark defaults and test fixture noise models were not turned into online adaptation.

Scope boundary: `sim/stir_skill.py` contains the separate twin spoon implementation and remains outside this brief’s
owned files. The twin tool’s inputs are configurable here, but that helper’s internal motion strategy still needs an
integrator-owned audit. No Station/IO code or robot JSON safety profile was modified.
