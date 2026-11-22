# tasks/blue_to_bin: "take the blue object and put it in the bin"

The demo task, built on the executor (the only owner of motion) and perception (world coordinates with σ).
Every motion is one `ActionProposal`, with a fresh observation, preconditions and an expiry. The task never sends a
motor command itself.

| file | what |
|---|---|
| `task.py` | goal, measured geometry (`Spec` ← `measured.json`), grasp choice (`across` ≤ 40 mm, else `rim` pinch), release points, **success predicate** `judge` |
| `plan.py` | scripted planner `Runner(ex, spec, mode=dry\|pick\|full)`: the steps, preconditions, replanning (object moved, missed, slipped), ≤ 3 attempts |
| `locate.py` | `BlueLocator`, the executor `Locator` adapter: blue blob on the table → cylinder of known size → centre, base height, σ |
| `skills.py` | `descend_guarded`, the adapter skill: guarded straight-down descent that does not close (`install(ex)`) |
| `measure.py` | read-only measurement of the lid and the tub from the overview → `measured.json`, `measured_overview.jpg` |
| `sim_trials.py` | N twin trials, lid randomised ±3 cm, ground truth only for reporting → `results/sim_scripted*.json` |
| `advisor_trials.py`, `advisor_prompt.md` | the same task with Claude (`claude -p`, executor MCP tools only) as the advisor |
| `serve.py` | the executor console with the adapters installed, optionally running the planner (sim or real) |
| `REAL_RUN.md` | the real-arm checklist (integrator + user) |

Steps of `full` (each line one proposal, except observe/locate/verify):
1. park, then a jaw check in the air (close empty, open);
2. observe → locate blue (σ ≤ 20 mm, resting on the table);
3. approach above;
4. re-observe; if it moved > 15 mm, re-plan the approach;
5. descend (guarded);
6. close;
7. test lift of 4 cm, then the overview must see the lid rise;
8. lift to 0.20 → carry above the bin at 0.20 → down over the bin to rim + 4 cm;
9. guarded lower inside the rim → release;
10. straight up → above park → park;
11. verify: the overview sees it inside the rim footprint and below the rim, and the gripper evidence agrees.

Tests: `python -m pytest tasks/tests` (geometry, the predicate on two real Station frames, the planner on the fake backend).

## NormaCore integration
- **Used:**
  - Station camera queues (`usbvideo/<hash>`) for the measurement: live via one `follow`, or the newest stored
    entries via `read_from_tail`, which also works while video is stalled;
  - the executor's Station backend for the real run;
  - the ElRobot URDF through the robot profile.
- **Could go upstream:**
  - a task-level "proposal script" format (steps with preconditions/expiry) runnable against any Station robot through the executor API;
  - the `read_from_tail` frame grab as a station_py helper.
