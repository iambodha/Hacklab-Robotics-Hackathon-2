# Plan: use the higher-level model to get a better small policy (ACT vs SmolVLA)

Nothing here has been installed or trained (the disk is ~2.8 GB free). The runnable parts exist:
`lerobot_export.py` (recorder -> LeRobot v3.0), `dataset_audit.py` (results in `AUDIT.md`),
`autolabel.py` (Gemini ER labels and filters episodes), `advisor_gemini.py` (Gemini as the evaluator
in the closed-loop eval).

## 1. Which skill

**Pick a thin flat object (sweet wrapper) off the table.** Reasons:
- It's the skill the scripted controller is worst at. On 21 Nov, `pick_at` held something after the
  lift in **7 of 12** attempts (58 %, counted from the gripper's own hold test in
  `sim/real_runs/2026-11-21/*.jsonl`; the same 12 attempts are the `grasp` items in `bench/`). The
  orange wrapper failed 3 times in a row in run 6. That gives a measured baseline to beat.
- Success is measurable without people: the gripper hold test (jaw position + load) after the lift,
  plus an independent overview check (Gemini `judge` -> `Outcome`).
- Demonstrations are cheap with the leader arm (teleop already works, LOG.md).
- Wiping stays scripted for the demo (contact-sensed strokes + Gemini choosing the region and the
  stroke spacing). A learned wipe is the follow-up if the pick experiment passes the gate.

## 2. Data

| | |
|---|---|
| Demos | **60** teleop demos: 6 object positions x 10 (SmolVLA's docs: 50 demos over 5 positions worked, 25 did not). Plus **2 held-out positions x 5** for the test set (never trained on). Same objects, 2-3 wrapper types. |
| Episode | from rest pose -> grasp -> lift 10 cm -> hold 1 s. About 8-12 s. Start each demo with a recorder marker. |
| Recording | `recorder` with the **leader bus connected** so `action = leader norm` (`AUDIT.md`: no recording has actions yet). Follower torque on, mirroring on. Station `resize_target` unchanged (224) unless the first ACT run fails on detail. |
| Disk | ~1.4 MB/s recorded -> 70 demos x 10 s ≈ 1.0 GB. Record in batches of 20, export, move off the laptop. |
| Labelling | `python -m models.autolabel <ids> --task "pick up the wrapper"`: Gemini ER segments approach / grasp / lift, writes the instruction, flags failed grasps, a person's hands in view, camera gaps. `keep=false` episodes are excluded, and each removal is listed for a human to confirm (the model can be wrong; removed demos aren't deleted). |
| Export | `python -m models.lerobot_export <ids> --labels models/labels --out models/exports/pick_wrapper --fps 30` -> LeRobot v3.0 (`observation.state`, `action`, two cameras). Validate with `lerobot` on the training machine (`LeRobotDataset(root=...)`) before training. |

**Splits (identical for both policies, saved as `splits.json`, seed 0):** train = 8 demos per
training position (48); val = 2 per training position (12); test = the 10 held-out-position demos
(offline metrics only) + the closed-loop trials below. Splits are by episode, never by frame.

## 3. Policies and compute

| | ACT | SmolVLA |
|---|---|---|
| Start | from scratch (ResNet-18 ImageNet backbone) | fine-tune `lerobot/smolvla_base` (450 M params, 907 MB checkpoint) |
| Config | lerobot defaults: chunk 100, n_action_steps 100, batch 8, lr 1e-5, ~50-80 k steps | docs: batch 64, 20 k steps ≈ 4 h on one A100; we'd start with batch 32, 10 k steps |
| Inputs | both cameras 224x299, state (8) | both cameras (padded to 512), state, the instruction from the auto-labeller |
| Train on | this Mac (MPS) is possible in principle, but the env + data don't fit in 2.8 GB; **rent a GPU** (1x L4/A10G is enough for ACT, about 1-2 h) | **rented GPU** (A100 40 GB or L40S) |
| Inference on the robot laptop | expected well under 33 ms per step on MPS (to be measured: gate item 3) | hundreds of ms per chunk on MPS expected; run it as async chunked inference (LeRobot async / real-time chunking) or on the GPU box over the LAN |

Both run as an executor skill backend (`policy_rollout`, source `policy:act` / `policy:smolvla`): the
executor process keeps the only connection that sends motion. Each chunk of joint targets passes the
same limits and guards as a scripted skill (range clamp, step size, workspace box, contact abort,
cancel). An advisor (Claude / Gemini) proposes `policy_rollout(policy, task, max_s)` like any other skill.

## 4. Admission gate (all must pass, in this order, before the policy runs without a hand on the stop)

1. **Observation/action mapping is right.** Feed one recorded demo through the policy's pre- and
   post-processors without the network (identity in place of the model). The output joint targets
   must equal the recorded leader -> follower targets within 1 tick: same joint order (j1..j7, gripper),
   same signs, Station's calibrated norm 0..1, and the ranges stored in the recording equal to the
   live calibrated ranges (otherwise the SetupId differs and the run is rejected).
2. **Held-out sanity (offline).** On the 10 held-out-position demos: action L1 per joint and gripper
   timing error. The policy must beat both "repeat the last state" and "replay the nearest training
   demo" by a clear margin. Plot 3 episodes open loop.
3. **Measured inference timing** on the machine that will run it, with the real camera queue:
   p50 / p95 / max per chunk over 300 calls. ACT: p95 < 33 ms (or chunked with a buffer that never
   runs dry). SmolVLA: p95 < the chunk's execution time minus one control period. Logged in the
   recorder as markers.
4. **Dry runs.** (a) Replay mode (`ROBOT_SOURCE=replay:...`): the policy runs on recorded
   observations, and its commands are logged, never sent. Every target stays within limits, and no step
   exceeds the executor's max step. (b) Twin preflight of the first chunk of every rollout, as for the
   scripted skills.
5. **Guarded closed loop on the arm.** Executor speed-limited, workspace box, contact abort, operator
   at the stop. **10 trials at training positions + 10 at the 2 held-out positions**, the same 20
   setups for the scripted `pick_at` baseline (re-measured on the same day). The success judge is
   independent of the policy: hold test after lift AND overview `Outcome` (Gemini judge; `None`
   counts as failure).
6. **Beats the existing skill.** Policy success at least **15 points** above scripted `pick_at` on the
   same 20 setups, with **zero** guard trips that needed the operator. With 20 trials that's evidence,
   not proof: repeat on 20 more before making the policy the default. If ACT and SmolVLA both pass,
   take the one with the better held-out-position score. On a tie, take the cheaper one at
   inference (ACT).

## 5. How the higher-level model raises the small policy's ceiling

- **Before training:** Gemini ER labels sub-tasks, writes the language instruction (SmolVLA is
  language-conditioned), and filters failed / interrupted demos (`autolabel.py`). Bad demos are the
  most common reason small imitation policies fail.
- **During evaluation:** Gemini is the independent success judge (`advisor_gemini.judge`, `residual`)
  -> `Outcome`, which feeds gate items 5 and 6 without a human scoring every trial.
- **At run time:** the advisor decides *when* to call the policy and when to fall back
  (`retry_grasp` -> scripted `pick_at`, or `stop`). Every rollout is recorded, so the policy's own
  successes, labelled the same way, become the next round of training data (DAgger-style, with the
  model as the filter).

## 6. Installs (not done now; exact commands and sizes)

Pinned versions are all > 7 days old on 22 Nov 2026 except transformers. Pin that to a release from
before 15 Nov.

```bash
# on the training machine (Linux + CUDA), or this Mac once there's >= 6 GB free
uv venv ~/venvs/lerobot --python 3.12
uv pip install --python ~/venvs/lerobot "lerobot[dataset,training]==0.7.0"   # torch 2.15.0 (macOS arm64 wheel 139 MB;
                                                                               # CUDA wheels ~0.9 GB + nvidia libs ~2-3 GB),
                                                                               # torchvision 0.29.2 (2 MB), opencv-python-headless (26 MB),
                                                                               # pyarrow 26 (37 MB), av 19 (19 MB), datasets, torchcodec
uv pip install --python ~/venvs/lerobot "lerobot[smolvla]==0.7.0" "transformers<5.24"   # transformers (13.5 MB wheel), accelerate, num2words
hf download lerobot/smolvla_base                                              # 907 MB (model.safetensors)
# SmolVLA also reads HuggingFaceTB/SmolVLM2-500M-Video-Instruct's processor/tokenizer (~5 MB); its full
# weights (2.0 GB) are only needed if lerobot is told to load VLM weights separately; check before downloading.
```

Disk estimate on macOS (not measured): ~1.5-2.5 GB for the env, + 0.9 GB SmolVLA weights,
+ ~1 GB data, + checkpoints (ACT ~200 MB each, SmolVLA ~0.9 GB each). This doesn't fit in the
current 2.8 GB. Train on the rented machine and copy back only the chosen checkpoint.
