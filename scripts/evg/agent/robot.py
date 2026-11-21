"""Robot access for the agent: real arm through Station (Robot), the MCP arm server (McpRobot) or recorded demo frames (ReplayRobot).

Realtime moves (Station): one coordinated command per move. Each joint gets its own speed so all joints arrive together (a straight
line in joint space, like the old sub-stepping, but without stopping at every sub-step). The arm is checked 20x per second: any arm joint
over the contact current stops the move and holds position. The servos are P-only and stop short, so a joint that has stopped moving
before its target gets its goal nudged (the project's grasp.py recipe).
"""
import asyncio, logging, sys, time
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS.parent / "norma-core" / "software/station/shared")); sys.path.insert(0, str(SCRIPTS.parent / "norma-core"))
sys.path.insert(0, str(SCRIPTS / "evg"))
import station_thin  # noqa: E402  (scripts/ is on sys.path now)
import common as C   # noqa: E402

FRAME_MAX_AGE = 3.0
CAM_STEP = 10          # take every 10th camera frame (~3 fps): see station_thin.py
POLL = 0.05            # check the arm 20x per second during a move
MIN_SPEED = 40         # steps/s floor for the slowest joint of a coordinated move
STALL_POLLS = 3        # a joint that has not moved for this many checks has stopped (arrived short, or blocked)


class RobotError(Exception):
    """fatal=True: we can no longer see the arm (e.g. Station state stream lost) -> the run must stop, not retry blindly."""
    def __init__(self, msg, fatal=False):
        super().__init__(msg); self.fatal = fatal


class Robot:
    def __init__(self, dry_run=False):
        self.dry_run, self.frames, self.arm = dry_run, {}, None

    async def connect(self):
        import arm
        from target.gen_python.protobuf.drivers.usbvideo import usbvideo
        self._usbvideo, self.mod = usbvideo, arm
        self.arm = await arm.Arm().connect()
        self.cams = C.camera_queues()
        for name, q in self.cams.items():
            queue = asyncio.Queue(); station_thin.follow_thin(self.arm.client, q, queue, step=CAM_STEP)
            asyncio.create_task(self._pump(name, queue))
        return self

    async def _pump(self, name, queue):
        while True:
            e = await queue.get()
            try:
                pack = self._usbvideo.RxEnvelopeReader(memoryview(bytes(e.Data))).get_frames()
                fr = (pack.get_frames_data() or []) if pack else []
                if fr: self.frames[name] = (time.time(), bytes(fr[-1]))
            except Exception:
                pass

    async def _state(self):
        """Joint state with one retry. Two timeouts in a row = the state stream is lost -> fatal (we are blind)."""
        for _ in range(2):
            try:
                return await self.arm.state()
            except TimeoutError:
                continue
            except SystemExit as e:
                raise RobotError(str(e), fatal=True)
        raise RobotError("lost the Station state stream (no joint state for 10 s); the arm keeps holding its last goal", fatal=True)

    def _frame(self, name):
        t, jpeg = self.frames.get(name, (0, None))
        return jpeg if time.time() - t < FRAME_MAX_AGE else None

    async def observe(self):
        st = await self._state()
        return dict(state=st, over=self._frame("over"), grip=self._frame("grip"))

    def _check_calibrated(self, st, joints):
        for j in joints:
            m = st[j]
            if not m["frozen"] or m["mx"] - m["mn"] < 300:
                raise RobotError(f"motor {j} is not calibrated; refusing to move")

    async def drive(self, target, stop=lambda: False, tol=6, timeout=None):
        """Coordinated closed-loop move of {joint: steps}. Returns (final_state, note). Raises RobotError on contact."""
        if self.dry_run:
            return None, "ok (dry-run: arm was not actually moved)"
        a, m = self.arm, self.mod
        st = await self._state()
        self._check_calibrated(st, target)
        target = {j: int(min(st[j]["mx"] - m.MARGIN, max(st[j]["mn"] + m.MARGIN, t))) for j, t in target.items()}
        dist = {j: abs(t - st[j]["pos"]) for j, t in target.items()}
        far = max(dist.values()) or 1
        items = []
        for j, t in target.items():
            speed = max(MIN_SPEED, round(m.SPEED * dist[j] / far))          # all joints arrive together
            items += [(j, 0x29, bytes([m.ACCEL])), (j, 0x2E, speed.to_bytes(2, "little")), (j, 0x28, b"\x01"), (j, 0x2A, t.to_bytes(2, "little"))]
        await a.write(items)
        timeout = timeout or (2.0 + 2.0 * far / m.SPEED)                     # time to cover the distance, with margin
        cmd, prev, still, t0 = dict(target), {j: st[j]["pos"] for j in target}, {j: 0 for j in target}, time.time()
        while time.time() - t0 < timeout and not stop():
            await asyncio.sleep(POLL)
            st = await self._state()
            hot = {i: st[i]["cur"] for i in range(1, 8) if st[i]["cur"] > m.CUR_LIMIT}
            if hot:
                await a.write([(i, 0x2A, st[i]["pos"].to_bytes(2, "little")) for i in range(1, 8)])
                raise RobotError(f"CONTACT: current over {m.CUR_LIMIT} mA {hot}; stopped and holding position")
            err = {j: target[j] - st[j]["pos"] for j in target}
            if all(abs(e) <= tol for e in err.values()):
                break
            nudge = []
            for j, e in err.items():
                still[j] = still[j] + 1 if abs(st[j]["pos"] - prev[j]) <= 1 else 0
                prev[j] = st[j]["pos"]
                if abs(e) > tol and still[j] >= STALL_POLLS:                     # stopped short (P-only servo): push the goal further
                    c = int(max(target[j] - 400, min(target[j] + 400, cmd[j] + round(0.5 * e))))
                    cmd[j] = int(min(st[j]["mx"] - m.MARGIN, max(st[j]["mn"] + m.MARGIN, c)))
                    nudge.append((j, 0x2A, cmd[j].to_bytes(2, "little"))); still[j] = 0
            if nudge:
                await a.write(nudge)
        if stop():
            return st, "stopped"
        left = {j: target[j] - st[j]["pos"] for j in target if abs(target[j] - st[j]["pos"]) > tol}
        return st, (f"did not fully arrive, remaining steps {left} (something may be touching)" if left else
                    f"arrived in {time.time() - t0:.1f}s")

    async def goto_path(self, target, st, stop, **_):
        """One coordinated move to a waypoint (all joints arrive together); contact and STOP are checked 20x per second."""
        return (await self.drive(target, stop))[1]

    async def grip(self, opening, stop=lambda: False):
        """Move the jaws; returns as soon as they arrive or stall on an object (no fixed wait)."""
        if self.dry_run:
            return "ok (dry-run: arm was not actually moved)"
        a, m = self.arm, self.mod
        st = await self._state(); g = st[8]
        self._check_calibrated(st, [8])
        goal = int(min(g["mx"] - m.MARGIN, max(g["mn"] + m.MARGIN, g["mn"] + opening * (g["mx"] - g["mn"]))))
        await a.write([(8, 0x29, bytes([m.ACCEL])), (8, 0x2E, m.SPEED.to_bytes(2, "little")), (8, 0x28, b"\x01"), (8, 0x2A, goal.to_bytes(2, "little"))])
        prev, still, t0 = g["pos"], 0, time.time()
        while time.time() - t0 < 3.0 and not stop():
            await asyncio.sleep(POLL)
            st = await self._state(); p = st[8]["pos"]
            if abs(goal - p) <= 8: break
            still = still + 1 if abs(p - prev) <= 1 else 0; prev = p
            if still >= STALL_POLLS and time.time() - t0 > 0.3: break        # stalled on an object = a grip
        return f"gripper at {(st[8]['pos'] - g['mn']) / (g['mx'] - g['mn']):.2f}, motor current {st[8]['cur']} mA ({time.time() - t0:.1f}s)"

    async def hold(self):
        st = await self._state()
        await self.arm.goto_steps({i: st[i]["pos"] for i in range(1, 9)}, wait=False)


class McpRobot:
    """The shared arm behind the MCP server (look / joint_state / move_to / gripper / set_torque). Moves are sent as normalized
    targets in the arm's own calibrated range; the server additionally refuses big jumps and aborts/holds if a joint stops tracking."""
    cur_unit = "load"

    class mod:
        MARGIN = 30

    def __init__(self, dry_run=False):
        self.dry_run, self.mcp, self.last = dry_run, None, {}

    async def connect(self):
        from agent import mcp_client
        self._m = mcp_client
        try:
            self.mcp = await asyncio.to_thread(mcp_client.connect)
        except mcp_client.McpError as e:
            raise RobotError(str(e))
        return self

    async def _call(self, name, args=None, timeout=None):
        try:
            return await asyncio.to_thread(self.mcp.call, name, args, timeout)
        except self._m.McpError as e:
            raise RobotError(str(e))

    async def observe(self):
        text, images = await self._call("look")
        st = self._m.parse_table(text)
        if len(st) != 8:
            raise RobotError("could not read the 8-joint table from the arm server")
        frames = {"over": None, "grip": None}
        for label, jpeg in images:
            for cid, name in self._m.CAMERAS.items():
                if cid in label and "STALE" not in label.upper():
                    frames[name] = jpeg
        self.last = st
        return dict(state=st, **frames)

    @staticmethod
    def _norm(j, steps, st):
        return min(1.0, max(0.0, (steps - st[j]["mn"]) / (st[j]["mx"] - st[j]["mn"])))

    async def drive(self, target, stop=lambda: False, **_):
        if self.dry_run:
            return None, "ok (dry-run: arm was not actually moved)"
        st = self.last or (await self.observe())["state"]
        text, _ = await self._call("move_to", {"targets": {f"j{j}": round(self._norm(j, t, st), 4) for j, t in target.items()}}, timeout=90)
        low = text.lower()
        if any(w in low for w in ("abort", "refus", "fault", "fail", "error")):
            raise RobotError(text.strip()[:300])
        st2 = (await self.observe())["state"]
        left = {j: target[j] - st2[j]["pos"] for j in target if abs(target[j] - st2[j]["pos"]) > 12}
        return st2, (f"did not fully arrive, remaining steps {left}" if left else "arrived")

    async def goto_path(self, target, st, stop, max_deg=6.0):
        n = max(1, int(max(abs(target[j] - st[j]["pos"]) for j in target) // round(max_deg * 4096 / 360)) + 1)
        note = ""
        for k in range(1, n + 1):
            if stop(): return "stopped"
            sub = {j: round(st[j]["pos"] + (target[j] - st[j]["pos"]) * k / n) for j in target}
            st2, note = await self.drive(sub, stop)
            if self.dry_run: return note
            st = st2 or st
        return note

    async def grip(self, opening, stop=lambda: False):
        if self.dry_run:
            return "ok (dry-run: arm was not actually moved)"
        text, _ = await self._call("gripper", {"open_fraction": round(opening, 3)}, timeout=60)
        return " ".join(text.split())[:200] or "gripper set"

    async def hold(self):
        if not self.dry_run:
            await self._call("set_torque", {"enabled": True})


class ReplayRobot:
    """Plays back a recorded demo as if it were the live arm: for testing prompts/models without hardware. Actions are only logged."""
    dry_run = True

    def __init__(self, demo_dir):
        import json
        self.dir = Path(demo_dir)
        self.rows = [json.loads(l) for l in (self.dir / "states.jsonl").read_text().splitlines() if l.strip()]
        self.over_files = sorted(self.dir.glob("over_*.jpg")); self.grip_files = sorted(self.dir.glob("grip_*.jpg"))
        self.i = 0
        # synthetic calibration range: wide, so clamps come from the demo envelope
        self.rng = {j: (min(r[str(j)][0] for r in self.rows) - 200, max(r[str(j)][0] for r in self.rows) + 200) for j in range(1, 9)}

    async def connect(self): return self

    async def observe(self):
        r = self.rows[min(self.i, len(self.rows) - 1)]
        st = {j: dict(pos=r[str(j)][0], goal=r[str(j)][1], cur=r[str(j)][2], mn=self.rng[j][0], mx=self.rng[j][1], frozen=True) for j in range(1, 9)}
        def pick(files):
            if not files: return None
            f = files[min(int(self.i / max(len(self.rows) - 1, 1) * (len(files) - 1)), len(files) - 1)]
            return f.read_bytes()
        return dict(state=st, over=pick(self.over_files), grip=pick(self.grip_files))

    def advance(self, n=15): self.i += n
    async def drive(self, *a, **k): return None, "ok (replay: arm was not actually moved)"
    async def goto_path(self, *a, **k): return "ok (replay: arm was not actually moved)"
    async def grip(self, *a, **k): return "ok (replay: arm was not actually moved)"
    async def hold(self): pass

    class mod:   # what run.py reads from the arm module
        MARGIN = 30
