"""Robot access for the agent: real arm through Station (Robot) or recorded demo frames (ReplayRobot).

Real moves reuse the project's closed-loop recipe from grasp.py: the servos are P-only and stop short, so we keep nudging the
goal until the measured position arrives, and abort (holding position) as soon as any arm joint exceeds the contact current.
"""
import asyncio, logging, sys, time
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS.parent / "norma-core" / "software/station/shared")); sys.path.insert(0, str(SCRIPTS.parent / "norma-core"))
import station_thin  # noqa: E402  (scripts/ is on sys.path now)

OVERVIEW_CAM = "usbvideo/5cd55831e21692d4f05fcaec75c351a5"
GRIPPER_CAM = "usbvideo/e091eeda8dcda172482f039057691cdb"
FRAME_MAX_AGE = 3.0
CAM_STEP = 10          # take every 10th camera frame (~3 fps): see station_thin.py


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
        for name, q in (("over", OVERVIEW_CAM), ("grip", GRIPPER_CAM)):
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

    async def drive(self, target, stop=lambda: False, tol=4, timeout=8.0):
        """Closed-loop move of {joint: steps}. Returns (final_state, note). Raises RobotError on contact."""
        if self.dry_run:
            return None, "ok (dry-run: arm was not actually moved)"
        a, m = self.arm, self.mod
        cmd = dict(target)
        try:
            await a.goto_steps(dict(cmd), wait=False)            # sets speed/accel, torque on, first goal
        except SystemExit as e:
            raise RobotError(str(e))
        except TimeoutError:
            raise RobotError("lost the Station state stream before the move started; nothing was commanded", fatal=True)
        t, ok, st = 0.0, 0, None
        while t < timeout and not stop():
            await asyncio.sleep(0.25); t += 0.25; st = await self._state()
            hot = {i: st[i]["cur"] for i in range(1, 8) if st[i]["cur"] > m.CUR_LIMIT}
            if hot:
                await a.write([(i, 0x2A, st[i]["pos"].to_bytes(2, "little")) for i in range(1, 8)])
                raise RobotError(f"CONTACT: current over {m.CUR_LIMIT} mA {hot}; stopped and holding position")
            err = {j: target[j] - st[j]["pos"] for j in target}
            ok = ok + 1 if all(abs(e) <= tol for e in err.values()) else 0
            if ok >= 2: break
            for j, e in err.items():
                if abs(e) > tol:
                    c = int(max(target[j] - 400, min(target[j] + 400, cmd[j] + round(0.5 * e))))
                    cmd[j] = int(min(st[j]["mx"] - m.MARGIN, max(st[j]["mn"] + m.MARGIN, c)))
            await a.write([(j, 0x2A, c.to_bytes(2, "little")) for j, c in cmd.items()])
        left = {j: target[j] - st[j]["pos"] for j in target if st and abs(target[j] - st[j]["pos"]) > tol}
        return st, (f"did not fully arrive, remaining steps {left} (something may be touching)" if left else "arrived")

    async def goto_path(self, target, st, stop, max_deg=6.0):
        """Joint-space interpolation in sub-steps of at most max_deg per joint, so no single jump is large."""
        n = max(1, max(abs(target[j] - st[j]["pos"]) for j in target) // round(max_deg * 4096 / 360) + 1)
        note = ""
        for k in range(1, int(n) + 1):
            if stop(): return "stopped"
            sub = {j: round(st[j]["pos"] + (target[j] - st[j]["pos"]) * k / n) for j in target}
            st2, note = await self.drive(sub, stop)
            if self.dry_run: return note
            st = st2 or st
        return note

    async def grip(self, opening, stop=lambda: False):
        if self.dry_run:
            return "ok (dry-run: arm was not actually moved)"
        st = await self._state(); g = st[8]
        try:
            await self.arm.goto_steps({8: g["mn"] + opening * (g["mx"] - g["mn"])}, timeout=4)
        except SystemExit as e:
            raise RobotError(str(e))
        except TimeoutError:
            raise RobotError("lost the Station state stream during the gripper move", fatal=True)
        st = await self._state()
        return f"gripper at {(st[8]['pos'] - g['mn']) / (g['mx'] - g['mn']):.2f}, motor current {st[8]['cur']} mA"

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
