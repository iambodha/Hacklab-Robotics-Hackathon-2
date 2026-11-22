"""Direct control of the follower arm through Station: state, FK/IK, joint and Cartesian moves.

  python3 scripts/arm.py pose                     joint angles, tool pose, where the gripper points on the table
  python3 scripts/arm.py hold                     torque on, goal = present position (use right after stopping teleop)
  python3 scripts/arm.py grip 0..1                gripper: 0 = closed end of range, 1 = open end (see pose output)
  python3 scripts/arm.py joint N DELTA_DEG        rotate one joint (1-7)
  python3 scripts/arm.py move DX DY DZ            move the tool tip by metres in base frame, keeping orientation
  python3 scripts/arm.py fwd D                    move the tool tip D metres along the gripper's pointing direction
  python3 scripts/arm.py relax                    torque off (arm will drop!)

Base frame: origin under the base, z up. Teleop must be off or mirroring will fight these commands.
Every move is clamped to the calibrated range minus a margin, and split so no joint jumps far in one command.
"""
import asyncio, logging, math, struct, sys, time, xml.etree.ElementTree as ET
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent / "norma-core"
sys.path.insert(0, str(ROOT / "software/station/shared")); sys.path.insert(0, str(ROOT))
from station_py import new_station_client, send_commands
from target.gen_python.protobuf.station import commands, drivers
from target.gen_python.protobuf.drivers.st3215 import st3215

BUS = "5B61033301"          # follower
MARGIN = 30                 # steps kept clear of each calibrated end
SPEED, ACCEL = 350, 15      # steps/s, servo accel units: deliberately slow
TIP = (0.0, 0.125, 0.0)     # tool tip in the gripper-base frame (estimate: between the jaw tips)
TABLE_Z = 0.0
CUR_LIMIT = 260            # mA; abort a move if any arm joint exceeds this (contact)

# ---------- tiny linear algebra ----------
def matmul(A, B): return [[sum(A[i][k] * B[k][j] for k in range(len(B))) for j in range(len(B[0]))] for i in range(len(A))]
def rot(axis, q):
    n = math.sqrt(sum(a * a for a in axis)); x, y, z = (a / n for a in axis); c, s, C = math.cos(q), math.sin(q), 1 - math.cos(q)
    return [[c + x*x*C, x*y*C - z*s, x*z*C + y*s], [y*x*C + z*s, c + y*y*C, y*z*C - x*s], [z*x*C - y*s, z*y*C + x*s, c + z*z*C]]
def apply(R, p, v): return [sum(R[i][k] * v[k] for k in range(3)) + p[i] for i in range(3)]
def solve(A, b):
    n = len(b); M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for i in range(n):
        p = max(range(i, n), key=lambda r: abs(M[r][i])); M[i], M[p] = M[p], M[i]
        for r in range(i + 1, n):
            f = M[r][i] / M[i][i]
            for c in range(i, n + 1): M[r][c] -= f * M[i][c]
    x = [0.0] * n
    for i in reversed(range(n)): x[i] = (M[i][n] - sum(M[i][c] * x[c] for c in range(i + 1, n))) / M[i][i]
    return x

# ---------- kinematics from the URDF ----------
def load_chain():
    joints = {j.get("name"): j for j in ET.parse(HERE / "elrobot_follower.urdf").getroot().findall("joint")}
    child_of = {j.find("parent").get("link"): j for j in joints.values() if j.get("type") in ("fixed", "revolute")}
    chain, link = [], "base_link"
    while link in child_of:
        j = child_of[link]; xyz = [float(v) for v in j.find("origin").get("xyz").split()]
        if j.get("type") == "revolute":
            lim = j.find("limit"); chain.append(("rev", xyz, [float(v) for v in j.find("axis").get("xyz").split()], float(lim.get("lower")), float(lim.get("upper"))))
        else: chain.append(("fix", xyz))
        link = j.find("child").get("link")
        if link == "Gripper_Base_v1_1": break
    return chain
CHAIN = load_chain()
LIMITS = [(c[3], c[4]) for c in CHAIN if c[0] == "rev"]      # joints 1..7

def fk(q):
    """q: 7 joint angles (rad). Returns (tip position, R) with R the gripper-base orientation in base frame."""
    R = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]; p = [0.0, 0.0, 0.0]; i = 0
    for c in CHAIN:
        p = apply(R, p, c[1])
        if c[0] == "rev": R = matmul(R, rot(c[2], q[i])); i += 1
    return apply(R, p, TIP), R

def pose_err(q, tp, tR):
    p, R = fk(q)
    e = [tp[i] - p[i] for i in range(3)]
    # orientation error as rotation vector of tR * R^T
    E = matmul(tR, [[R[j][i] for j in range(3)] for i in range(3)])
    w = [(E[2][1] - E[1][2]) / 2, (E[0][2] - E[2][0]) / 2, (E[1][0] - E[0][1]) / 2]
    return e + [0.3 * x for x in w]          # orientation weighted lower than position

def ik(q0, tp, tR, iters=200):
    q = q0[:]
    for _ in range(iters):
        e = pose_err(q, tp, tR)
        if math.sqrt(sum(x * x for x in e[:3])) < 3e-4 and math.sqrt(sum(x * x for x in e[3:])) < 3e-3: break
        J = []
        for k in range(7):
            d = q[:]; d[k] += 1e-5; e2 = pose_err(d, tp, tR); J.append([(e[i] - e2[i]) / 1e-5 for i in range(6)])
        JJt = [[sum(J[k][i] * J[k][j] for k in range(7)) + (1e-4 if i == j else 0) for j in range(6)] for i in range(6)]
        y = solve(JJt, e)
        dq = [sum(J[k][i] * y[i] for i in range(6)) for k in range(7)]
        s = min(1.0, 0.15 / max(1e-9, max(abs(x) for x in dq)))
        q = [min(LIMITS[k][1], max(LIMITS[k][0], q[k] + s * dq[k])) for k in range(7)]
    return q, pose_err(q, tp, tR)

# ---------- servo <-> joint mapping (same as station-viewer: angle = upper - norm * (upper - lower)) ----------
def u16(b, a): return struct.unpack_from("<H", b, a)[0]
def steps_to_angle(m, k): lo, hi = LIMITS[k]; return hi - (m["pos"] - m["mn"]) / (m["mx"] - m["mn"]) * (hi - lo)
def angle_to_steps(m, k, a): lo, hi = LIMITS[k]; return round(m["mn"] + (hi - a) / (hi - lo) * (m["mx"] - m["mn"]))

class Arm:
    async def connect(self):
        logging.basicConfig(level=logging.ERROR)
        self.client = await new_station_client("localhost", logging.getLogger("arm"))
        self.q = asyncio.Queue(); self.client.follow("st3215/inference", self.q)
        return self
    async def state(self, fresh=True):
        if fresh:
            while not self.q.empty(): self.q.get_nowait()
        while True:
            e = await asyncio.wait_for(self.q.get(), 5)
            for bus in st3215.InferenceStateReader(memoryview(bytes(e.Data))).get_buses() or []:
                if str(bus.get_bus().get_serial_number()) != BUS: continue
                out = {}
                for m in bus.get_motors() or []:
                    s = bytes(m.get_state() or b"")
                    if len(s) < 0x47: continue
                    out[m.get_id()] = dict(pos=u16(s, 0x38) & 4095, goal=u16(s, 0x2A) & 4095, torque=s[0x28], cur=u16(s, 0x45) & 0x7FFF,
                                           mn=m.get_range_min(), mx=m.get_range_max(), frozen=m.get_range_freezed())
                if len(out) == 8: return out
    async def write(self, items):
        """items: list of (motor_id, address, bytes)"""
        cmds = [commands.DriverCommand(type=drivers.StationCommandType.STC_ST3215_COMMAND, body=st3215.Command(
            target_bus_serial=BUS, write=st3215.ST3215WriteCommand(motor_id=i, address=a, value=v)).encode()) for i, a, v in items]
        await send_commands(self.client, cmds)
    async def goto_steps(self, targets, wait=True, tol=25, timeout=8.0):
        """targets: {motor_id: steps}. Clamped to calibrated range. Returns final state."""
        st = await self.state(); items = []
        for i, t in targets.items():
            m = st[i]
            if not m["frozen"] or m["mx"] - m["mn"] < 300: raise SystemExit(f"motor {i} is not calibrated; refusing to move")
            t = int(min(m["mx"] - MARGIN, max(m["mn"] + MARGIN, t))); targets[i] = t
            items += [(i, 0x29, bytes([ACCEL])), (i, 0x2E, SPEED.to_bytes(2, "little")), (i, 0x28, b"\x01"), (i, 0x2A, t.to_bytes(2, "little"))]
        await self.write(items)
        t0 = time.time()
        while wait:
            await asyncio.sleep(0.15); st = await self.state()
            hot = {i: st[i]["cur"] for i in range(1, 8) if st[i]["cur"] > CUR_LIMIT}
            if hot:      # pushing against something: stop where we are
                await self.write([(i, 0x2A, st[i]["pos"].to_bytes(2, "little")) for i in range(1, 8)])
                raise SystemExit(f"STOPPED: motor current over {CUR_LIMIT} mA {hot}; holding current position")
            if all(abs(st[i]["pos"] - t) <= tol for i, t in targets.items()) or time.time() - t0 > timeout: break
        return st
    async def angles(self, st=None):
        st = st or await self.state(); return [steps_to_angle(st[k + 1], k) for k in range(7)], st
    async def goto_angles(self, q):
        _, st = await self.angles()
        return await self.goto_steps({k + 1: angle_to_steps(st[k + 1], k, q[k]) for k in range(7)})
    async def cartesian(self, delta, step=0.01):
        """Move the tip by delta (m, base frame) in straight sub-steps, holding orientation."""
        q, _ = await self.angles(); p0, R0 = fk(q)
        n = max(1, math.ceil(math.sqrt(sum(d * d for d in delta)) / step))
        for s in range(1, n + 1):
            tp = [p0[i] + delta[i] * s / n for i in range(3)]
            q2, err = ik(q, tp, R0)
            perr = math.sqrt(sum(x * x for x in err[:3]))
            if perr > 0.004: print(f"  IK stopped at step {s}/{n}: residual {perr*1000:.1f} mm (out of reach or at a joint limit)"); break
            if max(abs(a - b) for a, b in zip(q, q2)) > 0.5: print(f"  IK jump too large at step {s}/{n}; stopping"); break
            await self.goto_angles(q2); q = q2
        return await self.report()
    async def report(self):
        q, st = await self.angles(); p, R = fk(q)
        fwd = [R[i][1] for i in range(3)]          # gripper pointing direction (+y of gripper base)
        print("joint  deg    steps [min..max]  goal  cur(mA) torque")
        for k in range(7):
            m = st[k + 1]; print(f"  {k+1}  {math.degrees(q[k]):7.1f}  {m['pos']:5} [{m['mn']}..{m['mx']}] {m['goal']:5} {m['cur']:5}   {m['torque']}")
        g = st[8]; print(f"  8  grip {(g['pos']-g['mn'])/(g['mx']-g['mn']):.2f}  {g['pos']:5} [{g['mn']}..{g['mx']}] {g['goal']:5} {g['cur']:5}   {g['torque']}")
        print(f"tip xyz (m): {p[0]:+.3f} {p[1]:+.3f} {p[2]:+.3f}   pointing: {fwd[0]:+.2f} {fwd[1]:+.2f} {fwd[2]:+.2f}")
        if fwd[2] < -0.05:
            t = (TABLE_Z - p[2]) / fwd[2]; print(f"pointing ray hits table z={TABLE_Z} at x={p[0]+t*fwd[0]:+.3f} y={p[1]+t*fwd[1]:+.3f}, {t*100:.1f} cm ahead of the tip")
        return q, p, R

async def main(a):
    arm = await Arm().connect(); op = a[0] if a else "pose"
    if op == "pose": await arm.report()
    elif op == "hold":
        st = await arm.state(); await arm.goto_steps({i: st[i]["pos"] for i in range(1, 9)}, wait=False); await asyncio.sleep(0.5); await arm.report()
    elif op == "relax": await arm.write([(i, 0x28, b"\x00") for i in range(1, 9)]); print("torque off")
    elif op == "grip":
        st = await arm.state(); g = st[8]; await arm.goto_steps({8: g["mn"] + float(a[1]) * (g["mx"] - g["mn"])}, timeout=4); await arm.report()
    elif op == "joint":
        q, _ = await arm.angles(); k = int(a[1]) - 1; q[k] += math.radians(float(a[2])); await arm.goto_angles(q); await arm.report()
    elif op == "move": await arm.cartesian([float(x) for x in a[1:4]])
    elif op == "fwd":
        q, _ = await arm.angles(); _, R = fk(q); d = float(a[1]); await arm.cartesian([R[i][1] * d for i in range(3)])
    else: print(__doc__)

if __name__ == "__main__": asyncio.run(main(sys.argv[1:]))
