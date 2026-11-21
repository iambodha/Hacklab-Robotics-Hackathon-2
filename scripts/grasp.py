"""Camera-guided joint moves for the follower arm (no kinematic model: measures image response directly).
  python3 scripts/grasp.py probe J DEG [J DEG ...]   move joint(s), report how the object moved in the gripper camera, move back
  python3 scripts/grasp.py step J DEG [J DEG ...]    move joint(s) and stay; report the object's new position
  python3 scripts/grasp.py goto J STEPS [J STEPS ...] move joint(s) to absolute servo positions (closed loop)
"""
import asyncio, math, sys
import arm, vision

async def settle(a, timeout=6.0):
    prev, same, t = None, 0, 0.0
    while t < timeout:
        await asyncio.sleep(0.2); t += 0.2; st = await a.state()
        cur = [st[i]["pos"] for i in range(1, 9)]
        same = same + 1 if prev and max(abs(x - y) for x, y in zip(cur, prev)) <= 2 else 0
        if same >= 3: return st
        prev = cur
    return st

async def move(a, deltas, tol=4, timeout=7.0):
    """deltas: {joint: degrees}, in servo direction (4096 steps = 360 deg).
    The servos run P-only with a low gain, so a plain goal write stops short of the target.
    This closes the loop: it keeps nudging the commanded goal until the measured position arrives."""
    st = await a.state()
    target = {}
    for j, d in deltas.items():
        m = st[j]; t = m["pos"] + round(d * 4096 / 360)
        target[j] = int(min(m["mx"] - arm.MARGIN, max(m["mn"] + arm.MARGIN, t)))
    cmd = dict(target); await a.goto_steps(dict(cmd), wait=False); t = 0.0; ok = 0
    while t < timeout:
        await asyncio.sleep(0.25); t += 0.25; st = await a.state()
        hot = {i: st[i]["cur"] for i in range(1, 8) if st[i]["cur"] > arm.CUR_LIMIT}
        if hot:
            await a.write([(i, 0x2A, st[i]["pos"].to_bytes(2, "little")) for i in range(1, 8)])
            raise SystemExit(f"STOPPED on contact: current {hot} mA; holding position")
        err = {j: target[j] - st[j]["pos"] for j in target}
        ok = ok + 1 if all(abs(e) <= tol for e in err.values()) else 0
        if ok >= 2: break
        for j, e in err.items():
            if abs(e) > tol:
                cmd[j] = int(max(target[j] - 400, min(target[j] + 400, cmd[j] + round(0.5 * e))))
                cmd[j] = int(min(st[j]["mx"] - arm.MARGIN, max(st[j]["mn"] + arm.MARGIN, cmd[j])))
        await a.write([(j, 0x2A, c.to_bytes(2, "little")) for j, c in cmd.items()])
    left = {j: target[j] - st[j]["pos"] for j in target if abs(target[j] - st[j]["pos"]) > tol}
    if left: print(f"  note: did not fully arrive, remaining steps {left}")
    return st

def show(tag, b, st=None):
    cur = " cur(mA) " + " ".join(f"{i}:{st[i]['cur']}" for i in range(1, 8)) if st else ""
    print(f"{tag}: " + (f"object ({b['x']:.0f},{b['y']:.0f}) area {b['area']}" if b else "object NOT FOUND") + cur)

async def main(argv):
    a = await arm.Arm().connect(); op = argv[0]
    deltas = {int(argv[i]): float(argv[i + 1]) for i in range(1, len(argv), 2)}
    if op == "goto":
        st = await a.state(); deltas = {j: (t - st[j]["pos"]) * 360 / 4096 for j, t in deltas.items()}
    b0 = await vision.find(a.client); show("before", b0)
    st = await move(a, deltas); await asyncio.sleep(0.3)
    b1 = await vision.find(a.client, near=(b0["x"], b0["y"]) if b0 else None); show("after ", b1, st)
    if b0 and b1: print(f"shift: dx {b1['x']-b0['x']:+.0f}px dy {b1['y']-b0['y']:+.0f}px area x{b1['area']/b0['area']:.2f}")
    if op == "probe":
        await move(a, {j: -d for j, d in deltas.items()}); await asyncio.sleep(0.3)
        show("back  ", await vision.find(a.client, near=(b0["x"], b0["y"]) if b0 else None))

asyncio.run(main(sys.argv[1:]))
