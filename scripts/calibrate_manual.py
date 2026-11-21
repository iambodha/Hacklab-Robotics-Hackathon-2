"""Guided manual calibration of one arm (the procedure that fixed the leader on 21 Nov).
  1. reset the bus's ranges (retried until all motors are unfrozen)
  2. you sweep every joint by hand to both ends, then press Enter
  3. freeze, verify, and print the spans next to the other arm's

usage: python3 scripts/calibrate_manual.py [BUS_SERIAL]     (default: leader 5B61035034)
Stop teleop first:  python3 scripts/teleop.py stop
"""
import subprocess, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
LEADER, FOLLOWER = "5B61035034", "5B61033301"

def state():
    out = subprocess.run([sys.executable, str(HERE / "arm_state.py"), "1"], capture_output=True, text=True).stdout
    buses, cur = {}, None
    for line in out.splitlines():
        p = line.split()
        if line.startswith("BUS"): cur = p[1]; buses[cur] = {}
        elif len(p) >= 10 and p[0].isdigit():
            buses[cur][int(p[0])] = dict(pos=int(p[1]), mn=int(p[7]), mx=int(p[8]), frozen=p[9] == "True")
    return buses

def span(m): return m["mx"] - m["mn"] if m["mx"] >= m["mn"] else 4096 - m["mn"] + m["mx"]
def cal(op, bus): subprocess.run([sys.executable, str(HERE / "calibrate.py"), op, bus], check=True, stdout=subprocess.DEVNULL)

def main(bus):
    other = FOLLOWER if bus == LEADER else LEADER
    if input(f"Reset calibration of bus {bus} and recalibrate by hand? Teleop must be off. [y/N] ").lower() != "y": return
    for attempt in range(1, 6):                      # reset can half-apply on bus timeouts
        cal("reset", bus); time.sleep(3)
        motors = state().get(bus, {})
        if motors and not any(m["frozen"] for m in motors.values()): break
        print(f"  reset attempt {attempt}: some motors still frozen, retrying")
    else:
        sys.exit("reset did not apply to all motors; check station-run/station.log")
    print("Ranges cleared. Sweep EVERY joint slowly to both ends (gripper too).")
    print("Stop where the other arm's joint would physically stop.")
    input("Press Enter when done... ")
    before = state()[bus]
    small = [i for i, m in sorted(before.items()) if span(m) < 300]
    if small and input(f"Joints {small} moved less than 300 steps. Freeze anyway? [y/N] ").lower() != "y":
        sys.exit("not frozen; sweep those joints and rerun (ranges are still recording)")
    cal("freeze", bus); time.sleep(8)
    s = state(); mine, theirs = s[bus], s.get(other, {})
    print(f"\njoint | {bus} min-max span frozen | {other} span")
    for i in sorted(mine):
        m, t = mine[i], theirs.get(i)
        print(f"{i:5} | {m['mn']:>5}-{m['mx']:<5} {span(m):>5} {str(m['frozen']):>6} | {span(t) if t else '-':>5}")
    if not all(m["frozen"] for m in mine.values()): print("WARNING: not all motors frozen; run calibrate.py freeze again")
    print("\nRestart teleop with: python3 scripts/teleop.py start")

main(sys.argv[1] if len(sys.argv) > 1 else LEADER)
