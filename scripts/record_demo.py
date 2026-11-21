"""Record a teleop demonstration: follower joint positions (10 Hz) + gripper/overview frames (2 Hz).
usage: python3 scripts/record_demo.py [seconds=120] [name=demo]   -> demos/<name>/states.jsonl + frames
Stop early with Ctrl-C or by creating demos/<name>/STOP."""
import asyncio, json, sys, time
from pathlib import Path
import arm, vision, station_thin
from target.gen_python.protobuf.drivers.usbvideo import usbvideo
OVERVIEW = "usbvideo/5cd55831e21692d4f05fcaec75c351a5"

async def frames(client, queue, out, tag, t0, stop):
    q = asyncio.Queue(maxsize=4); station_thin.follow_thin(client, queue, q, step=15); last = 0
    while not stop.is_set():
        try: e = await asyncio.wait_for(q.get(), 2)
        except asyncio.TimeoutError: continue
        if time.time() - last < 0.5: continue
        pack = usbvideo.RxEnvelopeReader(memoryview(bytes(e.Data))).get_frames()
        fr = (pack.get_frames_data() or []) if pack else []
        if fr: last = time.time(); (out / f"{tag}_{last - t0:07.2f}.jpg").write_bytes(bytes(fr[-1]))

async def main(secs, name):
    out = arm.HERE.parent / "demos" / name; out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("*"): f.unlink()
    a = await arm.Arm().connect(); t0 = time.time(); stop = asyncio.Event()
    tasks = [asyncio.create_task(frames(a.client, vision.GRIPPER_CAM, out, "grip", t0, stop)),
             asyncio.create_task(frames(a.client, OVERVIEW, out, "over", t0, stop))]
    n = 0
    with open(out / "states.jsonl", "w") as f:
        while time.time() - t0 < secs and not (out / "STOP").exists():
            st = await a.state(fresh=True)
            f.write(json.dumps({"t": round(time.time() - t0, 2), **{str(i): [st[i]["pos"], st[i]["goal"], st[i]["cur"]] for i in range(1, 9)}}) + "\n"); f.flush(); n += 1
            await asyncio.sleep(0.1)
    stop.set(); await asyncio.gather(*tasks, return_exceptions=True)
    print(f"recorded {n} samples over {time.time() - t0:.0f}s -> {out}")

asyncio.run(main(float(sys.argv[1]) if len(sys.argv) > 1 else 120, sys.argv[2] if len(sys.argv) > 2 else "demo"))
