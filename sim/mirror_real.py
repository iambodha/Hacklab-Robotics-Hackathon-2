"""Live mirror: the twin's joints follow the real follower arm (read-only), shown next to the real cameras.

  python3 sim/mirror_real.py [--layout home] [--port 8892] [--fps 15] [--q-tol 0.002] [--viewer] [--seconds N]

Follows Station's st3215/inference (raw ticks -> URDF radians via sim/real_joints.py; mapping configurable with
layout["arm"]["joint_mapping"] or TWIN_JOINT_MAP) and both camera queues. Serves http://127.0.0.1:<port>/ with a live
2x2 view (real overview | twin overview / real wrist | twin wrist) and /state.json (q, raw ticks, latency).
--viewer also opens the MuJoCo viewer (on macOS run it with `mjpython` instead of python3).
Latency = time the twin frame is ready minus the bus read time stamped by Station (system_stamp_ns).
Cheap enough to run next to the console: the twin re-renders only when a joint moved more than --q-tol (rad; -1 =
every frame), camera JPEGs are decoded only when new, and the 2x2 grid is sent at native size (the page scales it).
Measured 22 Nov (load avg ~4 from other jobs): still arm 14.9 fps, 6.5 ms median, 11 % of one core; rendering every
frame 14.9 fps, 25 ms, 25 %; --fps 30 --q-tol -1 28.8 fps, 27 ms, 41 %.
Nothing is ever written to Station: no torque, goals or calibration.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_scene  # noqa: E402  (sets MUJOCO_GL)
import compare_real  # noqa: E402
import mujoco  # noqa: E402
import real_joints  # noqa: E402
import station_io  # noqa: E402

STATE = {"jpeg": b"", "state": {}, "lock": threading.Lock()}
PAGE = b"""<!doctype html><title>twin mirror</title><body style="margin:0;background:#111;color:#ddd;font:13px monospace">
<img src="/stream" style="width:100%;max-width:1196px;aspect-ratio:598/448;display:block"><pre id=s></pre>
<script>setInterval(async()=>{const r=await fetch('/state.json');document.getElementById('s').textContent=JSON.stringify(await r.json(),null,1)},500)</script>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path == "/state.json":
            with STATE["lock"]:
                body = json.dumps(STATE["state"]).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers()
            self.wfile.write(body)
        elif self.path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame"); self.end_headers()
            last = None
            try:
                while True:
                    with STATE["lock"]:
                        jpg = STATE["jpeg"]
                    if jpg and jpg is not last:
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"); last = jpg
                    time.sleep(0.03)
            except (BrokenPipeError, ConnectionResetError):
                pass
        elif self.path == "/snapshot.jpg":
            with STATE["lock"]:
                jpg = STATE["jpeg"]
            self.send_response(200); self.send_header("Content-Type", "image/jpeg"); self.end_headers(); self.wfile.write(jpg)
        else:
            self.send_response(200); self.send_header("Content-Type", "text/html"); self.end_headers(); self.wfile.write(PAGE)


async def follow_camera(cli, cam, latest):
    """One subscription per camera for the whole run (perception.latest.Latest); keeps only the newest JPEG
    (bytes, sequence number). A camera that stops just stops updating; the loop keeps waiting for it."""
    sub = station_io.Latest.of(cli, station_io.QUEUES[cam])
    seq = 0
    while True:
        try:
            e = await sub.get(fresh=True, timeout=5.0)
        except TimeoutError:
            continue
        jpg = station_io.jpeg_of(e)                     # parse only the newest envelope
        if jpg is not None:
            seq += 1
            latest[cam] = (jpg, seq)


async def run(a):
    layout_file = build_scene.layout_path(a.layout)
    model, layout = compare_real.load_model(layout_file)
    data = mujoco.MjData(model)
    W, H = layout["image_size"]
    mapping = real_joints.load_mapping(layout)
    lo, hi = model.actuator_ctrlrange.T
    opt = mujoco.MjvOption(); opt.geomgroup[1] = 0
    renderer = mujoco.Renderer(model, H, W)
    viewer = None
    if a.viewer:
        from mujoco import viewer as mjviewer
        viewer = mjviewer.launch_passive(model, data, show_left_ui=False)
    cli = await station_io.connect()
    jf = station_io.JointFollower(cli)
    latest: dict = {}
    for cam in ("overview", "wrist"):
        asyncio.ensure_future(follow_camera(cli, cam, latest))
    lat, n, renders, t0, period = [], 0, 0, time.time(), 1.0 / a.fps
    cpu0 = time.process_time()
    blank = np.zeros((H, W, 3), np.uint8)
    twin = {name: blank for name in compare_real.CAMS}
    real = {name: blank for name in compare_real.CAMS}
    seen_seq, q_drawn, partial = {}, None, 0
    while a.seconds is None or time.time() - t0 < a.seconds:
        t_loop = time.time()
        raw, meta = await jf.read()
        if any(m not in raw for m in range(1, len(lo) + 1)):          # partial bus read: skip, keep the last pose
            partial += 1
            with STATE["lock"]:
                STATE["state"] = {**STATE["state"], "partial_frames_skipped": partial}
            continue
        q = real_joints.to_q(raw, meta["ranges"], lo, hi, mapping)
        dirty = False
        if q_drawn is None or np.abs(q - q_drawn).max() > a.q_tol:     # a still arm costs no renders
            real_joints.set_arm_q(model, data, q)
            for name, cam in compare_real.CAMS.items():
                renderer.update_scene(data, camera=cam, scene_option=opt)
                twin[name] = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
            q_drawn, dirty, renders = q, True, renders + 1
        for name in compare_real.CAMS:                                  # decode only frames not shown yet
            if name in latest and latest[name][1] != seen_seq.get(name):
                jpg_b, seen_seq[name] = latest[name]
                im = cv2.imdecode(np.frombuffer(jpg_b, np.uint8), cv2.IMREAD_COLOR)
                real[name] = im if im.shape[:2] == (H, W) else cv2.resize(im, (W, H))
                dirty = True
        t_ready = time.time()
        latency = (t_ready - meta["stamp_s"]) if meta.get("stamp_s") else None
        if latency is not None:
            lat.append(latency)
        n += 1
        if dirty:            # native resolution; the page scales it up (2x server-side resize + bigger JPEG were ~3x the encode)
            grid = np.vstack([np.hstack([real[name], twin[name]]) for name in compare_real.CAMS])
            cv2.putText(grid, f"real | twin   latency {latency * 1000:.0f} ms" if latency is not None else "real | twin",
                        (4, 12), 0, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
            ok, jpg = cv2.imencode(".jpg", grid, [cv2.IMWRITE_JPEG_QUALITY, 85])
            with STATE["lock"]:
                STATE["jpeg"] = jpg.tobytes()
        el = max(time.time() - t0, 1e-6)
        with STATE["lock"]:
            STATE["state"] = {"q_rad": np.round(q, 4).tolist(), "raw": raw, "mapping": mapping, "frames": n,
                              "renders": renders, "fps": round(n / el, 1), "partial_frames_skipped": partial,
                              "cpu_pct": round(100 * (time.process_time() - cpu0) / el, 1),
                              "cameras_live": sorted(latest),
                              "latency_ms": round(latency * 1000, 1) if latency is not None else None,
                              "latency_ms_median": round(float(np.median(lat[-100:])) * 1000, 1) if lat else None,
                              "receive_to_ready_ms": round((t_ready - meta["t_recv"]) * 1000, 1)}
        if viewer is not None:
            if not viewer.is_running():
                break
            viewer.sync()
        await asyncio.sleep(max(0.0, period - (time.time() - t_loop)))
    return STATE["state"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layout", default="home")
    ap.add_argument("--port", type=int, default=8892)
    ap.add_argument("--fps", type=float, default=15)
    ap.add_argument("--q-tol", type=float, default=2e-3, help="re-render the twin only when a joint moved more (rad)")
    ap.add_argument("--viewer", action="store_true")
    ap.add_argument("--seconds", type=float, help="stop after N seconds and print the last state (for tests)")
    a = ap.parse_args()
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"twin mirror on http://127.0.0.1:{a.port}/  (read-only; Ctrl-C to stop)", flush=True)
    try:
        print(json.dumps(asyncio.run(run(a))))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
