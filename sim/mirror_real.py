"""Live mirror: the twin's joints follow the real follower arm (read-only), shown next to the real cameras.

  python3 sim/mirror_real.py [--layout home] [--port 8892] [--fps 15] [--viewer] [--seconds N]

Follows Station's st3215/inference (raw ticks -> URDF radians via sim/real_joints.py; mapping configurable with
layout["arm"]["joint_mapping"] or TWIN_JOINT_MAP) and both camera queues. Serves http://127.0.0.1:<port>/ with a live
2x2 view (real overview | twin overview / real wrist | twin wrist) and /state.json (q, raw ticks, latency).
--viewer also opens the MuJoCo viewer (on macOS run it with `mjpython` instead of python3).
Latency = time the twin frame is ready minus the bus read time stamped by Station (system_stamp_ns).
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
<img src="/stream" style="width:100%;max-width:1196px;display:block"><pre id=s></pre>
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
    import station_io as s
    _, usbvideo, fr, _ = s._mods()
    q: asyncio.Queue = asyncio.Queue()
    cli.follow(s.QUEUES[cam], q)
    while True:
        e = await q.get()
        env = usbvideo.RxEnvelopeReader(memoryview(bytes(e.Data)))
        pack = env.get_frames()
        if env.get_type() == usbvideo.RxEnvelopeType.ET_FRAMES and pack is not None:
            frames = pack.get_frames_data() or []
            if pack.get_format().get_kind() == fr.FrameFormatKind.FF_JPEG and frames:
                latest[cam] = bytes(frames[-1])


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
    lat, n, t0, period = [], 0, time.time(), 1.0 / a.fps
    blank = np.zeros((H, W, 3), np.uint8)
    while a.seconds is None or time.time() - t0 < a.seconds:
        t_loop = time.time()
        raw, meta = await jf.read()
        q = real_joints.to_q(raw, meta["ranges"], lo, hi, mapping)
        real_joints.set_arm_q(model, data, q)
        tiles = []
        for name, cam in compare_real.CAMS.items():
            renderer.update_scene(data, camera=cam, scene_option=opt)
            twin = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
            real = cv2.imdecode(np.frombuffer(latest[name], np.uint8), cv2.IMREAD_COLOR) if name in latest else blank
            tiles.append(np.hstack([cv2.resize(real, (W, H)), twin]))
        grid = cv2.resize(np.vstack(tiles), None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        t_ready = time.time()
        latency = (t_ready - meta["stamp_s"]) if meta.get("stamp_s") else None
        if latency is not None:
            lat.append(latency)
        cv2.putText(grid, f"real | twin   q={np.round(q, 2).tolist()}   latency {latency * 1000:.0f} ms" if latency is not None
                    else "real | twin", (8, 18), 0, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        ok, jpg = cv2.imencode(".jpg", grid, [cv2.IMWRITE_JPEG_QUALITY, 80])
        n += 1
        with STATE["lock"]:
            STATE["jpeg"] = jpg.tobytes()
            STATE["state"] = {"q_rad": np.round(q, 4).tolist(), "raw": raw, "mapping": mapping, "frames": n,
                              "fps": round(n / max(time.time() - t0, 1e-6), 1),
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
