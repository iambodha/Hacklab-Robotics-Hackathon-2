"""Find the dark object in the gripper camera frame (pure stdlib + macOS `sips` for JPEG decoding).
  python3 scripts/vision.py            -> grabs a fresh gripper frame, prints blob centre (px), area, bbox
Used by grasp.py for camera-guided moves."""
import os
import asyncio, hashlib, logging, struct, subprocess, sys, tempfile
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent / "norma-core"
sys.path.insert(0, str(ROOT / "software/station/shared")); sys.path.insert(0, str(ROOT))
from target.gen_python.protobuf.drivers.usbvideo import usbvideo

GRIPPER_CAM = "usbvideo/e091eeda8dcda172482f039057691cdb"
FRAMES = HERE.parent / "frames"

def decode(jpeg):
    """JPEG bytes -> (w, h, rows of luminance 0..255)"""
    with tempfile.TemporaryDirectory() as d:
        src, dst = Path(d) / "a.jpg", Path(d) / "a.bmp"; src.write_bytes(jpeg)
        subprocess.run(["sips", "-s", "format", "bmp", str(src), "--out", str(dst)], check=True, capture_output=True)
        b = dst.read_bytes()
    off, = struct.unpack_from("<I", b, 10); w, h = struct.unpack_from("<ii", b, 18); bpp, = struct.unpack_from("<H", b, 28)
    n = bpp // 8; stride = (w * n + 3) & ~3; flip = h > 0; h = abs(h); rows = []
    for y in range(h):
        r = off + (h - 1 - y if flip else y) * stride
        rows.append([(b[r + x * n] * 29 + b[r + x * n + 1] * 150 + b[r + x * n + 2] * 77) >> 8 for x in range(w)])
    return w, h, rows

def blobs(w, h, lum, thresh=75, ymax=None):
    """Connected dark regions above the jaws. Returns list of dicts sorted by area."""
    ymax = ymax or int(h * 0.85); seen = [[False] * w for _ in range(h)]; out = []
    for y0 in range(ymax):
        for x0 in range(w):
            if seen[y0][x0] or lum[y0][x0] >= thresh: continue
            stack = [(x0, y0)]; seen[y0][x0] = True; px = []
            while stack:
                x, y = stack.pop(); px.append((x, y))
                for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                    if 0 <= nx < w and 0 <= ny < h and not seen[ny][nx] and lum[ny][nx] < thresh:
                        seen[ny][nx] = True; stack.append((nx, ny))
            xs = [p[0] for p in px]; ys = [p[1] for p in px]
            touches_bottom = max(ys) >= h - 2          # that's the gripper itself
            if len(px) >= 25 and not touches_bottom:
                out.append(dict(x=sum(xs) / len(px), y=sum(ys) / len(px), area=len(px), box=(min(xs), min(ys), max(xs), max(ys))))
    return sorted(out, key=lambda b: -b["area"])

async def grab(client, save=True):
    q = asyncio.Queue(); client.follow(GRIPPER_CAM, q)
    for _ in range(300):
        e = await asyncio.wait_for(q.get(), 5)
        pack = usbvideo.RxEnvelopeReader(memoryview(bytes(e.Data))).get_frames()
        fr = (pack.get_frames_data() or []) if pack else []
        if fr:
            jpeg = bytes(fr[-1])
            if save: FRAMES.mkdir(exist_ok=True); (FRAMES / "e091eeda.jpg").write_bytes(jpeg)
            return jpeg
    raise RuntimeError("no frame from gripper camera")

async def find(client, near=None):
    """Blob nearest to `near` (x, y), or the largest central one."""
    w, h, lum = decode(await grab(client)); bs = blobs(w, h, lum)
    if not bs: return None
    if near: return min(bs, key=lambda b: (b["x"] - near[0]) ** 2 + (b["y"] - near[1]) ** 2)
    return min(bs, key=lambda b: abs(b["x"] - w / 2) + abs(b["y"] - h / 2) - 0.2 * b["area"])

if __name__ == "__main__":
    from station_py import new_station_client
    async def main():
        logging.basicConfig(level=logging.ERROR)
        c = await new_station_client(os.environ.get("STATION_HOST", "localhost"), logging.getLogger("vision"))
        w, h, lum = decode(await grab(c)); print(f"frame {w}x{h}")
        for b in blobs(w, h, lum): print(f"  blob centre ({b['x']:.0f},{b['y']:.0f}) area {b['area']} box {b['box']}")
    asyncio.run(main())
