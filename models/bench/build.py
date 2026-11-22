"""Build the labelled benchmark set from recorded data (read-only sources) into models/bench/data/.

    uv run --no-project --with opencv-python-headless --with numpy python -m models.bench.build

Four item kinds (items.jsonl, one JSON object per line; images under data/frames/, gitignored):

  markers   new scene (22 Nov), recorder sessions: the ArUco marker centres. Pixel truth from OpenCV
            ArUco detection; world truth (mm) from perception's scene_calib.json for the calibrated ids.
            One marker hidden under a cable is not detected; its centre is hand-placed (flagged).
  objects   21 Nov real-run overview frames: purple wrapper, black object, white box, grey box. Boxes
            hand-labelled from the frames' pixel grid (598x448, about +-5 px); absent objects listed so
            hallucinated points count as false positives.
  grasp     21 Nov real runs: wrist + overview images returned by pick_at after the lift. Truth = the
            gripper's own hold test (jaw position + load), not a person's opinion.
  stain     synthetic: a brown stain painted onto a real overview frame of the new table, in three states
            (full, partly wiped, clean). Truth by construction: residual box and table area via calibration.
"""
from __future__ import annotations

import base64
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]                                          # the worktree
WS = ROOT.parent
DATA = HERE / "data"
FRAMES = DATA / "frames"
RUNS = ROOT / "sim" / "real_runs" / "2026-11-21"
RECORDINGS = WS / "recorder" / "recordings"
sys.path.insert(0, str(ROOT))
from sites import get_site
CAMS = {role: get_site().camera_hash(role)[:8] for role in ("overview", "wrist")}

sys.path.insert(0, str(ROOT))
from models import perception_bridge as pb  # noqa: E402

# Hand labels on the 598x448 grid frames the agent saw (u0, v0, u1, v1), see module docstring.
OBJECT_LABELS = {
    "run1_frames/002_overview.jpg": {"purple wrapper": [242, 147, 266, 172], "black object": [336, 152, 381, 193],
                                     "white box": [65, 55, 222, 195], "grey box": [385, 73, 525, 166]},
    "run3_frames/002_overview.jpg": {"purple wrapper": [308, 150, 332, 175], "black object": [334, 152, 379, 193],
                                     "white box": [65, 55, 195, 192], "grey box": [385, 48, 530, 168]},
    "run4_frames/002_overview.jpg": {"purple wrapper": [246, 145, 268, 172], "black object": [334, 145, 366, 190],
                                     "white box": [88, 48, 225, 192], "grey box": [378, 50, 525, 172]},
    "run4_frames/018_overview.jpg": {"purple wrapper": [247, 146, 268, 172], "black object": None,
                                     "white box": [88, 48, 225, 192], "grey box": [378, 50, 525, 172]},
}
HIDDEN_MARKER_PX = {"overview": {21: [267.0, 157.0]}}             # under the cable, hand-placed, +-4 px


def _write(name: str, data: bytes) -> str:
    FRAMES.mkdir(parents=True, exist_ok=True)
    (FRAMES / name).write_bytes(data)
    return name


def marker_items() -> list[dict]:
    sys.path.insert(0, str(pb.PERCEPTION_ROOT))
    from perception import aruco
    loc = pb.locator()
    world = {i: T[:3, 3].tolist() for i, T in loc.sc.markers.items()} if loc else {}
    items = []
    for rec in sorted(RECORDINGS.glob("2026*")):
        for cam, h8 in CAMS.items():
            files = sorted((rec / "cams" / h8).glob("*.jpg"), key=lambda p: int(p.stem))
            if not files:
                continue
            f = files[len(files) // 2]
            img = cv2.imread(str(f))
            det = {int(i): c.mean(0).round(2).tolist() for i, c in aruco.detect(img, "DICT_4X4_50").items()}
            gt = [{"id": i, "px": px, "source": "aruco", "xyz_world": world.get(i) if cam == "overview" else None}
                  for i, px in sorted(det.items())]
            gt += [{"id": i, "px": px, "source": "hand", "xyz_world": world.get(i) if cam == "overview" else None}
                   for i, px in HIDDEN_MARKER_PX.get(cam, {}).items() if i not in det]
            if not gt:
                continue
            name = _write(f"markers_{rec.name}_{cam}.jpg", f.read_bytes())
            items.append({"id": f"markers/{rec.name}/{cam}", "kind": "markers", "camera": cam, "images": [name],
                          "source": f"recording:{rec.name}@{f.stem}", "size": [img.shape[1], img.shape[0]],
                          "targets": ["the centre of each black-and-white square ArUco fiducial marker"],
                          "gt": {"points": gt}})
    return items


def object_items() -> list[dict]:
    items = []
    for rel, labels in OBJECT_LABELS.items():
        f = RUNS / rel
        name = _write("objects_" + rel.replace("/", "_"), f.read_bytes())
        img = cv2.imread(str(f))
        items.append({"id": f"objects/{rel}", "kind": "objects", "camera": "overview", "images": [name],
                      "source": f"sim/real_runs/2026-11-21/{rel}", "size": [img.shape[1], img.shape[0]],
                      "targets": list(labels), "gt": {"boxes": labels, "label_source": "hand, +-5 px"}})
    return items


def _tool_results(path: Path):
    tools = {}
    for line in path.read_text().splitlines():
        c = (json.loads(line).get("message") or {}).get("content")
        for b in c if isinstance(c, list) else []:
            if b.get("type") == "tool_use":
                tools[b["id"]] = (b["name"].split("__")[-1], b["input"])
            elif b.get("type") == "tool_result":
                name, args = tools.get(b["tool_use_id"], ("?", {}))
                cc = b["content"] if isinstance(b.get("content"), list) else []
                yield name, args, "".join(x.get("text", "") for x in cc if x.get("type") == "text"), \
                    [x for x in cc if x.get("type") == "image"]


def grasp_items() -> list[dict]:
    items = []
    for f in sorted(RUNS.glob("*run*.jsonl")):
        k = 0
        for name, args, text, imgs in _tool_results(f):
            if name != "pick_at" or len(imgs) != 2 or not (text.startswith("HOLDING") or text.startswith("NOT holding")):
                continue
            k += 1
            held = text.startswith("HOLDING")
            names = [_write(f"grasp_{f.stem}_{k}_{role}.jpg", base64.b64decode(im["source"]["data"]))
                     for role, im in zip(("wrist", "overview"), imgs)]
            items.append({"id": f"grasp/{f.stem}/{k}", "kind": "grasp", "camera": "wrist", "images": names,
                          "roles": ["wrist", "overview"], "source": f"sim/real_runs/2026-11-21/{f.name} pick_at #{k}",
                          "size": list(cv2.imdecode(np.frombuffer((FRAMES / names[0]).read_bytes(), np.uint8), 1).shape[1::-1]),
                          "gt": {"held": held, "sensor": text.split("\n")[0][:90], "pick_args": args}})
    return items


def _stain_mask(shape, centre, r, seed) -> np.ndarray:
    rng = np.random.default_rng(seed)
    m = np.zeros(shape[:2], np.float32)
    for _ in range(7):
        c = (int(centre[0] + rng.normal(0, r * 0.35)), int(centre[1] + rng.normal(0, r * 0.25)))
        ax = (max(3, int(r * rng.uniform(0.4, 0.8))), max(2, int(r * rng.uniform(0.25, 0.5))))
        cv2.ellipse(m, c, ax, float(rng.uniform(0, 180)), 0, 360, 1.0, -1)
    return cv2.GaussianBlur(m, (5, 5), 1.5)


def _box(mask: np.ndarray, thr: float = 0.3):
    ys, xs = np.nonzero(mask > thr)
    return None if not len(xs) else [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def _box_world(box) -> dict | None:
    ws = [pb.pixel_to_world("overview", u, v) for u, v in ((box[0], box[1]), (box[2], box[1]), (box[2], box[3]), (box[0], box[3]))]
    if not all(w.get("ok") for w in ws):
        return None
    pts = [w["xyz"][:2] for w in ws]
    area = abs(sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(pts, pts[1:] + pts[:1]))) / 2
    c = pb.pixel_to_world("overview", (box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
    return {"centre_xyz": c["xyz"], "box_area_cm2": round(area * 1e4, 1)}


def stain_items() -> list[dict]:
    recs = sorted(RECORDINGS.glob("2026*"))
    if not recs:
        return []
    src = sorted((recs[-1] / "cams" / CAMS["overview"]).glob("*.jpg"), key=lambda p: int(p.stem))
    base = cv2.imread(str(src[len(src) // 3]))
    colour = np.array([35, 62, 105], np.float32)                 # BGR coffee brown
    items = []
    for k, (centre, r) in enumerate([((150, 128), 18), ((112, 165), 15), ((200, 105), 13)]):
        full = _stain_mask(base.shape, centre, r, seed=k)
        wiped = full.copy()
        top = _box(full)
        cut = int(top[1] + 0.65 * (top[3] - top[1]))             # the wipe cleaned the upper 65 %
        wiped[:cut] = 0
        for state, m in (("full", full), ("partial", wiped), ("clean", np.zeros_like(full))):
            img = (base * (1 - 0.75 * m[..., None]) + colour * 0.75 * m[..., None]).clip(0, 255).astype(np.uint8)
            ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            name = _write(f"stain_{k}_{state}.jpg", jpg.tobytes())
            box = _box(m)
            items.append({"id": f"stain/{k}/{state}", "kind": "stain", "camera": "overview", "images": [name],
                          "source": f"synthetic on recording:{recs[-1].name}@{src[len(src) // 3].stem}",
                          "size": [img.shape[1], img.shape[0]],
                          "gt": {"done": state == "clean", "state": state, "residual_box": box,
                                 "residual_px": int((m > 0.3).sum()),
                                 "world": _box_world(box) if box else None}})
    return items


def main():
    items = marker_items() + object_items() + grasp_items() + stain_items()
    DATA.mkdir(parents=True, exist_ok=True)
    with open(DATA / "items.jsonl", "w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    counts = {}
    for it in items:
        counts[it["kind"]] = counts.get(it["kind"], 0) + 1
    print(json.dumps({"items": len(items), "by_kind": counts, "frames_kb": round(sum(p.stat().st_size for p in FRAMES.glob("*")) / 1e3)}))
    if pb.locator() is None:
        print("note: perception unavailable, no mm truth:", pb.last_error())


if __name__ == "__main__":
    main()
