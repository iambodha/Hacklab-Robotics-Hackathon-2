"""Printable A4 kit sheet from the spec (kit_a4.json): ArUco markers, labelled target cells, start-spot outlines, a
100 mm scale bar. One drawing routine makes both the print (PDF + PNG at the spec's dpi, 1:1) and the twin's texture
(sim/kitting_scene.py), so the twin sees exactly what is printed.

    python -m tasks.kitting.sheet [--spec tasks/kitting/kit_a4.json] [--out DIR]   # -> kit_a4_v1.pdf / .png

Print at 100 % ("actual size", no fit-to-page), then measure the scale bar: it must be 100 mm +- 1 mm.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .spec import Pose2D, Spec

MM_PER_IN = 25.4
LINE_MM = 0.6            # cell outline width
DASH_MM = (3.0, 2.0)     # start-spot outline dash / gap


def _font(px: int):
    try:
        return ImageFont.load_default(size=px)
    except TypeError:                     # Pillow < 10.1
        return ImageFont.load_default()


class Canvas:
    """Sheet-frame mm -> image pixels (row 0 = the top edge of the page)."""

    def __init__(self, spec: Spec, ppm: float):
        self.spec, self.ppm = spec, ppm
        self.W, self.H = spec.paper_mm
        self.size = (int(round(self.W * ppm)), int(round(self.H * ppm)))
        self.img = Image.new("L", self.size, 255)
        self.draw = ImageDraw.Draw(self.img)

    def px(self, pts) -> list[tuple[float, float]]:
        pts = np.atleast_2d(pts)
        return [((x + self.W / 2) * self.ppm, (self.H / 2 - y) * self.ppm) for x, y in pts]

    def text(self, xy_mm, s, size_mm, anchor="la"):
        self.draw.text(self.px(xy_mm)[0], s, fill=0, font=_font(max(6, int(size_mm * self.ppm))), anchor=anchor)

    def polygon(self, pts_mm, width_mm=LINE_MM, dashed=False):
        p = self.px(np.vstack([pts_mm, pts_mm[:1]]))
        w = max(1, int(round(width_mm * self.ppm)))
        if not dashed:
            self.draw.line(p, fill=0, width=w, joint="curve")
            return
        on, off = (d * self.ppm for d in DASH_MM)
        distance = 0.0
        period = on + off
        for a, b in zip(p[:-1], p[1:]):
            length = math.dist(a, b)
            if length == 0:
                continue
            # Integer dash indices avoid a floating-point zero-step loop at dash boundaries.
            for j in range(math.floor(distance / period), math.ceil((distance + length) / period)):
                lo = max(distance, j * period)
                hi = min(distance + length, j * period + on)
                if hi > lo:
                    t0, t1 = (lo-distance)/length, (hi-distance)/length
                    self.draw.line([(a[0]+(b[0]-a[0])*t0, a[1]+(b[1]-a[1])*t0),
                                    (a[0]+(b[0]-a[0])*t1, a[1]+(b[1]-a[1])*t1)], fill=0, width=w)
            distance += length


def footprint_mm(spec: Spec, cls: str, pose: Pose2D) -> np.ndarray:
    """Outline of an object of class cls lying at pose (sheet frame): a rectangle, or a 48-gon for round objects."""
    d = spec.class_dims(cls)
    a, b = d["half_long"] * 1000, d["half_across"] * 1000
    if d["round"]:
        t = np.linspace(0, 2 * np.pi, 48, endpoint=False)
        return pose.apply(np.c_[a * np.cos(t), a * np.sin(t)])
    return pose.apply([[-a, b], [a, b], [a, -b], [-a, -b]])


def draw(spec: Spec, ppm: float) -> Image.Image:
    """The sheet as a greyscale image at ppm pixels per mm."""
    import cv2
    cv = Canvas(spec, ppm)
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, spec.dictionary))
    s = spec.marker_size_mm
    for mid, pose in spec.markers().items():
        n = int(round(s * ppm))
        m = cv2.aruco.generateImageMarker(d, mid, n)
        m = Image.fromarray(m).rotate(pose.yaw_deg, expand=True, fillcolor=255) if pose.yaw_deg else Image.fromarray(m)
        centre = cv.px([[pose.x, pose.y]])[0]
        cv.img.paste(m, (int(round(centre[0] - m.width / 2)), int(round(centre[1] - m.height / 2))))
        cv.text([pose.x, pose.y - s / 2 - 1.0], f"id {mid}", 2.5, anchor="mt")
    who = {o["cell"]: o for o in spec.objects()}
    for cid, c in spec.cells().items():
        cv.polygon(spec.cell_corners_mm(cid))
        w, h = c["size"] / 2
        cv.text(c["center"] + [-w + 1.5, h - 1.5], c["label"], 5.0)
        if cid in who:
            cv.text(c["center"] + [0, -h + 1.5], who[cid]["class"], 3.0, anchor="md")
    for o in spec.objects():
        sp = spec.spots()[o["spot"]]
        cv.polygon(footprint_mm(spec, o["class"], sp), width_mm=0.4, dashed=True)
        if spec.classes()[o["class"]]["shape"] == "screw":     # the head end (-x of the spot): a solid bar
            c = spec.classes()[o["class"]]
            L, hd = (c["shank_length"] + c["head_height"]) / 2, c["head_diameter"] / 2
            cv.polygon(sp.apply([[-L, hd], [-L + c["head_height"], hd], [-L + c["head_height"], -hd], [-L, -hd]]),
                       width_mm=0.8)
        cv.text([sp.x, sp.y + 17.0], f"{o['spot']}  {o['name']} -> {o['cell']}", 2.8, anchor="mm")
        cv.draw.ellipse([*(np.array(cv.px([[sp.x, sp.y]])[0]) - 0.6 * ppm), *(np.array(cv.px([[sp.x, sp.y]])[0]) + 0.6 * ppm)],
                        fill=0)
    W, H = spec.paper_mm
    y = -H / 2 + 8.0
    cv.draw.line(cv.px([[-50.0, y], [50.0, y]]), fill=0, width=max(1, int(0.5 * ppm)))
    for x in (-50.0, 50.0):
        cv.draw.line(cv.px([[x, y - 2], [x, y + 2]]), fill=0, width=max(1, int(0.5 * ppm)))
    cv.text([0.0, y + 1.0], "100 mm: print at 100 % (actual size) and check this bar", 2.6, anchor="md")
    cv.text([0.0, H / 2 - 5.0], f"{spec.name}  (kitting sheet; bottom edge towards the arm)", 3.2, anchor="mt")
    return cv.img


def write_print(spec: Spec, out_dir: Path) -> dict:
    """PDF + PNG at the spec's print dpi, 1:1. Returns the paths and the page size the PDF declares."""
    dpi = int(spec.raw["paper"].get("print_dpi", 300))
    img = draw(spec, dpi / MM_PER_IN)
    out_dir.mkdir(parents=True, exist_ok=True)
    png, pdf = out_dir / f"{spec.name}.png", out_dir / f"{spec.name}.pdf"
    img.save(png, dpi=(dpi, dpi))
    # Exact A4 media box and lossless bitmap embedding preserve marker edges.
    from reportlab.pdfgen import canvas as pdfcanvas
    from reportlab.lib.utils import ImageReader
    w_mm, h_mm = spec.paper_mm
    points = [v / MM_PER_IN * 72 for v in (w_mm, h_mm)]
    page = pdfcanvas.Canvas(str(pdf), pagesize=points)
    page.drawImage(ImageReader(img), 0, 0, width=points[0], height=points[1])
    page.showPage()
    page.save()
    return {"png": str(png), "pdf": str(pdf), "pixels": list(img.size), "page_mm": [round(w_mm, 2), round(h_mm, 2)], "dpi": dpi}


def twin_texture(spec: Spec, path: Path, ppm: float = 4.0, levels=(0.0, 1.0)) -> Path:
    """RGB texture for the twin (image right = sheet +x, image up = sheet +y); levels = (ink, paper) grey 0..1, as
    sim/build_scene.marker_texture (printed black is not black under room light)."""
    g = np.asarray(draw(spec, ppm), np.float32) / 255.0
    ink, paper = levels
    v = (255 * (ink + g * (paper - ink))).round().astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.dstack([v] * 3)).save(path)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", default=None)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "print"))
    a = ap.parse_args()
    print(write_print(Spec.load(a.spec), Path(a.out)))


if __name__ == "__main__":
    main()
