"""ArUco detection helpers: dictionary auto-detection, multi-frame corner averaging."""
from __future__ import annotations

import cv2
import numpy as np

# Candidate dictionaries, most common first. AprilTag families are included because they print the same way.
DICT_NAMES = [
    "DICT_4X4_50", "DICT_4X4_100", "DICT_4X4_250", "DICT_4X4_1000",
    "DICT_5X5_50", "DICT_5X5_100", "DICT_5X5_250", "DICT_5X5_1000",
    "DICT_6X6_50", "DICT_6X6_100", "DICT_6X6_250", "DICT_6X6_1000",
    "DICT_7X7_50", "DICT_7X7_100", "DICT_7X7_250", "DICT_7X7_1000",
    "DICT_ARUCO_ORIGINAL", "DICT_APRILTAG_16h5", "DICT_APRILTAG_25h9", "DICT_APRILTAG_36h10", "DICT_APRILTAG_36h11",
]
# Prefer the smallest dictionary that explains the detections (ids from bigger dicts of the same family
# are a superset, so the _50 variant is preferred whenever all ids fit).
_FAMILY_SIZE = {n: int(n.rsplit("_", 1)[1]) if n.rsplit("_", 1)[1].isdigit() else 0 for n in DICT_NAMES}


def dictionary(name: str):
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))


def detector(name: str, refine: bool = True) -> cv2.aruco.ArucoDetector:
    p = cv2.aruco.DetectorParameters()
    if refine:
        p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        p.cornerRefinementWinSize = 3
        p.cornerRefinementMinAccuracy = 0.01
        p.cornerRefinementMaxIterations = 50
    # Small (299x224) frames: allow small markers and be generous with the adaptive threshold.
    p.minMarkerPerimeterRate = 0.02
    p.minDistanceToBorder = 1          # default 3 drops marker 21, whose corner sits 2 px from the overview frame edge
    p.adaptiveThreshWinSizeMin, p.adaptiveThreshWinSizeMax, p.adaptiveThreshWinSizeStep = 3, 33, 4
    return cv2.aruco.ArucoDetector(dictionary(name), p)


def detect(img: np.ndarray, name: str, edge_refine: bool = True) -> dict[int, np.ndarray]:
    """-> {id: (4,2) corners in OpenCV order TL, TR, BR, BL of the marker's own frame}.
    edge_refine: replace OpenCV's corners (biased 0.2-0.7 px inward on small, blurred markers) by
    intersections of lines fitted to sub-pixel edge points along each side (see refine_edges)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    corners, ids, _ = detector(name).detectMarkers(gray)
    if ids is None:
        return {}
    out = {int(i): c.reshape(4, 2).astype(float) for i, c in zip(ids.ravel(), corners)}
    if edge_refine:
        g32 = gray.astype(np.float32)
        out = {i: refine_edges(g32, c) for i, c in out.items()}
    return out


def _bilinear(img: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return cv2.remap(img, x.astype(np.float32), y.astype(np.float32), cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


def refine_edges(gray: np.ndarray, c: np.ndarray, iters: int = 2, half: float = 2.5, bits: int = 4,
                 max_shift: float = 3.0) -> np.ndarray:
    """Sub-pixel marker corners from the black square's four edges.

    For each side, sample profiles across the edge (inside = black square, outside = white border),
    locate the mid-intensity crossing (unbiased for a symmetric blur), fit a line through those
    points (total least squares) and intersect neighbouring lines. Falls back to the input corners
    if a side is too short or the contrast is too low."""
    c = c.astype(float).copy()
    c0 = c.copy()
    for _ in range(iters):
        centre = c.mean(0)
        lines = []
        for k in range(4):
            a, b = c[k], c[(k + 1) % 4]
            L = np.linalg.norm(b - a)
            if L < 6:
                return c
            t = (b - a) / L
            n = np.array([-t[1], t[0]])
            if (centre - a) @ n > 0:                    # make n point outward (towards the white border)
                n = -n
            cell = L / (bits + 2)                       # black border is one cell wide, white border outside
            h = float(np.clip(0.9 * cell, 1.0, half))
            s = np.linspace(0.2, 0.8, max(4, int(L * 0.6)))
            base = a + s[:, None] * (b - a)
            offs = np.linspace(-h, h, int(h * 8) + 1)
            P = base[:, None, :] + offs[None, :, None] * n
            prof = _bilinear(gray, P[..., 0], P[..., 1]).reshape(len(s), len(offs))
            pts = []
            m0 = len(offs) // 2
            for j, pr in enumerate(prof):
                lo, hi = pr[:m0].min(), pr[m0:].max()
                if hi - lo < 25:
                    continue
                mid = (lo + hi) / 2
                # rising (dark inside -> bright border) mid-level crossings. Normally the outermost one (the
                # profile ends on the white border); if the profile ends dark (thin border, then a dark
                # background, e.g. a marker at the table edge) the one nearest the initial edge.
                up = np.nonzero((pr[:-1] < mid) & (pr[1:] >= mid))[0]
                if not len(up):
                    continue
                i0 = int(up[-1] if pr[-1] >= mid else up[np.argmin(np.abs(offs[up] + offs[up + 1]))])
                f = (mid - pr[i0]) / (pr[i0 + 1] - pr[i0] + 1e-9)
                pts.append(base[j] + (offs[i0] + f * (offs[i0 + 1] - offs[i0])) * n)
            if len(pts) < 3:
                return c
            pts = np.array(pts)
            for _ in range(2):                              # total least squares, then drop outliers and refit
                m = pts.mean(0)
                _, _, vt = np.linalg.svd(pts - m)
                d = vt[0]
                res = np.abs((pts - m) @ np.array([-d[1], d[0]]))
                keep = res <= max(0.25, 3 * np.median(res))
                if keep.all() or keep.sum() < 3:
                    break
                pts = pts[keep]
            lines.append((m, d))
        new = []
        for k in range(4):
            (p1, d1), (p2, d2) = lines[(k - 1) % 4], lines[k]
            A = np.c_[d1, -d2]
            if abs(np.linalg.det(A)) < 1e-6:
                return c
            tt = np.linalg.solve(A, p2 - p1)
            new.append(p1 + tt[0] * d1)
        new = np.array(new)
        if np.linalg.norm(new - c0, axis=1).max() > max_shift:   # ran away: keep OpenCV's corners
            return c0
        c = new
    return c


def autodetect_dictionary(frames: list[np.ndarray]) -> tuple[str | None, dict[str, set[int]]]:
    """Try every dictionary on the frames; pick the one finding the most markers (ties: smallest dict).
    Returns (name, {dict name: ids seen})."""
    seen: dict[str, set[int]] = {}
    for name in DICT_NAMES:
        ids: set[int] = set()
        for f in frames:
            ids |= set(detect(f, name))
        if ids:
            seen[name] = ids
    if not seen:
        return None, seen
    best = max(seen, key=lambda n: (len(seen[n]), -_FAMILY_SIZE[n]))
    return best, seen


def average_corners(frames: list[np.ndarray], name: str, min_frac: float = 0.5):
    """Detect in every frame and take the per-corner median. -> ({id: (4,2) median}, {id: (4,2) std}, {id: count})."""
    per: dict[int, list[np.ndarray]] = {}
    for f in frames:
        for i, c in detect(f, name).items():
            per.setdefault(i, []).append(c)
    need = max(1, int(min_frac * len(frames)))
    med, std, cnt = {}, {}, {}
    for i, cs in per.items():
        if len(cs) >= need:
            a = np.stack(cs)
            med[i], std[i], cnt[i] = np.median(a, 0), a.std(0), len(cs)
    return med, std, cnt


def marker_object_points(size: float) -> np.ndarray:
    """Corner coordinates in the marker frame (centre origin, x right, y up, z out of the marker),
    in the same TL, TR, BR, BL order detectMarkers returns (matches cv2.SOLVEPNP_IPPE_SQUARE)."""
    h = size / 2
    return np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], float)
