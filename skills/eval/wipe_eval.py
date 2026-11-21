"""Independent wipe evaluator: overview images before/after -> Outcome + effective cleaning width.

Never uses the executor's claim. Deterministic (pure numpy, no randomness), so it can be re-run on saved images.

1. Rectify both images onto the board (homography from the calibrated camera or marker points).
2. Classify pixels (dirt by chroma, robust to shadows): see dirt.py.
3. Unknown (success=None) if: an image is stale, the arm is reported in view, the board is not fully in the image,
   new dark things appeared on the board (an occluder), or there was no measurable stain before.
4. Residual over the WHOLE board: dirt pushed out of the target area still counts.
   metric contamination_reduction = 1 - dirt_area_after / dirt_area_before (negative if the wipe smeared).
5. Effective cleaned width along the EXECUTED tool path: the cleaned mask (dirty before, not after) is compared to
   the mask predicted by sweeping a band of width w (with a constant lateral offset) along the executed strokes;
   the best-agreeing w is the observed width. It is only VALID when identifiable: the agreement must drop on both
   sides of the best w (dirt visible beyond the band edges), otherwise it is just a lower bound.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .._contracts import Outcome
from .dirt import DirtParams, classify
from .geometry import Board, rectify, segment_distance


@dataclass(frozen=True)
class EvalConfig:
    px_per_m: float = 500.0             # rectified resolution (2 mm per pixel)
    success_reduction: float = 0.90     # success = at least 90 % of the board's dirt area removed
    min_visible: float = 0.98           # fraction of the board that must be inside both images
    occluder_frac: float = 0.01         # new dark pixels (fraction of the board) that mean "something is in view"
    min_before_px: int = 40             # less stain than this before: nothing to measure
    w_grid_mm: tuple = (4, 90, 1)
    offset_grid_mm: tuple = (-12, 12, 1)
    agree_tol: float = 0.004            # widths within this agreement of the best form the interval
    max_interval_mm: float = 10.0       # wider interval = width not identifiable
    min_eval_px: int = 120
    dirt: DirtParams = DirtParams()


CFG = EvalConfig()


@dataclass
class WidthEstimate:
    w_observed: float | None
    valid: bool
    why: str = ""
    interval: tuple[float, float] | None = None
    offset: float | None = None          # lateral offset of the cleaned band from the TCP path (m, along the normal)
    agreement: float | None = None
    evidence: list[str] = field(default_factory=list)


@dataclass
class WipeEvaluation:
    outcome: Outcome
    width: WidthEstimate | None
    stats: dict


def _unknown(execution_id, why, evidence, stats) -> WipeEvaluation:
    return WipeEvaluation(Outcome(execution_id, "contamination_reduction", float("nan"), "fraction", None,
                                  list(evidence) + [f"unknown: {why}"]), None, {**stats, "unknown": why})


def fit_width(X, Y, valid, dirty_b, cleaned, strokes, cfg: EvalConfig = CFG) -> WidthEstimate:
    """Fit the effective band width (and a constant lateral offset) to the observed cleaned mask."""
    segs = [(np.asarray(a, float), np.asarray(b, float)) for a, b in strokes if np.hypot(*(np.subtract(b, a))) > 1e-4]
    if not segs:
        return WidthEstimate(None, False, "no executed strokes")
    d = segs[0][1] - segs[0][0]
    n = np.array([-d[1], d[0]]) / np.linalg.norm(d)
    ws = np.arange(*cfg.w_grid_mm) / 1000.0
    offs = np.arange(cfg.offset_grid_mm[0], cfg.offset_grid_mm[1] + 1e-9, cfg.offset_grid_mm[2]) / 1000.0
    reach = ws[-1] / 2 + max(abs(offs)) + 0.005
    E = valid & dirty_b & (segment_distance(X, Y, segs) <= reach)
    ne = int(E.sum())
    if ne < cfg.min_eval_px:
        return WidthEstimate(None, False, f"only {ne} stained pixels near the path (< {cfg.min_eval_px})")
    x, y, c = X[E], Y[E], cleaned[E]
    best = (-1.0, None, None, None)
    for o in offs:
        dist = segment_distance(x, y, [(a + o * n, b + o * n) for a, b in segs])
        score = ((dist[None, :] <= ws[:, None] / 2) == c[None, :]).mean(1)
        i = int(score.argmax())
        if score[i] > best[0] + 1e-12:
            best = (float(score[i]), float(ws[i]), float(o), score)
    agree, w, o, score = best
    near = ws[score >= agree - cfg.agree_tol]
    lo, hi = float(near.min()), float(near.max())
    span_mm = (hi - lo) * 1000
    at_edge = w <= ws[0] + 1e-9 or w >= ws[-1] - 1e-9
    valid_w = span_mm <= cfg.max_interval_mm and not at_edge
    why = ("identified" if valid_w else
           f"not identifiable: widths {lo * 1000:.0f}-{hi * 1000:.0f} mm fit equally well (stain does not extend past "
           f"the cleaned band, so this is a lower bound)" if not at_edge else "best width at the search limit")
    return WidthEstimate(w, valid_w, why, (lo, hi), o, agree)


def evaluate(before: np.ndarray, after: np.ndarray, *, H: np.ndarray, board: Board, execution_id: str,
             strokes=(), target=None, evidence=(), arm_in_view: bool | None = None, stale: bool = False,
             out_dir: str | Path | None = None, cfg: EvalConfig = CFG) -> WipeEvaluation:
    """before/after: overview images (HxWx3 uint8 RGB). H: image <- world-plane homography. strokes: executed
    tool-path segments [((x0, y0), (x1, y1)), ...] in world metres (measured TCP path, not the command).
    target: optional (x0, y0, x1, y1) for the inside/outside split (the metric itself is whole-board)."""
    evidence = list(evidence)
    stats = {"arm_in_view": arm_in_view, "stale": stale}
    if stale:
        return _unknown(execution_id, "stale image", evidence, stats)
    if arm_in_view:
        return _unknown(execution_id, "arm reported in view of the board", evidence, stats)
    rb, vb = rectify(before, H, board, cfg.px_per_m)
    ra, va = rectify(after, H, board, cfg.px_per_m)
    valid = vb & va
    vis = float(valid.mean())
    stats["visible"] = round(vis, 4)
    if vis < cfg.min_visible:
        return _unknown(execution_id, f"only {vis:.0%} of the board is inside the image", evidence, stats)
    cb, ca = classify(rb, valid, cfg.dirt), classify(ra, valid, cfg.dirt)
    new_dark = float((ca["dark"] & ~cb["dark"]).sum()) / max(int(valid.sum()), 1)
    old_dark = float(cb["dark"].sum()) / max(int(valid.sum()), 1)
    stats.update(new_dark_frac=round(new_dark, 4), before_dark_frac=round(old_dark, 4))
    if out_dir is not None:
        evidence += save_evidence(out_dir, execution_id, rb, ra, cb, ca)
    if new_dark > cfg.occluder_frac:
        return _unknown(execution_id, f"something dark appeared on {new_dark:.1%} of the board (arm or occluder)", evidence, stats)
    if old_dark > cfg.occluder_frac:
        return _unknown(execution_id, f"the before image has {old_dark:.1%} of the board occluded", evidence, stats)
    a_b, a_a = int(cb["dirt"].sum()), int(ca["dirt"].sum())
    px_area = 1.0 / cfg.px_per_m ** 2
    stats.update(dirt_before_px=a_b, dirt_after_px=a_a, dirt_before_m2=a_b * px_area, dirt_after_m2=a_a * px_area)
    if a_b < cfg.min_before_px:
        return _unknown(execution_id, f"no measurable stain before ({a_b} px)", evidence, stats)
    X, Y = board.grid(cfg.px_per_m)
    if target is not None:
        x0, y0, x1, y1 = target
        inside = (X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1)
        stats.update(dirt_after_inside_px=int((ca["dirt"] & inside).sum()), dirt_after_outside_px=int((ca["dirt"] & ~inside).sum()),
                     dirt_before_outside_px=int((cb["dirt"] & ~inside).sum()))
    red = 1.0 - a_a / a_b
    stats["contamination_reduction"] = round(red, 4)
    width = None
    if len(strokes):
        cleaned = cb["dirt"] & ~ca["dirt"] & ~ca["dark"]
        width = fit_width(X, Y, valid, cb["dirt"], cleaned, strokes, cfg)
        width.evidence = list(evidence)
        stats["width"] = {"w_observed_mm": None if width.w_observed is None else round(width.w_observed * 1000, 1),
                          "valid": width.valid, "why": width.why,
                          "interval_mm": None if width.interval is None else [round(v * 1000, 1) for v in width.interval],
                          "offset_mm": None if width.offset is None else round(width.offset * 1000, 1),
                          "agreement": None if width.agreement is None else round(width.agreement, 4)}
    out = Outcome(execution_id, "contamination_reduction", red, "fraction of board dirt area removed",
                  bool(red >= cfg.success_reduction), evidence)
    return WipeEvaluation(out, width, stats)


def save_evidence(out_dir, execution_id, rb, ra, cb, ca) -> list[str]:
    """One small PNG: before | after | masks (red = dirt after, green = cleaned, blue = new dark)."""
    from PIL import Image
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    m = np.zeros(rb.shape, np.float32) + 0.9
    m[cb["dirt"] & ~ca["dirt"]] = (0.2, 0.8, 0.2)
    m[ca["dirt"]] = (0.9, 0.1, 0.1)
    m[ca["dark"] & ~cb["dark"]] = (0.1, 0.2, 0.9)
    sep = np.ones((rb.shape[0], 3, 3), np.float32)
    img = np.concatenate([rb, sep, ra, sep, m], 1)
    p = out / f"{execution_id}_wipe_eval.png"
    Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(p)
    return [str(p)]
