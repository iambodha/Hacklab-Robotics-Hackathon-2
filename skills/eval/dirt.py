"""Pixel classes on a rectified board image: clean board / dirt / other (occluder, unknown).

Dirt is detected by CHROMA, not brightness: a coloured stain (cocoa, coffee, ketchup, ...) on a white/neutral board.
Shadows and exposure changes scale all channels together, so they change brightness but not chroma, and do not
become dirt. Colours are white-balanced on the board's brightest pixels (the clean board) first.

  dirt   chroma >= dirt_chroma
  board  chroma <  board_chroma and brightness >= board_min_rel of the clean-board level (shadows allowed)
  dark   not dirt and darker than board_min_rel (the arm, a gripper, a deep shadow): occluder candidates
  other  dark + faint smear (chroma between the two thresholds)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DirtParams:
    dirt_chroma: float = 0.16
    board_chroma: float = 0.10
    board_min_rel: float = 0.40     # shadows down to 40 % of the clean-board brightness still count as board
    dirt_min_bright: float = 0.15   # chroma of near-black pixels is noise: they are never dirt
    white_pct: float = 92.0         # percentile per channel taken as the clean-board colour
    min_blob_px: int = 3            # 3x3 majority filter removes isolated pixels


def _majority3(m: np.ndarray, k: int) -> np.ndarray:
    p = np.pad(m.astype(np.int16), 1)
    s = sum(p[i:i + m.shape[0], j:j + m.shape[1]] for i in range(3) for j in range(3))
    return m & (s >= k)


def classify(rect: np.ndarray, valid: np.ndarray, P: DirtParams = DirtParams()) -> dict:
    """{'dirt', 'board', 'other'} boolean masks + the white level used. `valid` = rectified pixel came from the image."""
    px = rect[valid] if valid.any() else rect.reshape(-1, 3)
    white = np.maximum(np.percentile(px, P.white_pct, axis=0), 1e-3)
    bal = rect / white
    mx, mn = bal.max(-1), bal.min(-1)
    chroma = (mx - mn) / np.maximum(mx, 1e-3)
    bright = bal.mean(-1)
    dirt = _majority3((chroma >= P.dirt_chroma) & (bright >= P.dirt_min_bright) & valid, P.min_blob_px + 2)
    dark = _majority3(valid & ~dirt & (bright < P.board_min_rel), P.min_blob_px + 2)
    board = (chroma < P.board_chroma) & ~dark & valid & ~dirt
    other = valid & ~dirt & ~board          # dark things + faint (between board and dirt chroma)
    return {"dirt": dirt, "board": board, "dark": dark, "other": other, "white": white.tolist(), "chroma": chroma}
