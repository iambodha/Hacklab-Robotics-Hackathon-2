"""Level-1 adaptation (online, every run): effective cleaning width estimate

    w_hat_{k+1} = (1 - alpha) * w_hat_k + alpha * w_observed

applied only when the evaluator's measurement is valid AND visible (Outcome.success is not None), clamped to the
parameter's bounds. Every attempt, accepted or not, returns a record with its evidence for skill memory.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..model import Param, SkillVersion

ALPHA = 0.6


@dataclass
class WidthMeasurement:
    w_observed: float | None          # metres
    valid: bool                       # identifiable: dirt on both sides of the cleaned band, enough samples
    why: str = ""
    interval: tuple[float, float] | None = None   # widths whose fit is within tolerance of the best
    evidence: list[str] = field(default_factory=list)


def update_width(version: SkillVersion, spec: Param, m: WidthMeasurement, visible: bool,
                 alpha: float = ALPHA) -> tuple[SkillVersion | None, dict]:
    old = float(version.params[spec.name])
    rec = {"param": spec.name, "old": old, "observed": m.w_observed, "alpha": alpha, "visible": visible,
           "valid": m.valid, "why": m.why, "interval": m.interval, "evidence": list(m.evidence),
           "bounds": [spec.lo, spec.hi], "from_version": version.version}
    if not visible:
        return None, {**rec, "accepted": False, "reason": "outcome not visible (success=None): no update"}
    if not m.valid or m.w_observed is None:
        return None, {**rec, "accepted": False, "reason": "width measurement not valid: no update"}
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be in (0, 1]")
    raw = (1 - alpha) * old + alpha * float(m.w_observed)
    new = spec.clamp(raw)
    if abs(new - old) < 1e-6:
        return None, {**rec, "accepted": False, "new": new, "reason": "no change"}
    v = version.child({spec.name: round(new, 5)}, "adaptation",
                      f"{spec.name} {old * 1000:.1f} -> {new * 1000:.1f} mm from observed {m.w_observed * 1000:.1f} mm")
    return v, {**rec, "accepted": True, "new": new, "raw": raw, "clamped": abs(raw - new) > 1e-9, "to_version": v.version}
