"""Images for System 2: the camera frames upscaled, with a labelled pixel grid, every detection's
id and every tracked entity's name, so a planner can point at things by id."""
from __future__ import annotations

import io

from PIL import Image, ImageDraw

from .world import World

SCALE = 2


def annotate(world: World, role: str) -> bytes | None:
    f = world.frame
    if f is None or role not in f.images:
        return None
    im = Image.fromarray(f.images[role]).resize((f.images[role].shape[1] * SCALE, f.images[role].shape[0] * SCALE),
                                                Image.BICUBIC)
    draw = ImageDraw.Draw(im, 'RGBA')
    for x in range(0, im.width, 50):
        draw.line([(x, 0), (x, im.height)], fill=(255, 64, 64, 60 if x % 100 else 110))
        if x:
            draw.text((x + 2, 1), str(x), fill=(255, 40, 40, 255))
    for y in range(0, im.height, 50):
        draw.line([(0, y), (im.width, y)], fill=(255, 64, 64, 60 if y % 100 else 110))
        if y:
            draw.text((2, y + 1), str(y), fill=(255, 40, 40, 255))
    for d in world.detections.values():
        if d.camera != role:
            continue
        x0, y0, x1, y1 = (v * SCALE for v in d.bbox)
        draw.rectangle([x0 - 2, y0 - 2, x1 + 1, y1 + 1], outline=(0, 160, 255, 255), width=2)
        draw.rectangle([x0 - 2, y0 - 14, x0 + 7 * len(d.id) + 2, y0 - 2], fill=(0, 160, 255, 220))
        draw.text((x0, y0 - 13), d.id, fill=(255, 255, 255, 255))
    cam = f.cameras.get(role)
    if cam is not None:
        for e in world.entities.values():
            uv, front = cam.project(e.xyz)
            if not front[0]:
                continue
            u, v = uv[0] * SCALE
            if not (0 <= u < im.width and 0 <= v < im.height):
                continue
            color = (40, 200, 60, 255) if e.visible or e.kind == 'point' else (240, 160, 0, 255)
            draw.ellipse([u - 4, v - 4, u + 4, v + 4], outline=color, width=2)
            draw.text((u + 6, v - 6), e.name + ('' if e.visible or e.kind == 'point' else ' (belief)'), fill=color)
        uv, front = cam.project(world.gripper.tcp)
        if front[0]:
            u, v = uv[0] * SCALE
            draw.line([(u - 6, v), (u + 6, v)], fill=(255, 0, 255, 255), width=2)
            draw.line([(u, v - 6), (u, v + 6)], fill=(255, 0, 255, 255), width=2)
    buf = io.BytesIO()
    im.save(buf, format='JPEG', quality=85)
    return buf.getvalue()


def to_camera_pixels(u: float, v: float) -> tuple[float, float]:
    """Pixel in an annotated image -> pixel in the camera frame used by perception."""
    return u / SCALE, v / SCALE
