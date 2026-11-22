"""Physics-only MCP test launcher for sandboxes without a graphics connection.

Replaces camera rendering with an explicitly labeled diagnostic image. Physics,
IK, tool calls, episodes and ground-truth judging remain real MuJoCo execution.
Never use this launcher to validate perception or camera rendering.
"""
import sys


def main():
    if "--mode" not in sys.argv or sys.argv[sys.argv.index("--mode") + 1] != "sim":
        raise SystemExit("headless test launcher only accepts --mode sim")
    import mujoco
    import numpy as np
    from PIL import Image, ImageDraw

    class DiagnosticRenderer:
        def __init__(self, model, height, width):
            im = Image.new("RGB", (width, height), "#303030")
            ImageDraw.Draw(im).text((8, 8), "RENDERING DISABLED\nPhysics-only test", fill="white")
            self.pixels = np.asarray(im)

        def update_scene(self, *args, **kwargs):
            pass

        def render(self):
            return self.pixels.copy()

        def close(self):
            pass

    mujoco.Renderer = DiagnosticRenderer
    from generaloid.__main__ import main as cli
    return cli(sys.argv[1:])


if __name__ == "__main__":
    sys.exit(main())
