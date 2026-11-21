"""SO-101 simulation MCP server entrypoint (`so101-sim`)."""
import os
from pathlib import Path

os.environ.setdefault('ROBOT_MODEL', 'so101')
os.environ.setdefault('ROBOT_SIM_LAYOUT', str(Path(__file__).resolve().parent / 'layout_so101.json'))
os.environ.setdefault('ROBOT_MCP_NAME', 'so101-sim')

import sim_arm_mcp as _srv  # noqa: E402
from sim_arm_mcp import *  # noqa: E402,F401,F403

if __name__ == '__main__':
    try:
        _srv.server.run()
    finally:
        _srv.arm.running = False
        if _srv.recorder:
            _srv.recorder.close()
            import make_replay
            make_replay.build(_srv.recorder.dir)
        if _srv.arm.viewer is not None:
            _srv.arm.viewer.close()
