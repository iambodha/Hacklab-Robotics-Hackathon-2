"""SO-101 real-arm MCP server entrypoint (`so101-real`)."""
import os

os.environ.setdefault("ROBOT_MODEL", "so101")
os.environ.setdefault("ROBOT_MCP_NAME", "so101-real")

from real_arm_mcp import *  # noqa: E402,F401,F403

if __name__ == "__main__":
    server.run()
