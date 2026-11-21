"""robot_mcp — MCP server for task-level robot control.

Exposes robots (NormaCore/ElRobot today; LeRobot SO-100/101 later) to an AI
agent through a small MCP surface: observe, bounded joint motion, and
verified skills (home / pick / place) that run as cancellable operations.

See README.md for architecture and setup.
"""

__version__ = "0.1.0"
