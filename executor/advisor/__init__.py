"""Background advisors: claude (Claude Code headless + executor-only MCP), gemini (Gemini Robotics ER), none."""
from __future__ import annotations

from .base import Advisor


def make_advisor(name: str, api_url: str, events, skills: dict, scene: dict, executor=None, on_done=None, **kw) -> Advisor:
    if name == "claude":
        from .claude import ClaudeAdvisor
        return ClaudeAdvisor(api_url, events, skills, scene, on_done=on_done, **kw)
    if name == "gemini":
        from .gemini_er import GeminiERAdvisor
        return GeminiERAdvisor(api_url, events, skills, scene, executor=executor, on_done=on_done, **kw)
    return Advisor(api_url, events, skills, scene, on_done=on_done)
