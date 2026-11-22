"""Attempt-by-attempt learning for the bottle-opening demo (prompts/bottle-contract.md).

episodes.py  the episode store: episodes.jsonl + attempt_<N>/ files, parameter bounds, history text for the
             agent, per-attempt report.md and SUMMARY.md, recorder sessions per attempt
twist.py     the composite cap-unscrewing skill (twist_cap / check_cap) against a small rig interface, so the
             real arm (sim/real_arm_mcp.py) and the twin (sim/sim_arm_mcp.py) run the same code
tools.py     the MCP tool bodies (episode_begin, episode_history, twist_cap, check_cap, episode_end) shared by
             both servers
judge.py     optional Gemini ER judgement "is the cap separated from the bottle?"
"""
