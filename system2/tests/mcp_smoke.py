"""End-to-end check of the MCP surface System 2 sees (needs the simulator; not run by pytest).

  python system2/tests/mcp_smoke.py      # spawns run_server.sh over stdio, like Claude does

Lists the tools, looks, tracks the smallest and largest overview detections, submits a goal,
cancels one mid-way, checks that bad goals are rejected with a useful message, and that motion
tools are not exposed.
"""
import asyncio
import json
import os
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

HERE = Path(__file__).resolve().parents[1]


def text(result):
    return '\n'.join(c.text for c in result.content if c.type == 'text')


async def main():
    env = {**os.environ, 'ROBOT_SIM_RECORD': os.environ.get('ROBOT_SIM_RECORD', '0')}
    params = StdioServerParameters(command='bash', args=[str(HERE / 'run_server.sh')], env=env)
    async with stdio_client(params) as (r, w), ClientSession(r, w) as s:
        await s.initialize()
        tools = sorted(t.name for t in (await s.list_tools()).tools)
        print('tools:', tools)
        assert not {'move_to', 'move_tcp', 'gripper', 'set_torque'} & set(tools), 'motion tools must not be exposed'
        res = await s.call_tool('look', {})
        kinds = [c.type for c in res.content]
        assert kinds.count('image') == 2, kinds
        summary = json.loads([c.text for c in res.content if c.type == 'text'][-1])
        dets = sorted((d for d in summary['untracked_detections'] if d['camera'] == 'overview'), key=lambda d: d['area_px'])
        print('detections:', [(d['id'], d['area_px']) for d in dets])
        print(text(await s.call_tool('track', {'name': 'cube', 'detection': dets[0]['id'], 'height_m': 0.02})))
        print(text(await s.call_tool('track', {'name': 'box', 'detection': dets[-1]['id'], 'height_m': 0.09, 'kind': 'container'})))
        bad = await s.call_tool('submit_goal', {'description': 'x', 'conditions': [{'type': 'pick', 'object': 'cube'}]})
        assert bad.isError and 'unknown condition type' in text(bad), text(bad)
        bad = await s.call_tool('submit_goal', {'description': 'x', 'conditions': [{'type': 'reach', 'b': 'sphere'}]})
        assert bad.isError and 'unknown entity' in text(bad), text(bad)
        print('invalid goals rejected with reasons')
        goal = {'description': 'jaws open, 6 cm above the cube',
                'conditions': [{'type': 'gripper', 'state': 'open'},
                               {'type': 'reach', 'a': 'gripper', 'b': 'cube', 'offset': [0, 0, 0.06], 'tol': 0.01}]}
        print(text(await s.call_tool('submit_goal', goal)))
        busy = await s.call_tool('submit_goal', goal)
        assert busy.isError and 'running' in text(busy)
        st = json.loads(text(await s.call_tool('goal_status', {'wait_s': 120})))
        print('goal:', st['status'], st['reason'][:160])
        assert st['status'] == 'succeeded', st
        print(text(await s.call_tool('submit_goal', {'description': 'far away', 'conditions': [
            {'type': 'reach', 'a': 'gripper', 'xyz': [-0.2, 0.25, 0.1], 'tol': 0.01}]})))
        await asyncio.sleep(4)
        st = json.loads(text(await s.call_tool('cancel_goal', {})))
        print('cancel:', st['status'], st['reason'])
        assert st['status'] == 'cancelled', st
        print(text(await s.call_tool('system_status', {})))
        print('MCP smoke test passed')


if __name__ == '__main__':
    sys.exit(asyncio.run(main()))
