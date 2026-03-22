from __future__ import annotations

from dataclasses import replace

from ha_lmapf.core.types import AgentState, HumanState, Observation, PlanBundle, StepAction, TimedPath
from ha_lmapf.humans.safety import inflate_cells
from ha_lmapf.local_tier.agent_controller import AgentController
from ha_lmapf.local_tier.conflict_resolution.priority_rules import PriorityRulesResolver
from ha_lmapf.local_tier.local_planner import AStarLocalPlanner
from ha_lmapf.local_tier.sensors import build_observation
from ha_lmapf.simulation.environment import Environment


class _SimStub:
    def __init__(self, env, agents, humans, plans, step=0):
        self.env = env
        self.agents = agents
        self.humans = humans
        self._plans = plans
        self.step = step
        # minimal metrics hook (optional)
        class _M:
            def __init__(self): self.replans = 0
            def add_replan(self, n): self.replans += int(n)
        self.metrics = _M()

    def plans(self):
        return self._plans


def test_agent_controller_respects_human_safety() -> None:
    env = Environment(width=5, height=5, blocked=set())

    # Agent at (2,2) wants to move to (2,3) next
    agents = {
        0: AgentState(agent_id=0, pos=(2, 2), goal=(2, 4), task_id="t0", carrying=True),
    }
    # Human occupies the next planned cell (2,3)
    humans = {
        0: HumanState(human_id=0, pos=(2, 3), velocity=(0, 0)),
    }

    # Global plan: (2,2) -> (2,3) -> (2,4)
    tp = TimedPath(cells=[(2, 2), (2, 3), (2, 4)], start_step=0)
    plans = PlanBundle(paths={0: tp}, created_step=0, horizon=2)

    sim = _SimStub(env, agents, humans, plans, step=0)

    obs = build_observation(agent_id=0, sim_state=sim, fov_radius=4)

    controller = AgentController(
        agent_id=0,
        local_planner=AStarLocalPlanner(),
        conflict_resolver=PriorityRulesResolver(),
        fov_radius=4,
        safety_radius=1,
    )

    action = controller.decide_action(sim, obs, rng=None)

    # Compute what cell would result from action
    cur = sim.agents[0].pos
    if action == StepAction.UP:
        nxt = (cur[0] - 1, cur[1])
    elif action == StepAction.DOWN:
        nxt = (cur[0] + 1, cur[1])
    elif action == StepAction.LEFT:
        nxt = (cur[0], cur[1] - 1)
    elif action == StepAction.RIGHT:
        nxt = (cur[0], cur[1] + 1)
    else:
        nxt = cur

    forbidden = inflate_cells({humans[0].pos}, radius=1, env=env)

    # With hard safety (default): the agent must not ENTER a new forbidden cell.
    # If it WAITs (staying at current position), that's safe behavior -
    # the agent is not moving into the safety buffer, even if already inside it.
    if action != StepAction.WAIT:
        assert nxt not in forbidden
    # WAIT is always acceptable when a human is nearby (conservative safety)
