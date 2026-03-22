from __future__ import annotations

from typing import Optional, Tuple

from ha_lmapf.core.grid import manhattan, neighbors
from ha_lmapf.core.interfaces import SimStateView
from ha_lmapf.core.types import Observation, StepAction
from ha_lmapf.local_tier.conflict_resolution.base import BaseConflictResolver, detect_imminent_conflict

Cell = Tuple[int, int]


class PriorityRulesResolver(BaseConflictResolver):
    """
    Communication-free deterministic resolver.

    Priority tuple (higher is better):
      (urgency, wait_steps, -agent_id)
    where urgency = -distance_to_goal, optionally boosted if wait_steps exceeds threshold.
    """

    def __init__(self, starvation_threshold: int = 10, boost: int = 50, allow_side_step: bool = True) -> None:
        self.starvation_threshold = int(max(1, starvation_threshold))
        self.boost = int(boost)
        self.allow_side_step = bool(allow_side_step)

    def resolve(
            self,
            agent_id: int,
            desired_cell: Cell,
            sim_state: SimStateView,
            observation: Observation,
            rng=None,  # unused
    ) -> StepAction:
        conflict = detect_imminent_conflict(agent_id, desired_cell, sim_state)
        if conflict is None:
            return self.action_toward(sim_state.agents[agent_id].pos, desired_cell)

        other_id = conflict.other_agent_id
        p_self = self._priority(agent_id, sim_state)
        p_other = self._priority(other_id, sim_state)

        # Deterministic: higher tuple wins; if tie, lower agent_id wins because -agent_id is higher
        if p_self > p_other:
            return self.action_toward(sim_state.agents[agent_id].pos, desired_cell)

        # Lose: wait or side-step safely
        if self.allow_side_step:
            side = self._safe_side_step(agent_id, sim_state, observation)
            if side is not None:
                return self.action_toward(sim_state.agents[agent_id].pos, side)

        return StepAction.WAIT

    def _priority(self, agent_id: int, sim_state: SimStateView) -> Tuple[int, int, int]:
        a = sim_state.agents[agent_id]
        if a.goal is None:
            dist = 10 ** 9
        else:
            dist = manhattan(a.pos, a.goal)
        urgency = -dist

        # Starvation prevention: boost urgency once wait exceeds threshold
        if a.wait_steps > self.starvation_threshold:
            urgency += self.boost

        return (urgency, int(a.wait_steps), -int(agent_id))

    def _safe_side_step(
            self,
            agent_id: int,
            sim_state: SimStateView,
            observation: Observation,
    ) -> Optional[Cell]:
        cur = sim_state.agents[agent_id].pos
        for nb in neighbors(cur):
            if not sim_state.env.is_free(nb):
                continue
            if nb in observation.blocked:
                continue
            c = detect_imminent_conflict(agent_id, nb, sim_state)
            if c is None:
                return nb
        return None
