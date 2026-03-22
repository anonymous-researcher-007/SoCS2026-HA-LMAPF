from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from ha_lmapf.core.interfaces import ConflictResolver, SimStateView
from ha_lmapf.core.types import Observation, StepAction

Cell = Tuple[int, int]


@dataclass
class GlobalOnlyController:
    """
    Baseline controller: follow Tier-1 global plan only.
    - No local replanning around humans.
    - If the next planned cell is occupied by a visible human (or otherwise blocked in observation), WAIT.
    - Still uses a ConflictResolver to avoid agent-agent conflicts.
    """
    agent_id: int
    conflict_resolver: ConflictResolver

    def decide_action(self, sim_state: SimStateView, observation: Observation, rng=None) -> StepAction:
        aid = self.agent_id
        cur = sim_state.agents[aid].pos

        desired_next = self._desired_from_global_plan(sim_state, aid)
        if desired_next is None:
            desired_next = cur

        # Human-aware check (no detours): if blocked due to visible human, WAIT
        # Observation.blocked already includes visible human occupied cells (per sensors.py).
        if desired_next in observation.blocked:
            return StepAction.WAIT

        # Static safety check
        if not sim_state.env.is_free(desired_next):
            return StepAction.WAIT

        # Resolve agent-agent conflicts
        return self.conflict_resolver.resolve(
            agent_id=aid,
            desired_cell=desired_next,
            sim_state=sim_state,
            observation=observation,
            rng=rng,
        )

    @staticmethod
    def _desired_from_global_plan(sim_state: SimStateView, aid: int) -> Optional[Cell]:
        # FIX: Call the method .plans() defined in the SimStateView protocol
        plans = sim_state.plans()
        if plans is None:
            return None

        path = plans.paths.get(aid)
        if path is None:
            return None

        # FIX: Use __call__ instead of cell_at
        return path(sim_state.step + 1)


def make_global_only_controllers(
        sim_state: SimStateView,
        conflict_resolver: ConflictResolver,
        _fov_radius: int,  # Prefix with _ to indicate intentional non-use
        _safety_radius: int,  # Prefix with _ to indicate intentional non-use
) -> Dict[int, GlobalOnlyController]:
    """
    Factory returning per-agent GlobalOnlyController instances.

    Parameters fov_radius and safety_radius are accepted to match the main controller factory signature,
    but are not used here because the baseline does not do safety inflation or replanning.
    """
    return {
        aid: GlobalOnlyController(agent_id=aid, conflict_resolver=conflict_resolver)
        for aid in sorted(sim_state.agents.keys())
    }
