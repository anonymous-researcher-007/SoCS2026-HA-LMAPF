from __future__ import annotations

from typing import Dict, Optional

from ha_lmapf.core.interfaces import SimStateView, GlobalPlanner
from ha_lmapf.core.types import PlanBundle, Task


# Allocator (Matches your task_allocator.py file name)


class RollingHorizonPlanner:
    """
    Tier-1 global planner scheduler for lifelong MAPF.
    """

    def __init__(
            self,
            horizon: int,
            replan_every: int,
            solver_name: str = "cbs",
            solver_impl: Optional[GlobalPlanner] = None,
            exhaustion_fraction: float = 0.4,
            safety_wait_fraction: float = 0.3,
    ) -> None:
        """
        Initialize the planner.

        Args:
            horizon: Planning horizon length.
            replan_every: Re-plan interval.
            solver_name: Name of solver to use if solver_impl is None.
                         Options: "cbs", "lacam", "lacam3", "lacam_official", "pibt2"
            solver_impl: (Optional) A specific solver instance.
                         If provided, this OVERRIDES 'solver_name'
            exhaustion_fraction: If this fraction of agents have stale global
                plans (due to path-exhausted or no-global-path local replans),
                trigger an early global replan regardless of the periodic
                schedule. Default 0.4 (40 % of agents).
            safety_wait_fraction: If this fraction of agents are stuck in
                consecutive SAFETY-WAIT (flagged via major_deviation), trigger
                an early global replan. Default 0.3 (30 % of agents).
        """
        self.horizon = int(horizon)
        self.replan_every = max(1, int(replan_every))
        self.exhaustion_fraction = float(exhaustion_fraction)
        self.safety_wait_fraction = float(safety_wait_fraction)

        # Setup Solver (Dependency Injection or Factory)
        if solver_impl is not None:
            self.solver = solver_impl
        else:
            # Use GlobalPlannerFactory for consistent solver creation
            from ha_lmapf.global_tier.planner_interface import GlobalPlannerFactory
            self.solver = GlobalPlannerFactory.create(solver_name)

        # State tracking
        self.last_planned_step: int = -99999
        self.completed_tasks_seen: int = 0
        # Minimum gap between two consecutive emergency replans (avoid spam)
        self._min_emergency_gap: int = max(3, replan_every // 4)
        self._last_emergency_step: int = -99999
        # Track whether the last global replan produced useful (non-WAIT) paths.
        # If the solver consistently fails (e.g. binary not found), emergency
        # replans would fire every min_gap steps and multiply the waste.
        # We suppress emergency triggers when the last replan was useless.
        self._last_replan_useful: bool = True

    def _exhaustion_trigger(self, sim_state: SimStateView) -> bool:
        """
        Return True if too many agents have stale global plans.

        A stale plan means the agent already triggered a local A* replan
        (path-exhausted or no-global-path) and the global plan is no longer
        guiding them.  When a large fraction of agents are in this state the
        coordinated plan has effectively collapsed; replanning sooner recovers
        throughput without waiting for the full periodic interval.
        """
        if self.exhaustion_fraction <= 0.0:
            return False
        n_agents = len(getattr(sim_state, "agents", {}))
        if n_agents == 0:
            return False
        stale = getattr(sim_state, "stale_global_plan_agents", None)
        if stale is None or not callable(stale):
            return False
        n_stale = len(stale())
        return (n_stale / n_agents) >= self.exhaustion_fraction

    def _safety_wait_trigger(self, sim_state: SimStateView) -> bool:
        """
        Return True if too many agents are stuck in consecutive SAFETY-WAITs.

        When multiple agents are simultaneously frozen by the human safety
        buffer they clog corridors, causing downstream agent-agent conflicts
        and further degrading throughput.  An early global replan redistributes
        agents onto less-congested routes.

        Note: the global planner does not model humans, so it cannot directly
        resolve the blocking.  The benefit is rerouting OTHER agents around
        the congestion that the frozen agents are creating.
        """
        if self.safety_wait_fraction <= 0.0:
            return False
        n_agents = len(getattr(sim_state, "agents", {}))
        if n_agents == 0:
            return False
        waiting = getattr(sim_state, "safety_wait_agents", None)
        if waiting is None or not callable(waiting):
            return False
        n_waiting = len(waiting())
        return (n_waiting / n_agents) >= self.safety_wait_fraction

    def step(self, sim_state: SimStateView, assignments: Dict[int, Task]) -> Optional[PlanBundle]:
        cur_step = int(sim_state.step)

        # --- A. Check Triggers ---
        periodic = (cur_step % self.replan_every == 0)

        # Check completed tasks
        completed_since = 0
        if hasattr(sim_state, "completed_tasks_since_last_plan"):
            completed_since = int(getattr(sim_state, "completed_tasks_since_last_plan"))

        # Check major deviation (set by agent controllers)
        deviation = bool(getattr(sim_state, "major_deviation", False))

        # Emergency: too many agents have exhausted / lost their global paths,
        # or too many agents are stuck in consecutive SAFETY-WAITs.
        # Guard: only fire if the previous global replan was useful (produced
        # non-trivial paths). A consistently failing solver (e.g. binary not
        # found) produces all-WAIT paths that immediately exhaust, which would
        # cause the emergency trigger to fire every min_gap steps and multiply
        # the overhead without any benefit.
        emergency_gap_ok = (cur_step - self._last_emergency_step) >= self._min_emergency_gap
        emergency_allowed = emergency_gap_ok and self._last_replan_useful
        exhaustion = emergency_allowed and self._exhaustion_trigger(sim_state)
        safety_blocked = emergency_allowed and (not exhaustion) and self._safety_wait_trigger(sim_state)
        if exhaustion or safety_blocked:
            self._last_emergency_step = cur_step

        if not (periodic or deviation or exhaustion or safety_blocked):
            return None

        # --- B. Global Planning ---

        # Build planning kwargs - always pass is_lifelong=True for rolling horizon
        # (PIBT2 uses this to select MAPD binary instead of MAPF)
        plan_kwargs = {
            "env": sim_state.env,
            "agents": sim_state.agents,
            "assignments": assignments,
            "step": cur_step,
            "horizon": self.horizon,
            "rng": None,
        }

        # Pass is_lifelong=True if the solver supports it (e.g., PIBT2)
        # This ensures PIBT2 uses the MAPD binary for lifelong experiments
        import inspect
        if "is_lifelong" in inspect.signature(self.solver.plan).parameters:
            plan_kwargs["is_lifelong"] = True

        plan = self.solver.plan(**plan_kwargs)

        # Assess whether this replan was useful: at least one agent received a
        # path that has more than one unique cell (i.e. not a pure WAIT path).
        if plan is not None and plan.paths:
            self._last_replan_useful = any(
                len(set(tp.cells)) > 1
                for tp in plan.paths.values()
                if tp is not None
            )
        else:
            self._last_replan_useful = False

        self.last_planned_step = cur_step
        self.completed_tasks_seen += completed_since

        return plan
