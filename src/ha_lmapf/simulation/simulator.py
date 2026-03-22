"""
Central Simulation Engine.

This module defines the `Simulator` class, which orchestrates the entire
Human-Aware Lifelong MAPF experiment. It manages the interaction between:
  1. The static Environment (grid map).
  2. The dynamic Task Stream (lifelong goals).
  3. The Two-Tier Planning System (Global RHCR + Local Reactive).
  4. The Physics Engine (movement, collisions).
  5. The Data Recorders (Metrics, Replay).
"""
from __future__ import annotations

import math
import re
import time as _time
from dataclasses import replace
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from ha_lmapf.core import ConflictResolver, LocalPlanner, TaskAllocator
from ha_lmapf.core.grid import manhattan
from ha_lmapf.core.metrics import MetricsTracker
from ha_lmapf.core.types import AgentState, HumanState, Metrics, Observation, PlanBundle, SimConfig, StepAction, Task
from ha_lmapf.humans.models import (
    AdversarialHumanModel,
    AisleFollowerHumanModel,
    HumanModel,
    MixedPopulationHumanModel,
    RandomWalkHumanModel,
    ReplayHumanModel,
)
from ha_lmapf.humans.prediction import MyopicPredictor
from ha_lmapf.io.replay import ReplayWriter
from ha_lmapf.io.task_stream import get_released_tasks, load_task_stream
from ha_lmapf.simulation.agent_dynamics import apply_action as apply_agent_action
from ha_lmapf.simulation.environment import Environment

# These will be generated later in your repo. We import now for wiring.
from ha_lmapf.global_tier.rolling_horizon import RollingHorizonPlanner
from ha_lmapf.global_tier.solvers.cbsh2_wrapper import CBSH2Solver
from ha_lmapf.task_allocator.task_allocator import (
    GreedyNearestTaskAllocator,
    HungarianTaskAllocator,
    AuctionBasedTaskAllocator,
)
from ha_lmapf.local_tier.agent_controller import AgentController
from ha_lmapf.local_tier.local_planner import AStarLocalPlanner
from ha_lmapf.local_tier.conflict_resolution.priority_rules import PriorityRulesResolver
from ha_lmapf.local_tier.conflict_resolution.token_passing import TokenPassingResolver
from ha_lmapf.local_tier.sensors import build_observation

Cell = Tuple[int, int]


class Simulator:
    """
    Two-tier Human-Aware Lifelong MAPF simulator.

    Architecture:
    - **Tier-1 (Global):** A Rolling Horizon Planner runs periodically (or upon major events)
      to assign tasks and compute high-level guidance paths using MAPF solvers (CBS/LaCAM).
    - **Tier-2 (Local):** Decentralized Agent Controllers run every timestep. They
      sense the local environment (Partial Observability), detect humans, and
      execute reactive avoidance or conflict resolution logic.

    Attributes:
        config (SimConfig): The configuration parameters for the run.
        env (Environment): The static grid map.
        agents (Dict[int, AgentState]): Current state of all agent agents.
        humans (Dict[int, HumanState]): Current state of all dynamic obstacles.
        metrics (MetricsTracker): Recorder for statistical performance.
    """

    # Explicit type hints for attributes initialized in __init__
    conflict_resolver: ConflictResolver
    global_planner: RollingHorizonPlanner
    local_planner: LocalPlanner
    task_allocator: TaskAllocator
    human_model_impl: HumanModel

    def __init__(self, config: SimConfig) -> None:
        """
        Initialize the simulation state.

        Args:
            config: A fully populated SimConfig object defining map, agents, humans, etc.
        """
        self.config = config
        self.rng = np.random.default_rng(int(config.seed))

        # 1. Load Environment
        self.env: Environment = Environment.load_from_map(config.map_path)

        # 2. Initialize Entity Containers
        # Place agents and humans (no overlap)
        self.agents: Dict[int, AgentState] = {}
        self.humans: Dict[int, HumanState] = {}

        # 3. Place entities on the map (Random free cells)
        # Place entities BEFORE initializing controllers
        self._place_entities()

        # 4. Load or Generate Task Stream
        self.tasks: List[Task] = []
        self._is_one_shot = getattr(config, "mode", "lifelong").lower() == "one_shot"
        self._task_mode = getattr(config, "task_mode", "poisson")
        self._task_counter: int = 0  # monotonic counter for generating task IDs

        self._task_arrival_rate = getattr(config, "task_arrival_rate", None)  # None = auto
        self._task_arrival_steps = math.ceil(config.steps * float(getattr(config, "task_arrival_percentage", 0.9)))
        self._task_arrival_steps = min(config.steps, max(1, self._task_arrival_steps))

        if config.task_stream_path:
            self.tasks = load_task_stream(config.task_stream_path)
        elif self._is_one_shot:
            self.tasks = self._generate_one_shot_tasks()
        elif self._task_mode == "immediate":
            # Li et al. (2022) style: one task per agent at step 0.
            # New tasks are generated on-demand when agents complete delivery.
            self.tasks = self._generate_initial_batch()
        else:
            self.tasks = self._generate_task_stream(
                max_steps=self._task_arrival_steps,
                release_rate=self._task_arrival_rate,
            )

        # Ensure there are enough step-0 tasks for all agents.
        # If a loaded or generated stream has fewer initial tasks than
        # agents, pad with additional pickup-delivery tasks so no agent
        # starts idle.
        if not self._is_one_shot:
            step0_count = sum(1 for t in self.tasks if t.release_step == 0)
            needed = len(self.agents) - step0_count
            if needed > 0:
                max_id = max(
                    (int(m.group()) for t in self.tasks
                     for m in [re.search(r'\d+', t.task_id)] if m),
                    default=-1,
                )
                self._task_counter = max(self._task_counter, max_id + 1)
                for i in range(needed):
                    tid = f"t{self._task_counter:07d}"
                    self._task_counter += 1
                    start = self.env.sample_free_cell(self.rng, exclude=set())
                    goal = self.env.sample_free_cell(self.rng, exclude={start})
                    self.tasks.append(Task(
                        task_id=tid, start=start, goal=goal, release_step=0
                    ))
                self.tasks.sort(key=lambda t: (t.release_step, t.task_id))

        # Initialize task counter from pre-generated tasks
        if self.tasks:
            max_id = max(
                (int(m.group()) for t in self.tasks
                 for m in [re.search(r'\d+', t.task_id)] if m),
                default=-1,
            )
            self._task_counter = max(self._task_counter, max_id + 1)

        self._pending_tasks: List[Task] = list(self.tasks)  # tasks not yet moved to open/released set
        self.open_tasks: List[Task] = []  # released. but not yet assigned or completed ones

        # Task lookup for pickup-delivery workflow
        self._task_by_id: Dict[str, Task] = {t.task_id: t for t in self.tasks}

        # 5. Planning State
        self._plans: Optional[PlanBundle] = None

        # Local replanned paths storage (for visualization)
        # Maps agent_id -> list of cells representing the local detour path
        self._local_paths: Dict[int, List[Cell]] = {}

        # Track agents whose global plans are stale (they locally replanned)
        # This set is cleared when a new global replan occurs
        self._stale_global_plan_agents: Set[int] = set()

        # Agents that received a new task mid-horizon (task completed before
        # global replan epoch). They plan locally until the next global replan,
        # which then folds them into the coordinated plan.
        self._mid_horizon_assigned: Set[int] = set()

        # Per-step event log for GUI/debugging: cleared at start of each step_once()
        self.step_events: List[str] = []

        # Triggers for Rolling Horizon
        # Bookkeeping for planner triggers
        self.step: int = 0
        self._tasks_completed_total: int = 0
        self._tasks_completed_since_last_plan: int = 0
        self._major_deviation_flag: bool = False  # optional trigger set by controllers/simulator
        self.deviation_threshold: float = float(getattr(config, "deviation_threshold", 1.0))

        # 6. Initialize: Global Planner
        time_budget_ms = float(getattr(config, "time_budget_ms", 0.0))
        solver_impl = self._make_solver_impl(
            config.global_solver,
            time_limit_sec=time_budget_ms / 1000.0 if time_budget_ms > 0 else None,
        )

        self.global_planner = RollingHorizonPlanner(
            horizon=int(config.horizon),
            replan_every=int(config.replan_every),
            solver_name=str(config.global_solver),
            solver_impl=solver_impl
        )

        # 7. Initialize: Task Allocator
        self.task_allocator = self._make_allocator(getattr(config, "task_allocator", "greedy"))

        # 8. Initialize: Local Planner
        self.human_model: HumanModel = self._make_human_model(config.human_model)
        self.human_predictor = MyopicPredictor(include_neighbors=True)

        hard_safety = bool(getattr(config, "hard_safety", True))
        self.local_planner = AStarLocalPlanner(hard_safety=hard_safety)

        # Select Conflict Resolution Strategy
        if config.communication_mode.lower() == "token":
            self.conflict_solver = TokenPassingResolver()
        else:
            self.conflict_solver = PriorityRulesResolver()

        # Create Agent Controllers (The "Brains" of the agents)
        # Initialize Controllers (Requires self.agents to be populated!)
        # Read ablation flags
        self.disable_local_replan = bool(getattr(config, "disable_local_replan", False))
        self.disable_conflict_resolution = bool(getattr(config, "disable_conflict_resolution", False))
        self._disable_safety = bool(getattr(config, "disable_safety", False))

        # If safety is disabled for ablation, override safety radius to 0
        eff_safety_radius = 0 if self._disable_safety else config.safety_radius

        fallback_wait_limit = int(getattr(config, "fallback_wait_limit", 5))
        self.controllers: Dict[int, AgentController] = {
            aid: AgentController(
                agent_id=aid,
                local_planner=self.local_planner,
                conflict_resolver=self.conflict_solver,
                fov_radius=config.fov_radius,
                safety_radius=eff_safety_radius,
                hard_safety=hard_safety and not self._disable_safety,
                fallback_wait_limit=fallback_wait_limit,
            )
            for aid in self.agents
        }

        # Execution delay state: maps agent_id -> remaining delay steps
        self._exec_delay_prob = float(getattr(config, "execution_delay_prob", 0.0))
        self._exec_delay_steps = int(getattr(config, "execution_delay_steps", 1))
        self._agent_delay: Dict[int, int] = {aid: 0 for aid in self.agents}

        # Commitment persistence parameters
        self._commit_horizon = int(getattr(config, "commit_horizon", 0))
        self._delay_threshold = float(getattr(config, "delay_threshold", 0.0))
        # Per-agent assignment tracking (populated by mark_task_assigned)
        self._assign_step: Dict[int, int] = {}  # agent_id → step of assignment
        self._assign_dist: Dict[int, float] = {}  # agent_id → d0 at assignment

        # Track decided next positions during sequential decision-making
        # This prevents collisions when agents make decisions in sequence
        self._decided_next_positions: Dict[int, Cell] = {}

        # 9. Setup Logging
        # Metrics and Replay
        self.metrics = MetricsTracker()

        for t in self.tasks:
            self.metrics.on_task_released(task_id=t.task_id, release_step=t.release_step)

        self.replay = ReplayWriter.from_config(
            map_path=config.map_path,
            seed=int(config.seed),
            config=config,
            tasks=self.tasks,
        )

        # Record initial positions at step 0
        self.replay.record(self.agents, self.humans)

    # ----------------------------------------------------------------------
    # Helpers / Constructors -----------------------------------------------

    @staticmethod
    def _make_allocator(name: str):
        """Create a task allocator by name."""
        name = (name or "greedy").lower()
        if name == "hungarian":
            return HungarianTaskAllocator()
        if name == "auction":
            return AuctionBasedTaskAllocator()
        return GreedyNearestTaskAllocator()

    @staticmethod
    def _make_solver_impl(name: str, time_limit_sec: float | None = None):
        """
        Create a solver implementation from config name using GlobalPlannerFactory.

        Supported names:
            - "cbs", "conflict_based_search": Python CBS (optimal)
            - "lacam", "lacam_like", "prioritized": Python LaCAM-like
            - "lacam3", "lacam3_cpp": Official LaCAM3 C++ executable
            - "lacam_official", "lacam_cpp": Official LaCAM C++ executable
            - "pibt2", "pibt2_cpp", "pibt": Official PIBT2 C++ executable
            - "cbs_pypi": CBS wrapper from cbs-mapf_pibt2 PyPI package

        Args:
            name: Solver name string.
            time_limit_sec: Optional solver time limit in seconds.
                            Passed as ``time_limit_sec`` kwarg to the factory.

        Returns:
            GlobalPlanner instance or None for unknown solvers
        """
        from ha_lmapf.global_tier.planner_interface import GlobalPlannerFactory

        # Special case for PyPI CBS wrapper name (redirect to CBSH2)
        if name == "cbs_pypi":
            return CBSH2Solver()

        kwargs = {}
        if time_limit_sec is not None:
            kwargs["time_limit_sec"] = time_limit_sec

        # Use the centralized factory for all other solvers
        try:
            return GlobalPlannerFactory.create(name, **kwargs)
        except ValueError:
            # Unknown solver - let RollingHorizonPlanner handle it
            return None

    def _generate_task_stream(self, max_steps: int, release_rate=None):
        """
        Deterministically generate a lifelong task stream if none provided.

        Each task has both a start (pickup) and goal (delivery) location.

        Initial batch: one task per agent released at step 0 so no agent
        starts idle. Subsequent tasks arrive via exponential inter-arrival
        times controlled by release_rate.

        release_rate: mean steps between task releases *per agent*.
            None (default) = auto-compute from map geometry: (H + W) / 3.
            This equals the expected task completion time for a random task
            on the map, giving a naturally balanced ~1× load for any fleet
            size or map dimensions — no per-config tuning required.
        """
        tasks: List[Task] = []
        counter = 0

        # Initial batch: one task per agent at step 0
        for _ in range(len(self.agents)):
            start = self.env.sample_free_cell(self.rng, exclude=set())
            goal = self.env.sample_free_cell(self.rng, exclude={start})
            tasks.append(Task(
                task_id=f"t{counter:07d}",
                start=start,
                goal=goal,
                release_step=0,
            ))
            counter += 1

        # Auto-compute release_rate from map geometry when not set explicitly.
        # Each lifelong task has TWO legs: travel to pickup + travel to delivery.
        # Average one-way Manhattan distance on H×W ≈ (H+W)/3, so a full task
        # cycle ≈ 2×(H+W)/3. Real-world overhead (congestion, safety waits,
        # conflict resolution) adds ~1.5× on top, giving an effective cycle of
        # ≈ H+W steps. Using this as release_rate keeps arrival ≈ service rate
        # for any fleet size and map, since both scale linearly with num_agents.
        if release_rate is None:
            release_rate = float(self.env.height + self.env.width)

        # Ongoing stream: Poisson arrivals after step 0.
        # release_rate is the mean inter-arrival time (steps) per agent.
        # System-wide mean inter-arrival = release_rate / num_agents.
        # This auto-scales: more agents → more tasks arrive, keeping load
        # proportional regardless of fleet size.
        # When effective_rate < 1.0 (many agents / low rate), multiple tasks
        # are released at the same step via Poisson batching rather than
        # clamping to 1 task/step, which would cause overload at large scales.
        n_agents = max(1, len(self.agents))
        effective_rate = float(release_rate) / n_agents  # mean steps between arrivals
        if effective_rate >= 1.0:
            # Normal case: one task at a time, exponential inter-arrivals
            current_step = float(self.rng.exponential(scale=effective_rate))
            while current_step < max_steps:
                start = self.env.sample_free_cell(self.rng, exclude=set())
                goal = self.env.sample_free_cell(self.rng, exclude={start})
                tasks.append(Task(
                    task_id=f"t{counter:07d}",
                    start=start,
                    goal=goal,
                    release_step=round(current_step),
                ))
                counter += 1
                current_step += float(self.rng.exponential(scale=effective_rate))
        else:
            # High-agent case: effective_rate < 1 means λ = 1/effective_rate > 1 task/step.
            # Use Poisson batching: draw the number of arrivals at each step from Poisson(λ).
            lam = 1.0 / effective_rate  # expected tasks per step
            for step in range(max_steps):
                n_arrivals = int(self.rng.poisson(lam))
                for _ in range(n_arrivals):
                    start = self.env.sample_free_cell(self.rng, exclude=set())
                    goal = self.env.sample_free_cell(self.rng, exclude={start})
                    tasks.append(Task(
                        task_id=f"t{counter:07d}",
                        start=start,
                        goal=goal,
                        release_step=step,
                    ))
                    counter += 1

        return tasks

    def _generate_initial_batch(self) -> List[Task]:
        """Generate one task per agent at step 0 for immediate-reassignment mode.

        In this mode (Li et al. 2022), each agent starts with exactly one task.
        New tasks are generated on-demand via _generate_on_demand_task() when
        an agent completes its delivery.
        """
        tasks: List[Task] = []
        for i in range(len(self.agents)):
            start = self.env.sample_free_cell(self.rng, exclude=set())
            goal = self.env.sample_free_cell(self.rng, exclude={start})
            tasks.append(Task(
                task_id=f"t{i:07d}",
                start=start,
                goal=goal,
                release_step=0,
            ))
        self._task_counter = len(tasks)
        return tasks

    def _generate_on_demand_task(self) -> Task:
        """Generate a single new task on-demand (immediate-reassignment mode).

        Called when an agent completes its delivery and needs a new task.
        The task is released at the current step and added to the open pool.
        """
        start = self.env.sample_free_cell(self.rng, exclude=set())
        goal = self.env.sample_free_cell(self.rng, exclude={start})
        task = Task(
            task_id=f"t{self._task_counter:07d}",
            start=start,
            goal=goal,
            release_step=self.step,
        )
        self._task_counter += 1
        self.tasks.append(task)
        self._task_by_id[task.task_id] = task
        self.open_tasks.append(task)
        self.metrics.on_task_released(task.task_id, self.step)
        return task

    def _generate_one_shot_tasks(self) -> List[Task]:
        """
        Generate one direct-to-goal task per agent for classical MAPF.

        No pickup phase: start is set to (-1, -1) so mark_task_assigned
        sends the agent directly to the goal. All released at step 0.
        """
        tasks: List[Task] = []
        used_goals: Set[Cell] = set()
        agent_positions: Set[Cell] = {a.pos for a in self.agents.values()}

        for aid in sorted(self.agents.keys()):
            goal = self.env.sample_free_cell(
                self.rng, exclude=used_goals | agent_positions
            )
            used_goals.add(goal)
            tasks.append(Task(
                task_id=f"oneshot_{aid}",
                start=(-1, -1),  # sentinel: skip pickup phase
                goal=goal,
                release_step=0,
            ))
        return tasks

    def _make_human_model(self, name: str) -> HumanModel:
        """
        Factory for human motion models.

        Supports all models from the paper's Human Motion Model Framework:
        - "random_walk": Random Walk with Inertia (Boltzmann softmax)
        - "aisle" / "aisle_follower" / "corridor": Aisle-Following (Boltzmann with phi field)
        - "adversarial" / "adversary": Adversarial congestion-seeking / agent-interfering
        - "mixed": Heterogeneous mixed population (per-human type sampling)
        - "replay": Deterministic trajectory playback

        Model-specific parameters are read from config.human_model_params.
        """
        name = (name or "").lower()
        params = self.config.human_model_params or {}

        if name in {"aisle", "aisle_follower", "corridor"}:
            return AisleFollowerHumanModel(
                alpha=float(params.get("alpha", 1.0)),
                beta=float(params.get("beta", 1.5)),
            )

        if name in {"adversarial", "adversary"}:
            return AdversarialHumanModel(
                gamma=float(params.get("gamma", 2.0)),
                lambda_=float(params.get("lambda", 0.5)),
            )

        if name == "mixed":
            return self._make_mixed_model(params)

        if name == "replay":
            traj_path = params.get("trajectory_path")
            if traj_path is None:
                raise ValueError(
                    "Replay human model requires 'trajectory_path' in human_model_params"
                )
            return ReplayHumanModel.from_json(str(traj_path), env=self.env)

        # Default: Random Walk with Inertia (Boltzmann)
        return RandomWalkHumanModel(
            beta_go=float(params.get("beta_go", 2.0)),
            beta_wait=float(params.get("beta_wait", -1.0)),
            beta_turn=float(params.get("beta_turn", 0.0)),
        )

    def _make_mixed_model(self, params: dict) -> MixedPopulationHumanModel:
        """
        Build a MixedPopulationHumanModel from config parameters.

        Expected params format:
            weights: {"random_walk": 0.4, "aisle": 0.4, "adversarial": 0.2}
            sub_params:
                random_walk: {beta_go: 2.0, beta_wait: -1.0, beta_turn: 0.0}
                aisle: {alpha: 1.0, beta: 1.5}
                adversarial: {gamma: 2.0, lambda: 0.5}
        """
        weights = params.get("weights", {"random_walk": 0.5, "aisle": 0.5})
        sub_params = params.get("sub_params", {})

        models = {}
        for model_name in weights:
            mp = sub_params.get(model_name, {})
            mn = model_name.lower()

            if mn in {"aisle", "aisle_follower", "corridor"}:
                models[model_name] = AisleFollowerHumanModel(
                    alpha=float(mp.get("alpha", 1.0)),
                    beta=float(mp.get("beta", 1.5)),
                )
            elif mn in {"adversarial", "adversary"}:
                models[model_name] = AdversarialHumanModel(
                    gamma=float(mp.get("gamma", 2.0)),
                    lambda_=float(mp.get("lambda", 0.5)),
                )
            else:
                models[model_name] = RandomWalkHumanModel(
                    beta_go=float(mp.get("beta_go", 2.0)),
                    beta_wait=float(mp.get("beta_wait", -1.0)),
                    beta_turn=float(mp.get("beta_turn", 0.0)),
                )

        return MixedPopulationHumanModel(models=models, weights=weights)

    def _place_entities(self) -> None:
        """Randomly spawn agents and humans on non-overlapping free cells."""
        occupied: Set[Cell] = set()

        # Place Agents
        for aid in range(int(self.config.num_agents)):
            cell = self.env.sample_free_cell(self.rng, exclude=occupied)
            occupied.add(cell)
            self.agents[aid] = AgentState(agent_id=aid, pos=cell)

        # Place Humans
        for hid in range(int(self.config.num_humans)):
            cell = self.env.sample_free_cell(self.rng, exclude=occupied)
            occupied.add(cell)
            self.humans[hid] = HumanState(human_id=hid, pos=cell, velocity=(0, 0))

    # ----------------------------------------------------------------------
    # Core Simulation ------------------------------------------------------

    def _release_tasks(self) -> None:
        """
        Move tasks from pending -> open_tasks when release_step <= current step.
        """
        if not self._pending_tasks:
            return

        released = get_released_tasks(self._pending_tasks, self.step)

        if not released:
            return

        released_ids = {t.task_id for t in released}

        self.open_tasks.extend(released)
        self.open_tasks.sort(key=lambda t: (t.release_step, t.task_id))
        self._pending_tasks = [t for t in self._pending_tasks if t.task_id not in released_ids]

        self.step_events.append(
            f"[TASKS] {len(released)} new task(s) released → open pool now has {len(self.open_tasks)}")

    def assign_tasks(self) -> Dict[int, Task]:
        """
        Assign tasks from pending list, honoring commitment persistence.

        Before running the allocator, each committed agent is checked against
        the three unlock conditions:
          1. Task completed  — handled in _check_task_completion(), not here.
          2. Horizon expired — step − t_assign ≥ commit_horizon  (if > 0).
          3. Excessive delay — manhattan(pos, goal) > delay_threshold × d0
                               where d0 is the distance at assignment time
                               (if delay_threshold > 0.0).
        Agents whose commitment is revoked are freed and their tasks are
        returned to the open pool so the allocator can reassign them.
        """
        # --- Commitment Persistence: check unlock conditions ---
        if self._commit_horizon > 0 or self._delay_threshold > 0.0:
            for aid in list(self.agents.keys()):
                a = self.agents[aid]
                if a.task_id is None or a.goal is None:
                    continue  # agent already idle

                should_break = False

                # Condition 2: commitment horizon expired
                if self._commit_horizon > 0:
                    t_assign = self._assign_step.get(aid)
                    if t_assign is not None and (self.step - t_assign) >= self._commit_horizon:
                        should_break = True

                # Condition 3: excessive delay relative to initial distance
                if not should_break and self._delay_threshold > 0.0:
                    d0 = self._assign_dist.get(aid, 0.0)
                    if d0 > 0.0:
                        current_dist = float(manhattan(a.pos, a.goal))
                        if current_dist > self._delay_threshold * d0:
                            should_break = True

                if should_break:
                    # Revoke commitment: return task to open pool, free agent
                    task = self._task_by_id.get(a.task_id)
                    if task is not None and not any(
                            t.task_id == task.task_id for t in self.open_tasks
                    ):
                        self.open_tasks.append(task)
                    self.agents[aid] = replace(a, goal=None, task_id=None, carrying=False)
                    self._assign_step.pop(aid, None)
                    self._assign_dist.pop(aid, None)
                    self.metrics.add_assignment_broken()

        # Run Allocator
        unassigned_tasks: Set[Task] = set(self.open_tasks)
        self.open_tasks.clear()

        assignments: Dict[int, Task] = self.task_allocator.assign(self.agents, unassigned_tasks, self.step)

        all_assignments = {}

        # Register Assignments
        for aid, task in assignments.items():
            self.mark_task_assigned(task, aid)

            unassigned_tasks.remove(task)

            # Clean current path to trigger local-replanner
            self.controllers[aid].clear_path(self)

            self.step_events.append(
                f"[ASSIGN-GLOBAL] Agent {aid} ← task {task.task_id} "
                f"(pickup={task.start}, delivery={task.goal})"
            )

            # For planner
            a = self.agents[aid]
            all_assignments[aid] = Task(
                task_id=task.task_id,
                start=a.pos,
                goal=a.goal,
                release_step=0
            )

        self.open_tasks.extend(unassigned_tasks)

        # Add existing assignments
        for aid, agent in self.agents.items():
            if aid not in all_assignments and agent.task_id is not None and agent.goal is not None:
                # For agents already working on a task, create a synthetic Task
                # Use agent's current position as start (they're already in progress)

                all_assignments[aid] = Task(
                    task_id=agent.task_id,
                    start=agent.pos,  # Current position as start for planning
                    goal=agent.goal,
                    release_step=0
                )

        return all_assignments

    def maybe_global_replan(self, assignments: Optional[Dict[int, Task]] = None) -> None:
        """
        Invoke Tier-1 Planner if specific triggers (time interval, task completion) are met.
        Updates self.plans with new global paths if replanning occurs.
        Instruments wall-clock planning time.
        """
        if assignments is None:
            self._release_tasks()
            assignments = self.assign_tasks()

        t0 = _time.perf_counter()
        plan = self.global_planner.step(self, assignments)
        elapsed_ms = (_time.perf_counter() - t0) * 1000.0

        if plan is not None:
            self._plans = plan

            # Determine trigger reason for logging
            periodic = (int(self.step) % self.global_planner.replan_every == 0)
            n_stale = len(self._stale_global_plan_agents)
            n_safe_wait = len(self.safety_wait_agents())
            n_agents = max(len(self.agents), 1)
            if periodic:
                trigger = "periodic"
            elif n_stale / n_agents >= getattr(self.global_planner, "exhaustion_fraction", 1.0):
                trigger = f"exhaustion({n_stale}/{n_agents})"
            elif n_safe_wait / n_agents >= getattr(self.global_planner, "safety_wait_fraction", 1.0):
                trigger = f"safety-wait({n_safe_wait}/{n_agents})"
            else:
                trigger = "major-deviation"
            mid_count = len(self._mid_horizon_assigned)
            mid_note = f", folding {mid_count} mid-horizon agent(s)" if mid_count else ""
            self.step_events.append(
                f"[GLOBAL-REPLAN] Tier-1 fired (trigger={trigger}, "
                f"agents={len(plan.paths)}, time={elapsed_ms:.1f}ms{mid_note})"
            )

            for aid, controller in self.controllers.items():
                if aid not in plan.paths:
                    controller.global_path = None
                else:
                    controller.global_path = plan.paths[aid]

            self._tasks_completed_since_last_plan = 0
            self._major_deviation_flag = False
            # Clear stale plans set - all agents now have fresh global plans
            self._stale_global_plan_agents.clear()
            self._local_paths.clear()  # Also clear local paths since we have fresh global plans
            # Mid-horizon assigned agents have now been folded into the global plan
            self._mid_horizon_assigned.clear()
            self.metrics.add_replan(1)
            self.metrics.add_global_replan(1)
            self.metrics.record_planning_time_ms(elapsed_ms)

    def _update_humans(self) -> None:
        """Advance all human positions using the stochastic motion model.

        Humans treat agent positions as obstacles - they will not move into
        cells occupied by agents.
        """
        # Collect current agent positions so humans treat them as obstacles
        agent_positions: Set[Cell] = {a.pos for a in self.agents.values()}
        self.humans = self.human_model.step(self.env, self.humans, self.rng, agent_positions)

    def _detect_collisions_and_near_misses(
            self,
            prev_pos: Dict[int, Cell],
            new_pos: Dict[int, Cell],
    ) -> None:
        """
        Check for safety violations after movement.

        Checks:
        1. Agent-Agent Vertex Collision: Two agents on same cell.
        2. Agent-Agent Edge Collision: Two agents swapped cells.
        3. Agent-Human Collision: Agent on same cell as human.
        4. Near Miss: Agent within L1 distance <= 1 of a human.
        """
        # Agent-Agent Vertex Collisions (same cell)
        cell_to_agents: Dict[Cell, List[int]] = {}
        for aid, p in new_pos.items():
            cell_to_agents.setdefault(p, []).append(aid)
        for cell, aids in cell_to_agents.items():
            if len(aids) > 1:
                # Count one collision event per additional agent (conservative)
                self.metrics.add_agent_agent_collision(len(aids) - 1)

        # Agent-Agent Edge Swap Collisions
        aids = sorted(new_pos.keys())
        for i in range(len(aids)):
            for j in range(i + 1, len(aids)):
                a, b = aids[i], aids[j]
                if prev_pos[a] == new_pos[b] and prev_pos[b] == new_pos[a] and prev_pos[a] != new_pos[a]:
                    self.metrics.add_agent_agent_collision(1)

        # Agent-Human proximity metrics.
        # Exclude stationary agents: if an agent didn't move this tick,
        # it's either idle or safety-waiting. In both cases the human
        # approached the agent, not the other way around.
        human_cells = {h.pos for h in self.humans.values()}
        moved_agents = {
            aid for aid in new_pos
            if aid in prev_pos and prev_pos[aid] != new_pos[aid]
        }

        # Agent-Human Collisions (same cell) — only if agent moved into it
        for aid in moved_agents:
            if new_pos[aid] in human_cells:
                self.metrics.add_agent_human_collision(1)

        # Near misses (Manhattan distance <= 1) — only for agents that moved
        for aid in moved_agents:
            p = new_pos[aid]
            for h in self.humans.values():
                if abs(p[0] - h.pos[0]) + abs(p[1] - h.pos[1]) <= 1:
                    self.metrics.add_near_miss(1)
                    break

        # Safety buffer violations — only for agents that moved
        safety_r = int(getattr(self.config, "safety_radius", 1))
        if human_cells and safety_r > 0:
            from ha_lmapf.humans.safety import inflate_cells
            forbidden = inflate_cells(human_cells, radius=safety_r, env=self.env)
            for aid in moved_agents:
                if new_pos[aid] in forbidden:
                    self.metrics.add_safety_violation(1)

        # Human passive waiting: humans stationary with an agent within safety_radius
        agent_cell_set = set(new_pos.values())
        for h in self.humans.values():
            if h.velocity == (0, 0):
                for ap in agent_cell_set:
                    if abs(h.pos[0] - ap[0]) + abs(h.pos[1] - ap[1]) <= safety_r:
                        self.metrics.add_human_passive_wait(1)
                        break

    def _maybe_complete_tasks(self) -> None:
        """
        Handle pickup-delivery task progression:

        Phase 1 (carrying=False): Agent navigates to start (pickup) location.
                                  When reached, picks up item (carrying=True)
                                  and goal is updated to delivery location.

        Phase 2 (carrying=True): Agent navigates to goal (delivery) location.
                                 When reached, task is completed and agent is freed.
        """
        for aid in sorted(self.agents.keys()):
            a = self.agents[aid]
            if a.goal is None or a.task_id is None:
                continue

            task = self._task_by_id.get(a.task_id)

            if task is None:
                continue

            if task.start != (-1, -1):  # Pick and Delivery
                if not a.carrying and a.pos != task.start:  # Phase 1 is not completed.
                    continue
                elif a.carrying and a.pos != task.goal:  # Phase 2 is not completed
                    continue
            else:
                if a.pos != task.goal:  # Task is not completed.
                    continue

            if not a.carrying:
                # Phase 1 complete: Agent reached pickup location
                # Now set goal to delivery location and mark as carrying
                self.agents[aid] = replace(
                    a,
                    goal=task.goal,
                    carrying=True,
                )

                # Reset d0 for the delivery phase so the delay_threshold
                # check is relative to the pickup→delivery distance, not the
                # original agent→pickup distance.
                self._assign_dist[aid] = float(manhattan(a.pos, task.goal))

                # Clear path and stale global_path so controller does not
                # compute deviation against the old Phase-1 path.
                self.controllers[aid].clear_path(self)
                self.controllers[aid].global_path = None

                self.step_events.append(
                    f"[PICKUP] Agent {aid} picked up task {task.task_id} "
                    f"at {a.pos} → now delivering to {task.goal}"
                )
            else:
                # Phase 2 complete: Agent reached delivery location.
                # Condition 1 of commitment persistence: task completed normally.
                self.metrics.on_task_completed(a.task_id, aid, self.step)
                self.metrics.update_makespan(self.step)

                # Sum-of-costs: path length = completion_step - assigned_step
                rec = self.metrics._tasks.get(a.task_id)
                if rec and rec.assigned_step is not None:
                    self.metrics.add_path_cost(self.step - rec.assigned_step)

                self._tasks_completed_total += 1
                self._tasks_completed_since_last_plan += 1

                # Count as a kept assignment (survived to natural completion)
                self.metrics.add_assignment_kept()
                self._assign_step.pop(aid, None)
                self._assign_dist.pop(aid, None)

                self.agents[aid] = replace(
                    a,
                    goal=None,
                    carrying=False,
                    task_id=None,
                    done_tasks=a.done_tasks + 1,
                )

                self.step_events.append(
                    f"[DELIVERY] Agent {aid} completed task {task.task_id} "
                    f"at {a.pos} (total done: {a.done_tasks + 1})"
                )

                # Mid-horizon assignment: immediately try to give this agent a
                # new task so it starts local planning without an idle step.
                self._try_mid_horizon_assign(aid)

    def _try_mid_horizon_assign(self, aid: int) -> bool:
        """
        Assign a new task to an agent that just completed its task mid-horizon.

        Called immediately inside _maybe_complete_tasks (step 9) so the agent
        begins local A* planning toward its new goal in the very next tick,
        without waiting for the step-2 assignment of the following step_once.

        The agent is recorded in _mid_horizon_assigned so the global planner
        folds it into the coordinated plan at the next replan epoch.

        Returns True if a task was successfully assigned.
        """
        # In immediate mode, generate a new task on-demand if pool is empty
        if not self.open_tasks and self._task_mode == "immediate":
            self._generate_on_demand_task()

        if not self.open_tasks:
            return False

        # Run allocator with only this agent against the open pool
        assignments = self.task_allocator.assign(
            {aid: self.agents[aid]}, set(self.open_tasks), self.step
        )
        if aid not in assignments:
            return False

        task = assignments[aid]
        self.open_tasks = [t for t in self.open_tasks if t.task_id != task.task_id]
        self.mark_task_assigned(task, aid)
        # Clear path so local controller immediately runs A* toward new goal
        self.controllers[aid].clear_path(self)
        self._mid_horizon_assigned.add(aid)

        self.step_events.append(
            f"[ASSIGN-MID-HORIZON] Agent {aid} ← task {task.task_id} "
            f"(pickup={task.start}, delivery={task.goal}) → local A* until next global replan"
        )
        return True

    def step_once(self) -> None:
        """
        Execute one full simulation tick (The "Game Loop").

        Order of Operations:
          1. Release new tasks.
          2. Assign new tasks.
          3. Tier-1 Planning (Global).
          4. Human Movement (Environment Dynamics).
          5. Tier-2 Execution (Local Sense-Plan-Act).
          6. Execution Delay Injection.
          7. Physics Update (Apply Moves).
          8. Collision Detection.
          9. Task Completion + Makespan/SOC tracking.
          10. Replay Recording.
        """
        step_t0 = _time.perf_counter()
        self.step_events.clear()

        # 1. Task Management
        self._release_tasks()

        # 2. Assign new tasks
        assignments = self.assign_tasks()

        # 3. Global Planning
        self.maybe_global_replan(assignments)

        # 4. Environment Dynamics (Humans)
        self._update_humans()

        # 5. Tier-2: Sense-Plan-Act
        # 5a. Build partial observations
        observations: Dict[int, Observation] = {}
        for aid in sorted(self.agents.keys()):
            observations[aid] = build_observation(aid, self, fov_radius=int(self.config.fov_radius))

        # 5b. Decide Actions (Local Controller)
        # Clear decided positions from previous step - agents will register as they decide
        self._decided_next_positions.clear()

        actions: Dict[int, StepAction] = {}
        prev_pos: Dict[int, Cell] = {aid: self.agents[aid].pos for aid in self.agents.keys()}

        for aid in sorted(self.agents.keys()):
            act = self.controllers[aid].decide_action(self, observations[aid], rng=self.rng)
            actions[aid] = act
            if act == StepAction.WAIT and self.agents[aid].goal is not None and self.agents[aid].task_id is not None:
                self.metrics.add_wait_steps(1)
            # Register this agent's intended next position for conflict detection
            # by later agents in this same decision loop
            next_pos = self._compute_next_position(prev_pos[aid], act)
            self._decided_next_positions[aid] = next_pos

        # 6. Execution Delay Injection (robust-MAPF)
        if self._exec_delay_prob > 0:
            for aid in sorted(self.agents.keys()):
                # Decrement existing delays
                if self._agent_delay[aid] > 0:
                    self._agent_delay[aid] -= 1
                    actions[aid] = StepAction.WAIT  # force wait during delay
                    continue
                # Probabilistically inject new delay
                if self.rng.random() < self._exec_delay_prob:
                    self._agent_delay[aid] = self._exec_delay_steps
                    actions[aid] = StepAction.WAIT
                    self.metrics.add_delay_event(1)
                    self.controllers[aid].clear_path(self)  # Clear the path when delay occurs

        # 7. Physics Update
        # 7a. Collision prevention: compute intended positions, then revert
        #     any agent that would cause a vertex or edge-swap collision.
        sorted_aids = sorted(self.agents.keys())
        intended: Dict[int, Cell] = {
            aid: self._compute_next_position(prev_pos[aid], actions[aid])
            for aid in sorted_aids
        }

        # Iteratively resolve: later agents (by id) yield to earlier ones
        changed = True
        while changed:
            changed = False
            claimed: Dict[Cell, int] = {}

            # Pre-claim cells of agents that are already waiting/staying in place.
            # A moving agent must never displace a stationary agent — without
            # pre-claiming, a lower-id mover could "win" the vertex check before
            # the staying agent is processed, leaving the staying agent with no
            # way to yield (intended == prev_pos → revert branch never fires).
            for aid in sorted_aids:
                if intended[aid] == prev_pos[aid]:
                    claimed[intended[aid]] = aid

            for aid in sorted_aids:
                nxt = intended[aid]
                conflict = False

                # Vertex conflict: cell already claimed by an earlier agent
                # (or pre-claimed by a stationary agent)
                if nxt in claimed and claimed[nxt] != aid:
                    conflict = True

                # Edge-swap conflict: agents would swap positions
                if not conflict:
                    for oid in sorted_aids:
                        if oid == aid:
                            continue
                        if (prev_pos[oid] == nxt and intended[oid] == prev_pos[aid]
                                and nxt != prev_pos[aid]):
                            conflict = True
                            break

                if conflict:
                    if intended[aid] != prev_pos[aid]:
                        # Revert this agent to WAIT
                        actions[aid] = StepAction.WAIT
                        intended[aid] = prev_pos[aid]
                        changed = True  # Re-check since this change may resolve/create others

                claimed[intended[aid]] = aid

        # 7b. Apply validated actions
        for aid in sorted_aids:
            self.agents[aid] = apply_agent_action(self.env, self.agents[aid], actions[aid])

        # 8. Safety Checks
        new_pos: Dict[int, Cell] = {aid: self.agents[aid].pos for aid in self.agents.keys()}
        self._detect_collisions_and_near_misses(prev_pos, new_pos)

        # 9. Logic Update (task completion with makespan/SOC tracking)
        self._maybe_complete_tasks()

        # 10. Logging
        self.replay.record(self.agents, self.humans)

        # Record per-step decision time
        step_elapsed_ms = (_time.perf_counter() - step_t0) * 1000.0
        self.metrics.record_decision_time_ms(step_elapsed_ms)

        # Advance Time
        self.step += 1

    def run(self, steps: Optional[int] = None) -> Metrics:
        """
        Run the simulation for a specified number of steps.

        In one-shot mode, terminates early when all agents have reached their goals.

        Args:
            steps: Number of steps to simulate. If None, uses config.steps.

        Returns:
            Final Metrics object.
        """
        total = int(steps if steps is not None else self.config.steps)
        for _ in range(total):
            self.step_once()
            # Early termination for one-shot mode: all agents idle (at goal)
            if self._is_one_shot and self._all_agents_idle():
                break
        return self.metrics.finalize(total_steps=self.step)

    def _all_agents_idle(self) -> bool:
        """Check if every agent has no goal (all tasks completed)."""
        return all(a.goal is None for a in self.agents.values())

    @staticmethod
    def _compute_next_position(cur: Cell, action: StepAction) -> Cell:
        """Compute the next position given current position and action."""
        if action == StepAction.WAIT:
            return cur
        if action == StepAction.UP:
            return (cur[0] - 1, cur[1])
        if action == StepAction.DOWN:
            return (cur[0] + 1, cur[1])
        if action == StepAction.LEFT:
            return (cur[0], cur[1] - 1)
        if action == StepAction.RIGHT:
            return (cur[0], cur[1] + 1)
        return cur

    # ----------------------------------------------------------------------
    # Compatibility hooks / properties -------------------------------------

    def plans(self) -> Optional[PlanBundle]:
        return self._plans

    def decided_next_positions(self) -> Dict[int, Cell]:
        """
        Return positions that agents have already decided to move to this timestep.

        Used by conflict resolution during sequential decision-making to prevent
        later agents from colliding with earlier agents' committed moves.
        """
        return self._decided_next_positions

    def local_paths(self) -> Dict[int, List[Cell]]:
        """Return the current local replanned paths for visualization."""
        return self._local_paths

    def set_local_path(self, agent_id: int, path: List[Cell]) -> None:
        """Store a local replanned path for an agent and mark their global plan as stale."""
        self._local_paths[agent_id] = path
        # Mark this agent's global plan as stale since they've locally replanned
        self._stale_global_plan_agents.add(agent_id)

    def clear_local_path(self, agent_id: int) -> None:
        """Clear the local path for an agent (e.g., when following global plan)."""
        self._local_paths.pop(agent_id, None)
        # Note: We do NOT remove from _stale_global_plan_agents here
        # Once an agent has deviated, their global plan remains stale until next global replan

    def is_global_plan_stale(self, agent_id: int) -> bool:
        """Check if an agent's global plan is stale (they locally replanned since last global replan)."""
        return agent_id in self._stale_global_plan_agents

    def stale_global_plan_agents(self) -> Set[int]:
        """Return the set of agents whose global plans are stale."""
        return self._stale_global_plan_agents

    # ------------------------------------------------------------------
    # Safety-wait tracking
    # ------------------------------------------------------------------

    def mark_safety_wait(self, agent_id: int) -> None:
        """Record that an agent is blocked in a SAFETY-WAIT this step."""
        if not hasattr(self, "_safety_wait_agents"):
            self._safety_wait_agents: Set[int] = set()
        self._safety_wait_agents.add(agent_id)

    def clear_safety_wait(self, agent_id: int) -> None:
        """Clear SAFETY-WAIT status once an agent moves successfully."""
        if hasattr(self, "_safety_wait_agents"):
            self._safety_wait_agents.discard(agent_id)

    def safety_wait_agents(self) -> Set[int]:
        """Return agents currently stuck in a SAFETY-WAIT."""
        if not hasattr(self, "_safety_wait_agents"):
            self._safety_wait_agents = set()
        return self._safety_wait_agents

    @property
    def completed_tasks_since_last_plan(self) -> int:
        """Number of tasks finished since the last global replan triggered."""
        return self._tasks_completed_since_last_plan

    @property
    def major_deviation(self) -> bool:
        """Flag indicating if a local controller has requested a global replan."""
        return self._major_deviation_flag

    def flag_major_deviation(self) -> None:
        """Allow controllers to signal that they are hopelessly stuck or large replans"""
        self._major_deviation_flag = True

    def mark_task_assigned(self, task: Task, agent_id: int) -> None:
        """
        Callback for the Task Allocator to register an assignment.
        Updates metrics and agent state.

        For pickup-delivery tasks:
        - If task.start is valid (not -1,-1), agent first goes to pickup location
        - Agent starts with carrying=False, will pick up when reaching start
        - If task.start is (-1,-1) (legacy format), agent goes directly to goal
        """
        self.metrics.on_task_assigned(task.task_id, agent_id, self.step)
        a = self.agents[agent_id]

        # Determine initial goal based on whether task has a valid start location
        if task.start != (-1, -1):
            # Pickup-delivery task: first go to pickup (start) location
            initial_goal = task.start
            initial_carrying = False
        else:
            # Legacy delivery-only task: go directly to goal
            initial_goal = task.goal
            initial_carrying = True

        self.agents[agent_id] = replace(
            a,
            goal=initial_goal,
            task_id=task.task_id,
            carrying=initial_carrying
        )

        # Record commitment baseline for this assignment
        self._assign_step[agent_id] = self.step
        self._assign_dist[agent_id] = float(manhattan(a.pos, initial_goal))
