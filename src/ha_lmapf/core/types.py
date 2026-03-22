"""
Core Type Definitions for Human-Aware Lifelong MAPF.

This module defines the fundamental data structures used throughout the
simulator and planning tiers. It includes:
  - Enums for agent actions.
  - Dataclasses for entity states (Agent, Human).
  - Structures for tasks, plans, and observations.
  - Metric containers for experiment logging.
  - Configuration schemas.

All types support serialization to dictionaries via `to_dict()` methods
to facilitate JSON logging and replay generation.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any, Literal


# Actions --------------------------------------------
class StepAction(Enum):
    """
    Enumeration of valid discrete actions an agent can take in a single time step.

    The grid system assumes a 4-connected graph (Von Neumann neighborhood) plus a wait action.
    """
    UP = "UP"  # Move (row - 1, col)
    DOWN = "DOWN"  # Move (row + 1, col)
    LEFT = "LEFT"  # Move (row, col - 1)
    RIGHT = "RIGHT"  # Move (row, col + 1)
    WAIT = "WAIT"  # Stay at current position (row, col)


# Core State Types ------------------------------------

@dataclass
class AgentState:
    """
    Represents the snapshot of an agent's status at a specific simulation step.

    Attributes:
        agent_id: Unique identifier for the agent.
        pos: Current grid position (row, col).
        goal: Current target position the agent is moving towards, if any.
        carrying: Boolean flag indicating if the agent is currently transporting a load.
        task_id: The ID of the task currently assigned to this agent, if any.
        done_tasks: Cumulative count of tasks completed by this agent so far.
        wait_steps: Cumulative count of steps the agent spent waiting (metrics).
    """
    agent_id: int
    pos: Tuple[int, int]
    goal: Optional[Tuple[int, int]] = None
    carrying: bool = False
    task_id: Optional[str] = None
    done_tasks: int = 0
    wait_steps: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize state to a dictionary for logging/JSON export."""
        return asdict(self)


@dataclass
class HumanState:
    """
    Represents the snapshot of a human's status at a specific simulation step.

    Attributes:
        human_id: Unique identifier for the human.
        pos: Current grid position (row, col).
        velocity: Approximate velocity vector (d_row, d_col) observed.
                  Default is (0,0) if stationary or unknown.
    """
    human_id: int
    pos: Tuple[int, int]
    velocity: Tuple[int, int] = (0, 0)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize state to a dictionary for logging/JSON export."""
        return asdict(self)


@dataclass
class Task:
    """
    Represents a pickup-delivery task to be performed by an agent.

    A task involves two phases:
      1. Pickup: Move to the start (pickup) location
      2. Delivery: Move from start to goal (delivery) location

    The task allocator assigns tasks to agents based on their distance
    to the start (pickup) location, not the goal.

    Attributes:
        task_id: Unique string identifier for the task.
        start: The pickup location grid cell (row, col).
        goal: The delivery destination grid cell (row, col).
        release_step: The simulation step when this task became available.
    """
    task_id: str
    start: Tuple[int, int]
    goal: Tuple[int, int]
    release_step: int

    def to_dict(self) -> Dict[str, Any]:
        """Serialize task to a dictionary for logging/JSON export."""
        return asdict(self)

    def __hash__(self):
        return hash(self.task_id)


# Planning Structures ---------------------------------

@dataclass
class TimedPath:
    """
    A sequence of time-indexed positions representing an agent's planned route.

    The path is anchored to a specific `start_step`. This allows the path
    object to answer "Where should I be at step t?" correctly.

    Attributes:
        cells: List of (row, col) tuples. cells[0] is the position at start_step.
        start_step: The global simulation step corresponding to cells[0].
    """
    cells: List[Tuple[int, int]]
    start_step: int

    def __call__(self, step: int) -> Tuple[int, int]:
        """
        Get the planned position at a specific global simulation step.

        Robustness behavior:
          - If step < start_step: Returns the first position (Wait at start).
          - If step > start_step + len(cells): Returns the last position (Stay at goal).

        Args:
            step: The global simulation time step.

        Returns:
            The (row, col) tuple for that step.
        """
        idx = step - self.start_step

        if idx < 0:
            return self.cells[0]

        if idx >= len(self.cells):
            return self.cells[-1]

        return self.cells[idx]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize path to a dictionary."""
        return {
            "cells": list(self.cells),
            "start_step": self.start_step,
        }


@dataclass
class PlanBundle:
    """
    A collection of paths for multiple agents generated by the global planner.

    Attributes:
        paths: Mapping from agent_id to their TimedPath.
        created_step: The simulation step when this plan was generated.
        horizon: The planning horizon used (how far into the future these paths go).
    """
    paths: Dict[int, Optional[TimedPath]]
    created_step: int
    horizon: int

    def to_dict(self) -> Dict[str, Any]:
        """Serialize bundle to a dictionary for logging."""
        return {
            "paths": {aid: path.to_dict() for aid, path in self.paths.items()},
            "created_step": self.created_step,
            "horizon": self.horizon,
        }


# Local Observation -----------------------------------

@dataclass
class Observation:
    """
    The sensory input available to a single agent during the execution phase.

    Reflects Partial Observability: only contains entities within the
    agent's Field of View (FOV).

    Attributes:
        visible_humans: Dict of {human_id: HumanState} currently inside FOV.
        visible_agents: Dict of {agent_id: AgentState} currently inside FOV.
        blocked: Set of coordinates considered 'unsafe' or 'occupied'
                 (e.g., inflated static obstacles or dynamic zones).
    """
    visible_humans: Dict[int, HumanState] = field(default_factory=dict)
    visible_agents: Dict[int, AgentState] = field(default_factory=dict)
    blocked: Set[Tuple[int, int]] = field(default_factory=set)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize observation to a dictionary."""
        return {
            "visible_humans": {hid: human.to_dict() for hid, human in self.visible_humans.items()},
            "visible_agents": {aid: agent.to_dict() for aid, agent in self.visible_agents.items()},
            "blocked": list(self.blocked),
        }


# Metrics ----------------------------------------------

@dataclass
class Metrics:
    """
    Container for aggregate performance statistics of a simulation run.

    Attributes:
        throughput: Rate of tasks completed per time step (completed / steps).
        completed_tasks: Total number of tasks finished.
        task_completion: The percentage of completed tasks completed.
        mean_flowtime: Average time (steps) from task release to completion.
        collisions_agent_agent: Count of agent-agent collisions detected.
        collisions_agent_human: Count of agent-human collisions detected.
        near_misses: Count of dangerous proximity events (e.g., distance <= 1).
        replans: Number of times local or global replanning was triggered.
        total_wait_steps: Sum of steps all agents spent waiting.
        steps: Total duration of the simulation run.
        safety_violations: Times an agent entered B_r(H_t) forbidden zone.
        safety_violation_rate: Safety violations per 1000 timesteps.
        global_replans: Number of Tier-1 global replanning events.
        local_replans: Number of Tier-2 local replanning events.
        intervention_rate: Global replans per 1000 timesteps (how often
            global planner must intervene due to local deadlocks).
        mean_service_time: Average steps from task assignment to completion.
        median_flowtime: Median flowtime across completed tasks.
        max_flowtime: Maximum flowtime across completed tasks.
        human_passive_wait_steps: Steps humans spent waiting due to agent proximity.
    """
    throughput: float = .0
    completed_tasks: int = 0
    total_released_tasks: int = 0
    task_completion: float = 0.
    mean_flowtime: float = .0
    collisions_agent_agent: int = 0
    collisions_agent_human: int = 0
    near_misses: int = 0
    replans: int = 0
    total_wait_steps: int = 0
    steps: int = 0
    safety_violations: int = 0
    safety_violation_rate: float = .0
    global_replans: int = 0
    local_replans: int = 0
    intervention_rate: float = .0
    mean_service_time: float = .0
    median_flowtime: float = .0
    max_flowtime: float = .0
    human_passive_wait_steps: int = 0
    # Timing metrics (wall-clock, in seconds)
    mean_planning_time_ms: float = .0
    p95_planning_time_ms: float = .0
    max_planning_time_ms: float = .0
    mean_decision_time_ms: float = .0
    p95_decision_time_ms: float = .0
    # Cost-based metrics
    makespan: int = 0
    sum_of_costs: int = 0
    # Delay robustness
    delay_events: int = 0
    immediate_assignments: int = 0
    assignments_kept: int = 0
    assignments_broken: int = 0
    # Per-step cumulative throughput timeline (for convergence analysis)
    throughput_timeline: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize metrics to a dictionary."""
        return asdict(self)


# Simulation Configuration -----------------------------

@dataclass
class SimConfig:
    """
    Configuration parameters defining a complete experiment setup.

    Attributes:
        map_path: Path to the static grid map file (.map format).
        task_stream_path: Path to pre-generated task file (optional).
                          If None, tasks may be generated randomly.
        seed: Random seed for reproducibility.
        steps: Maximum number of simulation steps to run.
        num_agents: Number of agent agents in the fleet.
        num_humans: Number of dynamic human obstacles to simulate.
        fov_radius: Radius (Manhattan distance) of agent's sensor field of view.
        safety_radius: Buffer distance to maintain around detected humans.
        global_solver: Name of the solver for Tier-1.
                       Python: "cbs" (optimal), "lacam" (fast, approximate).
                       Official C++: "lacam3", "lacam_official", "pibt2" (requires build).
        replan_every: Interval (steps) for triggering global re-planning.
        horizon: Time horizon for the global rolling planner.
        deviation_threshold: Min. change ratio between global and local paths to recall global planner
        communication_mode: Protocol for conflict resolution ("token", "priority").
        local_planner: Algorithm for Tier-2 local pathfinding ("astar").
        human_model: Motion model for humans.
                     Options: "random_walk", "aisle", "adversarial", "mixed", "replay".
        human_model_params: Model-specific parameters passed to the human model constructor.
                            - random_walk: {"beta_go": 2.0, "beta_wait": -1.0, "beta_turn": 0.0}
                            - aisle: {"alpha": 1.0, "beta": 1.5}
                            - adversarial: {"gamma": 2.0, "lambda": 0.5}
                            - mixed: {"weights": {"random_walk": 0.4, "aisle": 0.4,
                                      "adversarial": 0.2}, "sub_params": {...}}
                            - replay: {"trajectory_path": "path/to/replay.json"}
        hard_safety: If True, agents MUST NOT enter B_r(H_t) (hard constraint per the paper).
                     If False, safety buffer is a high-cost soft constraint (prevents deadlock).
        mode: Experiment mode.
              "lifelong" — continuous pickup-delivery task stream with rolling-horizon replanning.
              "one_shot" — classical MAPF: each agent gets one goal at step 0, planned once,
                           task done when goal reached. No pickup phase, no replanning.
        task_allocator: Strategy for assigning tasks to agents.
                        "greedy" — nearest-task by Manhattan distance (default).
                        "hungarian" — optimal matching via Hungarian algorithm.
                        "auction" — sequential single-item auction.
        execution_delay_prob: Probability [0,1] that an agent's move is delayed at each step.
        execution_delay_steps: Duration (in steps) of each injected delay.
        time_budget_ms: Max wall-clock ms per global planning call (0 = unlimited).
        task_arrival_rate: Mean steps between task releases *per agent* (higher = slower arrival).
            System-wide arrival rate = num_agents / task_arrival_rate tasks/step.
            None (default) = auto-compute as height + width, which accounts for the
            two-leg task cycle (pickup + delivery) plus typical congestion overhead,
            giving a balanced ~85% utilization for any fleet size or map dimensions.
        task_arrival_percentage: Fraction (0,1] of steps during which new tasks are released.
        disable_local_replan: Ablation flag — disable Tier-2 local replanning.
        disable_conflict_resolution: Ablation flag — disable decentralized conflict resolution.
        disable_safety: Ablation flag — disable human safety buffer entirely.
    """
    map_path: str
    task_stream_path: Optional[str] = None
    seed: int = 0
    steps: int = 1000
    num_agents: int = 1
    num_humans: int = 0
    fov_radius: int = 4
    safety_radius: int = 1
    global_solver: str = "cbs"
    replan_every: int = 25
    horizon: int = 50
    deviation_threshold: float = 1.0
    communication_mode: Literal["token", "priority"] = "token"
    local_planner: str = "astar"
    human_model: str = "random_walk"
    human_model_params: Dict[str, Any] = field(default_factory=dict)
    hard_safety: bool = True
    mode: Literal["lifelong", "one_shot"] = "lifelong"
    task_allocator: Literal["greedy", "hungarian", "auction"] = "greedy"
    # Delay robustness
    execution_delay_prob: float = 0.0
    execution_delay_steps: int = 1
    # Planning time budget (0 = unlimited)
    time_budget_ms: float = 0.0
    # Task generation mode:
    #   "poisson"   — pre-generate stream with Poisson inter-arrivals (original).
    #   "immediate"  — Li et al. (2022) style: one task per agent at step 0,
    #                  new task generated on-demand when agent completes delivery.
    #                  Guarantees every agent always has exactly one task.
    task_mode: Literal["poisson", "immediate"] = "poisson"
    # Task generation rate (mean steps between releases per agent).
    # Only used when task_mode="poisson".
    # None = auto-compute from map geometry: (height + width) / 3.
    task_arrival_rate: Optional[float] = None
    task_arrival_percentage: float = 0.9
    # Commitment persistence for task allocation.
    # commit_horizon: max steps an assignment stays locked (0 = disabled).
    # delay_threshold: revoke if current distance > threshold × d0, where
    #   d0 is the Manhattan distance to the goal at the moment of assignment
    #   (0.0 = disabled).  Both conditions are independent; either can fire.
    commit_horizon: int = 0
    delay_threshold: float = 0.0
    # Ablation flags
    disable_local_replan: bool = False
    disable_conflict_resolution: bool = False
    disable_safety: bool = False
    # When local replan is disabled, after this many consecutive safety-waits
    # the agent requests a global replan (0 = never).
    fallback_wait_limit: int = 5

    def to_dict(self) -> Dict[str, Any]:
        """Serialize config to a dictionary."""
        return asdict(self)
