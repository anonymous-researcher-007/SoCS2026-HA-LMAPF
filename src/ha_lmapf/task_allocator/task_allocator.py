from __future__ import annotations

from typing import Dict, List, Tuple, Optional, Iterable

import numpy as np

from ha_lmapf.core.grid import manhattan
from ha_lmapf.core.interfaces import TaskAllocator
from ha_lmapf.core.types import AgentState, Task

Cell = Tuple[int, int]


def _get_task_pickup_location(task: Task) -> Cell:
    """
    Get the pickup location for a task.

    For pickup-delivery tasks, this is task.start.
    For legacy delivery-only tasks (start = (-1,-1)), this falls back to task.goal.
    """
    if task.start != (-1, -1):
        return task.start
    return task.goal


class GreedyNearestTaskAllocator(TaskAllocator):
    """
    Greedy task allocator for lifelong pickup-delivery MAPF.

    For each released task (sorted by release_step, then id), assign it to the
    nearest available agent according to Manhattan distance to the task's
    START (pickup) location. Each agent receives at most one task per planning epoch.

    This ensures agents are assigned tasks they can reach quickly for pickup,
    rather than being assigned based on the delivery destination.
    """

    def assign(
            self,
            agents: Dict[int, AgentState],
            open_tasks: Iterable[Task],
            step: int,
            rng=None,  # Unused; kept for interface compatibility
    ) -> Dict[int, Task]:
        assignments: Dict[int, Task] = {}

        # Determine available agents: no current goal or already reached goal
        available_agents = {
            aid: a
            for aid, a in agents.items()
            if a.goal is None or a.pos == a.goal
        }

        if not available_agents or not open_tasks:
            return assignments

        # Deterministic ordering of tasks
        tasks_ordered = sorted(open_tasks, key=lambda t: (t.release_step, t.task_id))

        # Track which agents are still free to assign
        free_agents = dict(available_agents)

        for task in tasks_ordered:
            if not free_agents:
                break

            # Get the pickup location for this task
            pickup_loc = _get_task_pickup_location(task)

            # Choose nearest agent by Manhattan distance to PICKUP location (tie-break by agent ID)
            best_aid = min(
                free_agents.keys(),
                key=lambda aid: (manhattan(free_agents[aid].pos, pickup_loc), aid)
            )

            assignments[best_aid] = task

            del free_agents[best_aid]

        return assignments


class HungarianTaskAllocator(TaskAllocator):
    """
    Optimal task allocator using the Hungarian algorithm.

    Computes the globally optimal assignment that minimizes the total
    Manhattan distance from agents to their assigned task pickup locations.

    Falls back to greedy assignment if scipy is not available.
    """

    def assign(
            self,
            agents: Dict[int, AgentState],
            open_tasks: Iterable[Task],
            step: int,
            rng=None,
    ) -> Dict[int, Task]:
        assignments: Dict[int, Task] = {}

        # Determine available agents: no current goal or already reached goal
        available_agents = {
            aid: a
            for aid, a in agents.items()
            if a.goal is None or a.pos == a.goal
        }

        if not available_agents or not open_tasks:
            return assignments

        agent_ids = sorted(available_agents.keys())
        task_list = sorted(open_tasks, key=lambda t: (t.release_step, t.task_id))

        n_agents = len(agent_ids)
        n_tasks = len(task_list)

        # Build cost matrix
        cost_matrix = np.zeros((n_agents, n_tasks), dtype=np.float64)
        for i, aid in enumerate(agent_ids):
            agent_pos = available_agents[aid].pos
            for j, task in enumerate(task_list):
                pickup_loc = _get_task_pickup_location(task)
                cost_matrix[i, j] = manhattan(agent_pos, pickup_loc)

        try:
            from scipy.optimize import linear_sum_assignment
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            for i, j in zip(row_ind, col_ind):
                if j < n_tasks:  # Ensure task index is valid
                    assignments[agent_ids[i]] = task_list[j]

        except ImportError:
            # Fallback to greedy if scipy not available
            greedy = GreedyNearestTaskAllocator()
            return greedy.assign(agents, open_tasks, step, rng)

        return assignments


class AuctionBasedTaskAllocator(TaskAllocator):
    """
    Auction-based task allocator using sequential single-item auctions.

    Each task is auctioned off to the highest bidder (lowest distance),
    and agents bid based on their proximity to the task's pickup location.
    This provides a balance between optimality and computational efficiency.
    """

    def __init__(self, max_iterations: int = 100, epsilon: float = 0.01):
        """
        Initialize the auction allocator.

        Args:
            max_iterations: Maximum number of auction iterations.
            epsilon: Price increment for bidding.
        """
        self.max_iterations = max_iterations
        self.epsilon = epsilon

    def assign(
            self,
            agents: Dict[int, AgentState],
            open_tasks: Iterable[Task],
            step: int,
            rng=None,
    ) -> Dict[int, Task]:
        assignments: Dict[int, Task] = {}

        # Determine available agents
        available_agents = {
            aid: a
            for aid, a in agents.items()
            if a.goal is None or a.pos == a.goal
        }

        if not available_agents or not open_tasks:
            return assignments

        agent_ids = sorted(available_agents.keys())
        task_list = sorted(open_tasks, key=lambda t: (t.release_step, t.task_id))

        n_agents = len(agent_ids)
        n_tasks = len(task_list)

        # Compute benefit matrix (negative distance = higher benefit for closer tasks)
        max_dist = 0
        benefit_matrix = np.zeros((n_agents, n_tasks), dtype=np.float64)
        for i, aid in enumerate(agent_ids):
            agent_pos = available_agents[aid].pos
            for j, task in enumerate(task_list):
                pickup_loc = _get_task_pickup_location(task)
                dist = manhattan(agent_pos, pickup_loc)
                max_dist = max(max_dist, dist)
                benefit_matrix[i, j] = -dist

        # Normalize benefits to positive values
        benefit_matrix += max_dist + 1

        # Task prices (start at 0)
        prices = np.zeros(n_tasks, dtype=np.float64)

        # Agent assignments (-1 = unassigned)
        agent_to_task = {aid: -1 for aid in agent_ids}
        task_to_agent: Dict[int, int] = {}

        for _ in range(self.max_iterations):
            # Find unassigned agents
            unassigned = [aid for aid in agent_ids if agent_to_task[aid] == -1]
            if not unassigned:
                break

            for aid in unassigned:
                i = agent_ids.index(aid)

                # Compute net values for each task
                net_values = benefit_matrix[i] - prices
                best_task = int(np.argmax(net_values))
                best_value = net_values[best_task]

                # Find second best value for price increment
                net_values_copy = net_values.copy()
                net_values_copy[best_task] = -np.inf
                second_best = np.max(net_values_copy)

                # Only bid if beneficial
                if best_value <= 0:
                    continue

                # Compute bid increment
                bid_increment = best_value - second_best + self.epsilon

                # If task is already assigned, unassign previous owner
                if best_task in task_to_agent:
                    prev_owner = task_to_agent[best_task]
                    agent_to_task[prev_owner] = -1

                # Assign task and update price
                agent_to_task[aid] = best_task
                task_to_agent[best_task] = aid
                prices[best_task] += bid_increment

        # Build final assignments
        for aid, task_idx in agent_to_task.items():
            if task_idx >= 0 and task_idx < n_tasks:
                assignments[aid] = task_list[task_idx]

        return assignments
