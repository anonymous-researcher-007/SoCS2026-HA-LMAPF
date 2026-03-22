# Local Tier (Tier-2)

The local tier implements decentralized, reactive control for each agent. It handles partial observability, human-aware
navigation, and agent-agent conflict resolution.

## Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                    LOCAL TIER (Tier-2)                           │
│                  Per-Agent Control Loop                          │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│    ┌─────────┐     ┌─────────┐     ┌─────────┐     ┌──────────┐  │
│    │ SENSE   │ --> │ PREDICT │ --> │  PLAN   │ --> │ RESOLVE  │  │
│    │ (FOV)   │     │ (Humans)│     │  (A*)   │     │(Conflict)│  │
│    └─────────┘     └─────────┘     └─────────┘     └──────────┘  │
│         │                                               │        │
│         v                                               v        │
│    Observation                                      Action       │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

## Files

| File                   | Description                         |
|------------------------|-------------------------------------|
| `agent_controller.py`  | Per-agent Sense-Plan-Act controller |
| `local_planner.py`     | A* pathfinding with safety modes    |
| `sensors.py`           | FOV-based observation building      |
| `conflict_resolution/` | Agent-agent conflict resolvers      |

---

## agent_controller.py - Agent Controller

Main per-agent controller implementing the Sense-Plan-Act loop.

### AgentController Class

```python
class AgentController:
    """
    Tier-2 per-agent controller.

    Responsibilities:
        1. Follow global plan when path is clear
        2. Detect and avoid humans (safety buffer)
        3. Local replan around dynamic obstacles
        4. Resolve agent-agent conflicts
    """

    def __init__(
            self,
            agent_id: int,
            local_planner: LocalPlanner,
            conflict_resolver: ConflictResolver,
            fov_radius: int = 4,
            safety_radius: int = 1,
            hard_safety: bool = True
    ):
        ...

    def decide_action(
            self,
            sim_state: SimStateView,
            observation: Observation,
            rng
    ) -> StepAction:
        """
        Determine next action for this agent.

        Process:
            1. Check if global plan exists and is valid
            2. Build blocked set (humans + safety buffer)
            3. If next planned cell is blocked, local replan
            4. Resolve conflicts with other agents
            5. Return safe action
        """
```

### Decision Logic

```
decide_action():
    1. If no goal → WAIT
    2. If at goal → WAIT
    3. Get next cell from global plan
    4. Build blocked set from visible humans
    5. If next cell blocked by human:
       - Local replan around human
       - If no path found: WAIT
    6. Resolve agent-agent conflicts
    7. Return action toward next cell
```

### Usage

```python
from ha_lmapf.local_tier.agent_controller import AgentController
from ha_lmapf.local_tier.local_planner import AStarLocalPlanner
from ha_lmapf.local_tier.conflict_resolution import TokenPassingResolver

controller = AgentController(
    agent_id=0,
    local_planner=AStarLocalPlanner(hard_safety=True),
    conflict_resolver=TokenPassingResolver(),
    fov_radius=4,
    safety_radius=1
)

action = controller.decide_action(sim_state, observation, rng)
```

---

## local_planner.py - A* Local Planner

Single-agent pathfinding with human-aware safety modes.

### Safety Modes

| Mode            | Behavior                          | Use Case                   |
|-----------------|-----------------------------------|----------------------------|
| **Hard Safety** | Blocked cells are IMPASSABLE      | Strict safety requirements |
| **Soft Safety** | Blocked cells have HIGH COST (50) | Deadlock prevention        |

### AStarLocalPlanner Class

```python
class AStarLocalPlanner(LocalPlanner):
    """
    A* planner for local detours.

    Parameters:
        hard_safety: bool - If True, blocked cells impassable
                           If False, blocked cells high cost

    Features:
        - Fast single-agent planning
        - Respects static walls (always impassable)
        - Configurable safety mode for human zones
        - Expansion budget (MAX_EXPANSIONS = 500)
    """

    BLOCKED_CELL_COST = 50  # Cost for soft safety mode
    MAX_EXPANSIONS = 500  # Search budget

    def plan(
            self,
            env: Environment,
            start: Cell,
            goal: Cell,
            blocked: Set[Cell]
    ) -> List[Cell]:
        """
        Plan path avoiding blocked cells.

        Returns:
            List of cells from start to goal, or [] if no path
        """
```

### Usage

```python
from ha_lmapf.local_tier.local_planner import AStarLocalPlanner

# Hard safety - blocked cells impassable
planner = AStarLocalPlanner(hard_safety=True)
path = planner.plan(env, start=(0, 0), goal=(5, 5), blocked={(2, 2), (2, 3)})

# Soft safety - blocked cells high cost but passable
planner = AStarLocalPlanner(hard_safety=False)
path = planner.plan(env, start=(0, 0), goal=(5, 5), blocked={(2, 2), (2, 3)})
```

---

## sensors.py - Observation Building

Builds agent observations from limited field-of-view.

### build_observation Function

```python
def build_observation(
        agent_id: int,
        sim_state,  # SimStateView — provides agents, humans, env
        fov_radius: int,
) -> Observation:
    """
    Build local observation for an agent.

    Process:
        1. Filter humans within Manhattan distance <= fov_radius
        2. Filter other agents within Manhattan distance <= fov_radius
        3. blocked = static wall cells + cells directly occupied by visible humans
           (does NOT include the inflated safety buffer; that is computed in
            AgentController.decide_action() via inflate_cells())

    Returns:
        Observation with visible_humans, visible_agents, blocked
    """
```

### Visibility Calculation

```
FOV = Manhattan ball of radius r around agent position

Visible humans/agents: {x : manhattan(agent_pos, x.pos) <= fov_radius}
blocked = env.blocked ∪ {h.pos for h in visible_humans}
```

### Usage

```python
from ha_lmapf.local_tier.sensors import build_observation

obs = build_observation(agent_id=0, sim_state=simulator, fov_radius=4)

# obs.visible_humans: humans within FOV
# obs.visible_agents: other agents within FOV
# obs.blocked: static walls + cells directly occupied by visible humans
#   NOTE: inflated safety buffer (B_r(H_t)) is NOT in obs.blocked;
#         it is computed in AgentController using inflate_cells().
```

---

## Conflict Resolution Subdirectory

See [README_CONFLICT_RESOLUTION.md](conflict_resolution/README_CONFLICT_RESOLUTION.md).

| Resolver              | Communication | Description                     |
|-----------------------|---------------|---------------------------------|
| TokenPassingResolver  | Required      | Cell ownership via tokens       |
| PriorityRulesResolver | None          | Deterministic priority ordering |
| PIBTResolver          | None          | Push-based with backtracking    |

---

## Configuration

```yaml
# Local tier settings
fov_radius: 5           # Field-of-view radius
safety_radius: 1        # Safety buffer around humans
hard_safety: true       # Hard vs soft safety mode
communication_mode: "token"  # "token" or "priority"
```

---

## Integration with Global Tier

The local tier receives `PlanBundle` from global planning:

1. **Follow Plan**: When path clear, follow global plan
2. **Local Replan**: When blocked by humans, compute detour
3. **Fallback**: If no local path, WAIT for humans to move

```python
# In agent controller
if global_plan_exists:
    next_cell = plan.paths[agent_id](current_step + 1)
    if next_cell in blocked_by_humans:
        local_path = local_planner.plan(env, current_pos, goal, blocked)
        if local_path:
            next_cell = local_path[1]
        else:
            return StepAction.WAIT
```

---

## Related Modules

- [Conflict Resolution](conflict_resolution/README_CONFLICT_RESOLUTION.md) - Resolver implementations
- [Global Tier](../global_tier/README_GLOBAL_TIER.md) - Provides plan bundles
- [Humans](../humans/README_HUMANS.md) - Human prediction for safety