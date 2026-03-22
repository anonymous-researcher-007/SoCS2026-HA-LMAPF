# Simulation Module

The simulation module provides the central orchestration engine for Human-Aware Lifelong Multi-Agent Path Finding
under Partial Observability (HA-LMAPF) experiments. It manages the complete simulation lifecycle including entity dynamics, planning integration,
collision detection, and metrics recording.

## Module Overview

```
simulation/
├── __init__.py
├── simulator.py       # Central simulation engine
├── environment.py     # Static grid environment
├── agent_dynamics.py  # Agent physics and state transitions
└── events.py          # Discrete event definitions
```

## Components

### Simulator (`simulator.py`)

The `Simulator` class is the main orchestration engine that manages:

1. **Environment Loading** - Static grid maps from MovingAI format
2. **Task Stream Management** - Lifelong task release and tracking
3. **Two-Tier Planning** - Global (RHCR) and Local (reactive) integration
4. **Human Motion Simulation** - Stochastic movement models
5. **Collision Detection** - Safety violation tracking
6. **Metrics Recording** - Performance statistics collection

#### Key Methods

```python
class Simulator:
    def __init__(self, config: SimConfig) -> None:
        """Initialize simulation with configuration."""

    def step_once(self) -> None:
        """Execute one simulation tick (game loop)."""

    def run(self, steps: Optional[int] = None) -> Metrics:
        """Run simulation for specified steps."""
```

#### Simulation Step Order

Each `step_once()` call executes:

1. **Task Release** - Move pending tasks to open queue
2. **Task Assignment** - Run task allocator; assign open tasks to idle agents
3. **Global Planning** - Invoke Tier-1 planner (RHCR) with current assignments
4. **Human Movement** - Update human positions via motion models
5. **Sense-Plan-Act** - Build observations and decide agent actions
6. **Execution Delay** - Inject probabilistic delays (optional)
7. **Physics Update** - Apply actions to agent positions
8. **Collision Detection** - Log safety violations
9. **Task Completion** - Track makespan and sum-of-costs
10. **Replay Recording** - Save state for visualization

#### Supported Modes

- **Lifelong Mode**: Continuous task stream with pickup-delivery workflow
- **One-Shot Mode**: Classical MAPF with direct goal assignments

### Environment (`environment.py`)

The `Environment` class represents the static grid world:

```python
class Environment:
    def __init__(self, width: int, height: int, blocked: Set[Cell]) -> None:
        """Initialize grid with dimensions and obstacles."""

    def is_blocked(self, cell: Cell) -> bool:
        """Check if cell is blocked or out of bounds."""

    def is_free(self, cell: Cell) -> bool:
        """Check if cell is valid for occupancy."""

    def sample_free_cell(self, rng, exclude: Iterable[Cell] | None = None) -> Cell:
        """Sample random free cell for entity placement."""

    @classmethod
    def load_from_map(cls, path: str) -> "Environment":
        """Load from MovingAI .map file."""
```

#### Coordinate System

- Uses `(row, col)` format
- `(0, 0)` is top-left corner
- Rows increase downward, columns increase rightward

### Agent Dynamics (`agent_dynamics.py`)

Handles state transitions for agents:

```python
def apply_action(env, agent_state: AgentState, action: StepAction) -> AgentState:
    """
    Compute next agent state after applying action.

    - WAIT: Stay in place, increment wait_steps (consecutive streak counter)
    - Movement: Check static obstacles; if valid, update position and reset wait_steps to 0
    """
```

#### Physics Rules

- Static obstacles block movement (walls)
- Invalid moves result in implicit wait
- Dynamic constraints checked separately by Simulator

### Events (`events.py`)

Immutable event definitions for logging and metrics:

| Event             | Description                       |
|-------------------|-----------------------------------|
| `TaskAssigned`    | Agent assigned new task           |
| `TaskCompleted`   | Agent reached goal                |
| `HumanDetected`   | Agent sensed human in FOV         |
| `ReplanTriggered` | Path recalculation initiated      |
| `Collision`       | Safety violation (entity overlap) |
| `NearMiss`        | Close proximity without collision |

## Configuration

The simulator is configured via `SimConfig`:

```python
@dataclass
class SimConfig:
    map_path: str  # Path to .map file
    num_agents: int  # Number of agents
    num_humans: int  # Number of dynamic obstacles
    steps: int  # Simulation duration
    seed: int  # Random seed

    # Planning
    global_solver: str  # "cbs" or "lacam"
    horizon: int  # Planning horizon
    replan_every: int  # Replan interval

    # Perception
    fov_radius: int  # Field of view radius
    safety_radius: int  # Safety buffer around humans

    # Human Model
    human_model: str  # "random_walk", "aisle", "adversarial", etc.
    human_model_params: dict  # Model-specific parameters

    # Ablation Flags
    disable_local_replan: bool
    disable_conflict_resolution: bool
    disable_safety: bool
```

## Collision Detection

The simulator tracks multiple safety metrics:

```python
def _detect_collisions_and_near_misses(prev_pos, new_pos) -> None:
    """
    Checks:
    1. Agent-Agent Vertex Collision (same cell)
    2. Agent-Agent Edge Collision (swapped cells)
    3. Agent-Human Collision (agent on human cell)
    4. Near Miss (Manhattan distance <= 1)
    5. Safety Buffer Violation (inside B_r(H_t))
    6. Human Passive Waiting (human blocked by agent)
    """
```

## Task Workflow

### Pickup-Delivery Tasks

```
Phase 1: Agent navigates to start (pickup) location
         carrying = False
         goal = task.start

Phase 2: Agent navigates to goal (delivery) location
         carrying = True
         goal = task.goal

Completion: Agent freed for next task
            done_tasks incremented
```

### One-Shot Tasks

- No pickup phase (`start = (-1, -1)`)
- Agent goes directly to goal
- Simulation terminates when all agents idle

## Integration Points

### Global Planner Interface

```python
# Simulator provides state view for planners
sim_state.agents  # Current agent states
sim_state.env  # Static environment
sim_state.open_tasks  # Released but unassigned tasks
sim_state.plans()  # Current plan bundle

# Planner callbacks
sim_state.mark_task_assigned(task, agent_id)
```

### Local Controller Interface

```python
# Controller receives observation and state
controller.decide_action(sim_state, observation, rng)

# Observation contains:
observation.visible_humans  # Detected humans
observation.visible_agents  # Detected agents
observation.blocked  # Occupied cells
```

## Usage Example

```python
from ha_lmapf.core.types import SimConfig
from ha_lmapf.simulation.simulator import Simulator

# Configure simulation
config = SimConfig(
    map_path="maps/warehouse.map",
    num_agents=10,
    num_humans=5,
    steps=1000,
    seed=42,
    global_solver="lacam",
    horizon=50,
    replan_every=25,
    fov_radius=5,
    safety_radius=2,
    human_model="random_walk",
)

# Run simulation
sim = Simulator(config)
metrics = sim.run()

# Access results
print(f"Tasks completed: {metrics.completed_tasks}")
print(f"Agent-Human collisions: {metrics.collisions_agent_human}")
print(f"Makespan: {metrics.makespan}")
```

## Related Modules

- [Core Types](../core/README_CORE.md) - Data structures and interfaces
- [Global Tier](../global_tier/README_GLOBAL_TIER.md) - Planning algorithms
- [Local Tier](../local_tier/README_LOCAL_TIER.md) - Reactive control
- [Humans](../humans/README_HUMANS.md) - Motion models
- [I/O](../io/README_IO.md) - Map and task loading