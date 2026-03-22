# Human-Aware Lifelong Multi-Agent Path Finding under Partial Observability (HA-LMAPF)

A **two-tier planning framework** for Human-Aware Lifelong Multi-Agent Path Finding
under Partial Observability in dynamic warehouse environments.

This repository provides a complete research implementation
emphasizing clarity, reproducibility, and strong baselines.

---

## Table of Contents

- [Problem Definition](#problem-definition)
- [Proposed Approach](#proposed-approach)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Documentation](#documentation)
- [Benchmarks and Evaluation](#benchmarks-and-evaluation)
- [Reproducibility](#reproducibility)
- [License](#license)

---

## Problem Definition

### Lifelong MAPF in Human-Populated Warehouses

We address the problem of **Lifelong Multi-Agent Path Finding (L-MAPF)** in warehouse environments where:

<div align="center">

| Challenge                 | Description                                              |
|---------------------------|----------------------------------------------------------|
| **Continuous Operation**  | Agents operate indefinitely; tasks arrive continuously   |
| **Dynamic Obstacles**     | Unpredictable human workers move through the environment |
| **Partial Observability** | Agents have limited field-of-view (FOV)                  |
| **Safety Requirements**   | Agents must maintain safety buffers around humans        |
| **High Throughput**       | System must maximize task completion rate                |

</div>

### Classical MAPF vs. Our Setting

<div align="center">

| Aspect        | Classical MAPF   | Our Setting (HA-LMAPF)        |
|---------------|------------------|-------------------------------|
| Tasks         | One-shot         | Continuous stream             |
| Environment   | Static           | Dynamic (humans)              |
| Observability | Full             | Partial (limited FOV)         |
| Agents        | Cooperative only | Agents + unpredictable humans |
| Horizon       | Single plan      | Rolling horizon               |
| Safety        | Collision-free   | Safety buffers around humans  |

</div>

### Formal Problem Statement

Given:

- A grid environment `G = (V, E)` with static obstacles
- A set of `n` agents with positions `{p_1, ..., p_n}`
- A set of `m` humans with unpredictable trajectories `{h_1, ..., h_m}`
- A continuous stream of tasks `T = {(start, goal, release_time)}`
- Field-of-view radius `r_fov` for each agent
- Safety radius `r_safety` around each human

Objective:

- **Maximize throughput** (tasks completed per time step)
- **Minimize collisions** (agent-agent and agent-human)
- **Ensure safety** (agents stay outside human safety zones)
- **Minimize flowtime** (task completion time)

---

## Proposed Approach

### Two-Tier Planning Architecture

Our approach separates planning into two complementary tiers:


<div align="center">

![](architecture.png)

</div>

### Tier 1: Global Planning

The global planner operates on a **rolling horizon** and is responsible for:

1. **Task Allocation**: Assigns incoming tasks to available agents
    - **Greedy**: Nearest available agent (fast)
    - **Hungarian**: Optimal assignment (slower)
    - **Auction**: Distributed bidding

2. **Path Planning**: Computes collision-free paths for all agents
    - **CBS (Conflict-Based Search)**: Optimal, slower
    - **LaCAM**: Fast, bounded-suboptimal

3. **Periodic Replanning**: Recomputes plans every `K` steps to adapt to new tasks

### Tier 2: Local Execution

Each agent runs a local controller that:

1. **Observes**: Builds local observation from FOV
2. **Predicts**: Forecasts human positions (MyopicPredictor)
3. **Plans**: Local A* around dynamic obstacles
4. **Resolves**: Handles agent-agent conflicts
5. **Executes**: Applies safe action

#### Safety Modes

<div align="center">

| Mode            | Description                           | Use Case                   |
|-----------------|---------------------------------------|----------------------------|
| **Hard Safety** | Agents NEVER enter human safety zones | Strict safety requirements |
| **Soft Safety** | High cost but passable if necessary   | Deadlock prevention        |

</div>

#### Conflict Resolution Strategies

<div align="center">

| Strategy           | Communication | Description                              |
|--------------------|---------------|------------------------------------------|
| **Token Passing**  | Required      | Cell ownership via tokens, fair rotation |
| **Priority Rules** | None          | Deterministic priority by urgency/ID     |
| **PIBT**           | None          | Push-based with backtracking             |

</div>

---

## Project Structure

```
ha_lmapf/
├── README.md                          # This file
├── docs/
│   ├── GETTING_STARTED.md            # Beginner's guide
│   ├── README_TESTS.md               # Test documentation
│   ├── experimental_setup.md         # Paper experimental setup
│   ├── metrics.md                    # Metrics reference
│   ├── fine_tune.md                  # Hyperparameter tuning guide
│   └── proposed_approach.md          # Algorithm description
├── src/ha_lmapf/
│   ├── core/                         # Core types and utilities
│   ├── global_tier/                  # Tier-1 global planning
│   │   └── solvers/                  # MAPF solvers (CBS, LaCAM)
│   ├── local_tier/                   # Tier-2 local execution
│   │   └── conflict_resolution/      # Conflict resolvers
│   ├── humans/                       # Human models and prediction
│   ├── simulation/                   # Simulation engine
│   ├── io/                           # Input/Output utilities
│   ├── gui/                          # Visualization
│   ├── task_allocator/               # Task allocation algorithms
│   └── baselines/                    # Baseline implementations
├── scripts/                          # Runnable scripts
│   └── evaluation/                   # Evaluation and plotting
├── configs/                          # Configuration files
│   └── eval/                         # Paper evaluation configs
├── data/
│   ├── maps/                         # Map files
│   └── task_streams/                 # Task stream files
├── tests/                            # Test suite (359 tests)
└── logs/                             # Experiment outputs
```

### Module Documentation

<div align="center">

| Module                  | Description                              | Documentation                                                                                              |
|-------------------------|------------------------------------------|------------------------------------------------------------------------------------------------------------|
| **Core**                | Types, interfaces, grid utilities        | [README_CORE.md](src/ha_lmapf/core/README_CORE.md)                                                         |
| **Global Tier**         | Rolling horizon planner, task allocation | [README_GLOBAL_TIER.md](src/ha_lmapf/global_tier/README_GLOBAL_TIER.md)                                    |
| **Solvers**             | CBS, LaCAM MAPF algorithms               | [README_SOLVERS.md](src/ha_lmapf/global_tier/solvers/README_SOLVERS.md)                                    |
| **Local Tier**          | Agent controller, local planner          | [README_LOCAL_TIER.md](src/ha_lmapf/local_tier/README_LOCAL_TIER.md)                                       |
| **Conflict Resolution** | Token passing, priority rules, PIBT      | [README_CONFLICT_RESOLUTION.md](src/ha_lmapf/local_tier/conflict_resolution/README_CONFLICT_RESOLUTION.md) |
| **Humans**              | Motion models, prediction, safety        | [README_HUMANS.md](src/ha_lmapf/humans/README_HUMANS.md)                                                   |
| **Simulation**          | Environment, dynamics, events            | [README_SIMULATION.md](src/ha_lmapf/simulation/README_SIMULATION.md)                                       |
| **I/O**                 | Map loading, task streams, replay        | [README_IO.md](src/ha_lmapf/io/README_IO.md)                                                               |
| **GUI**                 | PyGame visualization                     | [README_GUI.md](src/ha_lmapf/gui/README_GUI.md)                                                            |
| **Baselines**           | Comparison implementations               | [README_BASELINES.md](src/ha_lmapf/baselines/README_BASELINES.md)                                          |

</div>

---

## Installation

### Requirements

- Python **3.10+**
- Minimal dependencies (no heavy ML frameworks)

### Install

```bash
# Clone the repository
git clone <REPO_URL>
cd ha_lmapf

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Optional: GUI support
pip install pygame

# Optional: Testing
pip install pytest
```

For detailed installation instructions, see [**GETTING_STARTED.md**](docs/GETTING_STARTED.md).

---

## Quick Start

### Run GUI Visualization

```bash
# Lifelong MAPF with configuration file
python scripts/run_gui.py --config configs/warehouse_small.yaml

# One-shot classical MAPF
python scripts/run_oneshot_gui.py --agents 15 --solver cbs

# Human-aware demonstration
python scripts/run_oneshot_hamapf_gui.py --agents 10 --humans 5
```

### Run Experiments

```bash
# Run baseline comparison
python scripts/evaluation/run_evaluation.py --group baselines --seeds 0 1 2 --out logs/results

# Run all experiments
python scripts/evaluation/run_evaluation.py --out logs/full_evaluation
```

### Run Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific category
python -m pytest tests/test_cbs_solver.py -v
```

---

## Documentation

<div align="center">

| Document                                                       | Description                                              |
|----------------------------------------------------------------|----------------------------------------------------------|
| [**GETTING_STARTED.md**](docs/GETTING_STARTED.md)              | Complete beginner's guide with step-by-step instructions |
| [**README_TESTS.md**](docs/README_TESTS.md)                    | Comprehensive test suite documentation                   |
| [**experimental_setup.md**](docs/experimental_setup.md)        | Paper experimental setup and parameters                  |
| [**metrics.md**](docs/metrics.md)                              | Detailed metrics reference                               |
| [**fine_tune.md**](docs/fine_tune.md)                          | Hyperparameter tuning guide                              |
| [**proposed_approach.md**](docs/proposed_approach.md)          | Algorithm and architecture description                   |

</div>

---

## Benchmarks and Evaluation

### Experiment Groups

<div align="center">

| Group           | Description                   | Experiments       |
|-----------------|-------------------------------|-------------------|
| `baselines`     | Compare our method vs. others | 6 baselines       |
| `scalability`   | Test with 10-500 agents       | 7 configurations  |
| `human_density` | Test with 0-20 humans         | 4 configurations  |
| `human_models`  | Different human behaviors     | 4 models          |
| `map_types`     | Different warehouse layouts   | 5 maps            |
| `ablations`     | Disable components            | 10 configurations |
| `no_humans`     | Compare without humans        | 9 configurations  |
| `classic_mapf`  | One-shot classical MAPF       | 14 configurations |

</div>

### Metrics

<div align="center">

| Metric                | Description                     |
|-----------------------|---------------------------------|
| **Throughput**        | Tasks completed per time step   |
| **Flowtime**          | Average task completion time    |
| **Makespan**          | Time until all tasks complete   |
| **Collisions**        | Agent-agent and agent-human     |
| **Safety Violations** | Entries into human safety zones |
| **Planning Time**     | Computation time per step       |

</div>

### Maps

We use standard MovingAI MAPF benchmark maps:

- Warehouse layouts
- Random obstacle maps
- Room and maze configurations

---

## Reproducibility

This repository ensures full reproducibility:

- **Seeds**: All randomness controlled via explicit seeds
- **Configs**: All parameters in YAML files under `configs/`
- **Logs**: Each run outputs:
    - `results.csv`: All metrics
    - `replay.json`: Full trajectories
- **Determinism**: All planners and resolvers are deterministic

---

## Baselines

<div align="center">

| Baseline         | Description                              |
|------------------|------------------------------------------|
| **GlobalOnly**   | Tier-1 only, no local replanning         |
| **PIBTOnly**     | Decentralized greedy, no global planning |
| **IgnoreHumans** | Our method, ignoring human observations  |
| **RHCR-like**    | Receding Horizon Collision Resolution    |
| **WHCA***        | Windowed Hierarchical Cooperative A*     |

</div>

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
