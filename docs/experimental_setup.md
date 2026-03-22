# Experimental Setup

## A. Environment and Maps

We evaluate on five map categories from standard MAPF benchmarks [1]:

| Map Type      | Instance               | Grid Size | Characteristics                                                               |
|---------------|------------------------|-----------|-------------------------------------------------------------------------------|
| **Warehouse** | warehouse-10-20-10-2-1 | 161 × 63  | Parallel aisles with shelving units; representative of logistics environments |
| **Random**    | random-32-32-20        | 32 × 32   | 20% randomly placed obstacles; tests general-purpose navigation               |
| **Room**      | room-32-32-4           | 32 × 32   | Partitioned rooms connected by doorways; creates bottleneck scenarios         |
| **Maze**      | maze-32-32-4           | 32 × 32   | Narrow corridors with many turns; stress-tests conflict resolution            |
| **Empty**     | empty-32-32            | 32 × 32   | Open grid with no internal obstacles; baseline topology                       |

All maps use the standard Moving AI format with 4-connected grids.

---

## B. Human Motion Models

We simulate human behavior using three motion models with Boltzmann (softmax) action selection:

### 1) Random Walk

Humans select actions based on directional inertia scores:

```
P(u | x_h, a_h) ∝ exp(s(u))
```

where:

- s(u) = β_go if u continues the current direction
- s(u) = β_wait if u is stationary
- s(u) = β_turn otherwise

**Default parameters:** β_go = 2.0, β_wait = −1.0, β_turn = 0.0

### 2) Aisle Follower

Humans prefer corridor-like regions:

```
s(u) = α · φ(u) + β · 1[u ∈ cont(x_h, a_h)]
```

where φ(u) = −dist(u, obstacles) is the aisle-likelihood field computed via multi-source BFS from obstacle cells.

**Default parameters:** α = 1.0, β = 1.5

### 3) Mixed Population

Each human is assigned a type z_h ~ Categorical({random_walk: 0.5, aisle: 0.5}) at initialization, maintaining
consistent individual behavior throughout the episode.

---

## C. Baseline Approaches

We compare against five baselines that ablate different architectural components:

| Baseline            | Global Plan | Local Replan | Human-Aware | Description                                                      |
|---------------------|:-----------:|:------------:|:-----------:|------------------------------------------------------------------|
| **Ours** (Two-Tier) |      ✓      |      ✓       |      ✓      | Full proposed system                                             |
| **Global-Only**     |      ✓      |      ✗       |   Partial   | Follows global plan rigidly; WAITs when blocked by humans        |
| **PIBT-Only**       |      ✗      |      ✓       |      ✓      | Decentralized greedy with priority inheritance; no coordination  |
| **RHCR**            |      ✓      |      ✗       |      ✗      | Standard Rolling-Horizon Collision Resolution [2]; human-unaware |
| **WHCA***           |      ✓      |      ✗       |      ✗      | Windowed Hierarchical Cooperative A* [3]; prioritized planning   |
| **Ignore-Humans**   |      ✓      |      ✓       |      ✗      | Two-tier architecture but blind to human obstacles               |

---

## D. Experiment Groups

We organize experiments into nine groups, each targeting a specific research question:

### 1) Baseline Comparison

Compares all six approaches on random-32-32-20 with 15 agents, 5 humans, 1000 steps.

### 2) Scalability

Evaluates throughput and planning time as fleet size increases.

| Parameter    | Values                             |
|--------------|------------------------------------|
| Agent counts | {10, 25, 50, 100, 200, 300, 500}   |
| Map          | warehouse-20-40-10-2-1 (321 × 123) |
| Solver       | LaCAM (scales polynomially)        |

### 3) Human Density

Measures the impact of human population on system performance.

| Parameter    | Values          |
|--------------|-----------------|
| Human counts | {0, 5, 10, 20}  |
| Agents       | 15              |
| Map          | random-32-32-20 |

### 4) Human Behavior Models

Tests robustness across different human motion patterns.

| Parameter | Values                    |
|-----------|---------------------------|
| Models    | random_walk, aisle, mixed |
| Humans    | 10                        |
| Agents    | 15                        |

### 5) Map Topology

Evaluates generalization across all five map categories.

| Parameter | Values                               |
|-----------|--------------------------------------|
| Maps      | warehouse, random, room, maze, empty |
| Agents    | 15–30 (adjusted per map)             |
| Humans    | 5                                    |
| Steps     | 1500                                 |

### 6) Ablation Studies

Isolates the contribution of each system component across two groups:

| Group                   | Ablation               | Component Disabled                           |
|-------------------------|------------------------|----------------------------------------------|
| A — Tier Architecture   | No Local Replan        | Tier-2 local replanning                      |
| A                       | No Conflict Resolution | Decentralized conflict resolution            |
| A                       | Global Only (RHCR)     | Both Tier-2 features                         |
| B — Safety & Perception | No Safety              | Human safety zones disabled                  |
| B                       | Soft Safety            | Replace hard constraints with cost penalties |

**Statistical methodology:**

- 30 seeds (default); same seed for control and ablation → paired design
- Friedman omnibus χ² test within each group before pairwise comparisons
- Wilcoxon signed-rank test (paired, two-sided) vs full_system
- Benjamini-Hochberg FDR correction within each group
- Rank-biserial r AND Cohen's d as standardised effect sizes
- Shapiro-Wilk test on paired differences (justifies non-parametric choice)
- Post-hoc power analysis (1−β) for each comparison
- 95% bootstrap confidence intervals (BCa when n ≥ 10)
- Multi-map generalization: warehouse_large, warehouse_small, room
- Scale sensitivity: 10:2, 20:5, 40:10 (agents:humans)
- Optional warm-up exclusion for steady-state metrics

### 7) Solver Comparison

Compares MAPF solvers at appropriate agent counts:

| Solver       | Agents | Type                      |
|--------------|--------|---------------------------|
| CBS          | 8      | Optimal                   |
| LaCAM        | 20     | Bounded-suboptimal        |
| LaCAM3 (C++) | 50     | Bounded-suboptimal        |
| PIBT2 (C++)  | 50     | Complete, anytime         |
| EECBS (C++)  | 40     | Bounded-suboptimal        |
| PBS (C++)    | 60     | Priority-based            |
| LNS2 (C++)   | 100    | Large Neighborhood Search |

### 8) Execution Delay Robustness

Injects stochastic delays to test real-world robustness.

| Parameter           | Values             |
|---------------------|--------------------|
| Delay probabilities | {0%, 5%, 10%, 20%} |
| Delay duration      | 2 steps per event  |

### 9) Task Arrival Rate

Varies workload intensity.

| Parameter     | Values                                      |
|---------------|---------------------------------------------|
| Arrival rates | {5, 10, 25, 50} steps between task releases |

---

## E. Default Parameters

Unless otherwise stated, experiments use the following configuration:

| Parameter            | Value         | Description                               |
|----------------------|---------------|-------------------------------------------|
| Simulation steps     | 1000          | Duration per experiment                   |
| FOV radius (ρ)       | 4             | Manhattan distance perception             |
| Safety radius (r)    | 1             | Buffer around humans                      |
| Hard safety          | True          | Forbidden zone as hard constraint         |
| Replan interval (K)  | 25            | Steps between global replans              |
| Horizon (H)          | 50            | Global planning depth                     |
| Task allocator       | Greedy        | Nearest-task by Manhattan distance        |
| Commit horizon (K_c) | 25            | Assignment lock duration                  |
| Delay threshold (α)  | 2.5           | ETA multiplier before reassignment        |
| Conflict resolution  | Token passing | With fairness rotation (k=5)              |
| Local planner        | A*            | Max 500 node expansions                   |
| Task arrival rate    | 10            | Steps between task releases               |
| Random seeds         | 30            | Per configuration (paired Wilcoxon + FDR) |

---

## F. Evaluation Metrics

We report the following metrics, grouped by category:

### Service Quality

| Metric                | Definition                                 | Goal     |
|-----------------------|--------------------------------------------|----------|
| **Throughput**        | completed_tasks / total_steps              | Maximize |
| **Mean flowtime**     | Mean steps from task release to completion | Minimize |
| **Median flowtime**   | Median flowtime (robust to outliers)       | Minimize |
| **Mean service time** | Mean steps from assignment to completion   | Minimize |

### Safety

| Metric                     | Definition                                 | Goal     |
|----------------------------|--------------------------------------------|----------|
| **Agent-agent collisions** | Robot-robot occupancy conflicts            | Zero     |
| **Agent-human collisions** | Robot-human occupancy conflicts            | Zero     |
| **Safety violations**      | Agent entries into B_r(H_t)                | Minimize |
| **Safety violation rate**  | Violations per 1000 timesteps              | Minimize |
| **Near misses**            | Proximity events (distance ≤ 1 from human) | Minimize |

### Efficiency

| Metric                | Definition                        | Goal     |
|-----------------------|-----------------------------------|----------|
| **Total wait steps**  | Cumulative agent idle time        | Minimize |
| **Global replans**    | Tier-1 replanning count           | —        |
| **Local replans**     | Tier-2 replanning count           | —        |
| **Intervention rate** | Global replans per 1000 timesteps | Minimize |

### Computational Cost

| Metric                 | Definition                          | Goal     |
|------------------------|-------------------------------------|----------|
| **Mean planning time** | Wall-clock ms per global replan     | Minimize |
| **P95 planning time**  | 95th percentile latency             | Minimize |
| **Mean decision time** | Wall-clock ms per step (all agents) | Minimize |

### Assignment Stability

| Metric                 | Definition                               | Goal                 |
|------------------------|------------------------------------------|----------------------|
| **Assignments kept**   | Commitments preserved due to persistence | Higher = more stable |
| **Assignments broken** | Commitments terminated (delay, expiry)   | Lower = more stable  |

---

## G. Reproducibility

All experiments are deterministic given a fixed random seed. Each configuration is run with 30 seeds (0–29) and we
report mean ± 95% bootstrap CI (BCa). Maps are sourced from the Moving AI benchmark repository [1]. The implementation,
configurations, and evaluation scripts are provided in the supplementary material.

---

## H. Running Experiments

### Quick Start

```bash
# Run all experiment groups (5 seeds each)
python scripts/evaluation/run_evaluation.py --out logs/eval

# Run specific group
python scripts/evaluation/run_evaluation.py --group scalability --out logs/eval

# Run with custom seeds
python scripts/evaluation/run_evaluation.py --seeds 0 1 2 3 4 5 6 7 8 9 --out logs/eval
```

### Generating Figures

```bash
# Generate all publication figures
python scripts/evaluation/plot_results.py --results logs/eval --out figures/
```

### Output Structure

```
logs/eval/
├── baselines/
│   ├── results.csv
│   └── summary.csv
├── scalability/
│   ├── results.csv
│   └── summary.csv
├── human_density/
│   └── ...
└── figures/
    ├── scalability.pdf
    ├── baselines.pdf
    └── ablation.pdf
```

---

## I. Hardware and Software

| Component    | Specification                                                    |
|--------------|------------------------------------------------------------------|
| CPU          | Intel Xeon / AMD EPYC (experiments are single-threaded per run)  |
| Memory       | 16+ GB RAM                                                       |
| Python       | 3.10+                                                            |
| Dependencies | NumPy, SciPy (Hungarian algorithm), PyYAML                       |
| C++ Solvers  | Optional: LaCAM3, PIBT2, EECBS, PBS, LNS2 (requires compilation) |

---

## References

[1] Stern et al., "Multi-Agent Pathfinding: Definitions, Variants, and Benchmarks," 2019

[2] Li et al., "Lifelong Multi-Agent Path Finding in Large-Scale Warehouses," 2021

[3] Silver, "Cooperative Pathfinding," 2005