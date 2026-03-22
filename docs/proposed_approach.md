# Proposed Approach

## A. Problem Formulation

We address Human-Aware Lifelong Multi-Agent Path Finding under Partial Observability (HA-LMAPF), defined as follows:

**Definition 1 (HA-LMAPF).** Given:
- A 4-connected grid graph G = (V, E) representing the environment
- A set of k agents A = {a₁, ..., aₖ} with positions s(t) = {s₁(t), ..., sₖ(t)} ⊂ V
- A set of n humans H = {h₁, ..., hₙ} with time-varying positions and velocities
- A continuous stream of pickup-delivery tasks T arriving at runtime
- A safety radius r defining forbidden zones around humans

Find collision-free paths that maximize task throughput while satisfying the safety constraint:

```
s_i(t) ∉ B_r(h_j(t)), ∀i ∈ {1,...,k}, ∀j ∈ {1,...,n}, ∀t
```

where B_r(h_j(t)) denotes the safety buffer of radius r (Manhattan distance) around human h_j at time t.

Each agent operates under **partial observability**, perceiving only entities within its field-of-view radius ρ.

---

## B. Two-Tier Hierarchical Architecture

We propose a hierarchical decomposition that separates global coordination from local reactivity:

| Tier | Scope | Frequency | Responsibility |
|------|-------|-----------|----------------|
| **Tier-1 (Global)** | Centralized | Every K steps | Task allocation + collision-free MAPF |
| **Tier-2 (Local)** | Per-agent | Every step | Human avoidance + conflict resolution |

This decomposition enables the global tier to optimize multi-agent coordination without modeling unpredictable human behavior, while the local tier provides real-time responsiveness to dynamic obstacles.

---

## C. Tier-1: Rolling Horizon Global Planning

### 1) Rolling Horizon Replanning

The global planner operates on a rolling horizon of H timesteps, replanning every K steps. At each replanning cycle, it:

1. **Collects open tasks** from the task stream
2. **Assigns tasks** to idle agents via the task allocator
3. **Computes collision-free paths** using a MAPF solver
4. **Distributes plans** to agent controllers

**Replanning Triggers:**
- Periodic: every K timesteps (default K=25)
- Deviation: when agents deviate significantly from planned paths

### 2) Task Allocation with Commitment Persistence

We employ a greedy nearest-task allocator that assigns each idle agent to the closest available task by Manhattan distance. To prevent *assignment thrashing*—oscillations caused by small perturbations—we introduce commitment persistence:

**Definition 2 (Commitment Persistence).** Once task τ is assigned to agent aᵢ, the assignment is locked for K steps. Reassignment occurs only when:
1. Agent completes the task
2. Commitment horizon expires (K steps elapsed)
3. Excessive delay: ETA(t) > α · ETA(t₀), where α is the delay threshold

This converts the allocator from a memoryless policy to a hysteresis controller, stabilizing the planning system.

### 3) Immediate Task Assignment

To reduce agent idle time, we decouple task assignment from global replanning. When an agent completes a task:
1. Immediately assign the nearest available task
2. Use local planning until the next global replan cycle

This reduces assignment latency from O(K) to O(1).

### 4) Global MAPF Solvers

We support multiple solvers with different optimality-speed trade-offs:

| Solver | Algorithm | Optimality | Use Case |
|--------|-----------|------------|----------|
| CBS | Conflict-Based Search | Optimal | ≤15 agents |
| LaCAM | Lazy Constraints Any-angle MAPF | Bounded-suboptimal | 15-50 agents |
| PIBT | Priority Inheritance with Backtracking | Complete | >50 agents |

---

## D. Tier-2: Decentralized Local Execution

Each agent executes a Sense-Plan-Act loop at every timestep:

### 1) Observation Construction (Sense)

Agent aᵢ constructs a local observation Oᵢ(t) containing:
- **Visible humans**: {hⱼ : d_M(sᵢ, hⱼ) ≤ ρ}
- **Visible agents**: {aⱼ : d_M(sᵢ, sⱼ) ≤ ρ, j ≠ i}
- **Blocked cells**: static obstacles ∪ human positions

where d_M denotes Manhattan distance and ρ is the FOV radius.

### 2) Safety Buffer Construction

From visible humans, construct the forbidden zone:

```
F(t) = ⋃_{h_j ∈ O_i(t)} B_r(h_j(t))
```

The safety radius r is inflated around each detected human position.

### 3) Local Replanning Decision (Plan)

Let p*ᵢ(t+1) denote the next position from the global plan. Local replanning is triggered when:
1. **No global plan available**: p*ᵢ(t+1) = ∅
2. **Plan enters forbidden zone**: p*ᵢ(t+1) ∈ F(t)
3. **Plan stagnates**: p*ᵢ(t+1) = sᵢ(t) while goal remains

When triggered, the local A* planner computes a detour avoiding F(t).

### 4) Local A* with Safety Constraints

**Algorithm 1: Safety-Aware Local A***

```
Input: start s, goal g, forbidden set F, mode (hard/soft)
Output: path P = [s, ..., g] or ∅

1:  OPEN ← {s}, g(s) ← 0, f(s) ← h(s,g)
2:  while OPEN ≠ ∅ do
3:    u ← argmin_{v∈OPEN} f(v)
4:    if u = g then return reconstruct_path(u)
5:    for each neighbor v of u do
6:      if v is static obstacle then continue
7:      if v ∈ F then
8:        if hard_mode then continue
9:        else cost ← C_blocked (high penalty)
10:     else cost ← 1
11:     if g(u) + cost < g(v) then
12:       g(v) ← g(u) + cost
13:       f(v) ← g(v) + h(v,g)
14:       parent(v) ← u
15: return ∅
```

**Hard vs. Soft Safety:**
- **Hard safety** (default): F(t) is an absolute barrier; agents wait if no path exists
- **Soft safety**: F(t) incurs high cost (C_blocked=50); allows traversal when necessary to prevent deadlock

### 5) Conflict Resolution (Act)

After determining the desired next position, agents resolve robot-robot conflicts through decentralized coordination.

**Conflict Types:**
- **Vertex conflict**: Two agents target the same cell
- **Edge conflict**: Two agents swap positions

**Priority-Based Resolution:**
Agents are ordered by priority tuple:

```
π(a_i) = (urgency_i, wait_steps_i, -id_i)
```

where:
- urgency = −d_M(sᵢ, gᵢ) (closer to goal = higher priority)
- wait_steps = starvation counter (prevents indefinite waiting)
- −id = deterministic tiebreaker

**Starvation Prevention:** If wait_steps > threshold (default 10), urgency is boosted by a constant B (default 50).

**Token Passing (with communication):** Adds fairness rotation—after K consecutive wins by the same agent, ownership rotates to the next-best contender.

---

## E. Human Motion Prediction

For global planning, we employ a conservative myopic predictor:

**Definition 3 (Myopic Prediction).** For each human hⱼ at position pⱼ, the predicted unsafe region over horizon H is:

```
B̂(h_j, t') = {p_j} ∪ N(p_j), ∀t' ∈ [t, t+H]
```

where N(pⱼ) denotes the 4-connected neighbors of pⱼ.

This static-shield approach assumes humans remain approximately stationary, erring on the side of caution rather than attempting complex trajectory prediction.

---

## F. System Parameters

| Parameter | Symbol | Default | Description |
|-----------|--------|---------|-------------|
| Planning horizon | H | 50 | Global plan depth (timesteps) |
| Replan interval | K | 25 | Steps between global replans |
| FOV radius | ρ | 4 | Agent perception range |
| Safety radius | r | 1 | Buffer around humans |
| Commit horizon | K_c | 25 | Task assignment lock duration |
| Delay threshold | α | 2.5 | ETA multiplier before reassignment |

---

## G. Algorithm Summary

**Algorithm 2: HA-LMAPF Two-Tier Execution**

```
Input: Environment G, agents A, humans H, task stream T
Output: Continuous task execution with safety guarantees

1:  Initialize global planner, local controllers
2:  for each timestep t do
3:    // Tier-1: Global Planning (periodic)
4:    if t mod K = 0 then
5:      tasks ← collect_open_tasks(T)
6:      assignments ← allocate_tasks(A, tasks)
7:      plans ← global_solve(G, A, assignments, H)
8:      distribute_plans(plans)
9:
10:   // Tier-2: Local Execution (every step)
11:   for each agent a_i ∈ A in parallel do
12:     O_i ← build_observation(a_i, ρ)
13:     F ← construct_forbidden_zone(O_i, r)
14:     p* ← get_global_plan_next(a_i)
15:     if needs_local_replan(p*, F) then
16:       p* ← local_astar(s_i, g_i, F)
17:     action ← resolve_conflicts(a_i, p*)
18:     execute(a_i, action)
19:
20:   // Update humans (external dynamics)
21:   update_humans(H)
```

---

## H. Key Design Decisions

This architecture achieves:

1. **Scalability**: Global tier handles 100+ agents via efficient MAPF solvers
2. **Responsiveness**: Local tier reacts in real-time to human dynamics
3. **Safety**: Configurable hard/soft constraints around humans
4. **Stability**: Commitment persistence prevents allocation thrashing

### Why Two Tiers?

| Single-Tier Approach | Limitation |
|---------------------|------------|
| Global-only | Cannot react to unpredictable humans in real-time |
| Local-only | No coordination; frequent deadlocks and collisions |
| Replan-on-conflict | Computational overhead; planning latency |

The two-tier approach combines the strengths of centralized optimization (global coordination, collision-free paths) with decentralized reactivity (real-time human avoidance, local conflict resolution).

### Why Commitment Persistence?

In lifelong MAPF settings, memoryless task allocators can cause "assignment thrashing":

```
Step 100: Agent A assigned Task X (distance 10)
Step 125: Task Y appears closer (distance 8) → Agent A reassigned to Y
Step 150: Task X now closer (distance 7) → Agent A reassigned back to X
```

This oscillation wastes planning effort and increases path length. Commitment persistence locks assignments for a finite horizon, converting the allocator from a memoryless policy to a hysteresis controller.

> "We use a greedy nearest-task allocator with commitment persistence: assignments remain fixed for a finite horizon and are only revised when infeasible or excessively delayed, preventing reassignment oscillations that could confound execution-level evaluation."
