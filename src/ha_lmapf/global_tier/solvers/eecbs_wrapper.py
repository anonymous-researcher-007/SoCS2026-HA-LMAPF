"""
EECBS (Explicit Estimation Conflict-Based Search) Official Wrapper

This module provides a wrapper for the official EECBS solver from:
https://github.com/Jiaoyang-Li/EECBS

EECBS is a bounded-suboptimal MAPF solver that uses explicit estimation
for conflict-based search. It guarantees solutions within a specified
suboptimality bound.

Cross-Platform Support:
    - Linux/macOS: Uses binary without extension (e.g., 'eecbs')
    - Windows: Uses .exe binary (e.g., 'eecbs.exe')

Usage:
    1. Clone and build EECBS:
       # Linux/macOS:
       sudo apt install libboost-all-dev
       git clone https://github.com/Jiaoyang-Li/EECBS.git && cd EECBS
       cmake -DCMAKE_BUILD_TYPE=RELEASE . && make
       cp eecbs src/ha_lmapf/global_tier/solvers/eecbs

       # Windows:
       git clone https://github.com/Jiaoyang-Li/EECBS.git && cd EECBS
       cmake -B build -DCMAKE_BUILD_TYPE=RELEASE && cmake --build build --config Release
       copy build\\Release\\eecbs.exe src\\ha_lmapf\\global_tier\\solvers\\eecbs.exe

    2. Use in code:
       from ha_lmapf.global_tier.solvers.eecbs_wrapper import EECBSSolver
       solver = EECBSSolver(suboptimality=1.2)  # 20% suboptimal
"""

from __future__ import annotations

import os
import platform
import re
import subprocess
import tempfile
from typing import Dict, List, Optional, Tuple

from ha_lmapf.core.types import AgentState, PlanBundle, Task, TimedPath

Cell = Tuple[int, int]

# Platform detection
IS_WINDOWS = platform.system() == "Windows"
BINARY_EXT = ".exe" if IS_WINDOWS else ""


class EECBSSolver:
    """
    Wrapper for the official EECBS C++ executable.

    EECBS (Explicit Estimation Conflict-Based Search) is a bounded-suboptimal
    MAPF solver that balances solution quality and computation time.

    Key parameter:
    - suboptimality (w): Bound factor. w=1.0 is optimal, w>1.0 allows suboptimality.

    Cross-platform compatible:
    - Linux/macOS: Looks for 'eecbs' binary
    - Windows: Looks for 'eecbs.exe' binary
    """

    DEFAULT_BINARY_LINUX = "eecbs"
    DEFAULT_BINARY_WINDOWS = "eecbs.exe"

    def __init__(
            self,
            binary_path: Optional[str] = None,
            time_limit_sec: float = 2.0,
            verbose: int = 0,
            suboptimality: float = 1.2,
    ) -> None:
        """
        Initialize the EECBS solver wrapper.

        Args:
            binary_path: Path to the eecbs executable. If None, auto-detects.
            time_limit_sec: Time limit for the solver in seconds.
            verbose: Verbosity level (0 = silent, 1+ = verbose).
            suboptimality: Suboptimality bound (w). 1.0 = optimal, >1.0 = bounded-suboptimal.
        """
        self.binary_path = self._find_binary(binary_path)
        self.time_limit_sec = time_limit_sec
        self.verbose = verbose
        self.suboptimality = max(1.0, suboptimality)

    @property
    def _default_binary(self) -> str:
        return self.DEFAULT_BINARY_WINDOWS if IS_WINDOWS else self.DEFAULT_BINARY_LINUX

    def _find_binary(self, binary_path: Optional[str]) -> str:
        if binary_path is not None:
            return binary_path

        solver_dir = os.path.dirname(__file__)

        if IS_WINDOWS:
            binary_names = ["eecbs.exe", "eecbs"]
        else:
            binary_names = ["eecbs"]

        search_paths = []
        for name in binary_names:
            search_paths.extend([
                os.path.join(solver_dir, name),
                os.path.join(solver_dir, "EECBS", name),
                os.path.join(solver_dir, "EECBS", "build", name),
                os.path.join(solver_dir, "EECBS", "build", "Release", name),
                os.path.join("build", name),
                name,
            ])

        for path in search_paths:
            if os.path.isfile(path):
                return path

        return os.path.join(solver_dir, self._default_binary)

    def plan(
            self,
            env,
            agents: Dict[int, AgentState],
            assignments: Dict[int, Task],
            step: int,
            horizon: int,
            rng=None,
    ) -> PlanBundle:
        """Compute collision-free paths using EECBS."""
        active_agents = self._get_active_agents(agents, assignments)

        if not active_agents:
            return PlanBundle(paths={}, created_step=step, horizon=horizon)

        with tempfile.TemporaryDirectory() as tmpdir:
            map_path = os.path.join(tmpdir, "map.map")
            scen_path = os.path.join(tmpdir, "agents.scen")
            output_path = os.path.join(tmpdir, "output.csv")
            paths_path = os.path.join(tmpdir, "paths.txt")

            map_filename = os.path.basename(map_path)
            self._write_map_file(env, map_path)
            agent_order = self._write_scenario_file(
                env, agents, assignments, active_agents, scen_path, map_filename
            )

            success = self._run_eecbs(
                map_path, scen_path, output_path, paths_path, len(agent_order)
            )

            if success and os.path.exists(paths_path):
                paths = self._parse_paths_file(paths_path, agent_order, step, horizon)
            else:
                paths = self._create_wait_paths(agents, active_agents, step, horizon)

        for aid in agents:
            if aid not in paths:
                paths[aid] = self._create_wait_path(agents[aid].pos, step, horizon)

        return PlanBundle(paths=paths, created_step=step, horizon=horizon)

    def _get_active_agents(
            self,
            agents: Dict[int, AgentState],
            assignments: Dict[int, Task],
    ) -> List[int]:
        active = []
        for aid, agent in agents.items():
            if agent.goal is not None:
                active.append(aid)
            elif aid in assignments:
                active.append(aid)
        return sorted(active)

    def _write_map_file(self, env, path: str) -> None:
        with open(path, 'w') as f:
            f.write("type octile\n")
            f.write(f"height {env.height}\n")
            f.write(f"width {env.width}\n")
            f.write("map\n")
            for r in range(env.height):
                row_str = ""
                for c in range(env.width):
                    if env.is_blocked((r, c)):
                        row_str += "@"
                    else:
                        row_str += "."
                f.write(row_str + "\n")

    def _write_scenario_file(
            self,
            env,
            agents: Dict[int, AgentState],
            assignments: Dict[int, Task],
            active_agents: List[int],
            path: str,
            map_filename: str = "map.map",
    ) -> List[int]:
        agent_order = []
        with open(path, 'w') as f:
            f.write("version 1\n")
            for aid in active_agents:
                agent = agents[aid]
                start = agent.pos
                if agent.goal is not None:
                    goal = agent.goal
                elif aid in assignments:
                    goal = assignments[aid].goal
                else:
                    goal = start
                start_col, start_row = start[1], start[0]
                goal_col, goal_row = goal[1], goal[0]
                f.write(f"0\t{map_filename}\t{env.width}\t{env.height}\t"
                        f"{start_col}\t{start_row}\t{goal_col}\t{goal_row}\t0.0\n")
                agent_order.append(aid)
        return agent_order

    def _run_eecbs(
            self,
            map_path: str,
            scen_path: str,
            output_path: str,
            paths_path: str,
            num_agents: int,
    ) -> bool:
        if not os.path.isfile(self.binary_path):
            print(f"[EECBS] ERROR: Binary not found at {self.binary_path}")
            print(f"[EECBS] Please build EECBS from https://github.com/Jiaoyang-Li/EECBS")
            if IS_WINDOWS:
                print(f"[EECBS] For Windows: copy build\\Release\\eecbs.exe to solvers\\eecbs.exe")
            else:
                print(f"[EECBS] For Linux/macOS: copy eecbs to solvers/eecbs")
            return False

        cmd = [
            self.binary_path,
            "-m", map_path,
            "-a", scen_path,
            "-o", output_path,
            f"--outputPaths={paths_path}",
            "-k", str(num_agents),
            "-t", str(int(self.time_limit_sec)),
            f"--suboptimality={self.suboptimality}",
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.time_limit_sec + 10,
            )
            if result.returncode != 0:
                if self.verbose > 0:
                    print(f"[EECBS] Solver returned code {result.returncode}")
                    print(f"[EECBS] stderr: {result.stderr}")
                return False
            return True
        except subprocess.TimeoutExpired:
            print(f"[EECBS] Solver timed out after {self.time_limit_sec}s")
            return False
        except FileNotFoundError:
            print(f"[EECBS] Binary not found: {self.binary_path}")
            return False
        except Exception as e:
            print(f"[EECBS] Execution error: {e}")
            return False

    def _parse_paths_file(
            self,
            paths_path: str,
            agent_order: List[int],
            start_step: int,
            horizon: int,
    ) -> Dict[int, TimedPath]:
        paths: Dict[int, TimedPath] = {}
        try:
            with open(paths_path, 'r') as f:
                lines = f.readlines()
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                match = re.match(r'^Agent\s+(\d+):\s*(.+)$', line, re.IGNORECASE)
                if match:
                    agent_idx = int(match.group(1))
                    path_str = match.group(2)
                    if agent_idx < len(agent_order):
                        aid = agent_order[agent_idx]
                        cells = self._parse_path_string(path_str)
                        if cells:
                            if len(cells) < horizon + 1:
                                cells = cells + [cells[-1]] * (horizon + 1 - len(cells))
                            elif len(cells) > horizon + 1:
                                cells = cells[:horizon + 1]
                            paths[aid] = TimedPath(cells=cells, start_step=start_step)
        except Exception as e:
            print(f"[EECBS] Error parsing paths file: {e}")
        return paths

    def _parse_path_string(self, path_str: str) -> List[Cell]:
        """Parse a path string like (row,col)->(row,col)->..."""
        cells = []
        # EECBS (Jiaoyang-Li) outputs (row, col) pairs
        coords = re.findall(r'\((\d+),(\d+)\)', path_str)
        for row_str, col_str in coords:
            row = int(row_str)
            col = int(col_str)
            cells.append((row, col))
        return cells

    def _create_wait_paths(
            self,
            agents: Dict[int, AgentState],
            active_agents: List[int],
            step: int,
            horizon: int,
    ) -> Dict[int, TimedPath]:
        paths = {}
        for aid in active_agents:
            paths[aid] = self._create_wait_path(agents[aid].pos, step, horizon)
        return paths

    def _create_wait_path(self, pos: Cell, step: int, horizon: int) -> TimedPath:
        cells = [pos] * (horizon + 1)
        return TimedPath(cells=cells, start_step=step)


# Aliases
EECBSCppSolver = EECBSSolver
BoundedSuboptimalCBSSolver = EECBSSolver
