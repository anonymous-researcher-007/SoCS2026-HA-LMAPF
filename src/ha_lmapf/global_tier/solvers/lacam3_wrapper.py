"""
LaCAM3 Official Executable Wrapper

This module provides a wrapper for the official LaCAM3 solver from:
https://github.com/Kei18/lacam3

Cross-Platform Support:
    - Linux/macOS: Uses binaries without extension (e.g., 'lacam3')
    - Windows: Uses .exe binaries (e.g., 'lacam3.exe')

Usage:
    1. Clone and build LaCAM3:
       # Linux/macOS:
       git clone --recursive https://github.com/Kei18/lacam3.git && cd lacam3
       cmake -B build && make -C build
       cp build/main src/ha_lmapf/global_tier/solvers/lacam3

       # Windows:
       git clone --recursive https://github.com/Kei18/lacam3.git && cd lacam3
       cmake -B build && cmake --build build --config Release
       copy build\\Release\\main.exe src\\ha_lmapf\\global_tier\\solvers\\lacam3.exe

    2. Use in code:
       from ha_lmapf.global_tier.solvers.lacam3_wrapper import LaCAM3Solver
       solver = LaCAM3Solver()  # Auto-detects platform
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


class LaCAM3Solver:
    """
    Wrapper for the official LaCAM3 C++ executable.

    LaCAM3 (Lazy Constraints Addition search for Multi-agent pathfinding)
    is a state-of-the-art MAPF solver that balances solution quality and speed.

    Cross-platform compatible:
    - Linux/macOS: Looks for 'lacam3' binary
    - Windows: Looks for 'lacam3.exe' binary

    Communication protocol:
    1. Write map file in MovingAI .map format
    2. Write scenario file in MovingAI .scen format
    3. Execute LaCAM3 binary with appropriate arguments
    4. Parse the result.txt output file

    Output format from LaCAM3:
        solution=
        0:(x1,y1),(x2,y2),...,
        1:(x1,y1),(x2,y2),...,
        ...
    Where (x,y) = (col, row) for each agent at each timestep.
    """

    # Default binary names (platform-specific)
    DEFAULT_BINARY_LINUX = "lacam3"
    DEFAULT_BINARY_WINDOWS = "lacam3.exe"

    def __init__(
            self,
            binary_path: Optional[str] = None,
            time_limit_sec: float = 1.0,
            verbose: int = 0,
    ) -> None:
        """
        Initialize the LaCAM3 solver wrapper.

        Args:
            binary_path: Path to the lacam3 executable. If None, auto-detects
                        based on platform:
                        - Linux/macOS: lacam3
                        - Windows: lacam3.exe
            time_limit_sec: Time limit for the solver in seconds
            verbose: Verbosity level (0-3)
        """
        self.binary_path = self._find_binary(binary_path)
        self.time_limit_sec = time_limit_sec
        self.verbose = verbose

    @property
    def _default_binary(self) -> str:
        """Get the default binary name for the current platform."""
        return self.DEFAULT_BINARY_WINDOWS if IS_WINDOWS else self.DEFAULT_BINARY_LINUX

    def _find_binary(self, binary_path: Optional[str]) -> str:
        """Find the LaCAM3 executable for the current platform."""
        # If a custom path is provided, use it directly
        if binary_path is not None:
            return binary_path

        solver_dir = os.path.dirname(__file__)

        # Platform-specific binary names to search
        if IS_WINDOWS:
            binary_names = ["lacam3.exe", "lacam3", "main.exe"]
        else:
            binary_names = ["lacam3", "main"]

        # Build search paths
        search_paths = []
        for name in binary_names:
            search_paths.extend([
                os.path.join(solver_dir, name),
                os.path.join(solver_dir, "lacam3", "build", name),
                os.path.join(solver_dir, "lacam3", "build", "Release", name),
                os.path.join("build", name),
                os.path.join("build", "Release", name),
                name,
                os.path.join(".", name),
            ])

        for path in search_paths:
            if os.path.isfile(path):
                return path

        # Return the default path - will fail later with helpful message
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
        """
        Compute collision-free paths using LaCAM3.

        Implements the GlobalPlanner protocol.

        Args:
            env: Environment with is_free(), is_blocked(), width, height
            agents: Dict mapping agent_id -> AgentState (with pos)
            assignments: Dict mapping agent_id -> Task (with goal)
            step: Current simulation step (paths start from this step)
            horizon: Planning horizon (paths will have horizon+1 cells)
            rng: Random number generator (unused, LaCAM3 has internal seed)

        Returns:
            PlanBundle with paths for all active agents
        """
        # Determine which agents need planning
        active_agents = self._get_active_agents(agents, assignments)

        if not active_agents:
            return PlanBundle(paths={}, created_step=step, horizon=horizon)

        # Create temporary files
        with tempfile.TemporaryDirectory() as tmpdir:
            map_path = os.path.join(tmpdir, "map.map")
            scen_path = os.path.join(tmpdir, "agents.scen")
            result_path = os.path.join(tmpdir, "result.txt")

            # Write input files
            map_filename = os.path.basename(map_path)  # "map.map"
            self._write_map_file(env, map_path)
            agent_order = self._write_scenario_file(
                env, agents, assignments, active_agents, scen_path, map_filename
            )

            # Run LaCAM3
            success = self._run_lacam3(map_path, scen_path, result_path, len(agent_order))

            if success and os.path.exists(result_path):
                # Parse results
                paths = self._parse_result_file(result_path, agent_order, step, horizon)
            else:
                # Fallback: WAIT paths for all agents
                paths = self._create_wait_paths(agents, active_agents, step, horizon)

        # Ensure all agents have paths (including inactive ones)
        for aid in agents:
            if aid not in paths:
                paths[aid] = self._create_wait_path(agents[aid].pos, step, horizon)

        return PlanBundle(paths=paths, created_step=step, horizon=horizon)

    def _get_active_agents(
            self,
            agents: Dict[int, AgentState],
            assignments: Dict[int, Task],
    ) -> List[int]:
        """Get list of agents that need planning (have goals)."""
        active = []
        for aid, agent in agents.items():
            # Agent has a current goal (from assignment or ongoing task)
            if agent.goal is not None:
                active.append(aid)
            elif aid in assignments:
                active.append(aid)
        return sorted(active)

    def _write_map_file(self, env, path: str) -> None:
        """Write environment to MovingAI .map format."""
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
        """
        Write MovingAI .scen format scenario file.

        Format (per line):
            bucket map_name map_width map_height start_col start_row goal_col goal_row optimal_dist

        Note: The map_name field must match the actual map filename used in the -m argument.

        Returns:
            List of agent IDs in the order they appear in the scenario file
        """
        agent_order = []

        with open(path, 'w') as f:
            f.write("version 1\n")

            for aid in active_agents:
                agent = agents[aid]
                start = agent.pos  # (row, col)

                # Determine goal
                if agent.goal is not None:
                    goal = agent.goal
                elif aid in assignments:
                    goal = assignments[aid].goal
                else:
                    goal = start  # WAIT

                # MovingAI format uses (col, row) for coordinates
                start_col, start_row = start[1], start[0]
                goal_col, goal_row = goal[1], goal[0]

                # bucket map_name width height start_col start_row goal_col goal_row dist
                # map_name must match the actual filename passed to LaCAM3
                f.write(f"0\t{map_filename}\t{env.width}\t{env.height}\t"
                        f"{start_col}\t{start_row}\t{goal_col}\t{goal_row}\t0.0\n")

                agent_order.append(aid)

        return agent_order

    def _run_lacam3(
            self,
            map_path: str,
            scen_path: str,
            result_path: str,
            num_agents: int,
    ) -> bool:
        """
        Execute the LaCAM3 binary.

        Returns:
            True if execution was successful, False otherwise
        """
        if not os.path.isfile(self.binary_path):
            print(f"[LaCAM3] ERROR: Binary not found at {self.binary_path}")
            print(f"[LaCAM3] Please build LaCAM3 from https://github.com/Kei18/lacam3")
            if IS_WINDOWS:
                print(f"[LaCAM3] For Windows: copy build\\Release\\main.exe to solvers\\lacam3.exe")
            else:
                print(f"[LaCAM3] For Linux/macOS: copy build/main to solvers/lacam3")
            return False

        cmd = [
            self.binary_path,
            "-m", map_path,
            "-i", scen_path,
            "-N", str(num_agents),
            "-o", result_path,
            "-t", str(self.time_limit_sec),
            "-v", str(self.verbose),
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.time_limit_sec + 5,  # Extra buffer
            )

            if result.returncode != 0:
                if self.verbose > 0:
                    print(f"[LaCAM3] Solver returned code {result.returncode}")
                    print(f"[LaCAM3] stderr: {result.stderr}")
                return False

            return True

        except subprocess.TimeoutExpired:
            print(f"[LaCAM3] Solver timed out after {self.time_limit_sec}s")
            return False
        except FileNotFoundError:
            print(f"[LaCAM3] Binary not found: {self.binary_path}")
            return False
        except Exception as e:
            print(f"[LaCAM3] Execution error: {e}")
            return False

    def _parse_result_file(
            self,
            result_path: str,
            agent_order: List[int],
            start_step: int,
            horizon: int,
    ) -> Dict[int, TimedPath]:
        """
        Parse LaCAM3 result.txt output.

        Format:
            ... metadata lines ...
            solution=
            0:(x1,y1),(x2,y2),...,
            1:(x1,y1),(x2,y2),...,
            ...

        Where (x,y) = (col, row) for each agent.
        """
        paths: Dict[int, TimedPath] = {}

        try:
            with open(result_path, 'r') as f:
                content = f.read()

            # Find solution section
            if "solution=" not in content:
                print("[LaCAM3] No solution found in result file")
                return paths

            # Extract solution lines
            solution_start = content.index("solution=") + len("solution=")
            solution_text = content[solution_start:].strip()

            # Parse timestep lines
            # Each line: "t:(x1,y1),(x2,y2),...,"
            agent_paths: Dict[int, List[Cell]] = {aid: [] for aid in agent_order}

            for line in solution_text.split('\n'):
                line = line.strip()
                if not line or ':' not in line:
                    continue

                # Parse "t:(x1,y1),(x2,y2),...,"
                match = re.match(r'^(\d+):(.+)$', line)
                if not match:
                    continue

                timestep = int(match.group(1))
                coords_str = match.group(2)

                # Extract all (x,y) pairs
                coord_pattern = r'\((\d+),(\d+)\)'
                coords = re.findall(coord_pattern, coords_str)

                if len(coords) != len(agent_order):
                    # Mismatch in agent count
                    continue

                for idx, (x_str, y_str) in enumerate(coords):
                    col = int(x_str)
                    row = int(y_str)
                    cell = (row, col)  # Convert to (row, col) format

                    aid = agent_order[idx]
                    agent_paths[aid].append(cell)

            # Create TimedPath objects
            for aid, cells in agent_paths.items():
                if cells:
                    # Pad or truncate to horizon+1
                    if len(cells) < horizon + 1:
                        # Pad with last position
                        cells = cells + [cells[-1]] * (horizon + 1 - len(cells))
                    elif len(cells) > horizon + 1:
                        cells = cells[:horizon + 1]

                    paths[aid] = TimedPath(cells=cells, start_step=start_step)

        except Exception as e:
            print(f"[LaCAM3] Error parsing result file: {e}")

        return paths

    def _create_wait_paths(
            self,
            agents: Dict[int, AgentState],
            active_agents: List[int],
            step: int,
            horizon: int,
    ) -> Dict[int, TimedPath]:
        """Create WAIT paths for all active agents (fallback)."""
        paths = {}
        for aid in active_agents:
            paths[aid] = self._create_wait_path(agents[aid].pos, step, horizon)
        return paths

    def _create_wait_path(self, pos: Cell, step: int, horizon: int) -> TimedPath:
        """Create a WAIT-in-place path."""
        cells = [pos] * (horizon + 1)
        return TimedPath(cells=cells, start_step=step)


# Alias for backward compatibility
LaCAM3CppPlanner = LaCAM3Solver
