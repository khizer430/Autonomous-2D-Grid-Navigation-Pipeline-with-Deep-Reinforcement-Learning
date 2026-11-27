from collections import deque
from typing import Any, Dict, List, Optional, Set, Tuple

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame


Coord = Tuple[int, int]


class GridWorldEnv(gym.Env):
    """2D grid-world navigation environment with random start/goal and static obstacles.

    Observation vector (shape=(8,), dtype=np.float32, normalized to [-1.0, 1.0]):
        [norm_agent_x, norm_agent_y, norm_dx_goal, norm_dy_goal,
         blocked_up, blocked_down, blocked_left, blocked_right]
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    # Action mapping: 0=Up, 1=Down, 2=Left, 3=Right
    ACTION_DELTAS: Dict[int, Coord] = {
        0: (0, -1),  # Up
        1: (0, 1),   # Down
        2: (-1, 0),  # Left
        3: (1, 0),   # Right
    }

    def __init__(
        self,
        grid_size: int = 10,
        num_obstacles: int = 14,
        max_steps: int = 100,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()

        if grid_size < 4:
            raise ValueError("grid_size must be at least 4.")

        max_allowed_obstacles = grid_size * grid_size - 2
        if not (0 <= num_obstacles <= max_allowed_obstacles):
            raise ValueError(
                f"num_obstacles must be between 0 and {max_allowed_obstacles}."
            )

        self.grid_size: int = grid_size
        self.num_obstacles: int = num_obstacles
        self.max_steps: int = max_steps
        self.render_mode: Optional[str] = render_mode

        # Discrete(4): 0=Up, 1=Down, 2=Left, 3=Right
        self.action_space: spaces.Discrete = spaces.Discrete(4)

        # 8-feature normalized state: agent pos, delta to goal, and 4-neighbor obstacle flags
        self.observation_space: spaces.Box = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(8,),
            dtype=np.float32,
        )

        # Episode state (initialized in reset)
        self._agent_pos: Coord = (0, 0)
        self._start_pos: Coord = (0, 0)
        self._goal_pos: Coord = (self.grid_size - 1, self.grid_size - 1)
        self._obstacles: Set[Coord] = set()
        self._step_count: int = 0

        # Pygame rendering state
        self._window_size: int = 600
        self._window: Optional[pygame.Surface] = None
        self._clock: Optional[pygame.time.Clock] = None

    def _is_cell_blocked(self, pos: Coord) -> bool:
        """Return True if a cell is outside grid bounds or occupied by an obstacle."""
        return (not self._is_within_bounds(pos)) or (pos in self._obstacles)

    def _get_obs(self) -> np.ndarray:
        """Return normalized position, goal delta, and 4-direction collision indicators."""
        denom = float(max(self.grid_size - 1, 1))
        ax, ay = self._agent_pos
        gx, gy = self._goal_pos

        norm_x = (ax / denom) * 2.0 - 1.0
        norm_y = (ay / denom) * 2.0 - 1.0
        dx_goal = (gx - ax) / denom
        dy_goal = (gy - ay) / denom

        blocked_flags = [
            1.0 if self._is_cell_blocked((ax + dx, ay + dy)) else -1.0
            for dx, dy in self.ACTION_DELTAS.values()
        ]

        obs = np.array(
            [norm_x, norm_y, dx_goal, dy_goal, *blocked_flags],
            dtype=np.float32,
        )
        return np.clip(obs, -1.0, 1.0)

    def _manhattan_to_goal(self, pos: Coord) -> int:
        """Compute Manhattan distance from `pos` to the goal."""
        return abs(self._goal_pos[0] - pos[0]) + abs(self._goal_pos[1] - pos[1])

    def _get_info(self) -> Dict[str, Any]:
        """Return auxiliary diagnostic information."""
        return {
            "start": self._start_pos,
            "goal": self._goal_pos,
            "obstacles": sorted(self._obstacles),
            "step_count": self._step_count,
            "manhattan_distance_to_goal": self._manhattan_to_goal(self._agent_pos),
            "reached_goal": self._agent_pos == self._goal_pos,
        }

    def _is_within_bounds(self, pos: Coord) -> bool:
        """Check whether `(x, y)` lies inside the grid."""
        x, y = pos
        return 0 <= x < self.grid_size and 0 <= y < self.grid_size

    def _has_valid_path(
        self, start: Coord, goal: Coord, obstacles: Set[Coord]
    ) -> bool:
        """Verify via Breadth-First Search (BFS) that `goal` is reachable from `start`."""
        if start in obstacles or goal in obstacles:
            return False

        queue: deque[Coord] = deque([start])
        visited: Set[Coord] = {start}

        while queue:
            curr = queue.popleft()
            if curr == goal:
                return True

            for dx, dy in self.ACTION_DELTAS.values():
                nxt = (curr[0] + dx, curr[1] + dy)
                if (
                    self._is_within_bounds(nxt)
                    and nxt not in obstacles
                    and nxt not in visited
                ):
                    visited.add(nxt)
                    queue.append(nxt)

        return False

    def _sample_layout(self) -> Tuple[Coord, Coord, Set[Coord]]:
        """Sample random start, goal, and solvable obstacle placement."""
        all_cells: List[Coord] = [
            (x, y) for y in range(self.grid_size) for x in range(self.grid_size)
        ]

        for _ in range(200):
            indices = self.np_random.choice(len(all_cells), size=2, replace=False)
            start: Coord = all_cells[int(indices[0])]
            goal: Coord = all_cells[int(indices[1])]

            if abs(start[0] - goal[0]) + abs(start[1] - goal[1]) < 2:
                continue

            candidate_pool: List[Coord] = [
                cell for cell in all_cells if cell != start and cell != goal
            ]
            obs_count = min(self.num_obstacles, len(candidate_pool))
            obs_indices = self.np_random.choice(
                len(candidate_pool), size=obs_count, replace=False
            )
            obstacles: Set[Coord] = {candidate_pool[int(i)] for i in obs_indices}

            if self._has_valid_path(start, goal, obstacles):
                return start, goal, obstacles

        start = (0, 0)
        goal = (self.grid_size - 1, self.grid_size - 1)
        candidate_pool = [
            cell for cell in all_cells if cell != start and cell != goal
        ]
        obs_count = min(self.num_obstacles, len(candidate_pool))
        obstacles = set()
        for cell in candidate_pool:
            if len(obstacles) >= obs_count:
                break
            trial_obs = obstacles | {cell}
            if self._has_valid_path(start, goal, trial_obs):
                obstacles = trial_obs

        return start, goal, obstacles

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to a new episode state."""
        super().reset(seed=seed)

        if options and "start" in options and "goal" in options:
            self._start_pos = tuple(options["start"])  # type: ignore[assignment]
            self._goal_pos = tuple(options["goal"])    # type: ignore[assignment]
            self._obstacles = set(
                tuple(c) for c in options.get("obstacles", [])  # type: ignore[arg-type]
            )
        else:
            self._start_pos, self._goal_pos, self._obstacles = self._sample_layout()

        self._agent_pos = self._start_pos
        self._step_count = 0

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), self._get_info()

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one discrete movement step in the grid world."""
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}; expected 0..3.")

        prev_dist = self._manhattan_to_goal(self._agent_pos)
        self._step_count += 1
        dx, dy = self.ACTION_DELTAS[int(action)]
        candidate_pos: Coord = (self._agent_pos[0] + dx, self._agent_pos[1] + dy)

        # Base step penalty (-1) + dense Manhattan progress shaping
        reward: float = -1.0
        terminated: bool = False

        if not self._is_within_bounds(candidate_pos):
            pass
        elif candidate_pos in self._obstacles:
            reward += -10.0
        else:
            self._agent_pos = candidate_pos

        curr_dist = self._manhattan_to_goal(self._agent_pos)
        reward += 0.5 * float(prev_dist - curr_dist)

        if self._agent_pos == self._goal_pos:
            reward += 100.0
            terminated = True

        truncated: bool = self._step_count >= self.max_steps

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), float(reward), terminated, truncated, self._get_info()

    def render(self) -> Optional[np.ndarray]:
        """Render the environment as an RGB array or in a Pygame window."""
        if self.render_mode is None:
            return None
        return self._render_frame()

    def _render_frame(self) -> Optional[np.ndarray]:
        """Draw the grid, start, goal, obstacles, and agent using Pygame."""
        if self._window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.set_caption("rl-nav: GridWorld")
            self._window = pygame.display.set_mode((self._window_size, self._window_size))

        if self._clock is None and self.render_mode == "human":
            self._clock = pygame.time.Clock()

        canvas = pygame.Surface((self._window_size, self._window_size))
        canvas.fill((248, 250, 252))

        cell_size: float = self._window_size / self.grid_size

        sx, sy = self._start_pos
        pygame.draw.rect(
            canvas,
            (191, 219, 254),
            pygame.Rect(sx * cell_size, sy * cell_size, cell_size, cell_size),
        )

        gx, gy = self._goal_pos
        pygame.draw.rect(
            canvas,
            (34, 197, 94),
            pygame.Rect(gx * cell_size, gy * cell_size, cell_size, cell_size),
        )

        for ox, oy in self._obstacles:
            pygame.draw.rect(
                canvas,
                (24, 24, 27),
                pygame.Rect(ox * cell_size, oy * cell_size, cell_size, cell_size),
            )

        for idx in range(self.grid_size + 1):
            pos = int(round(idx * cell_size))
            pygame.draw.line(
                canvas, (203, 213, 225), (0, pos), (self._window_size, pos), width=1
            )
            pygame.draw.line(
                canvas, (203, 213, 225), (pos, 0), (pos, self._window_size), width=1
            )

        ax, ay = self._agent_pos
        center = (
            int((ax + 0.5) * cell_size),
            int((ay + 0.5) * cell_size),
        )
        radius = max(4, int(cell_size * 0.32))
        pygame.draw.circle(canvas, (239, 68, 68), center, radius)

        if self.render_mode == "human":
            assert self._window is not None
            assert self._clock is not None
            self._window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self._clock.tick(self.metadata["render_fps"])
            return None

        return np.transpose(
            np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
        )

    def close(self) -> None:
        """Clean up Pygame resources."""
        if self._window is not None:
            pygame.display.quit()
            pygame.quit()
            self._window = None
            self._clock = None