"""
mars_terrain_env.py

MarsTerrainEnv: the custom Gymnasium environment described in Section 2.3
of the MarsPath proposal.

POMDP formulation
------------------
- True state: the full 50x50 terrain grid + rover position (never fully
  observed by the agent).
- Observation (partial): a 13x13 local terrain patch centered on the
  rover, plus a normalized direction-to-goal vector and a normalized
  distance-to-goal scalar. No global map is ever exposed.
- Action space: 8 discrete compass directions.
- Reward: +100 goal / -10 collision / +0.5 closer / -1 farther / -0.1
  per-step, exactly as specified in the proposal.
- Termination: goal reached (success), collision with impassable
  terrain (failure), or max_steps exceeded (timeout) -- tracked
  separately in `info` as specified in Section 2.3.

Design assumption (flagged for your methods/limitations section):
the grid boundary is treated as impassable. Moving off the edge of the
map counts as a collision. The proposal doesn't specify boundary
behavior explicitly, so this is a reasonable default but worth
mentioning as a modeling choice.
"""

from __future__ import annotations
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .terrain_generator import (
    TerrainConfig, generate_terrain, SAFE, ROCK, BOULDER, CRATER, IMPASSABLE
)

# 8 discrete actions -> (row_delta, col_delta). Row = north/south axis (row-- = north).
_ACTIONS = {
    0: (-1, 0),   # North
    1: (1, 0),    # South
    2: (0, 1),    # East
    3: (0, -1),   # West
    4: (-1, 1),   # Northeast
    5: (-1, -1),  # Northwest
    6: (1, 1),    # Southeast
    7: (1, -1),   # Southwest
}
ACTION_NAMES = ["N", "S", "E", "W", "NE", "NW", "SE", "SW"]

REWARD_GOAL = 100.0
REWARD_COLLISION = -10.0
REWARD_CLOSER = 0.5
REWARD_FARTHER = -1.0
REWARD_STEP = -0.1

PATCH_RADIUS = 6  # 6 -> 13x13 window, matches AutoNav-derived sensing range


class MarsTerrainEnv(gym.Env):
    metadata = {"render_modes": ["ansi"], "render_fps": 4}

    def __init__(
        self,
        difficulty: str = "medium",
        grid_size: int = 50,
        max_steps: int = 200,
        render_mode: str | None = None,
    ):
        super().__init__()
        self.difficulty = difficulty
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(8)

        patch_dim = 2 * PATCH_RADIUS + 1  # 13
        self.observation_space = spaces.Dict(
            {
                # Terrain codes 0-3 normalized to [0, 1]; shape (13, 13, 1)
                # so it plugs directly into a small CNN feature extractor
                # (SB3 MultiInputPolicy) without reshaping later.
                "terrain_patch": spaces.Box(
                    low=0.0, high=1.0, shape=(patch_dim, patch_dim, 1), dtype=np.float32
                ),
                "direction_to_goal": spaces.Box(
                    low=-1.0, high=1.0, shape=(2,), dtype=np.float32
                ),
                "distance_to_goal": spaces.Box(
                    low=0.0, high=1.0, shape=(1,), dtype=np.float32
                ),
            }
        )

        self._grid = None
        self._pos = None
        self._goal = None
        self._steps = 0
        self._max_possible_dist = float(np.hypot(grid_size, grid_size))
        self._rng = np.random.default_rng()

    # ------------------------------------------------------------------ #
    # Core Gymnasium API
    # ------------------------------------------------------------------ #
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        options = options or {}

        # Real-terrain evaluation path (Step 6): if a fixed grid/start/goal is
        # provided (e.g. loaded from a NASA DEM-derived .npz map), use it as-is
        # instead of procedurally generating synthetic terrain. This lets the
        # exact same agent/observation/reward code evaluate on real Jezero
        # Crater terrain with zero changes to the trained model.
        if "fixed_grid" in options:
            grid = options["fixed_grid"]
            if grid.shape != (self.grid_size, self.grid_size):
                raise ValueError(
                    f"fixed_grid shape {grid.shape} does not match env grid_size "
                    f"({self.grid_size}, {self.grid_size}). Resample the real-terrain "
                    f"map to this size first (see process_nasa_terrain.py's --grid-size)."
                )
            self._grid = grid.astype(np.int8)
            self._pos = tuple(int(v) for v in options["fixed_start"])
            self._goal = tuple(int(v) for v in options["fixed_goal"])
        else:
            difficulty = options.get("difficulty", self.difficulty)
            cfg = TerrainConfig(grid_size=self.grid_size, difficulty=difficulty)
            self._grid, self._pos, self._goal = generate_terrain(cfg, self._rng)

        self._steps = 0

        obs = self._get_obs()
        info = self._get_info(terminated_reason=None)
        return obs, info

    def step(self, action: int):
        assert self.action_space.contains(action), f"Invalid action {action}"
        self._steps += 1

        dr, dc = _ACTIONS[action]
        new_r, new_c = self._pos[0] + dr, self._pos[1] + dc

        prev_dist = self._dist_to_goal(self._pos)
        terminated = False
        truncated = False
        reason = None

        off_grid = not (0 <= new_r < self.grid_size and 0 <= new_c < self.grid_size)
        hit_obstacle = (not off_grid) and (self._grid[new_r, new_c] in IMPASSABLE)

        if off_grid or hit_obstacle:
            reward = REWARD_COLLISION
            terminated = True
            reason = "collision"
            # Rover stays in place on collision (doesn't move into the obstacle/off-grid).
        else:
            self._pos = (new_r, new_c)
            new_dist = self._dist_to_goal(self._pos)

            if self._pos == self._goal:
                reward = REWARD_GOAL
                terminated = True
                reason = "success"
            else:
                reward = REWARD_CLOSER if new_dist < prev_dist else REWARD_FARTHER
                reward += REWARD_STEP

        if not terminated and self._steps >= self.max_steps:
            truncated = True
            reason = "timeout"

        obs = self._get_obs()
        info = self._get_info(terminated_reason=reason)
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode != "ansi":
            return None
        symbols = {SAFE: ".", ROCK: ",", BOULDER: "#", CRATER: "O"}
        lines = []
        for r in range(self.grid_size):
            row = []
            for c in range(self.grid_size):
                if (r, c) == self._pos:
                    row.append("R")
                elif (r, c) == self._goal:
                    row.append("G")
                else:
                    row.append(symbols[int(self._grid[r, c])])
            lines.append("".join(row))
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _dist_to_goal(self, pos) -> float:
        return float(np.hypot(pos[0] - self._goal[0], pos[1] - self._goal[1]))

    def _get_obs(self) -> dict:
        r, c = self._pos
        size = self.grid_size
        patch_dim = 2 * PATCH_RADIUS + 1
        patch = np.full((patch_dim, patch_dim), BOULDER, dtype=np.int8)  # off-map = impassable

        r0, r1 = r - PATCH_RADIUS, r + PATCH_RADIUS + 1
        c0, c1 = c - PATCH_RADIUS, c + PATCH_RADIUS + 1
        src_r0, src_r1 = max(0, r0), min(size, r1)
        src_c0, src_c1 = max(0, c0), min(size, c1)
        dst_r0, dst_c0 = src_r0 - r0, src_c0 - c0
        dst_r1 = dst_r0 + (src_r1 - src_r0)
        dst_c1 = dst_c0 + (src_c1 - src_c0)
        patch[dst_r0:dst_r1, dst_c0:dst_c1] = self._grid[src_r0:src_r1, src_c0:src_c1]

        terrain_patch = (patch.astype(np.float32) / 3.0)[..., None]

        gr, gc = self._goal
        vec = np.array([gr - r, gc - c], dtype=np.float32)
        norm = np.linalg.norm(vec)
        direction = (vec / norm) if norm > 1e-8 else np.zeros(2, dtype=np.float32)

        dist = np.array([self._dist_to_goal(self._pos) / self._max_possible_dist], dtype=np.float32)

        return {
            "terrain_patch": terrain_patch,
            "direction_to_goal": direction.astype(np.float32),
            "distance_to_goal": dist,
        }

    def _get_info(self, terminated_reason: str | None) -> dict:
        return {
            "position": self._pos,
            "goal": self._goal,
            "steps": self._steps,
            "termination_reason": terminated_reason,  # "success" | "collision" | "timeout" | None
        }
