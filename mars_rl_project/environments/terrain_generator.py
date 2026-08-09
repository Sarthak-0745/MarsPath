"""
terrain_generator.py

Procedural terrain generation for MarsPath.

Terrain cell codes (matches Section 2.3 of the proposal):
    0 = safe ground   (passable)
    1 = rock          (passable, but flagged as rough terrain)
    2 = boulder       (impassable)
    3 = crater        (impassable)

Difficulty tiers control obstacle (boulder + crater) density, per the
proposal's "three difficulty tiers (easy, medium, hard, defined by
obstacle density)" (Section 2.4).

Design note / assumption (not fully specified in the proposal): every
generated grid is checked for start->goal reachability via BFS over
passable cells before being handed to the environment. If the random
layout is unsolvable, it is regenerated. This keeps training signal
clean -- an agent should never be punished for failing to solve a
literally unsolvable map. This is a modeling choice you should mention
in your methods section / limitations if you keep it.
"""

from __future__ import annotations
from collections import deque
from dataclasses import dataclass
import numpy as np

SAFE, ROCK, BOULDER, CRATER = 0, 1, 2, 3
IMPASSABLE = {BOULDER, CRATER}

DIFFICULTY_PRESETS = {
    # (rock_fraction, boulder_fraction, crater_fraction)
    "easy":   (0.08, 0.05, 0.02),
    "medium": (0.10, 0.10, 0.05),
    "hard":   (0.10, 0.16, 0.09),
}


@dataclass
class TerrainConfig:
    grid_size: int = 50
    difficulty: str = "medium"
    min_start_goal_distance: int = 20  # Chebyshev distance, forces a real navigation task


def _random_free_cell(rng: np.random.Generator, grid: np.ndarray) -> tuple[int, int]:
    size = grid.shape[0]
    while True:
        r, c = rng.integers(0, size), rng.integers(0, size)
        if grid[r, c] == SAFE:
            return int(r), int(c)


def _bfs_reachable(grid: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> bool:
    """8-connected BFS over passable cells (rock+safe), matching the agent's action space."""
    size = grid.shape[0]
    visited = np.zeros_like(grid, dtype=bool)
    q = deque([start])
    visited[start] = True
    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    while q:
        r, c = q.popleft()
        if (r, c) == goal:
            return True
        for dr, dc in neighbors:
            nr, nc = r + dr, c + dc
            if 0 <= nr < size and 0 <= nc < size and not visited[nr, nc]:
                if grid[nr, nc] not in IMPASSABLE:
                    visited[nr, nc] = True
                    q.append((nr, nc))
    return False


def generate_terrain(
    config: TerrainConfig, rng: np.random.Generator
) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
    """
    Returns (grid, start_pos, goal_pos).
    grid is a (grid_size, grid_size) int array of terrain codes.
    Regenerates until start/goal are on safe ground, far enough apart,
    and a passable path exists between them.
    """
    if config.difficulty not in DIFFICULTY_PRESETS:
        raise ValueError(f"Unknown difficulty '{config.difficulty}', choose from {list(DIFFICULTY_PRESETS)}")

    rock_p, boulder_p, crater_p = DIFFICULTY_PRESETS[config.difficulty]
    safe_p = 1.0 - rock_p - boulder_p - crater_p
    size = config.grid_size

    for _attempt in range(200):
        grid = rng.choice(
            [SAFE, ROCK, BOULDER, CRATER],
            size=(size, size),
            p=[safe_p, rock_p, boulder_p, crater_p],
        ).astype(np.int8)

        start = _random_free_cell(rng, grid)
        goal = _random_free_cell(rng, grid)

        cheby = max(abs(start[0] - goal[0]), abs(start[1] - goal[1]))
        if cheby < config.min_start_goal_distance:
            continue

        # Clear a small safe buffer around start/goal so the agent doesn't
        # spawn boxed in or facing an immediate forced collision.
        for pos in (start, goal):
            r, c = pos
            grid[max(0, r - 1):r + 2, max(0, c - 1):c + 2] = SAFE

        if _bfs_reachable(grid, start, goal):
            return grid, start, goal

    raise RuntimeError(
        "Failed to generate a solvable terrain after 200 attempts. "
        "Try a lower difficulty or smaller min_start_goal_distance."
    )
