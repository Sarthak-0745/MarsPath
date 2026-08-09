"""
render_failure_case.py

Visualizes a single real-terrain evaluation episode: the map, the
agent's actual step-by-step path, start/goal, and where the episode
ended (success / collision / timeout). Built to inspect specific
interesting cases -- e.g. jezero_map_03, which timed out for both the
DQN and PPO hard-difficulty models -- and produce a figure for the
report explaining WHY a given map is hard, not just that it failed.

Usage:
    python -m mars_rl_project.data_processing.render_failure_case \
        --model models/ppo_hard/final_model.zip --algo ppo \
        --map data/processed/jezero_map_03.npz
"""

from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from stable_baselines3 import DQN, PPO

from mars_rl_project.environments import MarsTerrainEnv
from mars_rl_project.environments.terrain_generator import SAFE, ROCK, BOULDER, CRATER

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ALGO_CLASSES = {"dqn": DQN, "ppo": PPO}

# Terrain colors: safe=light tan, rock=darker tan, boulder=dark gray, crater=near-black
TERRAIN_COLORS = ["#e8d9b5", "#c2a878", "#5a5a5a", "#1a1a1a"]
TERRAIN_CMAP = ListedColormap(TERRAIN_COLORS)


def run_episode_with_trace(model, grid: np.ndarray, start: tuple, goal: tuple, max_steps: int = 200):
    env = MarsTerrainEnv(grid_size=grid.shape[0], max_steps=max_steps)
    obs, info = env.reset(options={"fixed_grid": grid, "fixed_start": start, "fixed_goal": goal})

    path = [start]
    terminated = truncated = False
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        path.append(info["position"])

    outcome = info["termination_reason"] or "timeout"
    return path, outcome, info["steps"]


def render(grid: np.ndarray, start: tuple, goal: tuple, path: list, outcome: str,
           steps: int, title: str, out_path: Path, zoom_radius: int | None = None):
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(grid, cmap=TERRAIN_CMAP, vmin=0, vmax=3, origin="upper")

    path_arr = np.array(path)
    ax.plot(path_arr[:, 1], path_arr[:, 0], color="#00b0ff", linewidth=2.2,
            marker="o", markersize=3, alpha=0.9, label="Agent path", zorder=3)

    # Number the last N steps when zoomed in, so the exact order of movement
    # near the goal (e.g. oscillation between two cells) is readable, not just
    # visually implied by overlapping dots.
    if zoom_radius is not None:
        n_label = min(30, len(path))
        for i, (r, c) in enumerate(path[-n_label:]):
            step_num = len(path) - n_label + i
            ax.annotate(str(step_num), (c, r), fontsize=6, color="#003a5c",
                        xytext=(3, 3), textcoords="offset points", zorder=6)

    ax.scatter(*start[::-1], s=220, c="#2ecc71", marker="s", edgecolors="black",
               linewidths=1.5, zorder=4, label="Start")
    ax.scatter(*goal[::-1], s=260, c="#f1c40f", marker="*", edgecolors="black",
               linewidths=1.5, zorder=4, label="Goal")

    end_color = {"success": "#2ecc71", "collision": "#e74c3c", "timeout": "#e67e22"}.get(outcome, "gray")
    ax.scatter(path_arr[-1, 1], path_arr[-1, 0], s=180, facecolors="none",
               edgecolors=end_color, linewidths=3, zorder=5,
               label=f"End ({outcome}, step {steps})")

    legend_patches = [
        plt.Rectangle((0, 0), 1, 1, color=TERRAIN_COLORS[0], label="Safe"),
        plt.Rectangle((0, 0), 1, 1, color=TERRAIN_COLORS[1], label="Rock"),
        plt.Rectangle((0, 0), 1, 1, color=TERRAIN_COLORS[2], label="Boulder (impassable)"),
        plt.Rectangle((0, 0), 1, 1, color=TERRAIN_COLORS[3], label="Crater (impassable)"),
    ]
    terrain_legend = ax.legend(handles=legend_patches, loc="upper left",
                                bbox_to_anchor=(1.02, 1.0), title="Terrain", fontsize=9)
    ax.add_artist(terrain_legend)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 0.55), fontsize=9)

    if zoom_radius is not None:
        gr, gc = goal
        ax.set_xlim(gc - zoom_radius, gc + zoom_radius)
        ax.set_ylim(gr + zoom_radius, gr - zoom_radius)  # inverted y (origin="upper")

    ax.set_title(title, fontsize=13)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")

    if zoom_radius is not None:
        n_show = min(20, len(path))
        print(f"\nLast {n_show} positions (step: (row, col)):")
        for i, pos in enumerate(path[-n_show:]):
            step_num = len(path) - n_show + i
            print(f"  step {step_num:>3}: {pos}")


def main():
    parser = argparse.ArgumentParser(description="Render a single real-terrain evaluation episode")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--algo", choices=["dqn", "ppo"], required=True)
    parser.add_argument("--map", type=Path, required=True, help="Path to a single .npz real-terrain map")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--output", type=Path, default=None,
                         help="Output PNG path (default: <map_name>_<algo>_<run_name>.png next to the map)")
    parser.add_argument("--zoom-radius", type=int, default=None,
                         help="If set, crop the view to this many cells around the goal, and print/label "
                              "the last steps of the path -- use this to inspect near-goal behavior "
                              "(e.g. oscillation) that's invisible at full-map scale.")
    args = parser.parse_args()

    print(f"Loading {args.algo.upper()} model: {args.model}")
    model = ALGO_CLASSES[args.algo].load(str(args.model))

    print(f"Loading map: {args.map}")
    data = np.load(args.map)
    grid, start, goal = data["grid"], tuple(data["start"].tolist()), tuple(data["goal"].tolist())

    path, outcome, steps = run_episode_with_trace(model, grid, start, goal, args.max_steps)
    print(f"Outcome: {outcome} in {steps} steps")

    run_name = args.model.parent.name
    suffix = f"_zoom{args.zoom_radius}" if args.zoom_radius else ""
    out_path = args.output or (PROJECT_ROOT / "outputs" / f"{args.map.stem}_{run_name}{suffix}.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    title = f"{args.map.stem}  |  {run_name}  |  outcome: {outcome} ({steps} steps)"
    render(grid, start, goal, path, outcome, steps, title, out_path, zoom_radius=args.zoom_radius)


if __name__ == "__main__":
    main()
