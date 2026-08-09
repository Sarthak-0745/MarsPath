"""
compare_obstacle_geometry.py

Quantifies a specific hypothesis: that synthetic terrain (independent
per-cell random obstacles) produces mostly small, isolated obstacle
clusters, while real Jezero terrain (derived from continuous elevation)
produces large, connected, wall-like obstacle clusters -- and that this
geometric difference, not just obstacle density, is why the trained
agents suffer such a severe sim-to-real performance drop (see
evaluate_on_real_terrain.py results).

Method: find connected components of impassable cells (boulder + crater,
8-connectivity, matching the agent's 8-directional movement) in both
terrain types, and compare cluster size / elongation distributions.

Usage:
    python -m mars_rl_project.data_processing.compare_obstacle_geometry \
        --real-maps-dir data/processed_full --n-synthetic 50
"""

from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
from scipy import ndimage

from mars_rl_project.environments.terrain_generator import (
    TerrainConfig, generate_terrain, IMPASSABLE
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# 8-connectivity structuring element -- two impassable cells are considered
# part of the same obstacle cluster if they touch, including diagonally,
# matching the agent's own 8-directional movement (a diagonal gap is not
# a gap the agent can actually walk through).
STRUCTURE_8CONN = np.ones((3, 3), dtype=int)


def cluster_stats_for_grid(grid: np.ndarray) -> list[dict]:
    """Returns one dict per connected impassable cluster: size, bounding-box elongation."""
    impassable_mask = np.isin(grid, list(IMPASSABLE))
    labeled, n_clusters = ndimage.label(impassable_mask, structure=STRUCTURE_8CONN)

    stats = []
    for cluster_id in range(1, n_clusters + 1):
        coords = np.argwhere(labeled == cluster_id)
        size = len(coords)
        row_span = coords[:, 0].max() - coords[:, 0].min() + 1
        col_span = coords[:, 1].max() - coords[:, 1].min() + 1
        elongation = max(row_span, col_span) / min(row_span, col_span)
        stats.append({"size": size, "row_span": row_span, "col_span": col_span, "elongation": elongation})
    return stats


def summarize(all_stats: list[dict], label: str) -> dict:
    sizes = np.array([s["size"] for s in all_stats]) if all_stats else np.array([0])
    elongations = np.array([s["elongation"] for s in all_stats]) if all_stats else np.array([1.0])

    wall_like = np.array([s for s in all_stats if s["size"] >= 5 and s["elongation"] >= 2.0])
    n_wall_like = len(wall_like)

    print(f"\n{'='*60}\n{label}\n{'='*60}")
    print(f"Total obstacle clusters found : {len(all_stats)}")
    print(f"Cluster size (cells)  -- mean: {sizes.mean():.2f}  median: {np.median(sizes):.1f}  "
          f"max: {sizes.max()}  p90: {np.percentile(sizes, 90):.1f}")
    print(f"Elongation (long/short bbox side) -- mean: {elongations.mean():.2f}  "
          f"max: {elongations.max():.1f}")
    print(f"'Wall-like' clusters (size>=5 AND elongation>=2.0): {n_wall_like} "
          f"({100*n_wall_like/max(len(all_stats),1):.1f}% of all clusters)")
    print(f"Fraction of clusters that are a SINGLE isolated cell (size==1): "
          f"{100*np.mean(sizes==1):.1f}%")

    return {
        "n_clusters": len(all_stats), "mean_size": float(sizes.mean()),
        "median_size": float(np.median(sizes)), "max_size": int(sizes.max()),
        "mean_elongation": float(elongations.mean()), "n_wall_like": n_wall_like,
        "pct_single_cell": float(100 * np.mean(sizes == 1)),
    }


def main():
    parser = argparse.ArgumentParser(description="Compare obstacle cluster geometry: synthetic vs real terrain")
    parser.add_argument("--real-maps-dir", type=Path, default=PROJECT_ROOT / "data" / "processed_full")
    parser.add_argument("--n-synthetic", type=int, default=50,
                         help="Number of synthetic maps to generate per difficulty for comparison")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    # --- Synthetic terrain, all three difficulties pooled ---
    synthetic_stats = []
    for difficulty in ["easy", "medium", "hard"]:
        cfg = TerrainConfig(grid_size=50, difficulty=difficulty)
        for _ in range(args.n_synthetic):
            grid, _, _ = generate_terrain(cfg, rng)
            synthetic_stats.extend(cluster_stats_for_grid(grid))

    synth_summary = summarize(synthetic_stats, f"SYNTHETIC terrain ({args.n_synthetic}x3 maps, all difficulties pooled)")

    # --- Real terrain ---
    if not args.real_maps_dir.exists() or not list(args.real_maps_dir.glob("*.npz")):
        print(f"\nNo real-terrain maps found in {args.real_maps_dir} -- skipping real-terrain comparison.")
        return

    real_stats = []
    map_files = sorted(args.real_maps_dir.glob("*.npz"))
    for path in map_files:
        data = np.load(path)
        real_stats.extend(cluster_stats_for_grid(data["grid"]))

    real_summary = summarize(real_stats, f"REAL terrain ({len(map_files)} maps from {args.real_maps_dir.name})")

    # --- Head-to-head ---
    print(f"\n{'='*60}\nHEAD-TO-HEAD COMPARISON\n{'='*60}")
    print(f"{'Metric':<35}{'Synthetic':>14}{'Real':>14}")
    print(f"{'Mean cluster size (cells)':<35}{synth_summary['mean_size']:>14.2f}{real_summary['mean_size']:>14.2f}")
    print(f"{'Max cluster size (cells)':<35}{synth_summary['max_size']:>14}{real_summary['max_size']:>14}")
    print(f"{'Mean elongation':<35}{synth_summary['mean_elongation']:>14.2f}{real_summary['mean_elongation']:>14.2f}")
    print(f"{'% single-isolated-cell clusters':<35}{synth_summary['pct_single_cell']:>13.1f}%{real_summary['pct_single_cell']:>13.1f}%")
    print(f"{'Wall-like cluster count':<35}{synth_summary['n_wall_like']:>14}{real_summary['n_wall_like']:>14}")

    size_ratio = real_summary['mean_size'] / max(synth_summary['mean_size'], 0.01)
    print(
        f"\nReal-terrain obstacle clusters are on average {size_ratio:.1f}x larger than synthetic ones."
    )
    if real_summary['mean_size'] > synth_summary['mean_size'] * 1.5 and real_summary['mean_elongation'] > synth_summary['mean_elongation']:
        print(
            "This supports the hypothesis: real terrain has meaningfully larger, more elongated "
            "(wall-like) obstacle clusters than synthetic training terrain. Since the reward "
            "function shapes behavior via straight-line distance-to-goal (a potential-field-style "
            "signal), this class of obstacle geometry is a known failure case (local minima at "
            "concave/wall boundaries) -- consistent with the observed policy-cycling behavior."
        )
    else:
        print(
            "The size/elongation gap is smaller than expected -- the obstacle-geometry hypothesis "
            "may only partially explain the observed real-terrain failure rate; other factors "
            "(e.g. overall obstacle density, DEM classification noise) are likely also contributing."
        )


if __name__ == "__main__":
    main()
