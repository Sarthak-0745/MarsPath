"""
build_results_dataset.py

Builds ONE csv combining, per real-terrain map:
  - obstacle geometry (max/mean cluster size, cluster count)
  - the outcome of every model run you specify against that map
      (e.g. DQN deterministic, PPO deterministic, PPO stochastic)

This is the single data file the Results section should be built from --
everything else (tables, plots, correlation checks) can be derived from
this one CSV in a few lines of pandas, rather than re-running evaluations
every time a new figure is needed.

Usage:
    python -m mars_rl_project.data_processing.build_results_dataset \
        --maps-dir data/processed_full \
        --run "DQN=dqn:models/dqn_hard_fixed/final_model.zip:deterministic" \
        --run "PPO=ppo:models/ppo_hard/final_model.zip:deterministic" \
        --run "PPO_stochastic=ppo:models/ppo_hard/final_model.zip:stochastic" \
        --output outputs/results_dataset.csv

Each --run is: <label>=<algo>:<model_path>:<deterministic|stochastic>
Repeat --run as many times as you want columns in the output.
"""

from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import DQN, PPO

from mars_rl_project.environments import MarsTerrainEnv
from mars_rl_project.data_processing.evaluate_on_real_terrain import load_maps, detect_cycle
from mars_rl_project.data_processing.compare_obstacle_geometry import cluster_stats_for_grid

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ALGO_CLASSES = {"dqn": DQN, "ppo": PPO}


def parse_run_spec(spec: str) -> dict:
    """Parses 'LABEL=algo:model_path:mode' into a dict."""
    try:
        label, rest = spec.split("=", 1)
        algo, model_path, mode = rest.split(":", 2)
    except ValueError:
        raise ValueError(
            f"Bad --run spec '{spec}'. Expected format: "
            f"LABEL=algo:model_path:deterministic|stochastic"
        )
    if algo not in ALGO_CLASSES:
        raise ValueError(f"Unknown algo '{algo}' in --run spec (must be dqn or ppo)")
    if mode not in ("deterministic", "stochastic"):
        raise ValueError(f"Mode must be 'deterministic' or 'stochastic', got '{mode}'")
    return {"label": label, "algo": algo, "model_path": Path(model_path), "deterministic": mode == "deterministic"}


def run_episode(model, env, grid, start, goal, deterministic: bool, max_steps: int) -> dict:
    obs, info = env.reset(options={"fixed_grid": grid, "fixed_start": start, "fixed_goal": goal})
    path = [start]
    total_reward = 0.0
    terminated = truncated = False
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(int(action))
        path.append(info["position"])
        total_reward += reward

    outcome = info["termination_reason"] or "timeout"
    if outcome == "timeout":
        cycle_period = detect_cycle(path)
        if cycle_period is not None:
            outcome = f"cycle-{cycle_period}"
    return {"outcome": outcome, "reward": total_reward, "steps": info["steps"]}


def main():
    parser = argparse.ArgumentParser(description="Build a unified results CSV: obstacle geometry + every model's outcome, per map")
    parser.add_argument("--maps-dir", type=Path, default=PROJECT_ROOT / "data" / "processed_full")
    parser.add_argument("--run", action="append", required=True, dest="run_specs",
                         help="LABEL=algo:model_path:deterministic|stochastic -- repeatable")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "results_dataset.csv")
    parser.add_argument("--seed", type=int, default=0,
                         help="Seed for stochastic-mode action sampling reproducibility")
    args = parser.parse_args()

    runs = [parse_run_spec(s) for s in args.run_specs]
    print(f"Runs to evaluate: {[r['label'] for r in runs]}")

    print(f"Loading maps from: {args.maps_dir}")
    maps = load_maps(args.maps_dir)
    print(f"Loaded {len(maps)} maps")

    print("Loading models...")
    loaded_models = {}
    for r in runs:
        key = (r["algo"], str(r["model_path"]))
        if key not in loaded_models:
            loaded_models[key] = ALGO_CLASSES[r["algo"]].load(str(r["model_path"]))
    print(f"Loaded {len(loaded_models)} unique model(s) for {len(runs)} run(s)")

    env = MarsTerrainEnv(grid_size=maps[0]["grid"].shape[0], max_steps=args.max_steps)

    rows = []
    for m in maps:
        cluster_stats = cluster_stats_for_grid(m["grid"])
        sizes = [s["size"] for s in cluster_stats] if cluster_stats else [0]

        row = {
            "map_id": m["name"],
            "max_cluster_size": max(sizes),
            "mean_cluster_size": float(np.mean(sizes)),
            "n_clusters": len(cluster_stats),
        }

        for r in runs:
            model = loaded_models[(r["algo"], str(r["model_path"]))]
            if not r["deterministic"]:
                np.random.seed(args.seed)  # best-effort reproducibility for stochastic sampling
            result = run_episode(model, env, m["grid"], m["start"], m["goal"], r["deterministic"], args.max_steps)
            row[f"{r['label']}_outcome"] = result["outcome"]
            row[f"{r['label']}_reward"] = result["reward"]
            row[f"{r['label']}_steps"] = result["steps"]

        rows.append(row)
        outcome_summary = "  ".join(f"{r['label']}={row[r['label'] + '_outcome']}" for r in runs)
        print(f"  {m['name']}: max_cluster={row['max_cluster_size']}  {outcome_summary}")

    df = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\nSaved dataset: {args.output}  ({len(df)} rows, {len(df.columns)} columns)")

    # Quick built-in sanity check: does max_cluster_size actually differ between
    # outcomes for each run? This is the core correlation the whole script exists to check.
    print("\nQuick check -- mean max_cluster_size by outcome, per run:")
    for r in runs:
        col = f"{r['label']}_outcome"
        print(f"\n{r['label']}:")
        simplified = df[col].apply(lambda x: "cycle" if str(x).startswith("cycle") else x)
        print(df.groupby(simplified)["max_cluster_size"].agg(["mean", "count"]).to_string())


if __name__ == "__main__":
    main()
