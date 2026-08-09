"""
run_multiseed.py

Multi-seed runner for MarsPath — re-runs DQN and PPO training across
N seeds for each difficulty tier, evaluates each trained model on the
real-terrain map set, then aggregates results as mean ± std.

Usage (from the MarsPath project root, venv activated):

    # Minimal run: 3 seeds, all 6 model configs, real-terrain eval
    python -m mars_rl_project.run_multiseed --seeds 3 --maps-dir data/processed

    # Fewer configs while debugging:
    python -m mars_rl_project.run_multiseed --seeds 3 --maps-dir data/processed \
        --algos dqn --difficulties easy medium

    # Skip re-training if you already have saved models:
    python -m mars_rl_project.run_multiseed --seeds 3 --maps-dir data/processed \
        --eval-only

Results are written to:
    outputs/multiseed_results.json   <- full per-seed data
    outputs/multiseed_summary.csv    <- mean ± std table (paste into paper)
"""

from __future__ import annotations
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from mars_rl_project.environments import MarsTerrainEnv
from mars_rl_project.data_processing.evaluate_on_real_terrain import (
    load_maps,
    evaluate_model_on_maps,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

ALGO_CLASSES = {"dqn": DQN, "ppo": PPO}

# Hyperparameters matching your original paper exactly
DQN_KWARGS = dict(
    policy="MultiInputPolicy",
    learning_rate=5e-5,
    buffer_size=100_000,
    learning_starts=5_000,
    batch_size=64,
    gamma=0.99,
    train_freq=4,
    target_update_interval=500,
    exploration_fraction=0.3,
    exploration_final_eps=0.05,
    verbose=0,
    device="auto",
)

PPO_KWARGS = dict(
    policy="MultiInputPolicy",
    learning_rate=3e-4,
    n_steps=1024,
    batch_size=256,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    verbose=0,
    device="auto",
)

PPO_N_ENVS = 4  # matches original train_ppo.py


def make_env(difficulty: str):
    def _init():
        env = MarsTerrainEnv(difficulty=difficulty, max_steps=200)
        return Monitor(env)
    return _init


def train_one(algo: str, difficulty: str, seed: int, timesteps: int) -> Path:
    """Train a single model, save it, return the saved .zip path."""
    run_name = f"{algo}_{difficulty}_seed{seed}"
    save_dir = MODEL_DIR / run_name
    save_dir.mkdir(parents=True, exist_ok=True)
    final_path = save_dir / "final_model.zip"

    if final_path.exists():
        print(f"  [skip] {run_name} already trained — {final_path}")
        return final_path

    print(f"  Training {run_name} for {timesteps:,} steps …")
    t0 = time.time()

    if algo == "dqn":
        train_env = make_env(difficulty)()
        eval_env = make_env(difficulty)()
        model = DQN(env=train_env, seed=seed, **DQN_KWARGS)
    else:
        train_env = DummyVecEnv([make_env(difficulty) for _ in range(PPO_N_ENVS)])
        eval_env = make_env(difficulty)()
        model = PPO(env=train_env, seed=seed, **PPO_KWARGS)

    eval_cb = EvalCallback(
        eval_env,
        eval_freq=max(10_000 // (PPO_N_ENVS if algo == "ppo" else 1), 1),
        n_eval_episodes=20,
        deterministic=True,
        verbose=0,
    )

    model.learn(total_timesteps=timesteps, callback=eval_cb, progress_bar=True)
    model.save(str(save_dir / "final_model"))

    elapsed = time.time() - t0
    print(f"  Done in {elapsed/60:.1f} min → {final_path}")
    return final_path


def eval_one(algo: str, model_path: Path, maps: list[dict], deterministic: bool = True) -> dict:
    """Load a saved model and evaluate on real-terrain maps."""
    model = ALGO_CLASSES[algo].load(str(model_path))
    return evaluate_model_on_maps(model, maps, max_steps=200, deterministic=deterministic)


def aggregate(records: list[dict]) -> dict:
    """Compute mean ± std across seeds for scalar metrics."""
    keys = ["success_rate", "collision_rate", "timeout_rate", "cycle_rate", "mean_reward"]
    out = {}
    for k in keys:
        vals = [r[k] for r in records]
        out[f"{k}_mean"] = float(np.mean(vals))
        out[f"{k}_std"] = float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3, help="Number of random seeds (default 3)")
    parser.add_argument("--algos", nargs="+", default=["dqn", "ppo"], choices=["dqn", "ppo"])
    parser.add_argument("--difficulties", nargs="+", default=["easy", "medium", "hard"],
                        choices=["easy", "medium", "hard"])
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--maps-dir", type=Path, default=PROJECT_ROOT / "data" / "processed")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training, only run evaluation on already-saved models")
    parser.add_argument("--stochastic", action="store_true",
                        help="Use stochastic action selection during evaluation (default: deterministic)")
    args = parser.parse_args()
    deterministic = not args.stochastic

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    seeds = list(range(args.seeds))

    # ── Load real-terrain maps once ────────────────────────────────────
    if not args.maps_dir.exists() or not list(args.maps_dir.glob("*.npz")):
        raise FileNotFoundError(
            f"No .npz maps in {args.maps_dir}. Run process_nasa_terrain.py first."
        )
    maps = load_maps(args.maps_dir)
    print(f"Loaded {len(maps)} real-terrain maps from {args.maps_dir}\n")

    # ── Main loop ──────────────────────────────────────────────────────
    all_results = {}   # { "dqn_easy": [seed0_summary, seed1_summary, ...], ... }

    for algo in args.algos:
        for diff in args.difficulties:
            config_key = f"{algo}_{diff}"
            print(f"\n{'='*60}")
            print(f"Config: {config_key}  ({args.seeds} seeds)")
            print(f"{'='*60}")

            seed_results = []
            for seed in seeds:
                # Train
                if not args.eval_only:
                    model_path = train_one(algo, diff, seed, args.timesteps)
                else:
                    model_path = MODEL_DIR / f"{algo}_{diff}_seed{seed}" / "final_model.zip"
                    if not model_path.exists():
                        print(f"  [WARN] {model_path} not found — skipping seed {seed}")
                        continue

                # Evaluate
                mode_str = "stochastic" if args.stochastic else "deterministic"
                print(f"  Evaluating seed {seed} on {len(maps)} real-terrain maps [{mode_str}] …")
                summary = eval_one(algo, model_path, maps, deterministic=deterministic)
                summary["seed"] = seed
                seed_results.append(summary)
                print(
                    f"    success={summary['success_rate']*100:.1f}%  "
                    f"cycle={summary['cycle_rate']*100:.1f}%  "
                    f"reward={summary['mean_reward']:.2f}"
                )

            all_results[config_key] = seed_results

    # ── Save full results ──────────────────────────────────────────────
    mode_tag = "stochastic" if args.stochastic else "deterministic"
    results_path = OUTPUT_DIR / f"multiseed_results_{mode_tag}.json"
    # per_map lists aren't JSON-safe in a useful way; drop them for the aggregate file
    serialisable = {
        k: [{kk: vv for kk, vv in r.items() if kk != "per_map"} for r in v]
        for k, v in all_results.items()
    }
    with open(results_path, "w") as f:
        json.dump(serialisable, f, indent=2)
    print(f"\nFull results saved → {results_path}")

    # ── Build summary table (mean ± std) ──────────────────────────────
    summary_path = OUTPUT_DIR / f"multiseed_summary_{mode_tag}.csv"
    rows = []
    header = [
        "model",
        "success_mean", "success_std",
        "collision_mean", "collision_std",
        "timeout_mean", "timeout_std",
        "cycle_mean", "cycle_std",
        "reward_mean", "reward_std",
        "n_seeds",
    ]

    print(f"\n{'='*70}")
    print(f"{'Model':<20} {'Success':>12}  {'Cycle':>12}  {'Reward':>14}")
    print(f"{'='*70}")

    for config_key, seed_results in all_results.items():
        if not seed_results:
            continue
        agg = aggregate(seed_results)
        rows.append({
            "model": config_key,
            "success_mean": f"{agg['success_rate_mean']*100:.1f}",
            "success_std":  f"{agg['success_rate_std']*100:.1f}",
            "collision_mean": f"{agg['collision_rate_mean']*100:.1f}",
            "collision_std":  f"{agg['collision_rate_std']*100:.1f}",
            "timeout_mean":  f"{agg['timeout_rate_mean']*100:.1f}",
            "timeout_std":   f"{agg['timeout_rate_std']*100:.1f}",
            "cycle_mean":    f"{agg['cycle_rate_mean']*100:.1f}",
            "cycle_std":     f"{agg['cycle_rate_std']*100:.1f}",
            "reward_mean":   f"{agg['mean_reward_mean']:.2f}",
            "reward_std":    f"{agg['mean_reward_std']:.2f}",
            "n_seeds":       len(seed_results),
        })
        print(
            f"{config_key:<20} "
            f"{agg['success_rate_mean']*100:>5.1f}±{agg['success_rate_std']*100:<5.1f}  "
            f"{agg['cycle_rate_mean']*100:>5.1f}±{agg['cycle_rate_std']*100:<5.1f}  "
            f"{agg['mean_reward_mean']:>7.2f}±{agg['mean_reward_std']:<5.2f}"
        )

    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSummary table (mean ± std) saved → {summary_path}")
    print("Done. Copy multiseed_summary.csv values into your results tables.")


if __name__ == "__main__":
    main()
