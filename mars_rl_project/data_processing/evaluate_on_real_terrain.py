"""
evaluate_on_real_terrain.py

Step 6: evaluate an already-trained DQN/PPO model (from train_dqn.py /
train_ppo.py) on REAL Jezero Crater terrain maps (from
process_nasa_terrain.py), instead of synthetic terrain. This is the
"sim-to-real" test: does a policy that learned to navigate synthetic
obstacle fields actually transfer to genuine Mars topography?

Usage:
    python -m mars_rl_project.data_processing.evaluate_on_real_terrain \
        --model models/dqn_easy/final_model.zip \
        --maps-dir data/processed \
        --algo dqn
"""

from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import DQN, PPO

from mars_rl_project.environments import MarsTerrainEnv
from mars_rl_project.data_processing.render_failure_case import render

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

ALGO_CLASSES = {"dqn": DQN, "ppo": PPO}


def detect_cycle(path: list, max_period: int = 6, min_repeats: int = 3) -> int | None:
    """
    Checks whether the END of the path is a short repeating loop (as seen in
    jezero_map_03: a period-3 cycle the agent never escapes). Returns the
    cycle period (e.g. 3) if one is found ending at the final position,
    else None. This distinguishes "agent got stuck in a deterministic loop"
    from "agent is still genuinely searching/wandering" -- both currently
    show up as termination_reason='timeout', but they're different failure
    modes worth reporting separately.
    """
    n = len(path)
    for period in range(1, max_period + 1):
        needed = period * min_repeats
        if n < needed:
            continue
        tail = path[-needed:]
        block = tail[-period:]
        if all(tail[i] == block[i % period] for i in range(needed)):
            return period
    return None


def load_maps(maps_dir: Path) -> list[dict]:
    maps = []
    for path in sorted(maps_dir.glob("*.npz")):
        data = np.load(path)
        maps.append({
            "name": path.stem,
            "grid": data["grid"],
            "start": tuple(data["start"].tolist()),
            "goal": tuple(data["goal"].tolist()),
        })
    return maps


def evaluate_model_on_maps(model, maps: list[dict], max_steps: int = 200,
                            render_dir: Path | None = None, run_name: str = "model",
                            deterministic: bool = True) -> dict:
    env = MarsTerrainEnv(grid_size=maps[0]["grid"].shape[0], max_steps=max_steps)

    results = {"success": 0, "collision": 0, "timeout": 0}
    rewards, ep_lengths = [], []
    per_map_outcomes = []
    n_cycles = 0

    for m in maps:
        obs, info = env.reset(options={
            "fixed_grid": m["grid"], "fixed_start": m["start"], "fixed_goal": m["goal"],
        })
        path = [m["start"]]
        total_reward = 0.0
        terminated = truncated = False
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(int(action))
            path.append(info["position"])
            total_reward += reward

        outcome = info["termination_reason"] or "timeout"
        cycle_period = detect_cycle(path) if outcome == "timeout" else None
        if cycle_period is not None:
            n_cycles += 1
            outcome_label = f"timeout (cycle-{cycle_period})"
        else:
            outcome_label = outcome

        results[outcome] = results.get(outcome, 0) + 1
        rewards.append(total_reward)
        ep_lengths.append(info["steps"])
        per_map_outcomes.append((m["name"], outcome_label, total_reward, info["steps"]))

        if render_dir is not None:
            render_dir.mkdir(parents=True, exist_ok=True)
            out_path = render_dir / f"{m['name']}_{run_name}.png"
            title = f"{m['name']}  |  {run_name}  |  outcome: {outcome_label} ({info['steps']} steps)"
            render(m["grid"], m["start"], m["goal"], path, outcome, info["steps"], title, out_path)

    n = len(maps)
    summary = {
        "n_maps": n,
        "success_rate": results.get("success", 0) / n,
        "collision_rate": results.get("collision", 0) / n,
        "timeout_rate": results.get("timeout", 0) / n,
        "cycle_rate": n_cycles / n,
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_ep_length": float(np.mean(ep_lengths)),
        "per_map": per_map_outcomes,
    }
    return summary


def print_summary(label: str, summary: dict):
    print(f"\n{'='*60}")
    print(f"{label}")
    print(f"{'='*60}")
    print(f"Maps evaluated : {summary['n_maps']}")
    print(f"Success rate   : {summary['success_rate']*100:.1f}%")
    print(f"Collision rate : {summary['collision_rate']*100:.1f}%")
    print(f"Timeout rate   : {summary['timeout_rate']*100:.1f}%")
    print(f"  of which policy-cycle timeouts: {summary['cycle_rate']*100:.1f}% "
          f"(agent stuck in a short repeating loop, never a genuine collision/wander-forever)")
    print(f"Mean reward    : {summary['mean_reward']:.2f} (+/- {summary['std_reward']:.2f})")
    print(f"Mean ep length : {summary['mean_ep_length']:.1f}")
    print(f"\nPer-map results:")
    for name, outcome, reward, steps in summary["per_map"]:
        print(f"  {name:<20} {outcome:<10} reward={reward:>7.2f} steps={steps}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained model on real Jezero Crater terrain")
    parser.add_argument("--model", type=Path, required=True,
                         help="Path to a saved model .zip (from train_dqn.py or train_ppo.py)")
    parser.add_argument("--algo", choices=["dqn", "ppo"], required=True,
                         help="Which algorithm the model was trained with")
    parser.add_argument("--maps-dir", type=Path, default=PROJECT_ROOT / "data" / "processed",
                         help="Directory of .npz real-terrain maps from process_nasa_terrain.py")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--render-all", action="store_true",
                         help="Save a PNG (terrain + agent path + outcome) for EVERY map, not just "
                              "one -- same visual as render_failure_case.py, generated for the "
                              "whole evaluation set in one pass. Saved to outputs/eval_renders/<run_name>/")
    args = parser.parse_args()

    if not args.model.exists():
        raise FileNotFoundError(f"Model not found at {args.model}")
    if not args.maps_dir.exists() or not list(args.maps_dir.glob("*.npz")):
        raise FileNotFoundError(
            f"No .npz maps found in {args.maps_dir}. Run process_nasa_terrain.py first."
        )

    print(f"Loading {args.algo.upper()} model: {args.model}")
    model = ALGO_CLASSES[args.algo].load(str(args.model))

    print(f"Loading real-terrain maps from: {args.maps_dir}")
    maps = load_maps(args.maps_dir)
    print(f"Loaded {len(maps)} maps")

    run_name = args.model.parent.name
    render_dir = (PROJECT_ROOT / "outputs" / "eval_renders" / run_name) if args.render_all else None
    if render_dir:
        print(f"Rendering a PNG for every map to: {render_dir}")

    summary = evaluate_model_on_maps(model, maps, args.max_steps, render_dir=render_dir, run_name=run_name)
    print_summary(f"Real-terrain evaluation: {run_name}", summary)

    print(
        f"\nCompare these numbers against this model's synthetic eval/mean_reward "
        f"from its W&B run to see how well it transfers from synthetic training "
        f"terrain to genuine Mars topography."
    )


if __name__ == "__main__":
    main()
