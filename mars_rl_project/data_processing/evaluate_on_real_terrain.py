"""
evaluate_on_real_terrain.py

Project 2a update: evaluate a trained model -- masked (MaskableDQN /
MaskablePPO from Project 2a) or unmasked (plain DQN / PPO from MarsPath)
-- on real Jezero Crater terrain maps. Same 50-map stratified real-terrain
set as MarsPath; no changes to map loading, cycle detection, or the
success/collision/timeout accounting, so Project 2a numbers are directly
comparable to MarsPath's baseline numbers in the paper.

New in this version:
    --masked            wrap the eval env in HistoryActionMaskWrapper and
                         drive the model with masked predict() calls
                         (required for MaskableDQN/MaskablePPO models;
                         must be OMITTED for MarsPath's original
                         unmasked DQN/PPO models, which don't expect a
                         wrapped observation/action-selection path)
    --history-len N      masking window, must match what the model was
                         trained with (default 15)

With --masked, cycle detection (detect_cycle) is still run on the
resulting path for reporting, but a true cycle should now be
structurally impossible for the masked model within any window <=
history_len -- if detect_cycle still fires, that's a real finding
worth flagging (e.g. a longer-period cycle than the mask window
prevents, or a cycle that predates the mask -- shouldn't happen since
the mask is active from step 1, but report exactly what detect_cycle's
period distribution looks like either way).

Usage:
    python -m mars_rl_project.data_processing.evaluate_on_real_terrain \
        --model models/dqn_masked_easy/final_model.zip \
        --maps-dir data/processed \
        --algo dqn --masked
"""

from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import DQN, PPO
from sb3_contrib import MaskablePPO

from mars_rl_project.environments import MarsTerrainEnv
from mars_rl_project.masking.history_mask import HistoryActionMaskWrapper
from mars_rl_project.masking.maskable_dqn import MaskableDQN
from mars_rl_project.masking.eval_utils import get_action_masks
from mars_rl_project.data_processing.render_failure_case import render

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Masked and unmasked model classes both keyed by --algo; --masked picks
# which row of this table is used to .load() the model.
ALGO_CLASSES = {
    "dqn": {"unmasked": DQN, "masked": MaskableDQN},
    "ppo": {"unmasked": PPO, "masked": MaskablePPO},
}


def detect_cycle(path: list, max_period: int = 6, min_repeats: int = 3) -> int | None:
    """Unchanged from MarsPath -- checks whether the END of the path is a
    short repeating loop. See MarsPath's original docstring for details."""
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
                            deterministic: bool = True, masked: bool = False,
                            history_len: int = 15) -> dict:
    base_env = MarsTerrainEnv(grid_size=maps[0]["grid"].shape[0], max_steps=max_steps)
    env = HistoryActionMaskWrapper(base_env, history_len=history_len) if masked else base_env

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
            if masked:
                action_masks = get_action_masks(env)
                action, _ = model.predict(obs, deterministic=deterministic, action_masks=action_masks)
            else:
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
    if masked:
        # Diagnostic from history_mask.py design assumption 3 -- how often
        # did the eval env get boxed in on all 8 sides by its own recent
        # history across this whole 50-map run?
        summary["all_masked_events"] = env.get_wrapper_attr("all_masked_events")
        summary["all_masked_steps_total"] = env.get_wrapper_attr("total_steps")
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
    if "all_masked_events" in summary:
        n = summary["all_masked_steps_total"]
        k = summary["all_masked_events"]
        rate = (k / n * 100) if n else 0.0
        print(f"All-masked fallback fired: {k}/{n} steps ({rate:.3f}%) across this eval run")
    print(f"\nPer-map results:")
    for name, outcome, reward, steps in summary["per_map"]:
        print(f"  {name:<20} {outcome:<10} reward={reward:>7.2f} steps={steps}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained model on real Jezero Crater terrain")
    parser.add_argument("--model", type=Path, required=True,
                         help="Path to a saved model .zip (from train_dqn.py or train_ppo.py)")
    parser.add_argument("--algo", choices=["dqn", "ppo"], required=True,
                         help="Which algorithm the model was trained with")
    parser.add_argument("--masked", action="store_true",
                         help="Set for Project 2a models (MaskableDQN/MaskablePPO). Omit for "
                              "MarsPath's original unmasked DQN/PPO models.")
    parser.add_argument("--history-len", type=int, default=15,
                         help="Masking window -- must match training (default: 15). Ignored "
                              "unless --masked is set.")
    parser.add_argument("--maps-dir", type=Path, default=PROJECT_ROOT / "data" / "processed",
                         help="Directory of .npz real-terrain maps from process_nasa_terrain.py")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--render-all", action="store_true",
                         help="Save a PNG for every map to outputs/eval_renders/<run_name>/")
    args = parser.parse_args()

    if not args.model.exists():
        raise FileNotFoundError(f"Model not found at {args.model}")
    if not args.maps_dir.exists() or not list(args.maps_dir.glob("*.npz")):
        raise FileNotFoundError(
            f"No .npz maps found in {args.maps_dir}. Run process_nasa_terrain.py first."
        )

    model_cls = ALGO_CLASSES[args.algo]["masked" if args.masked else "unmasked"]
    print(f"Loading {'masked ' if args.masked else ''}{args.algo.upper()} model: {args.model}")
    model = model_cls.load(str(args.model))

    print(f"Loading real-terrain maps from: {args.maps_dir}")
    maps = load_maps(args.maps_dir)
    print(f"Loaded {len(maps)} maps")

    run_name = args.model.parent.name
    render_dir = (PROJECT_ROOT / "outputs" / "eval_renders" / run_name) if args.render_all else None
    if render_dir:
        print(f"Rendering a PNG for every map to: {render_dir}")

    summary = evaluate_model_on_maps(
        model, maps, args.max_steps, render_dir=render_dir, run_name=run_name,
        masked=args.masked, history_len=args.history_len,
    )
    print_summary(f"Real-terrain evaluation: {run_name}", summary)

    print(
        f"\nCompare these numbers against MarsPath's baseline (unmasked) real-terrain "
        f"numbers for the same difficulty/algorithm to see whether history-aware masking "
        f"reduced the timeout/cycle rate."
    )


if __name__ == "__main__":
    main()
