"""
shap_analysis.py

Step 7 (lightweight): SHAP-based feature attribution for the trained
DQN/PPO policies, used as SUPPORTING EVIDENCE for the mechanistic
account already established (Discussion): that cycling failures stem
from the reward's straight-line "potential field" character
overwhelming local terrain information under partial observability.

Rather than explaining all 172 raw input dimensions (13x13 terrain
patch pixels + 2 direction components + 1 distance scalar) -- which
is slow, and produces per-pixel attributions no more interpretable
than a single "the patch mattered this much" summary -- this script
groups the observation into three meaningful components and computes
exact Shapley values (via shap.KernelExplainer, which enumerates all
2^3 = 8 coalitions exactly for a 3-feature game, so this is exact, not
approximate) for each component's contribution to the model's score
for whichever action the policy actually chose at that state:

    1. terrain_patch  -- the local 13x13 view
    2. direction_to_goal -- the unit vector toward the goal
    3. distance_to_goal  -- normalized scalar distance

"Turning a component off" replaces it with its mean value across a
background sample of real-terrain states, matching the standard SHAP
definition of a missing feature's expected effect.

Usage (fast -- ~100-200 states, a couple minutes per model):
    python -m mars_rl_project.data_processing.shap_analysis \
        --model models/ppo_hard/final_model.zip --algo ppo \
        --maps-dir data/processed_full --n-states 150
"""

from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.utils import obs_as_tensor

from mars_rl_project.environments import MarsTerrainEnv
from mars_rl_project.data_processing.evaluate_on_real_terrain import load_maps, detect_cycle

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ALGO_CLASSES = {"dqn": DQN, "ppo": PPO}


def collect_states(model, algo: str, maps: list, max_steps: int = 200, deterministic: bool = True):
    """
    Runs the policy (deterministic) on every map, recording every visited
    state's observation, the action actually taken, and which outcome
    ('success' / 'collision' / 'cycle' / 'timeout') the episode ended in.
    """
    env = MarsTerrainEnv(grid_size=maps[0]["grid"].shape[0], max_steps=max_steps)
    records = []

    for m in maps:
        obs, info = env.reset(options={"fixed_grid": m["grid"], "fixed_start": m["start"], "fixed_goal": m["goal"]})
        path = [m["start"]]
        episode_obs_actions = []
        terminated = truncated = False
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=deterministic)
            episode_obs_actions.append((obs, int(action)))
            obs, reward, terminated, truncated, info = env.step(int(action))
            path.append(info["position"])

        outcome = info["termination_reason"] or "timeout"
        if outcome == "timeout" and detect_cycle(path) is not None:
            outcome = "cycle"

        for o, a in episode_obs_actions:
            records.append({"obs": o, "action": a, "outcome": outcome})

    return records


def build_score_fn(model, algo: str, device: str):
    """
    Returns score(obs_dict, action) -> scalar: DQN's Q-value for `action`,
    or PPO's policy probability for `action`. Both put the chosen action's
    score on a comparable "higher = more strongly preferred" scale.
    """
    def to_batch(obs_dict):
        return {k: obs_as_tensor(np.expand_dims(v, 0), device) for k, v in obs_dict.items()}

    if algo == "dqn":
        def score(obs_dict, action):
            with torch.no_grad():
                q_values = model.q_net(to_batch(obs_dict))
            return float(q_values[0, action].cpu().numpy())
    else:  # ppo
        def score(obs_dict, action):
            with torch.no_grad():
                dist = model.policy.get_distribution(to_batch(obs_dict))
                probs = dist.distribution.probs
            return float(probs[0, action].cpu().numpy())
    return score


def shapley_for_state(obs, action, score_fn, background_patch, background_dir, background_dist):
    """
    Exact Shapley values for the 3 grouped features (patch, direction,
    distance) via shap.KernelExplainer on a 3-binary-feature game. With
    only 3 features, KernelExplainer enumerates all 8 coalitions exactly.
    """
    def make_obs(patch_on, dir_on, dist_on):
        return {
            "terrain_patch": obs["terrain_patch"] if patch_on else background_patch,
            "direction_to_goal": obs["direction_to_goal"] if dir_on else background_dir,
            "distance_to_goal": obs["distance_to_goal"] if dist_on else background_dist,
        }

    def f(X):
        out = np.zeros(len(X))
        for i, row in enumerate(X):
            patch_on, dir_on, dist_on = row
            out[i] = score_fn(make_obs(patch_on, dir_on, dist_on), action)
        return out

    background = np.zeros((1, 3))
    explainer = shap.KernelExplainer(f, background)
    query = np.ones((1, 3))
    sv = explainer.shap_values(query, silent=True, nsamples="auto")
    return np.array(sv).flatten()  # [patch_contrib, direction_contrib, distance_contrib]


def main():
    parser = argparse.ArgumentParser(description="SHAP feature-group attribution for a trained policy")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--algo", choices=["dqn", "ppo"], required=True)
    parser.add_argument("--maps-dir", type=Path, default=PROJECT_ROOT / "data" / "processed_full")
    parser.add_argument("--n-states", type=int, default=150, help="Number of states to sample and explain")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument("--stochastic", action="store_true",
                        help="Use stochastic action selection when collecting states")
    args = parser.parse_args()

    print(f"Loading {args.algo.upper()} model: {args.model}")
    model = ALGO_CLASSES[args.algo].load(str(args.model))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading maps from: {args.maps_dir}")
    maps = load_maps(args.maps_dir)

    mode_str = "stochastic" if args.stochastic else "deterministic"
    print(f"Rolling out policy to collect states ({mode_str}, same procedure as main evaluation)...")
    records = collect_states(model, args.algo, maps, deterministic=not args.stochastic)
    print(f"Collected {len(records)} states across {len(maps)} maps")

    rng = np.random.default_rng(args.seed)
    sample_idx = rng.choice(len(records), size=min(args.n_states, len(records)), replace=False)
    sampled = [records[i] for i in sample_idx]
    print(f"Sampling {len(sampled)} states for SHAP analysis")

    # Background = mean patch / direction / distance across ALL collected states,
    # the standard SHAP convention for "this feature is absent" (marginalized
    # over the data distribution actually encountered).
    background_patch = np.mean([r["obs"]["terrain_patch"] for r in records], axis=0)
    background_dir = np.mean([r["obs"]["direction_to_goal"] for r in records], axis=0)
    background_dist = np.mean([r["obs"]["distance_to_goal"] for r in records], axis=0)

    score_fn = build_score_fn(model, args.algo, device)

    print("Computing exact Shapley values per state (patch / direction / distance)...")
    results = []
    for i, r in enumerate(sampled):
        sv = shapley_for_state(r["obs"], r["action"], score_fn, background_patch, background_dir, background_dist)
        results.append({"outcome": r["outcome"], "patch": sv[0], "direction": sv[1], "distance": sv[2]})
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(sampled)} states explained")

    # ---- Aggregate: mean |SHAP value| per feature group, split by outcome ----
    outcomes = sorted(set(r["outcome"] for r in results))
    groups = ["patch", "direction", "distance"]
    agg = {o: {g: [] for g in groups} for o in outcomes}
    for r in results:
        for g in groups:
            agg[r["outcome"]][g].append(abs(r[g]))

    run_name = args.model.parent.name
    print(f"\n{'='*60}\nMean |SHAP value| by feature group and outcome -- {run_name}\n{'='*60}")
    print(f"{'Outcome':<12}{'N':>5}{'Patch':>12}{'Direction':>12}{'Distance':>12}")
    for o in outcomes:
        n = len(agg[o]["patch"])
        vals = [np.mean(agg[o][g]) if agg[o][g] else 0.0 for g in groups]
        print(f"{o:<12}{n:>5}{vals[0]:>12.4f}{vals[1]:>12.4f}{vals[2]:>12.4f}")

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(outcomes))
    width = 0.25
    colors = {"patch": "#4a6fa5", "direction": "#e8935a", "distance": "#6fa54a"}
    for gi, g in enumerate(groups):
        means = [np.mean(agg[o][g]) if agg[o][g] else 0.0 for o in outcomes]
        ax.bar(x + (gi - 1) * width, means, width, label=g.replace("_", " ").title(), color=colors[g])
    ax.set_xticks(x)
    ax.set_xticklabels([o.title() for o in outcomes])
    ax.set_ylabel("Mean |Shapley value|")
    ax.set_title(f"Feature-Group Contribution to Action Choice \u2014 {run_name}")
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / f"shap_{run_name}.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"\nSaved plot: {out_path}")


if __name__ == "__main__":
    main()
