"""
evaluate_all_masked.py

Three-way real-terrain comparison across all 6 difficulty/algo configs:
  1. MarsPath baseline (unmasked DQN/PPO)      -- hardcoded from the paper
  2. Project 2a (history-only hard masking)     -- evaluated live
  3. Project 2b (terrain + history hard masking) -- evaluated live

Rows 2 and 3 are both run through evaluate_model_on_maps() directly
(no subprocess/stdout parsing), on the same loaded set of 50 maps, so
every number in the table comes from an identical evaluation procedure
except for which wrapper (if any) drove the model.

Run from the project root:
    python -m mars_rl_project.data_processing.evaluate_all_masked
"""

from __future__ import annotations
from pathlib import Path

import pandas as pd

from mars_rl_project.data_processing.evaluate_on_real_terrain import (
    ALGO_CLASSES, evaluate_model_on_maps, load_maps,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
MAPS_DIR = PROJECT_ROOT / "data" / "processed_full"
HISTORY_LEN = 15  # must match what all masked models were trained with

CONFIGS = [
    {"name": "DQN Easy", "algo": "dqn", "difficulty": "easy"},
    {"name": "DQN Medium", "algo": "dqn", "difficulty": "medium"},
    {"name": "DQN Hard", "algo": "dqn", "difficulty": "hard"},
    {"name": "PPO Easy", "algo": "ppo", "difficulty": "easy"},
    {"name": "PPO Medium", "algo": "ppo", "difficulty": "medium"},
    {"name": "PPO Hard", "algo": "ppo", "difficulty": "hard"},
]

# MarsPath baseline (unmasked), from the paper. NOTE: "cycle" is set equal
# to "timeout" for every row here -- double check against the paper's
# actual per-config cycle numbers before trusting "Cycle Reduction" too
# literally (memory notes stochastic PPO's cycle rate was 0.0% across the
# board, which wouldn't match timeout==cycle for the PPO rows below).
BASELINE = {
    "DQN Easy": {"success": 32.0, "collision": 59.2, "timeout": 8.8, "cycle": 8.8},
    "DQN Medium": {"success": 30.8, "collision": 36.8, "timeout": 32.4, "cycle": 32.4},
    "DQN Hard": {"success": 2.0, "collision": 3.6, "timeout": 94.4, "cycle": 94.4},
    "PPO Easy": {"success": 37.2, "collision": 25.2, "timeout": 37.6, "cycle": 37.6},
    "PPO Medium": {"success": 28.8, "collision": 14.8, "timeout": 56.4, "cycle": 56.4},
    "PPO Hard": {"success": 25.6, "collision": 6.4, "timeout": 68.0, "cycle": 68.0},
}


def model_path(algo: str, difficulty: str, terrain_mask: bool) -> Path:
    prefix = f"{algo}_maskedterrain" if terrain_mask else f"{algo}_masked"
    return MODEL_DIR / f"{prefix}_{difficulty}" / "final_model.zip"


def evaluate_one(algo: str, difficulty: str, terrain_mask: bool, maps: list[dict]) -> dict | None:
    """Load one masked model (Project 2a or 2b, per terrain_mask) and
    evaluate it on the already-loaded maps. Returns None if the model
    file doesn't exist (so callers can skip it gracefully) -- otherwise
    the summary dict with rates converted to percent (0-100)."""
    path = model_path(algo, difficulty, terrain_mask)
    if not path.exists():
        return None

    model = ALGO_CLASSES[algo]["masked"].load(str(path))
    summary = evaluate_model_on_maps(
        model, maps, masked=True, history_len=HISTORY_LEN, terrain_mask=terrain_mask,
    )
    return {
        "success": summary["success_rate"] * 100,
        "collision": summary["collision_rate"] * 100,
        "timeout": summary["timeout_rate"] * 100,
        "cycle": summary["cycle_rate"] * 100,
        "reward": summary["mean_reward"],
        "ep_length": summary["mean_ep_length"],
        "all_masked_events": summary.get("all_masked_events", 0),
        "all_masked_steps_total": summary.get("all_masked_steps_total", 0),
        "deadlock_events": summary.get("deadlock_events", 0),
    }


def print_metrics_line(label: str, m: dict):
    print(f"  [{label}] success={m['success']:.1f}%  collision={m['collision']:.1f}%  "
          f"timeout={m['timeout']:.1f}%  cycle={m['cycle']:.1f}%")
    if m["all_masked_steps_total"]:
        rate = m["all_masked_events"] / m["all_masked_steps_total"] * 100
        print(f"    all-masked fallback: {m['all_masked_events']}/{m['all_masked_steps_total']} steps ({rate:.3f}%)")
        if m["deadlock_events"]:
            drate = m["deadlock_events"] / m["all_masked_steps_total"] * 100
            print(f"    true terrain deadlocks: {m['deadlock_events']}/{m['all_masked_steps_total']} steps ({drate:.3f}%)")


def main():
    if not MAPS_DIR.exists() or not list(MAPS_DIR.glob("*.npz")):
        raise FileNotFoundError(f"No .npz maps found in {MAPS_DIR}. Run process_nasa_terrain.py first.")

    print(f"Loading real-terrain maps from: {MAPS_DIR}")
    maps = load_maps(MAPS_DIR)
    print(f"Loaded {len(maps)} maps\n")

    print("=" * 90)
    print("Evaluating Project 2a (history-only) and Project 2b (terrain+history) on Real Terrain")
    print("=" * 90)

    rows_2a, rows_2b = {}, {}
    for cfg in CONFIGS:
        name, algo, difficulty = cfg["name"], cfg["algo"], cfg["difficulty"]
        print(f"\n{name}")

        m2a = evaluate_one(algo, difficulty, terrain_mask=False, maps=maps)
        if m2a is None:
            print(f"  [2a] WARNING: model not found at {model_path(algo, difficulty, False)} -- skipping")
        else:
            print_metrics_line("2a", m2a)
            rows_2a[name] = m2a

        m2b = evaluate_one(algo, difficulty, terrain_mask=True, maps=maps)
        if m2b is None:
            print(f"  [2b] WARNING: model not found at {model_path(algo, difficulty, True)} -- skipping")
        else:
            print_metrics_line("2b", m2b)
            rows_2b[name] = m2b

    if not rows_2a and not rows_2b:
        print("\nNo models found -- nothing to compare. Check MODEL_DIR/MAPS_DIR paths.")
        return

    baseline_df = pd.DataFrame(BASELINE).T
    df_2a = pd.DataFrame(rows_2a).T if rows_2a else pd.DataFrame()
    df_2b = pd.DataFrame(rows_2b).T if rows_2b else pd.DataFrame()

    # Union of whatever configs we actually have results for, in CONFIGS order.
    all_names = [c["name"] for c in CONFIGS if c["name"] in rows_2a or c["name"] in rows_2b]

    comparison = pd.DataFrame(index=all_names)
    for metric, label in [("success", "Success"), ("collision", "Collision"),
                           ("timeout", "Timeout"), ("cycle", "Cycle")]:
        comparison[f"Baseline {label}"] = baseline_df.reindex(all_names)[metric]
        if not df_2a.empty:
            comparison[f"2a {label}"] = df_2a.reindex(all_names)[metric]
        if not df_2b.empty:
            comparison[f"2b {label}"] = df_2b.reindex(all_names)[metric]

    if not df_2a.empty:
        comparison["2a Reward"] = df_2a.reindex(all_names)["reward"]
    if not df_2b.empty:
        comparison["2b Reward"] = df_2b.reindex(all_names)["reward"]

    print("\n" + "=" * 90)
    print("COMPARISON TABLE: Baseline vs Project 2a vs Project 2b")
    print("=" * 90)
    print(comparison.round(1).to_string())

    output_path = PROJECT_ROOT / "results" / "masking_comparison.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_path)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
