"""
test_env.py

Week-2 milestone from the proposal timeline: "Custom Gymnasium environment
built and verified (check_env)".

Run with:
    python test_env.py
"""

from gymnasium.utils.env_checker import check_env
from mars_rl_project.environments import MarsTerrainEnv


def run_check_env():
    print("=" * 60)
    print("Running gymnasium.utils.env_checker.check_env ...")
    print("=" * 60)
    env = MarsTerrainEnv(difficulty="medium")
    check_env(env, skip_render_check=True)
    print("check_env passed with no errors.\n")


def run_random_rollout(episodes: int = 3, difficulty: str = "medium", seed: int = 0):
    print("=" * 60)
    print(f"Running {episodes} random-action rollout episode(s) on '{difficulty}' terrain")
    print("=" * 60)
    env = MarsTerrainEnv(difficulty=difficulty, max_steps=200, render_mode="ansi")

    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        total_reward = 0.0
        terminated = truncated = False
        while not (terminated or truncated):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

        print(
            f"Episode {ep}: steps={info['steps']:>3} "
            f"reward={total_reward:>7.2f} "
            f"outcome={info['termination_reason']}"
        )

    print()


def print_sample_map(difficulty: str = "medium", seed: int = 42):
    print("=" * 60)
    print(f"Sample rendered map ('{difficulty}' difficulty, seed={seed})")
    print("R = rover, G = goal, . = safe, , = rock, # = boulder, O = crater")
    print("=" * 60)
    env = MarsTerrainEnv(difficulty=difficulty, render_mode="ansi")
    env.reset(seed=seed)
    print(env.render())
    print()


def print_sample_observation():
    print("=" * 60)
    print("Sample observation shapes/dtypes")
    print("=" * 60)
    env = MarsTerrainEnv(difficulty="medium")
    obs, _ = env.reset(seed=0)
    for k, v in obs.items():
        print(f"  {k}: shape={v.shape} dtype={v.dtype} min={v.min():.3f} max={v.max():.3f}")
    print()


if __name__ == "__main__":
    run_check_env()
    print_sample_observation()
    run_random_rollout()
    print_sample_map()
    print("All smoke tests completed.")
