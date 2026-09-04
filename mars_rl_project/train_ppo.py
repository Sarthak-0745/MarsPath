"""
train_ppo.py

Step 4 of the MarsPath timeline: train a PPO agent on the same
MarsTerrainEnv, for direct comparison against DQN (Step 2/3).

PPO is on-policy (it learns only from data collected by its current
policy, then discards it) whereas DQN is off-policy (it learns from a
replay buffer of past experience, including old/stale data). This is
why the hyperparameters below look different from train_dqn.py even
though the environment, reward function, and evaluation setup are
identical -- keeping everything else matched is what makes the
DQN-vs-PPO comparison in your report fair.

Usage (from the MarsPath project root, venv activated):
    python -m mars_rl_project.train_ppo --difficulty easy --timesteps 500000
"""

from __future__ import annotations
import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

from mars_rl_project.environments import MarsTerrainEnv

try:
    import wandb
    from wandb.integration.sb3 import WandbCallback
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_ROOT / "logs"
MODEL_DIR = PROJECT_ROOT / "models"


def make_env(difficulty: str, max_steps: int = 200):
    def _init():
        env = MarsTerrainEnv(difficulty=difficulty, max_steps=max_steps)
        return Monitor(env)
    return _init


def main():
    parser = argparse.ArgumentParser(description="Train a PPO agent on MarsTerrainEnv")
    parser.add_argument("--difficulty", default="easy", choices=["easy", "medium", "hard"])
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--run-name", default=None,
                         help="Name for this run's logs/checkpoints (default: ppo_<difficulty>)")
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--checkpoint-freq", type=int, default=25_000)
    parser.add_argument("--dashboard", default="wandb", choices=["wandb", "tensorboard"])
    parser.add_argument("--wandb-project", default="marspath")
    args = parser.parse_args()

    use_wandb = args.dashboard == "wandb"
    if use_wandb and not WANDB_AVAILABLE:
        print("wandb not installed (pip install wandb) -- falling back to TensorBoard.")
        use_wandb = False

    run_name = args.run_name or f"ppo_{args.difficulty}"
    run_log_dir = LOG_DIR / run_name
    run_model_dir = MODEL_DIR / run_name
    run_log_dir.mkdir(parents=True, exist_ok=True)
    run_model_dir.mkdir(parents=True, exist_ok=True)

    wandb_run = None
    if use_wandb:
        config = {
            "difficulty": args.difficulty,
            "timesteps": args.timesteps,
            "algorithm": "PPO",
            "policy": "MultiInputPolicy",
        }
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=config,
            sync_tensorboard=True,
            monitor_gym=True,
            save_code=True,
        )

    # PPO needs multiple parallel environments to collect a batch of experience
    # before each update (unlike DQN, which updates from a replay buffer one
    # transition at a time). n_envs=1 still works -- SB3 just runs episodes
    # sequentially -- but 4 gives smoother, more stable gradient updates without
    # needing a beefy machine. Raise this if your CPU/GPU can handle more.
    from stable_baselines3.common.vec_env import DummyVecEnv
    n_envs = 4
    train_env = DummyVecEnv([make_env(args.difficulty) for _ in range(n_envs)])
    eval_env = make_env(args.difficulty)()  # single env is fine for eval

    model = PPO(
        policy="MultiInputPolicy",
        env=train_env,
        learning_rate=3e-4,     # PPO's default and a reasonable starting point (higher than DQN's,
                                 # since PPO's clipping mechanism guards against destabilizing updates
                                 # the way a lower learning rate protects DQN)
        n_steps=1024,            # steps collected per environment before each policy update
        batch_size=256,
        n_epochs=10,             # how many passes over each collected batch
        gamma=0.99,
        gae_lambda=0.95,         # bias/variance tradeoff for advantage estimation
        clip_range=0.2,          # PPO's signature trust-region-style clipping
        ent_coef=0.01,           # small entropy bonus to keep some exploration going
        verbose=1,
        tensorboard_log=str(LOG_DIR),
        device="auto",
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(args.checkpoint_freq // n_envs, 1),  # save_freq counts per-env steps under VecEnv
        save_path=str(run_model_dir),
        name_prefix="checkpoint",
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(run_model_dir / "best"),
        log_path=str(run_log_dir),
        eval_freq=max(args.eval_freq // n_envs, 1),
        n_eval_episodes=20,
        deterministic=True,
    )

    callbacks = [checkpoint_callback, eval_callback]
    if use_wandb:
        callbacks.append(WandbCallback(gradient_save_freq=0, model_save_path=None, verbose=0))

    print(f"Starting PPO training: difficulty={args.difficulty}, timesteps={args.timesteps}, n_envs={n_envs}")
    if use_wandb:
        print(f"W&B dashboard: {wandb_run.url}")
    else:
        print(f"TensorBoard logs: {LOG_DIR}  (run: python -m tensorboard --logdir logs)")
    print(f"Checkpoints/models: {run_model_dir}")

    model.learn(
        total_timesteps=args.timesteps,
        callback=callbacks,
        tb_log_name=run_name,
        progress_bar=True,
    )

    final_path = run_model_dir / "final_model"
    model.save(str(final_path))
    print(f"Training complete. Final model saved to {final_path}.zip")

    if use_wandb:
        wandb_run.finish()


if __name__ == "__main__":
    main()
