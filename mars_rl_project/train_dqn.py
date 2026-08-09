"""
train_dqn.py

Step 2 / Week 3 of the MarsPath timeline: train a DQN agent on synthetic
terrain using Stable-Baselines3.

Usage (from the MarsPath project root, venv activated):
    python -m mars_rl_project.train_dqn --difficulty easy --timesteps 500000

While training, open a second terminal (same folder, same venv) and run:
    tensorboard --logdir logs
Then open the printed http://localhost:6006 URL in a browser to watch
reward and success-rate curves update live.
"""

from __future__ import annotations
import argparse
from pathlib import Path

from stable_baselines3 import DQN
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
    """Factory so train/eval envs are built the same way but stay separate objects."""
    def _init():
        env = MarsTerrainEnv(difficulty=difficulty, max_steps=max_steps)
        return Monitor(env)  # Monitor records per-episode reward/length for TensorBoard
    return _init


def main():
    parser = argparse.ArgumentParser(description="Train a DQN agent on MarsTerrainEnv")
    parser.add_argument("--difficulty", default="easy", choices=["easy", "medium", "hard"],
                         help="Terrain difficulty to train on (default: easy)")
    parser.add_argument("--timesteps", type=int, default=500_000,
                         help="Total training timesteps (proposal Section 2.4 default: 500000)")
    parser.add_argument("--run-name", default=None,
                         help="Name for this run's logs/checkpoints (default: dqn_<difficulty>)")
    parser.add_argument("--eval-freq", type=int, default=10_000,
                         help="Timesteps between evaluation runs on held-out maps")
    parser.add_argument("--checkpoint-freq", type=int, default=25_000,
                         help="Timesteps between saved checkpoints")
    parser.add_argument("--dashboard", default="wandb", choices=["wandb", "tensorboard"],
                         help="Which dashboard to log to (default: wandb)")
    parser.add_argument("--wandb-project", default="marspath",
                         help="W&B project name (default: marspath)")
    args = parser.parse_args()

    use_wandb = args.dashboard == "wandb"
    if use_wandb and not WANDB_AVAILABLE:
        print("wandb not installed (pip install wandb) -- falling back to TensorBoard.")
        use_wandb = False

    run_name = args.run_name or f"dqn_{args.difficulty}"
    run_log_dir = LOG_DIR / run_name
    run_model_dir = MODEL_DIR / run_name
    run_log_dir.mkdir(parents=True, exist_ok=True)
    run_model_dir.mkdir(parents=True, exist_ok=True)

    wandb_run = None
    if use_wandb:
        config = {
            "difficulty": args.difficulty,
            "timesteps": args.timesteps,
            "algorithm": "DQN",
            "policy": "MultiInputPolicy",
        }
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=config,
            sync_tensorboard=True,   # SB3 always writes tensorboard logs too; wandb mirrors them
            monitor_gym=True,
            save_code=True,
        )

    train_env = make_env(args.difficulty)()
    eval_env = make_env(args.difficulty)()  # separate instance so eval doesn't disturb training state

    # MultiInputPolicy because our observation is a Dict (terrain_patch + direction + distance),
    # not a flat vector -- SB3 builds a small CNN branch for the image-like patch and an MLP
    # branch for the vector features, then concatenates them before the Q-network head.
    model = DQN(
        policy="MultiInputPolicy",
        env=train_env,
        learning_rate=5e-5,
        buffer_size=100_000,
        learning_starts=5_000,      # collect some random experience before training starts
        batch_size=64,
        gamma=0.99,                 # discount factor
        train_freq=4,
        target_update_interval=500,
        exploration_fraction=0.3,   # epsilon decays over first 30% of training
        exploration_final_eps=0.05,
        verbose=1,
        tensorboard_log=str(LOG_DIR),
        device="auto",              # uses CUDA automatically if available
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=str(run_model_dir),
        name_prefix="checkpoint",
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(run_model_dir / "best"),
        log_path=str(run_log_dir),
        eval_freq=args.eval_freq,
        n_eval_episodes=20,
        deterministic=True,         # no exploration noise during eval -- true performance measure
    )

    callbacks = [checkpoint_callback, eval_callback]
    if use_wandb:
        callbacks.append(WandbCallback(
            gradient_save_freq=0,   # set >0 to also log gradient histograms (slower, rarely needed)
            model_save_path=None,   # we already save via CheckpointCallback/EvalCallback above
            verbose=0,
        ))

    print(f"Starting training: difficulty={args.difficulty}, timesteps={args.timesteps}")
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
