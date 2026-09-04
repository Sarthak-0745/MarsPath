"""
train_ppo.py

Project 2a: train a PPO agent on MarsTerrainEnv WITH history-aware hard
action masking, for direct comparison against masked DQN (train_dqn.py)
and against MarsPath's unmasked PPO. PPO already supported native
masking in this codebase's dependency stack via sb3-contrib's
MaskablePPO, so this script needed less custom plumbing than
train_dqn.py -- but uses the SAME HistoryActionMaskWrapper (same
history window, same all-masked fallback, same diagnostics) so the
DQN-vs-PPO comparison reflects the algorithms, not different masking
implementations.

Differences from MarsPath's train_ppo.py:
  - The env is wrapped in HistoryActionMaskWrapper before Monitor.
    HistoryActionMaskWrapper already exposes `action_masks()` directly,
    which is exactly the interface MaskablePPO looks for -- no separate
    `ActionMasker` wrapper is needed on top.
  - `PPO` -> `MaskablePPO` (sb3_contrib).
  - `EvalCallback` -> `MaskableEvalCallback` (sb3_contrib), which is the
    library's own mask-aware eval callback -- unlike DQN, no custom
    callback needed to be written for this half of Project 2a.
  - New `--history-len` argument (default 15, matching train_dqn.py).

All other hyperparameters, n_envs=4 parallelism, and reward/eval setup
are identical to MarsPath's train_ppo.py so masking is the only
variable that changes.

Usage (from the project root, venv activated):
    python -m mars_rl_project.train_ppo --difficulty easy --timesteps 500000
"""

from __future__ import annotations
import argparse
from pathlib import Path

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from mars_rl_project.environments import MarsTerrainEnv
from mars_rl_project.masking.history_mask import HistoryActionMaskWrapper

try:
    import wandb
    from wandb.integration.sb3 import WandbCallback
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_ROOT / "logs"
MODEL_DIR = PROJECT_ROOT / "models"


def make_env(difficulty: str, max_steps: int, history_len: int):
    def _init():
        env = MarsTerrainEnv(difficulty=difficulty, max_steps=max_steps)
        env = HistoryActionMaskWrapper(env, history_len=history_len)
        return Monitor(env)
    return _init


def main():
    parser = argparse.ArgumentParser(description="Train a masked PPO agent on MarsTerrainEnv (Project 2a)")
    parser.add_argument("--difficulty", default="easy", choices=["easy", "medium", "hard"])
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--history-len", type=int, default=15,
                         help="Sliding window (in accepted moves) of recently-visited positions "
                              "that get hard-masked (default: 15, per Project 2a spec)")
    parser.add_argument("--run-name", default=None,
                         help="Name for this run's logs/checkpoints (default: ppo_masked_<difficulty>)")
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--checkpoint-freq", type=int, default=25_000)
    parser.add_argument("--dashboard", default="wandb", choices=["wandb", "tensorboard"])
    parser.add_argument("--wandb-project", default="marspath")
    args = parser.parse_args()

    use_wandb = args.dashboard == "wandb"
    if use_wandb and not WANDB_AVAILABLE:
        print("wandb not installed (pip install wandb) -- falling back to TensorBoard.")
        use_wandb = False

    run_name = args.run_name or f"ppo_masked_{args.difficulty}"
    run_log_dir = LOG_DIR / run_name
    run_model_dir = MODEL_DIR / run_name
    run_log_dir.mkdir(parents=True, exist_ok=True)
    run_model_dir.mkdir(parents=True, exist_ok=True)

    wandb_run = None
    if use_wandb:
        config = {
            "difficulty": args.difficulty,
            "timesteps": args.timesteps,
            "algorithm": "MaskablePPO",
            "policy": "MultiInputPolicy",
            "history_len": args.history_len,
        }
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=config,
            sync_tensorboard=True,
            monitor_gym=True,
            save_code=True,
        )

    n_envs = 4  # unchanged from MarsPath's train_ppo.py
    train_env = DummyVecEnv([make_env(args.difficulty, 200, args.history_len) for _ in range(n_envs)])
    eval_env = make_env(args.difficulty, 200, args.history_len)()

    model = MaskablePPO(
        policy="MultiInputPolicy",
        env=train_env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log=str(LOG_DIR),
        device="auto",
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(args.checkpoint_freq // n_envs, 1),
        save_path=str(run_model_dir),
        name_prefix="checkpoint",
    )
    eval_callback = MaskableEvalCallback(
        eval_env,
        best_model_save_path=str(run_model_dir / "best"),
        log_path=str(run_log_dir),
        eval_freq=max(args.eval_freq // n_envs, 1),
        n_eval_episodes=20,
        deterministic=True,
        use_masking=True,
    )

    callbacks = [checkpoint_callback, eval_callback]
    if use_wandb:
        callbacks.append(WandbCallback(gradient_save_freq=0, model_save_path=None, verbose=0))

    print(f"Starting masked PPO training: difficulty={args.difficulty}, timesteps={args.timesteps}, "
          f"n_envs={n_envs}, history_len={args.history_len}")
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

    # Diagnostic: all-masked fallback rate, aggregated across the n_envs
    # parallel training envs (see history_mask.py design assumption 3).
    per_env_masked = train_env.get_attr("all_masked_events")
    per_env_steps = train_env.get_attr("total_steps")
    total_masked, total_steps = sum(per_env_masked), sum(per_env_steps)
    rate = (total_masked / total_steps * 100) if total_steps else 0.0
    print(f"All-actions-masked fallback fired {total_masked}/{total_steps} steps "
          f"({rate:.3f}%) across all {n_envs} training envs.")

    if use_wandb:
        wandb_run.finish()


if __name__ == "__main__":
    main()
