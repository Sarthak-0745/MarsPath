"""
train_dqn.py

Project 2a: train a DQN agent on MarsTerrainEnv WITH history-aware hard
action masking, to test whether directly blocking the actions that cause
cycling fixes DQN's cycling failure mode (identified via SHAP in
MarsPath; stochastic action selection alone did not fix it there).

This is the masked counterpart of MarsPath's train_dqn.py. Differences
from that script:
  - The env is wrapped in HistoryActionMaskWrapper (masking/history_mask.py)
    before Monitor, so every step's `info` carries the current mask and
    `env.action_masks()` is available.
  - `DQN` -> `MaskableDQN` (masking/maskable_dqn.py), since SB3's DQN has
    no native masking support -- see that module's docstring for exactly
    what is and isn't masked (TL;DR: action *selection* is fully masked
    everywhere, including warm-up; the TD target is NOT mask-aware --
    flagged there as a scoping assumption).
  - `EvalCallback` -> `MaskedDQNEvalCallback` (masking/eval_utils.py),
    because SB3's stock EvalCallback calls `model.predict()` without an
    `action_masks` argument and would silently evaluate unmasked.
  - New `--history-len` argument (default 15, per "last ~15 steps").

Training from scratch, no dependency on MarsPath's saved models.

Usage (from the project root, venv activated):
    python -m mars_rl_project.train_dqn --difficulty easy --timesteps 500000
"""

from __future__ import annotations
import argparse
from pathlib import Path

from stable_baselines3.common.monitor import Monitor

from mars_rl_project.environments import MarsTerrainEnv
from mars_rl_project.masking.history_mask import HistoryActionMaskWrapper
from mars_rl_project.masking.maskable_dqn import MaskableDQN
from mars_rl_project.masking.eval_utils import MaskedDQNEvalCallback

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
    """Same factory pattern as MarsPath's train_dqn.py, but the env is now
    wrapped in HistoryActionMaskWrapper before Monitor -- Monitor still
    sits outermost so per-episode reward/length logging is unaffected."""
    def _init():
        env = MarsTerrainEnv(difficulty=difficulty, max_steps=max_steps)
        env = HistoryActionMaskWrapper(env, history_len=history_len)
        return Monitor(env)
    return _init


def main():
    parser = argparse.ArgumentParser(description="Train a masked DQN agent on MarsTerrainEnv (Project 2a)")
    parser.add_argument("--difficulty", default="easy", choices=["easy", "medium", "hard"],
                         help="Terrain difficulty to train on (default: easy)")
    parser.add_argument("--timesteps", type=int, default=500_000,
                         help="Total training timesteps (matches MarsPath default)")
    parser.add_argument("--history-len", type=int, default=15,
                         help="Sliding window (in accepted moves) of recently-visited positions "
                              "that get hard-masked (default: 15, per Project 2a spec)")
    parser.add_argument("--run-name", default=None,
                         help="Name for this run's logs/checkpoints (default: dqn_masked_<difficulty>)")
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

    run_name = args.run_name or f"dqn_masked_{args.difficulty}"
    run_log_dir = LOG_DIR / run_name
    run_model_dir = MODEL_DIR / run_name
    run_log_dir.mkdir(parents=True, exist_ok=True)
    run_model_dir.mkdir(parents=True, exist_ok=True)

    wandb_run = None
    if use_wandb:
        config = {
            "difficulty": args.difficulty,
            "timesteps": args.timesteps,
            "algorithm": "MaskableDQN",
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

    train_env = make_env(args.difficulty, max_steps=200, history_len=args.history_len)()
    eval_env = make_env(args.difficulty, max_steps=200, history_len=args.history_len)()

    # Same architecture/hyperparameters as MarsPath's DQN (5e-5 LR fix for
    # instability carries over unchanged) -- masking is the only variable
    # we want to change for a clean before/after comparison.
    model = MaskableDQN(
        policy="MultiInputPolicy",
        env=train_env,
        learning_rate=5e-5,
        buffer_size=100_000,
        learning_starts=5_000,
        batch_size=64,
        gamma=0.99,
        train_freq=4,
        target_update_interval=500,
        exploration_fraction=0.3,
        exploration_final_eps=0.05,
        verbose=1,
        tensorboard_log=str(LOG_DIR),
        device="auto",
    )

    from stable_baselines3.common.callbacks import CheckpointCallback
    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=str(run_model_dir),
        name_prefix="checkpoint",
    )
    eval_callback = MaskedDQNEvalCallback(
        eval_env,
        best_model_save_path=str(run_model_dir / "best"),
        log_path=str(run_log_dir),
        eval_freq=args.eval_freq,
        n_eval_episodes=20,
        deterministic=True,
    )

    callbacks = [checkpoint_callback, eval_callback]
    if use_wandb:
        callbacks.append(WandbCallback(gradient_save_freq=0, model_save_path=None, verbose=0))

    print(f"Starting masked DQN training: difficulty={args.difficulty}, timesteps={args.timesteps}, "
          f"history_len={args.history_len}")
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

    # Diagnostic: how often did the all-masked fallback (design assumption 3
    # in history_mask.py) fire during training? Worth reporting in the paper.
    all_masked = train_env.get_wrapper_attr("all_masked_events")
    total_steps = train_env.get_wrapper_attr("total_steps")
    rate = (all_masked / total_steps * 100) if total_steps else 0.0
    print(f"All-actions-masked fallback fired {all_masked}/{total_steps} steps "
          f"({rate:.3f}%) on the training env over its lifetime.")

    if use_wandb:
        wandb_run.finish()


if __name__ == "__main__":
    main()
