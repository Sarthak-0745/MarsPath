"""
eval_utils.py

SB3's stock `EvalCallback` / `evaluate_policy` call `model.predict(obs,
deterministic=...)` with no `action_masks` argument, so it would silently
evaluate MaskableDQN/MaskablePPO UNMASKED during training's periodic
eval -- which would both misreport progress and pick the wrong
"best_model" checkpoint. sb3-contrib solves this for PPO with
MaskableEvalCallback; this module provides the equivalent for our
custom MaskableDQN so both algorithms get an apples-to-apples,
mask-aware periodic eval during training.

`evaluate_masked_policy` is also reused directly by
evaluate_on_real_terrain.py for the final real-terrain evaluation, so
the exact same "fetch mask -> predict(..., action_masks=) -> step" loop
is used everywhere in Project 2a, not reimplemented three times.
"""

from __future__ import annotations

import os
import numpy as np
from stable_baselines3.common.callbacks import EventCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv


def get_action_masks(env) -> np.ndarray:
    """Same contract as sb3_contrib.common.maskable.utils.get_action_masks:
    works for either a single Env or a VecEnv, as long as every (sub-)env
    exposes an `action_masks()` method (HistoryActionMaskWrapper does)."""
    if isinstance(env, VecEnv):
        return np.stack(env.env_method("action_masks"))
    # gymnasium's Wrapper no longer auto-delegates attribute access to the
    # wrapped env, so a plain `env.action_masks()` fails whenever `env` is
    # e.g. a Monitor sitting on top of HistoryActionMaskWrapper. Walk the
    # wrapper chain explicitly instead.
    return env.get_wrapper_attr("action_masks")()


def evaluate_masked_policy(model, env, n_eval_episodes: int = 20, deterministic: bool = True,
                            return_episode_rewards: bool = False):
    """Minimal masked analogue of SB3's evaluate_policy, scoped to what
    train_dqn.py / evaluate_on_real_terrain.py actually need: no support
    for VecNormalize syncing or is_success callbacks, since MarsTerrainEnv
    uses neither. `env` may be a single Monitor-wrapped env or a VecEnv."""
    if not isinstance(env, VecEnv):
        env = DummyVecEnv([lambda: env])
    assert env.num_envs == 1, "evaluate_masked_policy expects a single-env VecEnv (n_envs=1)"

    episode_rewards, episode_lengths = [], []
    for _ in range(n_eval_episodes):
        obs = env.reset()
        done = False
        ep_reward, ep_len = 0.0, 0
        while not done:
            action_masks = get_action_masks(env)
            action, _ = model.predict(obs, deterministic=deterministic, action_masks=action_masks)
            obs, reward, dones, infos = env.step(action)
            done = bool(dones[0])
            ep_reward += float(reward[0])
            ep_len += 1
        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_len)

    if return_episode_rewards:
        return episode_rewards, episode_lengths
    return float(np.mean(episode_rewards)), float(np.std(episode_rewards))


class MaskedDQNEvalCallback(EventCallback):
    """DQN-side equivalent of sb3_contrib's MaskableEvalCallback: periodic
    mask-aware evaluation during training, best-model checkpointing, and
    an evaluations.npz log in the same format SB3's own EvalCallback
    produces (so any existing plotting code from MarsPath still works).
    """

    def __init__(self, eval_env, n_eval_episodes: int = 20, eval_freq: int = 10_000,
                 log_path: str | None = None, best_model_save_path: str | None = None,
                 deterministic: bool = True, verbose: int = 1):
        super().__init__(verbose=verbose)
        self.eval_env = eval_env if isinstance(eval_env, VecEnv) else DummyVecEnv([lambda: eval_env])
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.deterministic = deterministic
        self.best_model_save_path = best_model_save_path
        self.log_path = os.path.join(log_path, "evaluations") if log_path is not None else None
        self.best_mean_reward = -np.inf
        self.evaluations_timesteps: list[int] = []
        self.evaluations_results: list[list[float]] = []
        self.evaluations_length: list[list[int]] = []

    def _init_callback(self) -> None:
        if self.best_model_save_path is not None:
            os.makedirs(self.best_model_save_path, exist_ok=True)
        if self.log_path is not None:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            episode_rewards, episode_lengths = evaluate_masked_policy(
                self.model, self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                deterministic=self.deterministic,
                return_episode_rewards=True,
            )
            mean_reward, std_reward = float(np.mean(episode_rewards)), float(np.std(episode_rewards))

            if self.log_path is not None:
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)
                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                )

            self.logger.record("eval/mean_reward", mean_reward)
            self.logger.record("eval/mean_ep_length", float(np.mean(episode_lengths)))
            if self.verbose >= 1:
                print(f"Eval at {self.num_timesteps} steps: mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")

            if mean_reward > self.best_mean_reward:
                if self.verbose >= 1:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                self.best_mean_reward = mean_reward

        return True
