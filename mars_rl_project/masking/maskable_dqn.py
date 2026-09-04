"""
maskable_dqn.py

Stable-Baselines3 has no native action-masking support for DQN (only
sb3-contrib's MaskablePPO exists, and that machinery is on-policy-only).
This module adds it, scoped as narrowly as possible so it stays
correct against SB3 2.9's actual internals rather than a plausible-looking
reimplementation.

What IS masked
---------------
- Action *selection* at every decision point during training rollouts,
  including the random warm-up phase before `learning_starts` (both
  overridden below) -- so masked-invalid transitions never even enter
  the replay buffer.
- Action selection at inference time (`model.predict(..., action_masks=)`),
  matching sb3-contrib's MaskablePPO API for consistency, so
  evaluate_on_real_terrain.py can call both algorithms the same way.

What is NOT masked (flagged assumption)
-----------------------------------------
- The Bellman target computed in `DQN.train()` (untouched -- we do not
  override it). Doing this "properly" would mean the target's
  next-state max-Q should exclude whatever actions would have been
  masked out AT that next state -- but that mask is a function of the
  rover's position-history deque at that point in time, which is not
  recoverable from a `(obs, action, reward, next_obs, done)` tuple
  pulled from the replay buffer later. Making that correct would mean
  storing the mask itself in the replay buffer (a custom ReplayBuffer),
  which is a bigger change than "mask decisions." The scoped-down
  version implemented here -- mask what the agent is allowed to DO,
  leave the TD-target bootstrap as vanilla DQN -- still fully prevents
  cycling in practice (masking always applies at actual decision time)
  and is a common simplification for masked Q-learning. Flagging this
  explicitly in case you want a target-consistent version as a
  follow-up (would need a MaskedReplayBuffer storing the mask alongside
  each transition).
"""

from __future__ import annotations

import numpy as np
import torch as th
from stable_baselines3 import DQN


class MaskableDQN(DQN):
    """DQN with hard action masking at decision time.

    The training env (self.env, a VecEnv after SB3's internal wrapping)
    must expose an `action_masks()` method on every sub-env -- i.e. it
    should be built from HistoryActionMaskWrapper-wrapped envs. Masks
    are fetched via `env_method("action_masks")`, which gymnasium's
    Wrapper.__getattr__ delegates down through Monitor/etc. to the
    HistoryActionMaskWrapper instance.
    """

    def predict(
        self,
        observation,
        state=None,
        episode_start=None,
        deterministic: bool = False,
        action_masks: np.ndarray | None = None,
    ):
        self.policy.set_training_mode(False)
        obs_tensor, vectorized_env = self.policy.obs_to_tensor(observation)

        with th.no_grad():
            q_values = self.q_net(obs_tensor)  # shape (batch, n_actions)
        q_values_np = q_values.cpu().numpy()

        if action_masks is None:
            # No mask supplied -> behave like vanilla DQN (fully unmasked).
            # Used only as a safety fallback; every call site in
            # train_dqn.py / evaluate_on_real_terrain.py passes real masks.
            masks = np.ones_like(q_values_np, dtype=bool)
        else:
            masks = np.atleast_2d(np.asarray(action_masks, dtype=bool))

        masked_q = np.where(masks, q_values_np, -np.inf)
        greedy_actions = masked_q.argmax(axis=1)

        if not deterministic and np.random.rand() < self.exploration_rate:
            # Epsilon-greedy exploration, but sampled only from valid actions
            # -- this is the discrete analogue of DQN.predict()'s own
            # epsilon-greedy branch, just constrained to the mask.
            action = np.array(
                [np.random.choice(np.flatnonzero(masks[i])) for i in range(masks.shape[0])]
            )
        else:
            action = greedy_actions

        if not vectorized_env:
            action = action[0]
        return action, state

    def _sample_action(self, learning_starts, action_noise=None, n_envs: int = 1):
        # Mirrors OffPolicyAlgorithm._sample_action, simplified for a
        # Discrete action space (no Box rescaling needed), but with masks
        # respected in BOTH the random warm-up branch and the policy
        # branch -- the warm-up branch is the one vanilla SB3 bypasses
        # `self.predict()` for, so it has to be handled separately here.
        action_masks = np.array(self.env.env_method("action_masks"))  # (n_envs, n_actions)

        if self.num_timesteps < learning_starts and not (self.use_sde and self.use_sde_at_warmup):
            unscaled_action = np.array(
                [np.random.choice(np.flatnonzero(action_masks[i])) for i in range(n_envs)]
            )
        else:
            assert self._last_obs is not None, "self._last_obs was not set"
            unscaled_action, _ = self.predict(
                self._last_obs, deterministic=False, action_masks=action_masks
            )

        # Discrete case: no rescaling, buffer_action == action (see
        # OffPolicyAlgorithm._sample_action's own `else` branch).
        action = unscaled_action
        buffer_action = action
        return action, buffer_action
