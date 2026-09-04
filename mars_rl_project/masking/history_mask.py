"""
history_mask.py

Project 2a: history-aware action masking.

MarsPath's SHAP analysis (see MarsPath_Final.docx) showed that DQN's
"cycling" failure mode -- the agent oscillating between a small set of
cells instead of making forward progress -- corresponds to high absolute
terrain-patch attribution, and that stochastic action selection (tried
in MarsPath) did not fix it. This module implements the direct fix:
HARD-mask any action that would move the rover back to a position it
has occupied in roughly the last `history_len` steps, so a cycle of
that length becomes structurally impossible rather than merely
discouraged.

This wrapper is algorithm-agnostic -- the same masking logic, same
history window, and same edge-case handling apply whether it sits
under MaskableDQN (masking/maskable_dqn.py) or sb3-contrib's
MaskablePPO, which is what makes the DQN-vs-PPO comparison in Project
2a a comparison of the algorithms and not of two different masking
implementations.

Design assumptions (flagged per your request):

1. HISTORY_LEN = 15 by default, per "last ~15 steps." This is a sliding
   window of ACCEPTED positions only -- a collision does not move the
   rover (see mars_terrain_env.py: "Rover stays in place on collision"),
   so it does not consume a slot in the window.

2. Only history-based masking is applied here. Off-grid / impassable
   destinations are NOT masked -- that stays exactly as in MarsPath (a
   -10 collision penalty via the environment's own reward), since this
   project's scope is specifically the cycling fix, not terrain
   avoidance. (In practice this also means an off-grid/obstacle action
   is never itself masked by history, because a position the rover has
   never legally occupied can never appear in the visited window.)

3. Edge case -- all 8 actions masked: this can only happen when the
   rover is in the interior of the grid and every one of its 8
   neighboring cells was visited within the last `history_len` steps
   (a tight, contorted path). Rather than dropping the mask entirely
   (which would defeat hard-masking on exactly the step where it
   matters most), the wrapper relaxes to allow ONLY the action leading
   to the LEAST-recently-visited neighbor -- i.e. the neighbor that
   fell out of the window soonest / was visited longest ago among the
   currently-tracked history. Each occurrence is counted in
   `self.all_masked_events` so you can report how often it fires as a
   diagnostic in the paper.

4. The mask is exposed two ways: `action_masks()` (the method name
   sb3-contrib's ActionMasker / MaskablePPO looks for) and via
   `info["action_mask"]` after every `reset()`/`step()`, so
   evaluate_on_real_terrain.py can pull it without depending on
   sb3-contrib.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import gymnasium as gym

# Mirrors mars_terrain_env.py's action table exactly -- duplicated here
# (rather than imported) so this wrapper has zero dependency on internal
# names of MarsTerrainEnv beyond the public attributes used below.
_ACTIONS = {
    0: (-1, 0),   # North
    1: (1, 0),    # South
    2: (0, 1),    # East
    3: (0, -1),   # West
    4: (-1, 1),   # Northeast
    5: (-1, -1),  # Northwest
    6: (1, 1),    # Southeast
    7: (1, -1),   # Southwest
}
N_ACTIONS = 8


class HistoryActionMaskWrapper(gym.Wrapper):
    """Hard-masks actions that would revisit a recently-occupied cell.

    Wrap directly around MarsTerrainEnv (before Monitor):
        env = HistoryActionMaskWrapper(MarsTerrainEnv(...), history_len=15)
    """

    def __init__(self, env: gym.Env, history_len: int = 15):
        super().__init__(env)
        self.history_len = history_len
        self._visited: deque = deque(maxlen=history_len)
        # Diagnostic counters, cumulative across all episodes this wrapper
        # instance sees -- report these in the paper's masking-behavior
        # section (e.g. "all-masked fallback fired on X% of steps").
        self.all_masked_events = 0
        self.total_steps = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._visited.clear()
        self._visited.append(tuple(int(v) for v in self.unwrapped._pos))
        info["action_mask"] = self._compute_mask()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._visited.append(tuple(int(v) for v in self.unwrapped._pos))
        self.total_steps += 1
        info["action_mask"] = self._compute_mask()
        return obs, reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        """Public hook used by sb3-contrib's ActionMasker/MaskablePPO and
        by MaskableDQN. Safe to call any time after reset()."""
        return self._compute_mask()

    def _compute_mask(self) -> np.ndarray:
        pos = self.unwrapped._pos
        visited_set = set(self._visited)
        mask = np.ones(N_ACTIONS, dtype=bool)
        for a, (dr, dc) in _ACTIONS.items():
            new_pos = (pos[0] + dr, pos[1] + dc)
            if new_pos in visited_set:
                mask[a] = False

        if not mask.any():
            # See design assumption 3 above.
            self.all_masked_events += 1
            visited_list = list(self._visited)
            recency = {
                a: visited_list.index((pos[0] + dr, pos[1] + dc))
                for a, (dr, dc) in _ACTIONS.items()
            }
            best_action = min(recency, key=recency.get)  # smallest index = oldest visit
            mask[:] = False
            mask[best_action] = True

        return mask
