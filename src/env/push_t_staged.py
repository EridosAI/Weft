"""Staged Push-T environment wrapper (Tier A, Session 5).

Stage 0a is a single default gym-pusht configuration with no visual variation.
Later stages (0b / 0c) will add multi-config visual variation and smooth
morph transitions respectively; this wrapper is structured so those stages
slot in via the `stage` argument without changing the encoder/trainer glue.

Frame protocol:
  - Underlying gym-pusht env renders at 96×96 RGB uint8.
  - Random actions are sampled uniformly and held for 4 env steps; the
    saved frame is the one at the end of the 4-step hold.
  - Effective experience frame rate: 10 Hz (every 4th env step).
  - Output frames are upscaled to 224×224 RGB uint8 for V-JEPA 2 input.
"""

from __future__ import annotations

from typing import Tuple

import gymnasium as gym
import gym_pusht  # noqa: F401 — registers gym_pusht/PushT-v0
import numpy as np


_UNDERLYING_RENDER_SIZE = 96
_ENCODER_FRAME_SIZE = 224
_ACTION_HOLD_ENV_STEPS = 4


class PushTStagedEnv:
    """Random-policy Push-T rollouts with 10 Hz effective frame extraction.

    Usage:
        env = PushTStagedEnv(stage="0a", seed=42)
        env.reset()
        for _ in range(1000):
            frame = env.next_frame()   # (224, 224, 3) uint8 in [0, 255]

    Design notes:
      - `reset()` is called by `next_frame()` implicitly if the env has not
        been reset or if an episode has terminated. Callers do not need to
        manage episodes themselves for Stage 0a rollouts.
      - Action policy for Stage 0a: sample from `env.action_space` every
        `_ACTION_HOLD_ENV_STEPS` env steps, hold constant between.
    """

    def __init__(self, stage: str = "0a", seed: int = 42) -> None:
        if stage != "0a":
            raise NotImplementedError(
                f"Stage {stage} is not yet wired up; Tier A Session 5 implements stage 0a only."
            )
        self.stage = stage
        self._seed = int(seed)
        self._env = gym.make("gym_pusht/PushT-v0", obs_type="pixels")
        self._rng = np.random.default_rng(seed)
        self._current_action: np.ndarray | None = None
        self._env_step_counter: int = 0
        self._needs_reset: bool = True

    # ---- public API ---------------------------------------------------------

    def reset(self) -> np.ndarray:
        obs, _info = self._env.reset(seed=self._seed)
        self._current_action = self._sample_action()
        self._env_step_counter = 0
        self._needs_reset = False
        return self._upscale_frame(obs)

    def next_frame(self) -> np.ndarray:
        """Advance `_ACTION_HOLD_ENV_STEPS` underlying steps and return the
        upscaled frame after the hold. Re-samples the action at step 0 of
        the hold; carries it through the remaining 3 steps."""
        if self._needs_reset:
            self.reset()

        obs = None
        for i in range(_ACTION_HOLD_ENV_STEPS):
            # Resample action at the start of every 4-step block.
            if self._env_step_counter % _ACTION_HOLD_ENV_STEPS == 0:
                self._current_action = self._sample_action()
            obs, _r, terminated, truncated, _info = self._env.step(self._current_action)
            self._env_step_counter += 1
            if terminated or truncated:
                # Auto-reset so `next_frame()` keeps producing frames across
                # episode boundaries; encoder/memory-bank do not care.
                obs, _info = self._env.reset(seed=self._seed + self._env_step_counter)
                self._current_action = self._sample_action()
                # Finish the block with fresh state so total env-step count is
                # consistent with the 4-step cadence.
                remaining = _ACTION_HOLD_ENV_STEPS - (i + 1)
                for _ in range(remaining):
                    obs, _r, terminated, truncated, _info = self._env.step(self._current_action)
                    self._env_step_counter += 1
                    if terminated or truncated:
                        obs, _info = self._env.reset(seed=self._seed + self._env_step_counter)
                break
        assert obs is not None
        return self._upscale_frame(obs)

    def close(self) -> None:
        self._env.close()

    @property
    def action_space(self) -> gym.spaces.Space:
        return self._env.action_space

    @property
    def observation_space(self) -> gym.spaces.Space:
        return self._env.observation_space

    @property
    def env_step_counter(self) -> int:
        return self._env_step_counter

    # ---- helpers ------------------------------------------------------------

    def _sample_action(self) -> np.ndarray:
        low = self._env.action_space.low
        high = self._env.action_space.high
        return self._rng.uniform(low=low, high=high).astype(self._env.action_space.dtype)

    def _upscale_frame(self, frame: np.ndarray) -> np.ndarray:
        if frame.ndim != 3 or frame.shape != (
            _UNDERLYING_RENDER_SIZE,
            _UNDERLYING_RENDER_SIZE,
            3,
        ):
            raise ValueError(
                f"unexpected underlying frame shape {frame.shape}; expected "
                f"({_UNDERLYING_RENDER_SIZE}, {_UNDERLYING_RENDER_SIZE}, 3)"
            )
        return _nearest_upscale(frame, _ENCODER_FRAME_SIZE)


def _nearest_upscale(frame: np.ndarray, target: int) -> np.ndarray:
    """Nearest-neighbour upscale a (H, W, 3) uint8 frame to (target, target, 3).

    Chosen over bilinear upsampling to preserve the sharp edges characteristic
    of Push-T's block rendering — we do not want the upscale step to
    artificially blur the T-block or the background/goal rectangles.
    """
    h, w, _ = frame.shape
    ys = (np.arange(target) * h // target).astype(np.int64)
    xs = (np.arange(target) * w // target).astype(np.int64)
    return np.ascontiguousarray(frame[ys[:, None], xs[None, :], :])


def frame_to_encoder_tensor(frame: np.ndarray) -> "torch.Tensor":
    """Convert a `(224, 224, 3)` uint8 frame to an ImageNet-normalised
    `(3, 224, 224)` float32 tensor on CPU. Separate helper so tests can
    exercise the env wrapper without a torch import at module level."""
    import torch

    if frame.dtype != np.uint8:
        raise TypeError(f"expected uint8 frame; got {frame.dtype}")
    arr = frame.astype(np.float32) / 255.0
    arr = (arr - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
        [0.229, 0.224, 0.225], dtype=np.float32
    )
    tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    return tensor


__all__ = ["PushTStagedEnv", "frame_to_encoder_tensor"]
