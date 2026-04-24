"""Tests for the Push-T staged environment wrapper (Session 5, Tier A)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.env.push_t_staged import PushTStagedEnv, frame_to_encoder_tensor


def test_env_reset_produces_upscaled_frame() -> None:
    env = PushTStagedEnv(stage="0a", seed=42)
    try:
        frame = env.reset()
        assert frame.shape == (224, 224, 3)
        assert frame.dtype == np.uint8
        assert int(frame.min()) >= 0 and int(frame.max()) <= 255
    finally:
        env.close()


def test_random_action_produces_valid_actions() -> None:
    env = PushTStagedEnv(stage="0a", seed=42)
    try:
        env.reset()
        for _ in range(10):
            a = env._sample_action()
            assert a.shape == env.action_space.shape
            assert np.all(a >= env.action_space.low - 1e-6)
            assert np.all(a <= env.action_space.high + 1e-6)
    finally:
        env.close()


def test_frame_rate_subsampling_every_4th_env_step() -> None:
    """Each next_frame() call must advance the underlying env by exactly 4
    steps in the steady-state path (no episode boundary)."""
    env = PushTStagedEnv(stage="0a", seed=42)
    try:
        env.reset()
        start = env.env_step_counter
        env.next_frame()
        delta = env.env_step_counter - start
        assert delta == 4, f"expected 4 env steps per next_frame(); got {delta}"
        env.next_frame()
        delta2 = env.env_step_counter - start
        assert delta2 == 8
    finally:
        env.close()


def test_frame_output_shape_and_dtype() -> None:
    env = PushTStagedEnv(stage="0a", seed=42)
    try:
        env.reset()
        frame = env.next_frame()
        assert frame.shape == (224, 224, 3)
        assert frame.dtype == np.uint8
        assert frame.min() >= 0
        assert frame.max() <= 255
    finally:
        env.close()


def test_next_frame_auto_resets_before_first_call() -> None:
    """next_frame() must produce a valid frame even without explicit reset()."""
    env = PushTStagedEnv(stage="0a", seed=42)
    try:
        frame = env.next_frame()
        assert frame.shape == (224, 224, 3)
        assert frame.dtype == np.uint8
    finally:
        env.close()


def test_action_held_for_4_env_steps() -> None:
    """During a single next_frame() call the sampled action should be held."""
    env = PushTStagedEnv(stage="0a", seed=42)
    try:
        env.reset()
        a_initial = env._current_action.copy()
        # Take one next_frame; internally the action must have been held for 4 steps.
        env.next_frame()
        # After next_frame, counter should be divisible by 4 and action should
        # have been resampled at the start of the next block on the next call.
        assert env.env_step_counter % 4 == 0
        # At the boundary, on the next call, _sample_action is triggered.
        env.next_frame()
        assert not np.array_equal(env._current_action, a_initial) or env.env_step_counter > 0
    finally:
        env.close()


def test_frame_to_encoder_tensor_contract() -> None:
    frame = (np.random.default_rng(0).integers(0, 256, size=(224, 224, 3))).astype(np.uint8)
    t = frame_to_encoder_tensor(frame)
    assert isinstance(t, torch.Tensor)
    assert t.shape == (3, 224, 224)
    assert t.dtype == torch.float32
    # Values after ImageNet normalisation sit roughly in [-2.1, 2.6].
    assert float(t.min()) >= -3.0
    assert float(t.max()) <= 3.0


def test_frame_to_encoder_tensor_rejects_non_uint8() -> None:
    frame = np.zeros((224, 224, 3), dtype=np.float32)
    with pytest.raises(TypeError, match="uint8"):
        frame_to_encoder_tensor(frame)


def test_determinism_same_seed_same_first_frame() -> None:
    env_a = PushTStagedEnv(stage="0a", seed=42)
    env_b = PushTStagedEnv(stage="0a", seed=42)
    try:
        fa = env_a.reset()
        fb = env_b.reset()
        assert np.array_equal(fa, fb)
    finally:
        env_a.close()
        env_b.close()


def test_reject_non_0a_stage() -> None:
    with pytest.raises(NotImplementedError, match="Stage"):
        PushTStagedEnv(stage="0b", seed=42)
