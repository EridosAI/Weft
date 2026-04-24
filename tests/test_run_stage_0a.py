"""Tests for scripts/run_stage_0a.py — Stage 0a training driver.

These tests exercise the driver end-to-end via injected factories. Real
V-JEPA 2 loading and real gym-pusht rollouts are NOT exercised here — the
pre-flight smoke test covers the real-stack pipeline integration. These
tests focus on driver-specific logic: config loading, episode-boundary
metadata population, SIGTERM exit path, and the clean-exit artifact set.
"""

from __future__ import annotations

import importlib.util
import json
import signal
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pytest
import torch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Import the driver as a module. `scripts/` is not a package, so load by path.
_DRIVER_PATH = _ROOT / "scripts" / "run_stage_0a.py"
_spec = importlib.util.spec_from_file_location("run_stage_0a", _DRIVER_PATH)
assert _spec is not None and _spec.loader is not None
run_stage_0a = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_stage_0a)

from src.predictor.trajectory_predictor import TrajectoryPredictor  # noqa: E402


# ---- fakes ------------------------------------------------------------------


class _FakeActionSpace:
    def __init__(self) -> None:
        self.low = np.array([-1.0, -1.0], dtype=np.float32)
        self.high = np.array([1.0, 1.0], dtype=np.float32)
        self.dtype = np.float32


class _FakeInnerEnv:
    """Minimal gym-style env matching what `PushTStagedEnv._env` does internally.

    `terminate_every` — if set, returns `terminated=True` every N step() calls
    after a reset. `reset()` restarts the per-episode counter.
    """

    def __init__(self, terminate_every: Optional[int] = None) -> None:
        self.terminate_every = terminate_every
        self.step_counter = 0
        self.reset_counter = 0
        self.action_space = _FakeActionSpace()
        self.observation_space = None

    def reset(self, seed: Optional[int] = None) -> Any:
        self.step_counter = 0
        self.reset_counter += 1
        return np.zeros((96, 96, 3), dtype=np.uint8), {}

    def step(self, action: Any) -> Any:
        self.step_counter += 1
        terminated = (
            self.terminate_every is not None
            and self.step_counter > 0
            and self.step_counter % self.terminate_every == 0
        )
        return np.zeros((96, 96, 3), dtype=np.uint8), 0.0, terminated, False, {}

    def close(self) -> None:
        pass


class _FakeStagedEnv:
    """Duck-typed stand-in for `PushTStagedEnv` that mirrors the inner-reset
    auto-recovery pattern. `_ResetTracker` monkey-patches `_env.reset`; the
    fake faithfully reproduces the pattern so the tracker sees the same
    signal on both real and fake envs.
    """

    _ACTION_HOLD = 4

    def __init__(self, seed: int = 42, terminate_every: Optional[int] = None) -> None:
        self._env = _FakeInnerEnv(terminate_every=terminate_every)
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._env_step_counter = 0
        self._needs_reset = True
        self._current_action: Optional[np.ndarray] = None

    def reset(self) -> np.ndarray:
        self._env.reset(seed=self._seed)
        self._current_action = self._sample_action()
        self._env_step_counter = 0
        self._needs_reset = False
        return np.zeros((256, 256, 3), dtype=np.uint8)

    def next_frame(self) -> np.ndarray:
        if self._needs_reset:
            self.reset()
        for i in range(self._ACTION_HOLD):
            if self._env_step_counter % self._ACTION_HOLD == 0:
                self._current_action = self._sample_action()
            _obs, _r, terminated, truncated, _ = self._env.step(self._current_action)
            self._env_step_counter += 1
            if terminated or truncated:
                self._env.reset(seed=self._seed + self._env_step_counter)
                self._current_action = self._sample_action()
                remaining = self._ACTION_HOLD - (i + 1)
                for _ in range(remaining):
                    _o, _r2, t2, tr2, _i = self._env.step(self._current_action)
                    self._env_step_counter += 1
                    if t2 or tr2:
                        self._env.reset(seed=self._seed + self._env_step_counter)
                break
        return np.zeros((256, 256, 3), dtype=np.uint8)

    def close(self) -> None:
        self._env.close()

    def _sample_action(self) -> np.ndarray:
        return self._rng.uniform(low=-1.0, high=1.0, size=2).astype(np.float32)


class _FakeEncoder(torch.nn.Module):
    """Frozen stand-in for `FrozenVJepa2Encoder`. Returns random (B, D) embeddings."""

    def __init__(self, embed_dim: int = 1024, on_encode: Optional[Any] = None) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self._gen = torch.Generator()
        self._gen.manual_seed(0)
        self._on_encode = on_encode
        self.calls = 0

    @torch.no_grad()
    def encode_frame(self, tensor: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        if self._on_encode is not None:
            self._on_encode(self.calls)
        b = tensor.shape[0] if tensor.dim() == 4 else 1
        return torch.randn(b, self.embed_dim, generator=self._gen)


def _tiny_predictor_factory(cfg: Dict[str, Any]) -> TrajectoryPredictor:
    """Fast predictor for tests. Architecture mirrors tests/test_online_loop.py."""
    return TrajectoryPredictor(
        embed_dim=cfg["predictor"]["embed_dim"],
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
        mlp_dim=128,
        window_size=cfg["window_size"],
        dropout=0.0,
    )


def _tiny_predictor_param_count(cfg: Dict[str, Any]) -> int:
    p = _tiny_predictor_factory(cfg)
    return sum(t.numel() for t in p.parameters() if t.requires_grad)


def _write_test_config(tmp_path: Path, overrides: Optional[Dict[str, Any]] = None) -> Path:
    """Writes a stage_0a-shaped config into tmp_path, redirecting output dirs
    to tmp_path so tests do not write into the real results/ or checkpoints/.
    """
    cfg = {
        "stage": "0a",
        "seed": 42,
        "total_frames": 50000,
        "window_size": 16,
        "encoder": {
            "checkpoint": "facebook/vjepa2-vitl-fpc64-256",
            "embed_dim": 1024,
        },
        "predictor": {
            "embed_dim": 1024,
            "hidden_dim": 512,
            "num_layers": 4,
            "num_heads": 8,
            "mlp_dim": 2048,
            "dropout": 0.0,
        },
        "memory_bank": {"max_size": 200000, "rebuild_interval": 1000},
        "optimizer": {
            "lr": 0.0003,
            "weight_decay": 0.01,
            "betas": [0.9, 0.95],
            "warmup_steps": 5000,
        },
        "masking": {
            "initial_mask_count": 1,
            "mask_count_cap": 4,
            "plateau_window": 10000,
            "plateau_threshold": 0.05,
        },
        "logging": {"checkpoint_interval": 10000, "grad_log_interval": 100},
        "paths": {
            "log_dir": str(tmp_path / "logs"),
            "results_dir": str(tmp_path / "results"),
            "checkpoint_dir": str(tmp_path / "checkpoints"),
        },
    }
    if overrides:
        def _deep_update(d, u):
            for k, v in u.items():
                if isinstance(v, dict) and isinstance(d.get(k), dict):
                    _deep_update(d[k], v)
                else:
                    d[k] = v
        _deep_update(cfg, overrides)
    import yaml as _yaml
    p = tmp_path / "stage_0a_test.yaml"
    p.write_text(_yaml.safe_dump(cfg))
    return p


def _monkeypatch_paths_to_tmp(monkeypatch, tmp_path: Path) -> None:
    """Redirect the driver's `_ROOT`-relative writes to `tmp_path`.

    The driver computes `results_dir` and `checkpoint_dir` as
    `_ROOT / cfg['paths'][...]`. If the config's paths are already
    absolute (tmp_path), `_ROOT / <abs>` stays absolute in tmp_path.
    """
    # No-op: config uses absolute paths via _write_test_config. Kept for
    # readability at call sites.
    _ = monkeypatch
    _ = tmp_path


# ---- tests ------------------------------------------------------------------


def test_1_dry_run_end_to_end(tmp_path: Path) -> None:
    """Dry-run exits 0, writes launch_info.txt, no checkpoint, bank populated,
    predictor params changed from init."""
    run_stage_0a._reset_shutdown_state()

    cfg_path = _write_test_config(tmp_path)
    cfg = run_stage_0a._load_config(cfg_path)

    results_dir = Path(cfg["paths"]["results_dir"]) / "stage_0a"
    run_stage_0a._write_launch_info(
        results_dir, cfg_path, cfg_path.read_text(),
        pid=12345, device=torch.device("cpu"), dry_run=True,
    )

    # Capture predictor + memory bank references so we can inspect post-run.
    captured: Dict[str, Any] = {}

    def capture_predictor_factory(c: Dict[str, Any]) -> TrajectoryPredictor:
        pred = _tiny_predictor_factory(c)
        captured["predictor"] = pred
        captured["init_params"] = [t.detach().clone() for t in pred.parameters()]
        return pred

    original_build_bank = run_stage_0a._build_memory_bank

    def capture_bank(c: Dict[str, Any]):
        bank = original_build_bank(c)
        captured["bank"] = bank
        return bank

    run_stage_0a._build_memory_bank = capture_bank  # type: ignore[assignment]
    try:
        exit_code = run_stage_0a.run(
            cfg,
            device=torch.device("cpu"),
            dry_run=True,
            encoder_factory=lambda c, d: _FakeEncoder(c["encoder"]["embed_dim"]),
            env_factory=lambda c: _FakeStagedEnv(seed=int(c["seed"])),
            predictor_factory=capture_predictor_factory,
            expected_predictor_params=_tiny_predictor_param_count(cfg),
            tensorboard_enabled=False,
            stdout_stride=1,
        )
    finally:
        run_stage_0a._build_memory_bank = original_build_bank  # type: ignore[assignment]

    assert exit_code == 0
    assert (results_dir / "launch_info.txt").is_file()

    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])
    assert not ckpt_dir.exists() or not list(ckpt_dir.glob("*.pt")), (
        "dry-run must not write any .pt checkpoints"
    )
    assert not (results_dir / "training_complete.json").exists()
    assert not (results_dir / "FATAL_ERROR.md").exists()
    assert not list(results_dir.glob("checkpoint_*.json"))

    # Memory bank populated with W + _DRY_RUN_TRAIN_STEPS entries.
    bank = captured["bank"]
    expected = cfg["window_size"] + run_stage_0a._DRY_RUN_TRAIN_STEPS
    assert len(bank) == expected, f"expected {expected} bank entries, got {len(bank)}"

    # Predictor state changed from init (gradients applied).
    predictor = captured["predictor"]
    post_params = [t.detach().clone() for t in predictor.parameters()]
    diffs = [not torch.equal(a, b) for a, b in zip(captured["init_params"], post_params)]
    assert any(diffs), "predictor params unchanged — dry-run did not apply any gradient steps"


def test_2_episode_boundary_flag_recorded(tmp_path: Path) -> None:
    """Memory bank records `episode_boundary_flag` True on frames whose
    `next_frame()` call fired an `_env.reset`, False otherwise. Frame 0 is
    always True (the initial reset from `_needs_reset=True`)."""
    run_stage_0a._reset_shutdown_state()

    # terminate_every=15 env steps. ACTION_HOLD=4 means 4 env steps per frame.
    # First frame triggers initial reset → boundary True.
    # Then every ~4 frames one of the internal steps hits termination → one
    # more reset → boundary True for that frame.
    overrides = {"total_frames": 16 + 14}  # 30 frames total (16 warmup + 14)
    cfg_path = _write_test_config(tmp_path, overrides=overrides)
    cfg = run_stage_0a._load_config(cfg_path)

    # Patch the memory bank builder inside run() to capture the bank after
    # the loop exits. Instead of patching, we construct one externally and
    # rely on observation via monkeypatch of _build_memory_bank.
    import types
    captured: Dict[str, Any] = {}
    original_build = run_stage_0a._build_memory_bank

    def _capture(c):
        mb = original_build(c)
        captured["bank"] = mb
        return mb

    run_stage_0a._build_memory_bank = _capture  # type: ignore[assignment]
    try:
        exit_code = run_stage_0a.run(
            cfg,
            device=torch.device("cpu"),
            dry_run=True,  # use small frame count
            encoder_factory=lambda c, d: _FakeEncoder(c["encoder"]["embed_dim"]),
            env_factory=lambda c: _FakeStagedEnv(
                seed=int(c["seed"]), terminate_every=15
            ),
            predictor_factory=_tiny_predictor_factory,
            expected_predictor_params=_tiny_predictor_param_count(cfg),
            tensorboard_enabled=False,
            stdout_stride=10_000,  # suppress chatty stdout
        )
    finally:
        run_stage_0a._build_memory_bank = original_build  # type: ignore[assignment]

    assert exit_code == 0
    bank = captured["bank"]
    assert len(bank) >= 10, f"bank should have observed frames; got {len(bank)}"

    flags = [m.extra.get("episode_boundary_flag") for m in bank._metadata]
    # Frame 0 is always True.
    assert flags[0] is True, f"frame 0 must be episode boundary; got {flags[0]}"
    # Must be a mix of True and False across the run.
    assert any(f is True for f in flags[1:]), "no mid-run boundaries detected"
    assert any(f is False for f in flags), "no non-boundary frames detected"
    # Types must all be bool.
    assert all(isinstance(f, bool) for f in flags), f"non-bool flag observed: {flags}"


def test_3_sigterm_exit_path(tmp_path: Path) -> None:
    """Simulate SIGTERM mid-run: driver exits 0, writes sigterm-tagged
    checkpoint + `training_sigterm.json`, no `training_complete.json`.

    We inject the shutdown flag from inside the fake encoder on its N-th
    call — this fires after `stats is not None` has evaluated at least once
    (i.e., after W warmup frames), which is the code path to exercise.
    """
    run_stage_0a._reset_shutdown_state()

    cfg_path = _write_test_config(tmp_path)
    cfg = run_stage_0a._load_config(cfg_path)

    # After 20 encoder calls, set the shutdown flag. Call 17 is the first
    # post-warmup training step (W=16, so call 17 produces the first stats).
    trigger_call = 20

    def _trigger_shutdown(calls: int) -> None:
        if calls == trigger_call:
            run_stage_0a._SHUTDOWN_REQUESTED = True
            run_stage_0a._SHUTDOWN_SIGNAL = signal.SIGTERM

    exit_code = run_stage_0a.run(
        cfg,
        device=torch.device("cpu"),
        dry_run=False,  # so the driver writes final checkpoint + JSON
        encoder_factory=lambda c, d: _FakeEncoder(
            c["encoder"]["embed_dim"], on_encode=_trigger_shutdown
        ),
        env_factory=lambda c: _FakeStagedEnv(seed=int(c["seed"])),
        predictor_factory=_tiny_predictor_factory,
        expected_predictor_params=_tiny_predictor_param_count(cfg),
        tensorboard_enabled=False,
        stdout_stride=1,
    )

    assert exit_code == 0
    results_dir = Path(cfg["paths"]["results_dir"]) / "stage_0a"
    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])

    # sigterm-tagged JSON written; `training_complete.json` NOT written.
    sigterm_json = results_dir / "training_sigterm.json"
    assert sigterm_json.is_file(), "training_sigterm.json must be written on SIGTERM"
    assert not (results_dir / "training_complete.json").exists()

    snap = json.loads(sigterm_json.read_text())
    assert snap["exit_reason"] == "sigterm"
    assert snap["signal"] == int(signal.SIGTERM)

    # Final checkpoint with `_sigterm` tag.
    sigterm_ckpts = list(ckpt_dir.glob("stage_0a_step*_sigterm_*.pt"))
    assert len(sigterm_ckpts) == 1, f"expected 1 sigterm checkpoint, got {sigterm_ckpts}"

    # Reset flag before leaving the test so the rest of the suite is clean.
    run_stage_0a._reset_shutdown_state()


def test_4_config_integrity(tmp_path: Path) -> None:
    """`_load_config` parses the real `configs/stage_0a.yaml` successfully
    with all required fields and expected types."""
    real_cfg_path = _ROOT / "configs" / "stage_0a.yaml"
    cfg = run_stage_0a._load_config(real_cfg_path)

    assert cfg["stage"] == "0a"
    assert isinstance(cfg["seed"], int) and cfg["seed"] == 42
    assert cfg["total_frames"] == 50_000
    assert cfg["window_size"] == 16
    assert cfg["encoder"]["checkpoint"] == "facebook/vjepa2-vitl-fpc64-256"
    assert cfg["encoder"]["embed_dim"] == 1024
    assert cfg["predictor"]["hidden_dim"] == 512
    assert cfg["predictor"]["num_layers"] == 4
    assert cfg["predictor"]["num_heads"] == 8
    assert cfg["predictor"]["mlp_dim"] == 2048
    assert cfg["memory_bank"]["max_size"] == 200_000
    assert cfg["memory_bank"]["rebuild_interval"] == 1000
    assert cfg["optimizer"]["lr"] == 3e-4
    assert cfg["optimizer"]["warmup_steps"] == 5000
    assert cfg["masking"]["initial_mask_count"] == 1
    assert cfg["masking"]["mask_count_cap"] == 4
    assert cfg["logging"]["checkpoint_interval"] == 10_000

    # Missing-key rejection path.
    bad = tmp_path / "bad.yaml"
    bad.write_text("stage: 0a\nseed: 42\n")
    with pytest.raises(ValueError, match="missing required keys"):
        run_stage_0a._load_config(bad)

    # Wrong stage rejection.
    bad2 = tmp_path / "wrong_stage.yaml"
    import yaml as _yaml
    cfg_copy = dict(cfg)
    cfg_copy["stage"] = "0b"
    bad2.write_text(_yaml.safe_dump(cfg_copy))
    with pytest.raises(ValueError, match="config stage="):
        run_stage_0a._load_config(bad2)


def test_5_resume_cli_flag_rejected(tmp_path: Path) -> None:
    """`--resume` at the CLI is explicitly deferred; driver must refuse."""
    cfg_path = _write_test_config(tmp_path)
    fake_ckpt = tmp_path / "fake.pt"
    fake_ckpt.write_bytes(b"")
    rc = run_stage_0a.main([
        "--config", str(cfg_path),
        "--resume", str(fake_ckpt),
    ])
    assert rc == 2
