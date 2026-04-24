"""Stage 0a training driver — 50,000 frames, single Push-T config.

Consumes `configs/stage_0a.yaml` and composes the existing modules
(encoder + env + memory bank + predictor + online trainer) into a single
online loop. Writes per-checkpoint JSON snapshots + predictor `.pt`
checkpoints tagged with the git commit hash, a `launch_info.txt`
artifact, and either `training_complete.json` (clean completion) or
`training_sigterm.json` (clean shutdown on SIGTERM/SIGINT).

Design decisions (see §1.1/§1.2 of the driver-implementation instruction):

1. Episode-boundary flagging. `PushTStagedEnv.next_frame` handles
   `env.reset()` internally and does not surface a signal to the caller.
   Modifying `src/env/push_t_staged.py` is out of scope for this driver.
   The driver installs a narrow, driver-local monkey-patch on the
   underlying gym env's `reset` method via `_ResetTracker`, removes it
   on shutdown, and records `episode_boundary_flag` in
   `FrameMetadata.extra`. This preserves the `src/` module as the single
   source of truth for env behaviour while still producing the metadata
   signal required for post-hoc analysis at reset frames.

2. SIGTERM/SIGINT deferral. Signal handler sets a module-level flag
   only; the main loop finishes the current training step and then
   writes a final checkpoint tagged `_sigterm` before exiting 0. This
   avoids corrupting mid-step state under WSL shutdown / power events /
   manual Ctrl-C.

Resume-from-checkpoint is NOT supported. `OnlineTrainer.save_checkpoint`
persists predictor/optimizer/scheduler/counters but not env RNG, the
trainer's mask sampler RNG, or the ring-buffer contents, so bit-exact
continuation is not possible with the current checkpoint format.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, Optional

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch  # noqa: E402
import yaml  # noqa: E402

from src.encoders.frozen_vjepa2 import FrozenVJepa2Encoder  # noqa: E402
from src.env.push_t_staged import PushTStagedEnv, frame_to_encoder_tensor  # noqa: E402
from src.memory.memory_bank import FrameMetadata, MemoryBank  # noqa: E402
from src.predictor.trajectory_predictor import TrajectoryPredictor  # noqa: E402
from src.training.online_loop import OnlineTrainer, TrainingConfig  # noqa: E402


_STAGE = "0a"
_DEFAULT_CONFIG = _ROOT / "configs" / "stage_0a.yaml"
# Post-LN-removal predictor trainable param count (HANDOFF Session 3 + pre-Stage-0a corrections).
_EXPECTED_TRAINABLE_PREDICTOR_PARAMS = 13_668_864
_DRY_RUN_TRAIN_STEPS = 10

_REQUIRED_CFG_KEYS = (
    "stage", "seed", "total_frames", "window_size",
    "encoder", "predictor", "memory_bank",
    "optimizer", "masking", "logging", "paths",
)


# ---- signal handling ---------------------------------------------------------

_SHUTDOWN_REQUESTED = False
_SHUTDOWN_SIGNAL: Optional[int] = None


def _signal_handler(signum: int, _frame: Any) -> None:
    global _SHUTDOWN_REQUESTED, _SHUTDOWN_SIGNAL
    _SHUTDOWN_REQUESTED = True
    _SHUTDOWN_SIGNAL = signum


def _install_signal_handlers() -> None:
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)


def _reset_shutdown_state() -> None:
    """Test hook: clear the module-level shutdown flags between test cases."""
    global _SHUTDOWN_REQUESTED, _SHUTDOWN_SIGNAL
    _SHUTDOWN_REQUESTED = False
    _SHUTDOWN_SIGNAL = None


# ---- driver-local episode-boundary detector ---------------------------------


class _ResetTracker:
    """Detects when `env._env.reset` fires during `env.next_frame()`.

    `PushTStagedEnv.next_frame` auto-resets on `terminated or truncated`
    without surfacing a signal. Rather than modify the env wrapper (out of
    scope for this driver), this class monkey-patches the underlying gym
    env's `reset` method to set a one-shot flag. The driver reads the flag
    after each `next_frame()` call to populate `episode_boundary_flag` in
    memory-bank metadata, then clears it.
    """

    def __init__(self, env: PushTStagedEnv) -> None:
        inner = env._env
        self._env = env
        self._original_reset = inner.reset
        self._flag = False

        def _patched_reset(*args: Any, **kwargs: Any) -> Any:
            self._flag = True
            return self._original_reset(*args, **kwargs)

        inner.reset = _patched_reset  # type: ignore[method-assign]
        self._installed = True

    def check_and_clear(self) -> bool:
        out = self._flag
        self._flag = False
        return out

    def close(self) -> None:
        if self._installed:
            self._env._env.reset = self._original_reset  # type: ignore[method-assign]
            self._installed = False


# ---- config loading ---------------------------------------------------------


def _load_config(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"config not found: {path}")
    cfg = yaml.safe_load(path.read_text())
    if not isinstance(cfg, dict):
        raise ValueError(f"config root must be a mapping; got {type(cfg).__name__}")
    missing = [k for k in _REQUIRED_CFG_KEYS if k not in cfg]
    if missing:
        raise ValueError(f"config missing required keys: {missing}")
    if cfg["stage"] != _STAGE:
        raise ValueError(f"config stage={cfg['stage']!r}; expected {_STAGE!r}")
    return cfg


def _config_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


# ---- component builders -----------------------------------------------------


def _build_encoder_default(cfg: Dict[str, Any], device: torch.device) -> torch.nn.Module:
    return FrozenVJepa2Encoder(
        checkpoint=cfg["encoder"]["checkpoint"], device=device
    ).eval()


def _build_env_default(cfg: Dict[str, Any]) -> PushTStagedEnv:
    return PushTStagedEnv(stage="0a", seed=int(cfg["seed"]))


def _build_predictor(cfg: Dict[str, Any]) -> TrajectoryPredictor:
    p = cfg["predictor"]
    return TrajectoryPredictor(
        embed_dim=p["embed_dim"],
        hidden_dim=p["hidden_dim"],
        num_layers=p["num_layers"],
        num_heads=p["num_heads"],
        mlp_dim=p["mlp_dim"],
        window_size=cfg["window_size"],
        dropout=p["dropout"],
    )


def _build_memory_bank(cfg: Dict[str, Any]) -> MemoryBank:
    return MemoryBank(
        embed_dim=cfg["encoder"]["embed_dim"],
        max_size=cfg["memory_bank"]["max_size"],
        rebuild_interval=cfg["memory_bank"]["rebuild_interval"],
    )


def _build_trainer(
    cfg: Dict[str, Any],
    predictor: torch.nn.Module,
    device: torch.device,
    tensorboard_enabled: bool = True,
) -> OnlineTrainer:
    p = cfg["predictor"]
    t_cfg = TrainingConfig(
        stage=_STAGE,
        window_size=cfg["window_size"],
        embed_dim=p["embed_dim"],
        lr=cfg["optimizer"]["lr"],
        weight_decay=cfg["optimizer"]["weight_decay"],
        betas=tuple(cfg["optimizer"]["betas"]),  # type: ignore[arg-type]
        warmup_steps=cfg["optimizer"]["warmup_steps"],
        initial_mask_count=cfg["masking"]["initial_mask_count"],
        mask_count_cap=cfg["masking"]["mask_count_cap"],
        plateau_window=cfg["masking"]["plateau_window"],
        plateau_threshold=cfg["masking"]["plateau_threshold"],
        checkpoint_interval=cfg["logging"]["checkpoint_interval"],
        grad_log_interval=cfg["logging"]["grad_log_interval"],
        log_dir=_ROOT / cfg["paths"]["log_dir"],
        results_dir=_ROOT / cfg["paths"]["results_dir"],
        seed=int(cfg["seed"]),
    )
    return OnlineTrainer(predictor, t_cfg, device=device, tensorboard_enabled=tensorboard_enabled)


# ---- invariants -------------------------------------------------------------


def _assert_pre_loop_invariants(
    encoder: torch.nn.Module,
    predictor: torch.nn.Module,
    memory_bank: MemoryBank,
    trainer: OnlineTrainer,
    expected_predictor_params: int,
) -> None:
    enc_trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    if enc_trainable != 0:
        raise RuntimeError(f"encoder must be frozen; has {enc_trainable} trainable params")
    pred_trainable = sum(p.numel() for p in predictor.parameters() if p.requires_grad)
    if pred_trainable != expected_predictor_params:
        raise RuntimeError(
            f"predictor trainable params = {pred_trainable}; expected {expected_predictor_params}"
        )
    if len(memory_bank) != 0:
        raise RuntimeError(f"memory bank must start empty; has {len(memory_bank)} entries")
    if len(trainer.ring_buffer) != 0:
        raise RuntimeError(
            f"trainer ring buffer must start empty; has {len(trainer.ring_buffer)} entries"
        )


# ---- launch info ------------------------------------------------------------


def _git_commit(short: bool = False) -> str:
    try:
        cmd = ["git", "-C", str(_ROOT), "rev-parse"]
        if short:
            cmd.append("--short")
        cmd.append("HEAD")
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def _write_launch_info(
    results_dir: Path,
    config_path: Path,
    config_text: str,
    pid: int,
    device: torch.device,
    dry_run: bool,
) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    lines = [
        f"Launch timestamp (UTC): {ts}",
        f"PID: {pid}",
        f"Device: {device}",
        f"Dry run: {dry_run}",
        f"Git commit: {_git_commit()}",
        f"Config path: {config_path}",
        f"Config sha256[:12]: {_config_hash(config_text)}",
    ]
    (results_dir / "launch_info.txt").write_text("\n".join(lines) + "\n")


def _append_progress(results_dir: Path, line: str) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    with (results_dir / "progress_log.txt").open("a") as f:
        f.write(line + "\n")


# ---- NaN/Inf guard ----------------------------------------------------------


def _is_finite(x: float) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


# ---- main run ---------------------------------------------------------------


def run(
    cfg: Dict[str, Any],
    device: torch.device,
    dry_run: bool = False,
    encoder_factory: Optional[Callable[[Dict[str, Any], torch.device], torch.nn.Module]] = None,
    env_factory: Optional[Callable[[Dict[str, Any]], Any]] = None,
    predictor_factory: Optional[Callable[[Dict[str, Any]], torch.nn.Module]] = None,
    expected_predictor_params: int = _EXPECTED_TRAINABLE_PREDICTOR_PARAMS,
    tensorboard_enabled: bool = True,
    stdout_stride: int = 1000,
) -> int:
    """Execute the Stage 0a loop end-to-end. Returns process-level exit code.

    Factories are exposed for tests so the encoder/env/predictor can be
    substituted without loading V-JEPA 2 or gym-pusht. Production callers
    (`main()`) leave them as None.
    """
    _install_signal_handlers()

    results_dir = _ROOT / cfg["paths"]["results_dir"] / f"stage_{_STAGE}"
    checkpoint_dir = _ROOT / cfg["paths"]["checkpoint_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    encoder = (encoder_factory or _build_encoder_default)(cfg, device)
    env = (env_factory or _build_env_default)(cfg)
    predictor = (predictor_factory or _build_predictor)(cfg).to(device)
    memory_bank = _build_memory_bank(cfg)
    trainer = _build_trainer(cfg, predictor, device, tensorboard_enabled=tensorboard_enabled)

    _assert_pre_loop_invariants(
        encoder, predictor, memory_bank, trainer, expected_predictor_params
    )

    window_size = int(cfg["window_size"])
    checkpoint_interval = int(cfg["logging"]["checkpoint_interval"])
    commit_short = _git_commit(short=True)

    if dry_run:
        total_frames = window_size + _DRY_RUN_TRAIN_STEPS
    else:
        total_frames = int(cfg["total_frames"])

    print(
        f"[stage_{_STAGE}] device={device} total_frames={total_frames} "
        f"window_size={window_size} checkpoint_interval={checkpoint_interval} "
        f"dry_run={dry_run} commit={commit_short}",
        flush=True,
    )
    print(
        f"[stage_{_STAGE}] predictor trainable params: "
        f"{sum(p.numel() for p in predictor.parameters() if p.requires_grad)}",
        flush=True,
    )

    # Deterministic mask-sampling / init where possible.
    torch.manual_seed(int(cfg["seed"]))

    reset_tracker = _ResetTracker(env)
    t0 = time.time()
    fatal_exc: Optional[BaseException] = None
    exit_reason = "complete"
    last_stats: Optional[Dict[str, Any]] = None

    try:
        for frame_idx in range(total_frames):
            frame = env.next_frame()
            boundary = reset_tracker.check_and_clear()

            tensor = frame_to_encoder_tensor(frame).to(device)
            emb = encoder.encode_frame(tensor).squeeze(0)
            if emb.dim() != 1 or emb.shape[0] != cfg["encoder"]["embed_dim"]:
                raise RuntimeError(
                    f"encoder returned unexpected shape {tuple(emb.shape)}; "
                    f"expected (1, {cfg['encoder']['embed_dim']})"
                )

            memory_bank.append(
                emb.detach(),
                FrameMetadata(
                    frame_idx=frame_idx,
                    stage=_STAGE,
                    config="default",
                    extra={"episode_boundary_flag": bool(boundary)},
                ),
            )

            stats = trainer.observe_frame(emb.detach())

            if stats is not None:
                last_stats = stats
                for key in ("loss_next", "loss_masked", "grad_norm"):
                    if not _is_finite(stats[key]):
                        raise RuntimeError(
                            f"non-finite {key}={stats[key]!r} at train step {stats['step']} "
                            f"(frame {frame_idx})"
                        )

                step = int(stats["step"])
                if stdout_stride > 0 and step % stdout_stride == 0:
                    elapsed = time.time() - t0
                    print(
                        f"[stage_{_STAGE}] step={step} frame={frame_idx + 1}/{total_frames} "
                        f"loss_next={stats['loss_next']:.6f} "
                        f"loss_masked={stats['loss_masked']:.6f} "
                        f"mask_count={stats['mask_count']} "
                        f"elapsed={elapsed:.1f}s",
                        flush=True,
                    )

                if not dry_run and checkpoint_interval > 0 and step % checkpoint_interval == 0:
                    ckpt_path = checkpoint_dir / f"stage_{_STAGE}_step{step}_{commit_short}.pt"
                    trainer.save_checkpoint(ckpt_path)
                    elapsed = time.time() - t0
                    _append_progress(
                        results_dir,
                        f"step={step} loss_next={stats['loss_next']:.6f} "
                        f"loss_masked={stats['loss_masked']:.6f} "
                        f"mask_ratio={stats['mask_count'] / window_size:.4f} "
                        f"grad_norm={stats['grad_norm']:.6f} "
                        f"elapsed={elapsed:.1f}s checkpoint={ckpt_path.name}",
                    )
                    print(f"[stage_{_STAGE}] wrote checkpoint {ckpt_path.name}", flush=True)

            if _SHUTDOWN_REQUESTED:
                exit_reason = "sigterm"
                break

    except BaseException as e:
        fatal_exc = e
        exit_reason = "fatal"
    finally:
        reset_tracker.close()
        try:
            env.close()
        except Exception:
            pass

    wall_s = time.time() - t0

    if exit_reason == "fatal" and fatal_exc is not None:
        tb_text = "".join(
            traceback.format_exception(type(fatal_exc), fatal_exc, fatal_exc.__traceback__)
        )
        fatal_path = results_dir / "FATAL_ERROR.md"
        fatal_path.write_text(
            "# Stage 0a Fatal Error\n\n"
            f"- Wall-clock: {wall_s:.1f}s\n"
            f"- Frames observed: {trainer.step_count}\n"
            f"- Training steps: {trainer.train_step_count}\n"
            f"- Memory bank size: {len(memory_bank)}\n"
            f"- Last stats: `{last_stats!r}`\n\n"
            "## Traceback\n\n```\n" + tb_text + "```\n"
        )
        print(
            f"[stage_{_STAGE}] FATAL: {type(fatal_exc).__name__}: {fatal_exc}",
            flush=True, file=sys.stderr,
        )
        print(f"[stage_{_STAGE}] wrote {fatal_path.name}", flush=True, file=sys.stderr)
        return 1

    final_step = trainer.train_step_count
    final_checkpoint_name: Optional[str] = None
    if not dry_run:
        tag = "" if exit_reason == "complete" else f"_{exit_reason}"
        ckpt_path = checkpoint_dir / f"stage_{_STAGE}_step{final_step}{tag}_{commit_short}.pt"
        trainer.save_checkpoint(ckpt_path)
        final_checkpoint_name = ckpt_path.name
        print(f"[stage_{_STAGE}] wrote final checkpoint {ckpt_path.name}", flush=True)

        final_snapshot = {
            "exit_reason": exit_reason,
            "signal": _SHUTDOWN_SIGNAL,
            "wall_clock_seconds": wall_s,
            "frames_observed": trainer.step_count,
            "training_steps": final_step,
            "memory_bank_size": len(memory_bank),
            "mask_count": trainer.mask_count,
            "mask_ratio": trainer.mask_count / window_size,
            "git_commit": _git_commit(),
            "final_checkpoint": final_checkpoint_name,
            "total_frames": total_frames,
        }
        name = "training_complete.json" if exit_reason == "complete" else f"training_{exit_reason}.json"
        (results_dir / name).write_text(json.dumps(final_snapshot, indent=2))
        print(f"[stage_{_STAGE}] wrote {name}", flush=True)

    print(
        f"[stage_{_STAGE}] DONE reason={exit_reason} wall={wall_s:.1f}s "
        f"frames={trainer.step_count} train_steps={final_step}",
        flush=True,
    )
    return 0


# ---- CLI --------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=f"Stage {_STAGE} training driver")
    parser.add_argument(
        "--config",
        type=Path,
        default=_DEFAULT_CONFIG,
        help="path to stage config yaml (default: configs/stage_0a.yaml)",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help=(
            "DEFERRED: resume-from-checkpoint is not supported. The current "
            "checkpoint format does not persist env RNG, mask sampler RNG, or "
            "ring-buffer state required for bit-exact continuation. Run afresh."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=f"build components, run {_DRY_RUN_TRAIN_STEPS} training steps, skip checkpointing, exit 0",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="override device (default: cuda if available, else cpu)",
    )
    args = parser.parse_args(argv)

    if args.resume is not None:
        print(
            "[stage_0a] --resume is deferred; see --help. Aborting.",
            file=sys.stderr,
        )
        return 2

    cfg_path: Path = args.config.resolve()
    cfg_text = cfg_path.read_text()
    cfg = _load_config(cfg_path)

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    results_dir = _ROOT / cfg["paths"]["results_dir"] / f"stage_{_STAGE}"
    _write_launch_info(
        results_dir, cfg_path, cfg_text, pid=os.getpid(),
        device=device, dry_run=args.dry_run,
    )

    return run(cfg, device=device, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
