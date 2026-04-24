"""Pre-flight smoke test for the Weft Tier A pipeline (Session 5).

Runs 1000 frames of the Stage 0a configuration end-to-end:

    env -> encoder -> memory_bank -> predictor -> loss -> optimizer step

This is a pipeline validator, not a gate evaluator. The pass/fail criteria
listed in SESSION_BATCH_INSTRUCTIONS.md §Session 5 determine whether to
proceed with a full Stage 0a run.

Usage:
    python scripts/preflight_smoke_test.py

Outputs:
    results/preflight/smoke_report.json — all metric values
    results/preflight/SMOKE_REPORT.md  — human-readable summary
    results/preflight/FAILURE_REPORT.md — written only on failure
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Ensure project root is importable when running from anywhere.
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


PREFLIGHT_FRAMES = 1000
PREFLIGHT_STAGE = "preflight"
REPORT_DIR = _ROOT / "results" / "preflight"


def _load_stage_0a_config() -> Dict[str, Any]:
    path = _ROOT / "configs" / "stage_0a.yaml"
    return yaml.safe_load(path.read_text())


def _build_trainer(cfg_yaml: Dict[str, Any], device: torch.device) -> OnlineTrainer:
    p = cfg_yaml["predictor"]
    predictor = TrajectoryPredictor(
        embed_dim=p["embed_dim"],
        hidden_dim=p["hidden_dim"],
        num_layers=p["num_layers"],
        num_heads=p["num_heads"],
        mlp_dim=p["mlp_dim"],
        window_size=cfg_yaml["window_size"],
        dropout=p["dropout"],
    )
    t_cfg = TrainingConfig(
        stage=PREFLIGHT_STAGE,
        window_size=cfg_yaml["window_size"],
        embed_dim=p["embed_dim"],
        lr=cfg_yaml["optimizer"]["lr"],
        weight_decay=cfg_yaml["optimizer"]["weight_decay"],
        betas=tuple(cfg_yaml["optimizer"]["betas"]),  # type: ignore[arg-type]
        warmup_steps=cfg_yaml["optimizer"]["warmup_steps"],
        initial_mask_count=cfg_yaml["masking"]["initial_mask_count"],
        mask_count_cap=cfg_yaml["masking"]["mask_count_cap"],
        plateau_window=cfg_yaml["masking"]["plateau_window"],
        plateau_threshold=cfg_yaml["masking"]["plateau_threshold"],
        checkpoint_interval=500,  # force at least one snapshot within 1000 steps
        grad_log_interval=cfg_yaml["logging"]["grad_log_interval"],
        log_dir=_ROOT / cfg_yaml["paths"]["log_dir"],
        results_dir=_ROOT / cfg_yaml["paths"]["results_dir"],
        seed=cfg_yaml["seed"],
    )
    trainer = OnlineTrainer(predictor, t_cfg, device=device, tensorboard_enabled=False)
    return trainer


def _summarise(values: List[float]) -> Dict[str, float | int]:
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "min": min(values),
        "max": max(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
    }


def _is_finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Weft pre-flight smoke test")
    parser.add_argument(
        "--frames",
        type=int,
        default=PREFLIGHT_FRAMES,
        help="number of frames to run (default 1000)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="override device (default: cuda if available, else cpu)",
    )
    args = parser.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    cfg = _load_stage_0a_config()
    torch.manual_seed(cfg["seed"])

    print(f"[preflight] device={device} frames={args.frames}")
    t0 = time.time()

    encoder = FrozenVJepa2Encoder(
        checkpoint=cfg["encoder"]["checkpoint"], device=device
    ).eval()
    memory_bank = MemoryBank(
        embed_dim=cfg["encoder"]["embed_dim"],
        max_size=cfg["memory_bank"]["max_size"],
        rebuild_interval=cfg["memory_bank"]["rebuild_interval"],
    )
    trainer = _build_trainer(cfg, device)

    env = PushTStagedEnv(stage="0a", seed=cfg["seed"])

    # Per-step logs
    step_losses_next: List[float] = []
    step_losses_masked: List[float] = []
    grad_norms: List[float] = []
    predicted_norms: List[float] = []
    encoder_emb_norms: List[float] = []
    training_step_ids: List[int] = []

    checkpoint_saved = False
    ckpt_path = _ROOT / "checkpoints" / "preflight.pt"

    for frame_idx in range(args.frames):
        frame = env.next_frame()
        tensor = frame_to_encoder_tensor(frame).to(device)
        emb = encoder.encode_frame(tensor).squeeze(0)  # (1024,) float32
        encoder_emb_norms.append(float(emb.detach().norm().item()))

        memory_bank.append(
            emb.detach(),
            FrameMetadata(frame_idx=frame_idx, stage=PREFLIGHT_STAGE, config="default"),
        )

        stats = trainer.observe_frame(emb.detach())
        if stats is not None:
            step_losses_next.append(stats["loss_next"])
            step_losses_masked.append(stats["loss_masked"])
            grad_norms.append(stats["grad_norm"])
            predicted_norms.append(trainer._recent_predicted_norms[-1])
            training_step_ids.append(stats["step"])

            # Save a checkpoint mid-run the first time we hit a boundary.
            if not checkpoint_saved and stats["step"] % 250 == 0:
                trainer.save_checkpoint(ckpt_path)
                checkpoint_saved = True

        if (frame_idx + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(
                f"[preflight] frame {frame_idx + 1}/{args.frames} "
                f"train_steps={trainer.train_step_count} elapsed={elapsed:.1f}s"
            )

    # Exercise FAISS retrieval: pick a stored embedding and verify top-1 is itself.
    faiss_ok = False
    faiss_top1_score = float("nan")
    try:
        probe_idx = min(50, len(memory_bank) - 1)
        _probe_emb, _ = memory_bank.get_window(probe_idx, 1)
        idx, score, _ = memory_bank.retrieve(_probe_emb[0], k=1)
        faiss_ok = int(idx[0]) == probe_idx
        faiss_top1_score = float(score[0])
    except Exception as e:
        faiss_ok = False
        faiss_top1_score = float("nan")
        print(f"[preflight] FAISS exercise raised: {e}")

    # ---- evaluate pass criteria --------------------------------------------
    expected_mem_size = args.frames
    mem_size_within_5pct = abs(len(memory_bank) - expected_mem_size) <= 0.05 * expected_mem_size

    # Loss-reduction check: compare mean of first ~step 100 window vs ~step 1000 window.
    loss_early = statistics.fmean(step_losses_next[:50]) if len(step_losses_next) >= 50 else float("nan")
    loss_late = statistics.fmean(step_losses_next[-50:]) if len(step_losses_next) >= 50 else float("nan")
    loss_decreased = (
        _is_finite(loss_early) and _is_finite(loss_late) and loss_late < loss_early
    )

    grad_median = statistics.median(grad_norms) if grad_norms else float("nan")
    grad_finite = _is_finite(grad_median)
    grad_in_range = _is_finite(grad_median) and (1e-6 <= grad_median <= 100.0)

    pred_norm_mean = statistics.fmean(predicted_norms) if predicted_norms else float("nan")
    enc_norm_mean = statistics.fmean(encoder_emb_norms) if encoder_emb_norms else float("nan")
    pred_norm_ratio = (
        pred_norm_mean / enc_norm_mean if enc_norm_mean and _is_finite(enc_norm_mean) else float("nan")
    )
    pred_norm_within = _is_finite(pred_norm_ratio) and (0.5 <= pred_norm_ratio <= 2.0)

    no_nan_inf = (
        all(_is_finite(v) for v in step_losses_next)
        and all(_is_finite(v) for v in step_losses_masked)
        and all(_is_finite(v) for v in grad_norms)
        and all(_is_finite(v) for v in predicted_norms)
        and all(_is_finite(v) for v in encoder_emb_norms)
    )

    pass_criteria = {
        "no_nan_inf": no_nan_inf,
        "loss_decreased": loss_decreased,
        "memory_bank_size_within_5pct": mem_size_within_5pct,
        "checkpoint_saved": checkpoint_saved and ckpt_path.is_file(),
        "stop_gradient_assertions_ok": True,  # if we got here, none fired
        "grad_norm_median_finite": grad_finite,
        "grad_norm_median_in_range": grad_in_range,
        "predicted_norm_within_half_to_two_x": pred_norm_within,
        "faiss_index_and_retrieve_ok": faiss_ok,
    }
    all_pass = all(pass_criteria.values())

    wall_clock_s = time.time() - t0

    report: Dict[str, Any] = {
        "overall": "PASS" if all_pass else "FAIL",
        "frames_run": args.frames,
        "training_steps": trainer.train_step_count,
        "wall_clock_seconds": wall_clock_s,
        "device": str(device),
        "memory_bank_size": len(memory_bank),
        "checkpoint_saved": checkpoint_saved,
        "checkpoint_path": str(ckpt_path) if checkpoint_saved else None,
        "faiss_top1_self_score": faiss_top1_score,
        "loss_next_summary": _summarise(step_losses_next),
        "loss_masked_summary": _summarise(step_losses_masked),
        "grad_norm_summary": _summarise(grad_norms),
        "predicted_norm_summary": _summarise(predicted_norms),
        "encoder_emb_norm_summary": _summarise(encoder_emb_norms),
        "loss_early_window_mean": loss_early,
        "loss_late_window_mean": loss_late,
        "predicted_to_encoder_norm_ratio": pred_norm_ratio,
        "pass_criteria": pass_criteria,
    }

    smoke_json_path = REPORT_DIR / "smoke_report.json"
    smoke_json_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"[preflight] wrote {smoke_json_path}")

    # Human-readable markdown
    md_lines = [
        "# Pre-flight Smoke Report",
        "",
        f"- **Overall:** {report['overall']}",
        f"- Device: `{device}`",
        f"- Frames run: {args.frames}",
        f"- Training steps: {trainer.train_step_count}",
        f"- Wall-clock: {wall_clock_s:.1f} s",
        f"- Memory bank entries: {len(memory_bank)}",
        f"- FAISS top-1 self score: {faiss_top1_score:.6f}",
        f"- Checkpoint saved: {checkpoint_saved} ({ckpt_path})",
        "",
        "## Pass criteria",
        "",
    ]
    for k, v in pass_criteria.items():
        md_lines.append(f"- `{k}`: **{'PASS' if v else 'FAIL'}**")
    md_lines.extend(
        [
            "",
            "## Loss trajectory (next-step MSE)",
            f"- Early-window mean (first 50 train steps): `{loss_early:.6f}`",
            f"- Late-window mean (last 50 train steps):  `{loss_late:.6f}`",
            f"- Loss decreased: **{loss_decreased}**",
            "",
            "## Gradient norms",
            f"- Median: `{grad_median:.6f}`",
            f"- Finite: **{grad_finite}** ; within `[1e-6, 100]`: **{grad_in_range}**",
            "",
            "## Embedding norms",
            f"- Encoder mean: `{enc_norm_mean:.4f}`",
            f"- Predictor mean: `{pred_norm_mean:.4f}`",
            f"- Ratio (pred / enc): `{pred_norm_ratio:.4f}` ; within `[0.5, 2.0]`: **{pred_norm_within}**",
            "",
        ]
    )
    if not all_pass:
        md_lines.extend(["", "## Failure summary", ""])
        for k, v in pass_criteria.items():
            if not v:
                md_lines.append(f"- **{k}** did not pass.")
    (REPORT_DIR / "SMOKE_REPORT.md").write_text("\n".join(md_lines))
    print(f"[preflight] wrote {REPORT_DIR / 'SMOKE_REPORT.md'}")

    if not all_pass:
        fail_lines = [
            "# Pre-flight Failure Report",
            "",
            f"Pre-flight smoke test FAILED at {args.frames} frames on `{device}`.",
            "",
            "## Failed criteria",
            "",
        ]
        for k, v in pass_criteria.items():
            if not v:
                fail_lines.append(f"- `{k}`")
        fail_lines.extend(
            [
                "",
                "## Metric snapshot",
                "```json",
                json.dumps(report, indent=2, default=str),
                "```",
            ]
        )
        (REPORT_DIR / "FAILURE_REPORT.md").write_text("\n".join(fail_lines))
        print(f"[preflight] wrote {REPORT_DIR / 'FAILURE_REPORT.md'}")

    env.close()
    print(f"[preflight] overall: {report['overall']}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
