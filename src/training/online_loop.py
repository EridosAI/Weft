"""Online single-pass training loop for the trajectory predictor (Tier A, Session 4).

One training step per new frame arrival. Context is a ring buffer of the most
recent W encoder embeddings; next-step target is the newly-arrived embedding;
masked-position targets are the (detached) context positions that were
replaced by the mask token. MSE on both; stop-gradient on every target,
asserted explicitly at the top of every step per the spec's never-remove
requirement.

AdamW(lr=3e-4, betas=(0.9, 0.95), weight_decay=0.01) with a cosine warmup
over the first 5000 steps and no subsequent LR decay.

Masking schedule: starts at mask_count = 1 (ratio 1/W). A `PlateauTrigger`
compares the mean masked loss over the most-recent `plateau_window` steps
to the preceding `plateau_window` steps every `plateau_window` steps and
fires when relative improvement < `plateau_threshold`; each fire increments
mask_count by 1, capped by `mask_count_cap`.

Logging: TensorBoard for scalars every step (next-step loss, masked loss,
mask count, LR, gradient-norm stats every 100 steps). JSON snapshot of
aggregate statistics written every `checkpoint_interval` steps to
`results/<stage>/checkpoint_<step>.json`.
"""

from __future__ import annotations

import json
import math
import statistics
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Deque, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

try:
    from torch.utils.tensorboard import SummaryWriter  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    SummaryWriter = None  # type: ignore[assignment]


@dataclass
class TrainingConfig:
    stage: str = "0a"
    window_size: int = 16
    embed_dim: int = 1024
    lr: float = 3e-4
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.95)
    warmup_steps: int = 5000
    initial_mask_count: int = 1
    mask_count_cap: int = 4  # Stage 0a target: 0.25 of W=16; raised per stage.
    plateau_window: int = 10_000
    plateau_threshold: float = 0.05
    checkpoint_interval: int = 10_000
    grad_log_interval: int = 100
    log_dir: Path = field(default_factory=lambda: Path("logs"))
    results_dir: Path = field(default_factory=lambda: Path("results"))
    seed: int = 42


class PlateauTrigger:
    """Fires `should_advance()` when the recent masked-loss mean has not improved
    by `threshold` (relative) over the prior window of the same size."""

    def __init__(self, window: int = 10_000, threshold: float = 0.05) -> None:
        self.window = window
        self.threshold = threshold
        # Two adjacent windows of size `window`: "previous" and "current".
        self._losses: Deque[float] = deque(maxlen=2 * window)

    def observe(self, loss_value: float) -> None:
        self._losses.append(float(loss_value))

    def should_advance(self) -> bool:
        if len(self._losses) < 2 * self.window:
            return False
        prev = list(self._losses)[: self.window]
        curr = list(self._losses)[self.window :]
        prev_mean = statistics.fmean(prev)
        curr_mean = statistics.fmean(curr)
        if prev_mean <= 0:
            return False
        rel_improvement = (prev_mean - curr_mean) / prev_mean
        return rel_improvement < self.threshold

    def reset(self) -> None:
        self._losses.clear()


def _cosine_warmup_lambda(warmup_steps: int):
    def lr_lambda(step: int) -> float:
        if warmup_steps <= 0:
            return 1.0
        if step >= warmup_steps:
            return 1.0
        # half-cosine ramp from 0 to 1 over [0, warmup_steps]
        return 0.5 * (1.0 - math.cos(math.pi * step / warmup_steps))

    return lr_lambda


class OnlineTrainer:
    def __init__(
        self,
        predictor: nn.Module,
        config: TrainingConfig,
        device: Optional[torch.device | str] = None,
        tensorboard_enabled: bool = True,
    ) -> None:
        self.predictor = predictor
        self.config = config
        self.device = torch.device(device) if device is not None else next(predictor.parameters()).device
        self.predictor.to(self.device)

        self.optimizer = AdamW(
            self.predictor.parameters(),
            lr=config.lr,
            betas=config.betas,
            weight_decay=config.weight_decay,
        )
        self.scheduler = LambdaLR(self.optimizer, _cosine_warmup_lambda(config.warmup_steps))

        self.ring_buffer: Deque[torch.Tensor] = deque(maxlen=config.window_size)
        self.step_count: int = 0
        self.train_step_count: int = 0
        self.mask_count: int = config.initial_mask_count
        self.plateau = PlateauTrigger(config.plateau_window, config.plateau_threshold)

        self._recent_next_losses: Deque[float] = deque(maxlen=config.checkpoint_interval)
        self._recent_masked_losses: Deque[float] = deque(maxlen=config.checkpoint_interval)
        self._recent_grad_norms: Deque[float] = deque(maxlen=config.checkpoint_interval)
        self._recent_predicted_norms: Deque[float] = deque(maxlen=config.checkpoint_interval)
        self._recent_target_norms: Deque[float] = deque(maxlen=config.checkpoint_interval)

        self.writer = None
        if tensorboard_enabled and SummaryWriter is not None:
            log_path = config.log_dir / f"stage_{config.stage}"
            log_path.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir=str(log_path))

        self._rng = torch.Generator()
        self._rng.manual_seed(config.seed)

    # ---- public API ---------------------------------------------------------

    def observe_frame(self, embedding: torch.Tensor) -> Optional[Dict[str, float]]:
        """Consume one encoder embedding and, if a full context window is
        available, run one training step against it as the next-step target.

        Returns a dict of per-step statistics on training steps, or None
        during warmup (before W frames accumulated)."""
        if embedding.dim() != 1 or embedding.shape[0] != self.config.embed_dim:
            raise ValueError(
                f"embedding must be 1-D of size {self.config.embed_dim}; got {tuple(embedding.shape)}"
            )
        emb = embedding.detach().to(self.device)

        stats: Optional[Dict[str, float]] = None
        if len(self.ring_buffer) == self.config.window_size:
            # Context is the current ring buffer; new embedding is the next-step target.
            context = torch.stack(list(self.ring_buffer), dim=0).unsqueeze(0)  # (1, W, D)
            target_next = emb  # (D,)
            stats = self._training_step(context=context, target_next=target_next)

        self.ring_buffer.append(emb)
        self.step_count += 1
        return stats

    # ---- training step ------------------------------------------------------

    def _sample_mask_positions(self, batch: int) -> torch.Tensor:
        k = min(self.mask_count, self.config.window_size)
        if k <= 0:
            return torch.zeros(batch, 0, dtype=torch.long, device=self.device)
        positions: List[List[int]] = []
        for _ in range(batch):
            perm = torch.randperm(self.config.window_size, generator=self._rng)[:k]
            positions.append(perm.tolist())
        return torch.tensor(positions, dtype=torch.long, device=self.device)

    def _training_step(self, context: torch.Tensor, target_next: torch.Tensor) -> Dict[str, float]:
        # Stop-gradient on targets — asserted per SESSION_BATCH_INSTRUCTIONS.md §Session 4.
        # DO NOT REMOVE THESE ASSERTIONS. A missing stop-gradient has been a prior
        # failure mode: the encoder is frozen in Tier A so there is no model to
        # update, but asserting keeps the training loop robust against future
        # refactors where targets might come from a trainable branch.
        assert not context.requires_grad, "Stop-gradient not applied to context target"
        assert not target_next.requires_grad, "Stop-gradient not applied to next-step target"

        self.predictor.train()
        mask_positions = self._sample_mask_positions(batch=context.shape[0])

        # Before masking, record the (detached) masked-position targets from context.
        # Indexing a detached tensor keeps requires_grad=False.
        b = context.shape[0]
        k = mask_positions.shape[1]
        if k > 0:
            batch_idx = torch.arange(b, device=self.device).unsqueeze(1).expand(b, k)
            target_masked = context[batch_idx, mask_positions].detach()
        else:
            target_masked = context.new_zeros((b, 0, self.config.embed_dim))
        assert not target_masked.requires_grad, "Stop-gradient not applied to masked-position targets"

        outputs = self.predictor(context, mask_positions)
        predicted_next = outputs["predicted_next"].squeeze(0)  # (D,)
        predicted_masked = outputs["predicted_masked"]          # (B, K, D)

        loss_next = F.mse_loss(predicted_next, target_next)
        if k > 0:
            loss_masked = F.mse_loss(predicted_masked, target_masked)
        else:
            loss_masked = torch.zeros((), device=self.device)
        loss = loss_next + loss_masked

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = self._total_grad_norm()
        self.optimizer.step()
        self.scheduler.step()

        self.train_step_count += 1
        loss_next_v = float(loss_next.detach().item())
        loss_masked_v = float(loss_masked.detach().item())

        self._recent_next_losses.append(loss_next_v)
        self._recent_masked_losses.append(loss_masked_v)
        self._recent_grad_norms.append(grad_norm)
        self._recent_predicted_norms.append(float(predicted_next.detach().norm().item()))
        self._recent_target_norms.append(float(target_next.detach().norm().item()))

        self.plateau.observe(loss_masked_v)
        if (
            self.train_step_count % self.config.plateau_window == 0
            and self.mask_count < self.config.mask_count_cap
            and self.plateau.should_advance()
        ):
            self.mask_count += 1
            self.plateau.reset()

        self._log(loss_next_v, loss_masked_v, grad_norm)
        if self.train_step_count % self.config.checkpoint_interval == 0:
            self._dump_checkpoint_snapshot()

        return {
            "step": self.train_step_count,
            "loss_next": loss_next_v,
            "loss_masked": loss_masked_v,
            "loss": loss_next_v + loss_masked_v,
            "grad_norm": grad_norm,
            "mask_count": self.mask_count,
            "lr": float(self.scheduler.get_last_lr()[0]),
        }

    # ---- helpers ------------------------------------------------------------

    def _total_grad_norm(self) -> float:
        total = 0.0
        for p in self.predictor.parameters():
            if p.grad is not None:
                total += float(p.grad.detach().data.norm(2).item()) ** 2
        return math.sqrt(total)

    def _log(self, loss_next: float, loss_masked: float, grad_norm: float) -> None:
        if self.writer is None:
            return
        s = self.train_step_count
        self.writer.add_scalar("loss/next_step", loss_next, s)
        self.writer.add_scalar("loss/masked", loss_masked, s)
        self.writer.add_scalar("mask/count", self.mask_count, s)
        self.writer.add_scalar("mask/ratio", self.mask_count / self.config.window_size, s)
        self.writer.add_scalar("lr", float(self.scheduler.get_last_lr()[0]), s)
        if s % self.config.grad_log_interval == 0:
            recent = list(self._recent_grad_norms)[-self.config.grad_log_interval :] or [grad_norm]
            self.writer.add_scalar("grad/norm_max", max(recent), s)
            self.writer.add_scalar("grad/norm_median", statistics.median(recent), s)
            self.writer.add_scalar(
                "grad/norm_std", statistics.pstdev(recent) if len(recent) > 1 else 0.0, s
            )

    def _dump_checkpoint_snapshot(self) -> None:
        results_dir = self.config.results_dir / f"stage_{self.config.stage}"
        results_dir.mkdir(parents=True, exist_ok=True)
        path = results_dir / f"checkpoint_{self.train_step_count}.json"
        snapshot = {
            "step": self.train_step_count,
            "mask_count": self.mask_count,
            "mask_ratio": self.mask_count / self.config.window_size,
            "lr": float(self.scheduler.get_last_lr()[0]),
            "loss_next": _agg(self._recent_next_losses),
            "loss_masked": _agg(self._recent_masked_losses),
            "grad_norm": _agg(self._recent_grad_norms),
            "predicted_norm": _agg(self._recent_predicted_norms),
            "target_norm": _agg(self._recent_target_norms),
            "config": _config_to_dict(self.config),
        }
        path.write_text(json.dumps(snapshot, indent=2))

    # ---- checkpoint save/load -----------------------------------------------

    def save_checkpoint(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "predictor": self.predictor.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "step_count": self.step_count,
                "train_step_count": self.train_step_count,
                "mask_count": self.mask_count,
            },
            path,
        )

    def load_checkpoint(self, path: Path | str) -> None:
        path = Path(path)
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.predictor.load_state_dict(state["predictor"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.step_count = int(state["step_count"])
        self.train_step_count = int(state["train_step_count"])
        self.mask_count = int(state["mask_count"])


def _agg(values: Deque[float]) -> Dict[str, float]:
    if not values:
        return {"n": 0}
    lst = list(values)
    return {
        "n": len(lst),
        "min": min(lst),
        "max": max(lst),
        "mean": statistics.fmean(lst),
        "median": statistics.median(lst),
        "std": statistics.pstdev(lst) if len(lst) > 1 else 0.0,
    }


def _config_to_dict(cfg: TrainingConfig) -> Dict[str, object]:
    d = asdict(cfg)
    d["log_dir"] = str(cfg.log_dir)
    d["results_dir"] = str(cfg.results_dir)
    return d
