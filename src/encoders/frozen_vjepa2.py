"""Frozen V-JEPA 2 visual encoder wrapper for Tier A.

V-JEPA 2 is a JEPA-family ViT without a CLS token. Per-frame embeddings are
produced by mean-pooling patch tokens from the final layer. This preserves
scene content with smooth geometry across adjacent frames, suitable for
trajectory prediction targets.

Checkpoint: `facebook/vjepa2-vitl-fpc64-256` (ViT-L/16, 64-frame / 256-patch).
Frame embedding dim: D = 1024 (verified at load time against config.hidden_size).

All parameters are frozen. All forward passes run under `torch.no_grad()`.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from transformers import VJEPA2Model


_DEFAULT_CHECKPOINT = "facebook/vjepa2-vitl-fpc64-256"
_EXPECTED_EMBED_DIM = 1024
_EXPECTED_FRAME_SIZE = 224


class FrozenVJepa2Encoder(nn.Module):
    """Frozen V-JEPA 2 ViT-L encoder producing per-frame embeddings of shape (B, D).

    Usage:
        encoder = FrozenVJepa2Encoder().eval()
        emb = encoder.encode_frame(frame)   # frame: (C, H, W) or (B, C, H, W)
                                            # emb:   (B, D), D = 1024

    Input contract:
        - Pixel tensor of shape (C, H, W) or (B, C, H, W).
        - H == W == 224. C == 3 (RGB).
        - Expected to be already preprocessed (ImageNet normalisation as used
          by VJEPA2VideoProcessor). This module does NOT preprocess; callers
          pass normalised float tensors.

    Output contract:
        - Tensor of shape (B, D) with D == 1024, mean-pooled over the final
          layer's patch-token sequence.
    """

    def __init__(
        self,
        checkpoint: str = _DEFAULT_CHECKPOINT,
        device: Optional[torch.device | str] = None,
    ) -> None:
        super().__init__()
        self.checkpoint = checkpoint
        self.model = VJEPA2Model.from_pretrained(checkpoint)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        if trainable != 0:
            raise RuntimeError(
                f"Freeze check failed: {trainable} trainable params remain; expected 0."
            )

        self.embed_dim: int = int(self.model.config.hidden_size)
        if self.embed_dim != _EXPECTED_EMBED_DIM:
            raise RuntimeError(
                f"Unexpected embedding dim: got {self.embed_dim}, expected "
                f"{_EXPECTED_EMBED_DIM} for {checkpoint}."
            )

        if device is not None:
            self.model.to(device)

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def _validate_and_batch(self, frame: torch.Tensor) -> torch.Tensor:
        if not isinstance(frame, torch.Tensor):
            raise TypeError(f"frame must be a torch.Tensor, got {type(frame).__name__}")

        if frame.dim() == 3:
            # (C, H, W) -> (1, C, H, W)
            frame = frame.unsqueeze(0)
        elif frame.dim() != 4:
            raise ValueError(
                f"frame must have shape (C, H, W) or (B, C, H, W); got shape {tuple(frame.shape)}"
            )

        b, c, h, w = frame.shape
        if c != 3:
            raise ValueError(f"expected 3 channels (RGB); got C={c}, shape={tuple(frame.shape)}")
        if h != _EXPECTED_FRAME_SIZE or w != _EXPECTED_FRAME_SIZE:
            raise ValueError(
                f"expected {_EXPECTED_FRAME_SIZE}x{_EXPECTED_FRAME_SIZE} frames; got HxW={h}x{w}"
            )
        return frame

    @torch.no_grad()
    def encode_frame(self, frame: torch.Tensor) -> torch.Tensor:
        """Encode one or more frames to (B, D) embeddings via mean-pool of patch tokens."""
        frame = self._validate_and_batch(frame).to(self.device)

        # NOTE: T=1 chosen for Tier A to avoid cross-boundary blending in Stage 0b.
        # V-JEPA 2 expects pixel_values_videos with shape (B, T, C, H, W).
        # Consider T>1 sliding clip for Stage 0c smooth morphs (see HANDOFF.md).
        videos = frame.unsqueeze(1)  # (B, T=1, C, H, W)

        outputs = self.model(pixel_values_videos=videos, skip_predictor=True)
        last_hidden = outputs.last_hidden_state  # (B, N_patches, D)
        if last_hidden.dim() != 3 or last_hidden.shape[-1] != self.embed_dim:
            raise RuntimeError(
                f"unexpected last_hidden_state shape {tuple(last_hidden.shape)}; "
                f"expected (B, N, {self.embed_dim})"
            )

        embedding = last_hidden.mean(dim=1)  # (B, D)
        assert embedding.shape == (frame.shape[0], self.embed_dim), (
            f"output shape mismatch: got {tuple(embedding.shape)}, "
            f"expected {(frame.shape[0], self.embed_dim)}"
        )
        return embedding

    def forward(self, frame: torch.Tensor) -> torch.Tensor:  # noqa: D401
        """Alias for ``encode_frame`` so the module is usable as a standard nn.Module."""
        return self.encode_frame(frame)
