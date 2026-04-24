"""Inward PAM — trajectory predictor transformer (Tier A, Session 3).

Standard pre-LayerNorm transformer encoder over a context window of W frame
embeddings plus a query position at index W.

  - Frame embedding D_in (1024 for frozen V-JEPA 2) is projected linearly to
    hidden dim H = 512.
  - W+1 learnable temporal position embeddings are added to every token.
  - A single learnable mask token (at hidden dim H) replaces the projected
    embedding at masked positions. The query position always receives the
    mask token (no embedding to mask — it's the position being predicted).
  - The transformer output at each position of interest (query and masked)
    is projected back to D_out = D_in via a linear layer. No final
    output-side LayerNorm — predictions are left unnormalised so the
    predictor can learn to match encoder target magnitude under MSE.

No stop-gradient is applied here; that is handled in the training loop
(see src/training/online_loop.py).

Forward contract:
    context:        (B, W, D_in) frame embeddings
    mask_positions: (B, K) LongTensor of indices in [0, W)
    Returns:
        {"predicted_next":  (B, D_out),
         "predicted_masked":(B, K, D_out)}
    Empty K=0 is supported and returns a (B, 0, D_out) tensor.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class TrajectoryPredictor(nn.Module):
    def __init__(
        self,
        embed_dim: int = 1024,
        hidden_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        mlp_dim: int = 2048,
        window_size: int = 16,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.window_size = window_size
        self.num_layers = num_layers

        self.input_proj = nn.Linear(embed_dim, hidden_dim)
        # Learnable mask token, shared by all masked positions and by the query.
        self.mask_token = nn.Parameter(torch.zeros(hidden_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        # Learnable temporal position embeddings: W context slots + 1 query slot.
        self.position_embeddings = nn.Parameter(torch.zeros(window_size + 1, hidden_dim))
        nn.init.trunc_normal_(self.position_embeddings, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # pre-LayerNorm
        )
        # enable_nested_tensor=False because norm_first (pre-LN) is incompatible
        # with the nested-tensor fast path — setting it explicitly silences the
        # PyTorch warning without changing behaviour.
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )

        self.output_proj = nn.Linear(hidden_dim, embed_dim)

    def forward(
        self,
        context: torch.Tensor,
        mask_positions: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if context.dim() != 3:
            raise ValueError(
                f"context must be (B, W, D); got shape {tuple(context.shape)}"
            )
        b, w, d_in = context.shape
        if w != self.window_size:
            raise ValueError(
                f"context window size {w} does not match configured W={self.window_size}"
            )
        if d_in != self.embed_dim:
            raise ValueError(
                f"context embedding dim {d_in} does not match configured D={self.embed_dim}"
            )
        if not isinstance(mask_positions, torch.Tensor):
            raise TypeError("mask_positions must be a torch.LongTensor")
        if mask_positions.dim() != 2 or mask_positions.shape[0] != b:
            raise ValueError(
                f"mask_positions must be (B, K) with B={b}; got {tuple(mask_positions.shape)}"
            )
        if mask_positions.dtype not in (torch.long, torch.int64):
            raise TypeError(f"mask_positions must be long/int64; got {mask_positions.dtype}")
        if mask_positions.numel() > 0 and (
            mask_positions.min().item() < 0 or mask_positions.max().item() >= self.window_size
        ):
            raise ValueError(
                f"mask_positions values must be in [0, {self.window_size}); got range "
                f"[{mask_positions.min().item()}, {mask_positions.max().item()}]"
            )

        k = mask_positions.shape[1]

        # Project inputs to hidden dim: (B, W, H)
        projected = self.input_proj(context)

        # Apply mask token at mask_positions. Broadcast mask_token over (B, K, H).
        if k > 0:
            batch_idx = torch.arange(b, device=context.device).unsqueeze(1).expand(b, k)
            projected = projected.clone()
            projected[batch_idx, mask_positions] = self.mask_token.to(projected.dtype)

        # Append the query slot at position W with the mask token: (B, W+1, H)
        query_token = self.mask_token.to(projected.dtype).view(1, 1, -1).expand(b, 1, -1)
        tokens = torch.cat([projected, query_token], dim=1)

        # Add temporal position embeddings to every token.
        tokens = tokens + self.position_embeddings.to(projected.dtype).unsqueeze(0)

        # Transformer forward: (B, W+1, H)
        encoded = self.encoder(tokens)

        # Query output at position W -> project back to D_out -> (B, D_out)
        query_out = encoded[:, self.window_size, :]
        predicted_next = self.output_proj(query_out)

        # Masked outputs at mask_positions -> (B, K, D_out)
        if k > 0:
            masked_out = encoded[batch_idx, mask_positions]  # (B, K, H)
            predicted_masked = self.output_proj(masked_out)
        else:
            predicted_masked = context.new_zeros((b, 0, self.embed_dim))

        return {"predicted_next": predicted_next, "predicted_masked": predicted_masked}
