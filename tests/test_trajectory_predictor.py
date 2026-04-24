"""Tests for the inward PAM trajectory predictor (Session 3, Tier A)."""

from __future__ import annotations

import pytest
import torch

from src.predictor.trajectory_predictor import TrajectoryPredictor


W = 16
D = 1024
B = 2


@pytest.fixture()
def model() -> TrajectoryPredictor:
    torch.manual_seed(0)
    return TrajectoryPredictor(
        embed_dim=D,
        hidden_dim=512,
        num_layers=4,
        num_heads=8,
        mlp_dim=2048,
        window_size=W,
        dropout=0.0,
    ).eval()


def _context(batch: int = B) -> torch.Tensor:
    return torch.randn(batch, W, D)


def test_forward_shapes_k4(model: TrajectoryPredictor) -> None:
    torch.manual_seed(1)
    ctx = _context()
    mask = torch.tensor([[0, 3, 7, 12], [1, 5, 9, 14]], dtype=torch.long)
    out = model(ctx, mask)
    assert out["predicted_next"].shape == (B, D)
    assert out["predicted_masked"].shape == (B, 4, D)


def test_forward_k0_only_query(model: TrajectoryPredictor) -> None:
    ctx = _context()
    mask = torch.zeros(B, 0, dtype=torch.long)
    out = model(ctx, mask)
    assert out["predicted_next"].shape == (B, D)
    assert out["predicted_masked"].shape == (B, 0, D)


def test_forward_k_equals_W_minus_1(model: TrajectoryPredictor) -> None:
    """All but one context position masked, plus the query."""
    ctx = _context()
    positions = list(range(W - 1))  # 0..14, leave 15 unmasked
    mask = torch.tensor([positions, positions], dtype=torch.long)
    out = model(ctx, mask)
    assert out["predicted_next"].shape == (B, D)
    assert out["predicted_masked"].shape == (B, W - 1, D)


def test_parameter_count_logged_and_reasonable(
    model: TrajectoryPredictor, capsys: pytest.CaptureFixture[str]
) -> None:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # Log — the spec's note is "should be in the ~5-10M range"; our spec'd
    # architecture (4 layers, H=512, MLP=2048, D=1024) actually lands around
    # ~13.7M. Captured for HANDOFF notes; assert a loose reasonable range.
    print(f"TrajectoryPredictor params: total={total:,} trainable={trainable:,}")
    assert total == trainable, "all params should be trainable by default"
    assert 5_000_000 <= total <= 25_000_000, (
        f"param count {total:,} is outside the reasonable range; "
        f"spec hint was ~5-10M but architecture as-specified lands higher"
    )


def test_output_embedding_dim_round_trip(model: TrajectoryPredictor) -> None:
    """Output embedding dim must match input embedding dim D=1024."""
    ctx = _context()
    mask = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=torch.long)
    out = model(ctx, mask)
    assert out["predicted_next"].shape[-1] == D
    assert out["predicted_masked"].shape[-1] == D


def test_gradient_flow(model: TrajectoryPredictor) -> None:
    """Backward from a dummy loss must produce non-zero grads on all params."""
    model.train()
    torch.manual_seed(2)
    ctx = _context()
    mask = torch.tensor([[0, 3, 7, 12], [1, 5, 9, 14]], dtype=torch.long)
    out = model(ctx, mask)
    loss = out["predicted_next"].pow(2).mean() + out["predicted_masked"].pow(2).mean()
    loss.backward()

    params_with_zero_grad: list[str] = []
    for name, p in model.named_parameters():
        if p.grad is None:
            params_with_zero_grad.append(f"{name} (grad is None)")
        elif p.grad.abs().sum().item() == 0.0:
            params_with_zero_grad.append(f"{name} (grad sum 0)")
    assert not params_with_zero_grad, (
        f"{len(params_with_zero_grad)} params have no gradient: "
        + ", ".join(params_with_zero_grad[:5])
    )


def test_mask_token_injection_changes_output(model: TrajectoryPredictor) -> None:
    """Sanity: masking a position changes the masked-position output
    (otherwise the mask token is being ignored)."""
    ctx = _context()
    mask_empty = torch.zeros(B, 0, dtype=torch.long)
    mask_one = torch.tensor([[5], [5]], dtype=torch.long)

    out_empty = model(ctx, mask_empty)
    out_masked = model(ctx, mask_one)
    # predicted_next depends on attention that now sees mask_token at pos 5;
    # it should differ from the no-mask run.
    assert not torch.allclose(
        out_empty["predicted_next"], out_masked["predicted_next"], atol=1e-5
    ), "masking a context position did not change the predicted_next output"


def test_rejects_wrong_window(model: TrajectoryPredictor) -> None:
    bad_ctx = torch.randn(B, W - 1, D)  # window 15 instead of 16
    mask = torch.zeros(B, 0, dtype=torch.long)
    with pytest.raises(ValueError, match="window size"):
        model(bad_ctx, mask)


def test_rejects_wrong_embed_dim(model: TrajectoryPredictor) -> None:
    bad_ctx = torch.randn(B, W, D + 1)
    mask = torch.zeros(B, 0, dtype=torch.long)
    with pytest.raises(ValueError, match="embedding dim"):
        model(bad_ctx, mask)


def test_rejects_mask_out_of_range(model: TrajectoryPredictor) -> None:
    ctx = _context()
    bad_mask = torch.tensor([[W], [0]], dtype=torch.long)  # W is out of range
    with pytest.raises(ValueError, match=r"values must be in"):
        model(ctx, bad_mask)
