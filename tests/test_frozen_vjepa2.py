"""Tests for the frozen V-JEPA 2 encoder wrapper (Session 2, Tier A)."""

from __future__ import annotations

import pytest
import torch

from src.encoders.frozen_vjepa2 import FrozenVJepa2Encoder


@pytest.fixture(scope="module")
def encoder() -> FrozenVJepa2Encoder:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc = FrozenVJepa2Encoder(device=device).eval()
    return enc


def _random_frame(batch: int | None, device: torch.device) -> torch.Tensor:
    shape = (3, 224, 224) if batch is None else (batch, 3, 224, 224)
    return torch.randn(*shape, device=device)


def test_forward_output_shape(encoder: FrozenVJepa2Encoder) -> None:
    frame = _random_frame(batch=2, device=encoder.device)
    emb = encoder.encode_frame(frame)
    assert emb.shape == (2, 1024)
    assert emb.dtype == torch.float32 or emb.dtype == torch.float16
    assert emb.device.type == encoder.device.type


def test_freeze_no_trainable_params(encoder: FrozenVJepa2Encoder) -> None:
    trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    assert trainable == 0, f"expected 0 trainable params; got {trainable}"
    for p in encoder.model.parameters():
        assert p.requires_grad is False


def test_deterministic_for_identical_input(encoder: FrozenVJepa2Encoder) -> None:
    torch.manual_seed(0)
    frame = _random_frame(batch=1, device=encoder.device)
    a = encoder.encode_frame(frame)
    b = encoder.encode_frame(frame)
    assert torch.allclose(a, b, atol=0.0, rtol=0.0), (
        "expected exact determinism on identical input with frozen encoder in eval mode"
    )


def test_single_vs_batched_equivalence(encoder: FrozenVJepa2Encoder) -> None:
    """(C, H, W) input should give the same embedding as (1, C, H, W) input,
    and stacking a batch should give per-row results equal to single-frame results."""
    torch.manual_seed(1)
    unbatched = _random_frame(batch=None, device=encoder.device)  # (C, H, W)
    emb_u = encoder.encode_frame(unbatched)
    emb_b1 = encoder.encode_frame(unbatched.unsqueeze(0))
    assert emb_u.shape == (1, 1024)
    assert torch.allclose(emb_u, emb_b1, atol=1e-5, rtol=1e-5)

    torch.manual_seed(2)
    f1 = _random_frame(batch=None, device=encoder.device)
    f2 = _random_frame(batch=None, device=encoder.device)
    batch = torch.stack([f1, f2], dim=0)
    emb_batch = encoder.encode_frame(batch)
    emb_1 = encoder.encode_frame(f1)
    emb_2 = encoder.encode_frame(f2)
    assert torch.allclose(emb_batch[0:1], emb_1, atol=1e-4, rtol=1e-4)
    assert torch.allclose(emb_batch[1:2], emb_2, atol=1e-4, rtol=1e-4)


def test_rejects_wrong_channel_count(encoder: FrozenVJepa2Encoder) -> None:
    bad = torch.randn(1, 4, 224, 224, device=encoder.device)
    with pytest.raises(ValueError, match="expected 3 channels"):
        encoder.encode_frame(bad)


def test_rejects_wrong_spatial_size(encoder: FrozenVJepa2Encoder) -> None:
    bad = torch.randn(1, 3, 96, 96, device=encoder.device)
    with pytest.raises(ValueError, match="expected 224x224"):
        encoder.encode_frame(bad)


def test_rejects_wrong_dimensionality(encoder: FrozenVJepa2Encoder) -> None:
    bad = torch.randn(5, device=encoder.device)
    with pytest.raises(ValueError, match=r"shape \(C, H, W\) or \(B, C, H, W\)"):
        encoder.encode_frame(bad)


def test_rejects_non_tensor(encoder: FrozenVJepa2Encoder) -> None:
    with pytest.raises(TypeError, match="must be a torch.Tensor"):
        encoder.encode_frame([[0, 1, 2]])  # type: ignore[arg-type]


def test_embed_dim_attribute(encoder: FrozenVJepa2Encoder) -> None:
    assert encoder.embed_dim == 1024


def test_no_grad_on_outputs(encoder: FrozenVJepa2Encoder) -> None:
    frame = _random_frame(batch=1, device=encoder.device)
    emb = encoder.encode_frame(frame)
    assert emb.requires_grad is False, "frozen encoder outputs should not require grad"
