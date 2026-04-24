"""Tests for the append-only FAISS memory bank (Session 4, Tier A)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.memory.memory_bank import MemoryBank, FrameMetadata


D = 1024


def _rand_embedding(seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(D, generator=g)


def test_append_retrieve_roundtrip_1000() -> None:
    bank = MemoryBank(embed_dim=D, max_size=2000, rebuild_interval=500)
    for i in range(1000):
        bank.append(_rand_embedding(i), FrameMetadata(frame_idx=i))
    assert len(bank) == 1000
    # Probe one of the stored embeddings: the top hit should be itself with cosine ~1.0.
    probe = _rand_embedding(42)
    indices, scores, metas = bank.retrieve(probe, k=5)
    assert indices[0] == 42
    assert scores[0] == pytest.approx(1.0, abs=1e-4)
    assert metas[0].frame_idx == 42


def test_l2_normalisation_applied_on_storage() -> None:
    bank = MemoryBank(embed_dim=D, max_size=100, rebuild_interval=10)
    big = torch.randn(D) * 17.5  # large-scale vector
    bank.append(big, FrameMetadata(frame_idx=0))
    stored = bank._store[0]  # internal access for this test only
    norm = float(np.linalg.norm(stored))
    assert norm == pytest.approx(1.0, abs=1e-5), (
        f"stored vector should be L2-normalised; got norm={norm}"
    )


def test_metadata_preserved_through_retrieval() -> None:
    bank = MemoryBank(embed_dim=D, max_size=100, rebuild_interval=10)
    for i in range(50):
        meta = FrameMetadata(
            frame_idx=i,
            stage="0b",
            config=f"cfg_{i % 4}",
            transition_zone_flag=(i % 7 == 0),
            extra={"timestamp": i * 0.1},
        )
        bank.append(_rand_embedding(i), meta)
    _, _, metas = bank.retrieve(_rand_embedding(12), k=3)
    assert metas[0].frame_idx == 12
    assert metas[0].config == "cfg_0"  # 12 % 4
    assert metas[0].transition_zone_flag is False
    assert metas[0].extra["timestamp"] == pytest.approx(1.2)


def test_rebuild_index_is_idempotent() -> None:
    """Running rebuild_index twice should not change retrieval results."""
    bank = MemoryBank(embed_dim=D, max_size=100, rebuild_interval=1_000_000)
    for i in range(30):
        bank.append(_rand_embedding(i), FrameMetadata(frame_idx=i))
    bank.rebuild_index()
    i1, s1, _ = bank.retrieve(_rand_embedding(5), k=5)
    bank.rebuild_index()
    i2, s2, _ = bank.retrieve(_rand_embedding(5), k=5)
    assert np.array_equal(i1, i2)
    assert np.allclose(s1, s2)


def test_probe_retrieves_itself_top_hit_score_near_one() -> None:
    """FAISS correctness sanity check: the closest neighbour of a stored vector
    should be itself with cosine ~1.0."""
    bank = MemoryBank(embed_dim=D, max_size=200, rebuild_interval=10)
    for i in range(100):
        bank.append(_rand_embedding(i), FrameMetadata(frame_idx=i))
    for target in (0, 33, 77, 99):
        i, s, _ = bank.retrieve(_rand_embedding(target), k=1)
        assert i[0] == target
        assert s[0] == pytest.approx(1.0, abs=1e-4)


def test_automatic_rebuild_every_interval() -> None:
    bank = MemoryBank(embed_dim=D, max_size=200, rebuild_interval=50)
    for i in range(49):
        bank.append(_rand_embedding(i), FrameMetadata(frame_idx=i))
    assert bank._index is None  # no rebuild yet
    bank.append(_rand_embedding(49), FrameMetadata(frame_idx=49))
    assert bank._index is not None  # rebuild triggered at 50 appends


def test_get_window_returns_contiguous_slice() -> None:
    bank = MemoryBank(embed_dim=D, max_size=100, rebuild_interval=1000)
    for i in range(20):
        bank.append(_rand_embedding(i), FrameMetadata(frame_idx=i))
    emb, metas = bank.get_window(5, 4)
    assert emb.shape == (4, D)
    assert [m.frame_idx for m in metas] == [5, 6, 7, 8]
    # Rows are already L2-normalised.
    for row in emb:
        assert float(np.linalg.norm(row)) == pytest.approx(1.0, abs=1e-5)


def test_get_window_out_of_range_raises() -> None:
    bank = MemoryBank(embed_dim=D, max_size=100, rebuild_interval=1000)
    for i in range(5):
        bank.append(_rand_embedding(i), FrameMetadata(frame_idx=i))
    with pytest.raises(IndexError):
        bank.get_window(3, 10)


def test_append_rejects_zero_norm_embedding() -> None:
    bank = MemoryBank(embed_dim=D, max_size=100, rebuild_interval=1000)
    with pytest.raises(ValueError, match="near-zero norm"):
        bank.append(torch.zeros(D), FrameMetadata(frame_idx=0))


def test_append_rejects_wrong_dim() -> None:
    bank = MemoryBank(embed_dim=D, max_size=100, rebuild_interval=1000)
    with pytest.raises(ValueError, match="embedding dim"):
        bank.append(torch.randn(D + 1), FrameMetadata(frame_idx=0))


def test_growth_on_overflow() -> None:
    bank = MemoryBank(embed_dim=D, max_size=4, rebuild_interval=1000)
    for i in range(10):
        bank.append(_rand_embedding(i), FrameMetadata(frame_idx=i))
    assert len(bank) == 10
    assert bank.max_size >= 10  # grew as needed
    # Verify retrieval still works correctly after growth.
    indices, scores, _ = bank.retrieve(_rand_embedding(7), k=1)
    assert indices[0] == 7
    assert scores[0] == pytest.approx(1.0, abs=1e-4)
