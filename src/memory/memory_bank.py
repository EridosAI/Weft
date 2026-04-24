"""Append-only memory bank with FAISS IndexFlatIP retrieval (Tier A, Session 4).

Every experienced frame's encoder embedding is stored contiguously in CPU
memory along with per-frame metadata (frame_idx, stage, config,
transition_zone_flag, plus arbitrary extra keys). No eviction, no
subsampling, no compression in Tier A.

Retrieval is exact cosine similarity implemented as Inner Product on
L2-normalised vectors via `faiss.IndexFlatIP`. The index is rebuilt every
`rebuild_interval` appends (default 1000) rather than on every insert, for
efficiency; callers can also call `rebuild_index()` explicitly.

The backing store is a pre-allocated NumPy array sized to `max_size`
(default 200 000). If the store fills, it doubles in size. Never shrinks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np
import torch


_EPS = 1e-12


@dataclass
class FrameMetadata:
    frame_idx: int
    stage: str = "0a"
    config: str = "default"
    transition_zone_flag: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)


def _to_float32_numpy(embedding: Any) -> np.ndarray:
    """Normalise input to a contiguous float32 numpy row vector."""
    if isinstance(embedding, torch.Tensor):
        vec = embedding.detach().to(torch.float32).cpu().numpy()
    else:
        vec = np.asarray(embedding, dtype=np.float32)
    if vec.ndim == 2 and vec.shape[0] == 1:
        vec = vec[0]
    if vec.ndim != 1:
        raise ValueError(f"embedding must be 1-D (or (1, D)); got shape {vec.shape}")
    return np.ascontiguousarray(vec, dtype=np.float32)


def _l2_normalise_rows(x: np.ndarray) -> np.ndarray:
    """L2-normalise each row of a 2-D array. Returns a new array."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, _EPS)
    return x / norms


class MemoryBank:
    def __init__(
        self,
        embed_dim: int = 1024,
        max_size: int = 200_000,
        rebuild_interval: int = 1000,
    ) -> None:
        if embed_dim <= 0 or max_size <= 0 or rebuild_interval <= 0:
            raise ValueError("embed_dim, max_size, rebuild_interval must all be positive")
        self.embed_dim = embed_dim
        self.max_size = max_size
        self.rebuild_interval = rebuild_interval

        self._store: np.ndarray = np.zeros((max_size, embed_dim), dtype=np.float32)
        self._metadata: List[FrameMetadata] = []
        self._size: int = 0
        self._last_index_size: int = 0
        self._index: Optional[faiss.Index] = None

    def __len__(self) -> int:
        return self._size

    @property
    def size(self) -> int:
        return self._size

    def _ensure_capacity(self) -> None:
        if self._size >= self.max_size:
            new_max = self.max_size * 2
            new_store = np.zeros((new_max, self.embed_dim), dtype=np.float32)
            new_store[: self.max_size] = self._store
            self._store = new_store
            self.max_size = new_max

    def append(self, embedding: Any, metadata: FrameMetadata | Dict[str, Any]) -> int:
        """Append one embedding with its metadata. Returns the assigned frame index (0-based)."""
        self._ensure_capacity()
        vec = _to_float32_numpy(embedding)
        if vec.shape[0] != self.embed_dim:
            raise ValueError(f"embedding dim {vec.shape[0]} != bank embed_dim {self.embed_dim}")

        # L2-normalise before storage so FAISS IndexFlatIP gives exact cosine similarity.
        # This is explicit, tested, and commented per CODING_STANDARDS.md §7.4.
        norm = float(np.linalg.norm(vec))
        if norm < _EPS:
            raise ValueError("embedding has near-zero norm; cannot normalise")
        self._store[self._size] = vec / norm

        if not isinstance(metadata, FrameMetadata):
            metadata = FrameMetadata(**metadata)
        self._metadata.append(metadata)

        idx = self._size
        self._size += 1

        if self._size - self._last_index_size >= self.rebuild_interval:
            self.rebuild_index()

        return idx

    def rebuild_index(self) -> None:
        """Rebuild the FAISS IndexFlatIP from the current store contents."""
        index = faiss.IndexFlatIP(self.embed_dim)
        if self._size > 0:
            index.add(self._store[: self._size])
        self._index = index
        self._last_index_size = self._size

    def _ensure_index_current(self) -> None:
        if self._index is None or self._last_index_size != self._size:
            self.rebuild_index()

    def get_window(
        self, start_idx: int, window_size: int
    ) -> Tuple[np.ndarray, List[FrameMetadata]]:
        """Return a contiguous window of L2-normalised embeddings and their metadata."""
        if start_idx < 0 or window_size <= 0 or start_idx + window_size > self._size:
            raise IndexError(
                f"window [{start_idx}:{start_idx + window_size}) out of range for "
                f"bank of size {self._size}"
            )
        end = start_idx + window_size
        embeddings = self._store[start_idx:end].copy()
        metas = self._metadata[start_idx:end]
        return embeddings, metas

    def retrieve(
        self, query_embedding: Any, k: int
    ) -> Tuple[np.ndarray, np.ndarray, List[FrameMetadata]]:
        """Top-k nearest neighbours to `query_embedding` by cosine similarity.

        Returns (indices, scores, metadatas):
          - indices:  np.ndarray of shape (k,)
          - scores:   np.ndarray of shape (k,) — cosine similarity values
          - metadatas: list of FrameMetadata, length k
        """
        if self._size == 0:
            raise RuntimeError("memory bank is empty; cannot retrieve")
        if k <= 0:
            raise ValueError(f"k must be positive; got {k}")
        k = min(k, self._size)
        self._ensure_index_current()
        assert self._index is not None

        vec = _to_float32_numpy(query_embedding).reshape(1, self.embed_dim)
        vec = _l2_normalise_rows(vec)

        scores, indices = self._index.search(vec, k)  # each (1, k)
        indices = indices[0]
        scores = scores[0]
        metadatas = [self._metadata[i] for i in indices]
        return indices, scores, metadatas
