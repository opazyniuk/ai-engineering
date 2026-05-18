"""Abstract base for all vector DB wrappers used in the benchmark."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np


class VectorDB(ABC):
    """Common interface for FAISS / Qdrant / Chroma / pgvector wrappers.

    All implementations assume:
      * vectors are float32, L2-normalized → inner product == cosine similarity.
      * ids are strings (because qrels uses str ids).
    """

    name: str = "BASE"  # short label for plots/CSV

    @abstractmethod
    def index(self, vectors: np.ndarray, ids: List[str]) -> None:
        """Build the index from (N, dim) float32 vectors with parallel str ids."""

    @abstractmethod
    def search(self, query_vec: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        """Return [(doc_id, score), ...] of length up to top_k.

        query_vec is 1D — implementations must reshape if needed.
        score is the raw similarity (higher = better for cosine/IP).
        """

    @abstractmethod
    def disk_size_mb(self) -> float:
        """Index size on disk in MB. Return 0.0 for purely in-memory DBs."""

    def cleanup(self) -> None:  # noqa: B027 — intentional no-op default
        """Close connections, remove temp files. Default: no-op."""


def assert_index_inputs(vectors: np.ndarray, ids: List[str]) -> None:
    """Cheap defensive checks — catch dumb mistakes early."""
    if vectors.dtype != np.float32:
        raise TypeError(f"vectors must be float32, got {vectors.dtype}")
    if vectors.ndim != 2:
        raise ValueError(f"vectors must be 2D, got shape {vectors.shape}")
    if len(ids) != vectors.shape[0]:
        raise ValueError(f"len(ids)={len(ids)} but vectors has {vectors.shape[0]} rows")
