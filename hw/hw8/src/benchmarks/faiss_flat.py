"""FAISS Flat (brute-force inner product) — exact, slow, 100% recall.

Baseline against which all approximate methods are measured.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np

from .base import VectorDB, assert_index_inputs


class FaissFlatDB(VectorDB):
    name = "FAISS-Flat"

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.index_obj: faiss.Index | None = None
        self.int_to_id: List[str] = []
        self._disk_path: Path | None = None

    def index(self, vectors: np.ndarray, ids: List[str]) -> None:
        assert_index_inputs(vectors, ids)
        if vectors.shape[1] != self.dim:
            raise ValueError(f"expected dim={self.dim}, got {vectors.shape[1]}")
        self.index_obj = faiss.IndexFlatIP(self.dim)
        # IndexFlat is implicitly contiguous: id == row index
        self.index_obj.add(vectors)
        self.int_to_id = list(ids)

    def search(self, query_vec: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        assert self.index_obj is not None, "call .index() first"
        q = query_vec.reshape(1, -1).astype(np.float32, copy=False)
        scores, idx = self.index_obj.search(q, top_k)
        return [(self.int_to_id[i], float(s)) for i, s in zip(idx[0], scores[0]) if i != -1]

    def disk_size_mb(self) -> float:
        assert self.index_obj is not None, "call .index() first"
        if self._disk_path is None:
            tmp = tempfile.NamedTemporaryFile(suffix=".faiss", delete=False)
            tmp.close()
            self._disk_path = Path(tmp.name)
            faiss.write_index(self.index_obj, str(self._disk_path))
        return self._disk_path.stat().st_size / (1024 * 1024)

    def cleanup(self) -> None:
        if self._disk_path and self._disk_path.exists():
            self._disk_path.unlink()
            self._disk_path = None
