"""FAISS HNSW (Hierarchical Navigable Small World) — approximate, fast.

Key knob for Pareto frontier: `ef_search`. Change it between query passes
without rebuilding the index.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np

from .base import VectorDB, assert_index_inputs


class FaissHNSWDB(VectorDB):
    def __init__(
        self,
        dim: int,
        M: int = 32,
        ef_construction: int = 200,
        ef_search: int = 64,
    ) -> None:
        self.dim = dim
        self.M = M
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.name = f"FAISS-HNSW(M={M},efC={ef_construction},efS={ef_search})"
        self.index_obj: faiss.IndexHNSWFlat | None = None
        self.int_to_id: List[str] = []
        self._disk_path: Path | None = None

    def index(self, vectors: np.ndarray, ids: List[str]) -> None:
        assert_index_inputs(vectors, ids)
        if vectors.shape[1] != self.dim:
            raise ValueError(f"expected dim={self.dim}, got {vectors.shape[1]}")
        # METRIC_INNER_PRODUCT because our vectors are L2-normalized → cosine
        self.index_obj = faiss.IndexHNSWFlat(self.dim, self.M, faiss.METRIC_INNER_PRODUCT)
        self.index_obj.hnsw.efConstruction = self.ef_construction
        self.index_obj.hnsw.efSearch = self.ef_search
        self.index_obj.add(vectors)
        self.int_to_id = list(ids)

    def set_ef(self, ef_search: int) -> None:
        """Change ef_search without rebuilding the index."""
        assert self.index_obj is not None, "call .index() first"
        self.ef_search = ef_search
        self.index_obj.hnsw.efSearch = ef_search
        self.name = f"FAISS-HNSW(M={self.M},efC={self.ef_construction},efS={ef_search})"

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
