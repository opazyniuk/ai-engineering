"""Chroma — embedded persistent vector DB (SQLite + hnswlib).

Stores data under ./chroma_db/. cleanup() deletes that directory.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import List, Tuple

import chromadb
import numpy as np
from tqdm import tqdm

from .base import VectorDB, assert_index_inputs

ADD_BATCH = 1000  # well under Chroma's ~5461 hard limit


class ChromaDB(VectorDB):
    def __init__(
        self,
        dim: int,
        M: int = 32,
        ef_construction: int = 200,
        ef_search: int = 64,
        persist_dir: str | Path = "chroma_db",
    ) -> None:
        self.dim = dim
        self.M = M
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.persist_dir = Path(persist_dir)
        self.name = f"Chroma(M={M},efC={ef_construction},efS={ef_search})"
        self.client: chromadb.ClientAPI | None = None
        self.collection = None

    def index(self, vectors: np.ndarray, ids: List[str]) -> None:
        assert_index_inputs(vectors, ids)
        # start from clean state — otherwise disk_size includes leftovers
        if self.persist_dir.exists():
            shutil.rmtree(self.persist_dir)
        self.persist_dir.mkdir(parents=True)

        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.collection = self.client.create_collection(
            name="bench",
            metadata={
                "hnsw:space": "cosine",
                "hnsw:M": self.M,
                "hnsw:construction_ef": self.ef_construction,
                "hnsw:search_ef": self.ef_search,
            },
        )

        n = len(ids)
        for start in tqdm(range(0, n, ADD_BATCH), desc="chroma.add", unit="batch"):
            end = min(start + ADD_BATCH, n)
            self.collection.add(
                ids=ids[start:end],
                embeddings=vectors[start:end].tolist(),
            )

    def search(self, query_vec: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        assert self.collection is not None, "call .index() first"
        q = query_vec.reshape(1, -1).astype(np.float32, copy=False).tolist()
        res = self.collection.query(query_embeddings=q, n_results=top_k)
        # res["ids"] is List[List[str]], res["distances"] is List[List[float]]
        ids = res["ids"][0]
        dists = res["distances"][0]
        # for "cosine" Chroma returns 1 - cosine_similarity, convert back to similarity
        return [(_id, 1.0 - float(d)) for _id, d in zip(ids, dists)]

    def disk_size_mb(self) -> float:
        if not self.persist_dir.exists():
            return 0.0
        total = 0
        for f in self.persist_dir.rglob("*"):
            if f.is_file():
                total += f.stat().st_size
        return total / (1024 * 1024)

    def cleanup(self) -> None:
        self.client = None
        self.collection = None
        if self.persist_dir.exists():
            shutil.rmtree(self.persist_dir)
