"""Qdrant — Rust-based vector DB, REST + gRPC. Runs in Docker (hw_qdrant)."""

from __future__ import annotations

import os
import subprocess
import time
from typing import List, Tuple

import numpy as np
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from .base import VectorDB, assert_index_inputs

load_dotenv()

QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))
QDRANT_GRPC_PORT = 6334
QDRANT_CONTAINER = "hw_qdrant"
COLLECTION = "bench"


class QdrantDB(VectorDB):
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
        self.name = f"Qdrant(m={M},efC={ef_construction},efS={ef_search})"
        self.int_to_id: List[str] = []
        self.client = QdrantClient(
            host=QDRANT_HOST,
            port=QDRANT_PORT,
            grpc_port=QDRANT_GRPC_PORT,
            prefer_grpc=True,
            timeout=120,
        )

    def index(self, vectors: np.ndarray, ids: List[str]) -> None:
        assert_index_inputs(vectors, ids)

        # clean state — drop collection if exists
        if self.client.collection_exists(COLLECTION):
            self.client.delete_collection(COLLECTION)

        self.client.create_collection(
            collection_name=COLLECTION,
            vectors_config=qm.VectorParams(
                size=self.dim,
                distance=qm.Distance.COSINE,
            ),
            hnsw_config=qm.HnswConfigDiff(
                m=self.M,
                ef_construct=self.ef_construction,
            ),
        )

        self.int_to_id = list(ids)
        n = len(ids)
        self.client.upload_collection(
            collection_name=COLLECTION,
            vectors=vectors,
            ids=list(range(n)),
            batch_size=256,
            parallel=4,
        )

        # wait for full indexing — async by default
        deadline = time.perf_counter() + 1800  # 30 min hard cap
        while time.perf_counter() < deadline:
            info = self.client.get_collection(COLLECTION)
            indexed = info.indexed_vectors_count or 0
            total = info.points_count or 0
            if indexed >= total and total == n:
                break
            time.sleep(1)
        else:
            raise TimeoutError(f"Qdrant indexing did not finish: indexed={indexed}/{n}")

    def search(self, query_vec: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        q = query_vec.astype(np.float32, copy=False).tolist()
        hits = self.client.query_points(
            collection_name=COLLECTION,
            query=q,
            limit=top_k,
            search_params=qm.SearchParams(hnsw_ef=self.ef_search),
            with_payload=False,
            with_vectors=False,
        ).points
        return [(self.int_to_id[int(h.id)], float(h.score)) for h in hits]

    def set_ef(self, ef_search: int) -> None:
        """Change ef_search without rebuilding the index."""
        self.ef_search = ef_search
        self.name = f"Qdrant(m={self.M},efC={self.ef_construction},efS={ef_search})"

    def disk_size_mb(self) -> float:
        try:
            r = subprocess.run(
                ["docker", "exec", QDRANT_CONTAINER, "du", "-sb",
                 f"/qdrant/storage/collections/{COLLECTION}"],
                capture_output=True, text=True, check=True, timeout=10,
            )
            return int(r.stdout.split()[0]) / (1024 * 1024)
        except Exception as e:
            print(f"[qdrant disk_size_mb] failed: {e}")
            return 0.0

    def cleanup(self) -> None:
        try:
            if self.client.collection_exists(COLLECTION):
                self.client.delete_collection(COLLECTION)
        except Exception:
            pass
