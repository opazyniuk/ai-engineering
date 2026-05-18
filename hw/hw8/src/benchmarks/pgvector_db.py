"""pgvector — Postgres extension. Runs in Docker (hw_pgvector).

Strategy: CREATE TABLE → COPY all vectors → CREATE HNSW INDEX (faster than
incremental indexing). ef_search is per-session via `SET hnsw.ef_search`.
"""

from __future__ import annotations

import io
import os
import struct
import time
from typing import List, Tuple

import numpy as np
import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from tqdm import tqdm

from .base import VectorDB, assert_index_inputs

load_dotenv()

PG_DSN = (
    f"host={os.environ.get('POSTGRES_HOST', 'localhost')} "
    f"port={os.environ.get('POSTGRES_PORT', '5434')} "
    f"user={os.environ.get('POSTGRES_USER', 'bench')} "
    f"password={os.environ.get('POSTGRES_PASSWORD', 'bench')} "
    f"dbname={os.environ.get('POSTGRES_DB', 'bench')}"
)
TABLE = "bench"


class PgvectorDB(VectorDB):
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
        self.name = f"pgvector(m={M},efC={ef_construction},efS={ef_search})"
        self.int_to_id: List[str] = []
        self.conn: psycopg.Connection | None = None

    def _connect(self) -> psycopg.Connection:
        conn = psycopg.connect(PG_DSN, autocommit=True)
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(conn)
        return conn

    def index(self, vectors: np.ndarray, ids: List[str]) -> None:
        assert_index_inputs(vectors, ids)
        self.conn = self._connect()
        cur = self.conn.cursor()

        # clean state
        cur.execute(f"DROP TABLE IF EXISTS {TABLE}")
        cur.execute(f"CREATE TABLE {TABLE} (id INTEGER PRIMARY KEY, embedding vector({self.dim}))")

        self.int_to_id = list(ids)
        n = len(ids)

        # COPY BINARY — fastest bulk path
        print(f"  pgvector COPY {n:,d} rows...")
        t0 = time.perf_counter()
        with cur.copy(f"COPY {TABLE} (id, embedding) FROM STDIN WITH (FORMAT BINARY)") as copy:
            copy.set_types(["int4", "vector"])
            for i in tqdm(range(n), desc="  copy", unit="row", mininterval=2.0):
                copy.write_row((i, vectors[i]))
        copy_secs = time.perf_counter() - t0
        print(f"  COPY done in {copy_secs:.1f}s ({n/copy_secs:.0f} rows/sec)")

        # CREATE INDEX after data is loaded
        print(f"  CREATE INDEX HNSW (m={self.M}, ef_construction={self.ef_construction})...")
        t0 = time.perf_counter()
        cur.execute(
            f"CREATE INDEX ON {TABLE} USING hnsw (embedding vector_cosine_ops) "
            f"WITH (m = {self.M}, ef_construction = {self.ef_construction})"
        )
        print(f"  INDEX built in {time.perf_counter()-t0:.1f}s")

        cur.execute(f"SET hnsw.ef_search = {self.ef_search}")
        # ANALYZE for planner
        cur.execute(f"ANALYZE {TABLE}")

    def set_ef(self, ef_search: int) -> None:
        assert self.conn is not None
        self.ef_search = ef_search
        self.name = f"pgvector(m={self.M},efC={self.ef_construction},efS={ef_search})"
        self.conn.execute(f"SET hnsw.ef_search = {ef_search}")

    def search(self, query_vec: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        assert self.conn is not None
        qv = query_vec.astype(np.float32, copy=False)
        # <=> returns cosine distance ∈ [0, 2]; similarity = 1 - distance
        rows = self.conn.execute(
            f"SELECT id, embedding <=> %s AS d FROM {TABLE} ORDER BY d LIMIT %s",
            (qv, top_k),
        ).fetchall()
        return [(self.int_to_id[int(_id)], 1.0 - float(d)) for _id, d in rows]

    def disk_size_mb(self) -> float:
        assert self.conn is not None
        row = self.conn.execute(
            f"SELECT pg_total_relation_size('{TABLE}')"
        ).fetchone()
        return int(row[0]) / (1024 * 1024)

    def cleanup(self) -> None:
        if self.conn is not None:
            try:
                self.conn.execute(f"DROP TABLE IF EXISTS {TABLE}")
            except Exception:
                pass
            self.conn.close()
            self.conn = None
