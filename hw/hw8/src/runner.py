"""Benchmark all vector DBs on a unified protocol and dump results.csv.

Usage:
  python src/runner.py                            # full: 523K corpus, ~1-2h
  python src/runner.py --subset 50000 --num-queries 1000   # quick dev
  python src/runner.py --only faiss_hnsw,qdrant
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Set, Tuple

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from benchmarks.base import VectorDB
from benchmarks.chroma_db import ChromaDB
from benchmarks.faiss_flat import FaissFlatDB
from benchmarks.faiss_hnsw import FaissHNSWDB
from benchmarks.pgvector_db import PgvectorDB
from benchmarks.qdrant_db import QdrantDB
from metrics import mrr_at_k, percentiles, recall_at_k

DIM = 1536
TOP_K = 10
WARMUP_QUERIES = 50
NUM_REPEATS = 3
EF_SWEEP = [16, 32, 64, 128, 256]


def load_qrels(path: Path) -> Dict[str, Set[str]]:
    qrels: Dict[str, Set[str]] = {}
    with path.open() as f:
        next(f)  # header
        for line in f:
            qid, did, score = line.strip().split("\t")
            if int(score) > 0:
                qrels.setdefault(qid, set()).add(did)
    return qrels


def load_data(data_dir: Path, subset: int | None) -> tuple:
    print("loading corpus + queries + qrels...")
    corpus = np.load(data_dir / "corpus_embeddings.npy", mmap_mode="r")
    corpus_ids = json.load((data_dir / "corpus_ids.json").open())
    queries = np.load(data_dir / "query_embeddings.npy", mmap_mode="r")
    query_ids = json.load((data_dir / "query_ids.json").open())
    qrels = load_qrels(data_dir / "qrels.tsv")

    if subset:
        corpus = np.ascontiguousarray(corpus[:subset], dtype=np.float32)
        corpus_ids = corpus_ids[:subset]
        corpus_set = set(corpus_ids)
        # filter qrels: keep only relevant docs that are in subset
        qrels = {q: (r & corpus_set) for q, r in qrels.items()}
        qrels = {q: r for q, r in qrels.items() if r}
    else:
        corpus = np.ascontiguousarray(corpus, dtype=np.float32)

    # keep only queries that HAVE qrels (otherwise recall is meaningless)
    query_set = set(qrels.keys())
    keep_idx = [i for i, qid in enumerate(query_ids) if qid in query_set]
    queries = np.ascontiguousarray(queries[keep_idx], dtype=np.float32)
    query_ids = [query_ids[i] for i in keep_idx]

    print(f"  corpus={len(corpus_ids):,d}  queries={len(query_ids):,d}  qrels-queries={len(qrels):,d}")
    return corpus, corpus_ids, queries, query_ids, qrels


def measure_db(
    db: VectorDB,
    queries: np.ndarray,
    query_ids: List[str],
    qrels: Dict[str, Set[str]],
    flat_topk: Dict[str, List[str]] | None,
    *,
    num_queries: int | None = None,
) -> Dict:
    """Run warmup + 3 repeats. Returns metrics dict (without index_time / disk)."""
    if num_queries:
        queries = queries[:num_queries]
        query_ids = query_ids[:num_queries]

    n_q = len(queries)

    # WARMUP
    for i in range(min(WARMUP_QUERIES, n_q)):
        db.search(queries[i], top_k=TOP_K)

    all_latencies: List[List[float]] = []
    retrieved_per_query: List[List[str]] = [[] for _ in range(n_q)]

    for repeat in range(NUM_REPEATS):
        latencies = []
        for i in tqdm(range(n_q), desc=f"  {db.name} [rep {repeat+1}/{NUM_REPEATS}]",
                      leave=False, mininterval=2.0):
            t0 = time.perf_counter()
            results = db.search(queries[i], top_k=TOP_K)
            latencies.append((time.perf_counter() - t0) * 1000)
            if repeat == 0:
                retrieved_per_query[i] = [d for d, _ in results]
        all_latencies.append(latencies)

    lat_arr = np.median(np.array(all_latencies), axis=0)
    pcts = percentiles(lat_arr)

    recalls_q, mrrs_q = [], []
    for i, qid in enumerate(query_ids):
        rel = qrels.get(qid, set())
        if not rel:
            continue
        recalls_q.append(recall_at_k(retrieved_per_query[i], rel, TOP_K))
        mrrs_q.append(mrr_at_k(retrieved_per_query[i], rel, TOP_K))

    recall_flat = None
    if flat_topk is not None:
        recalls_f = []
        for i, qid in enumerate(query_ids):
            flat_set = set(flat_topk.get(qid, []))
            if not flat_set:
                continue
            hits = len(set(retrieved_per_query[i]) & flat_set)
            recalls_f.append(hits / min(TOP_K, len(flat_set)))
        recall_flat = float(np.mean(recalls_f)) if recalls_f else None

    return {
        "num_queries": n_q,
        "latency_p50_ms": round(pcts["p50"], 3),
        "latency_p95_ms": round(pcts["p95"], 3),
        "latency_p99_ms": round(pcts["p99"], 3),
        "latency_mean_ms": round(pcts["mean"], 3),
        "recall_qrels@10": round(float(np.mean(recalls_q)), 4) if recalls_q else 0.0,
        "mrr_qrels@10": round(float(np.mean(mrrs_q)), 4) if mrrs_q else 0.0,
        "recall_flat@10": round(recall_flat, 4) if recall_flat is not None else None,
    }


def benchmark_db(
    db: VectorDB,
    corpus: np.ndarray,
    corpus_ids: List[str],
    queries: np.ndarray,
    query_ids: List[str],
    qrels: Dict[str, Set[str]],
    flat_topk: Dict[str, List[str]] | None,
    num_queries: int | None,
) -> Dict:
    """Full bench: index + measure."""
    print(f"\n=== {db.name} ===")
    t0 = time.perf_counter()
    db.index(corpus, corpus_ids)
    index_time = round(time.perf_counter() - t0, 2)
    disk = round(db.disk_size_mb(), 1)
    print(f"  index_time={index_time}s  disk={disk} MB")

    m = measure_db(db, queries, query_ids, qrels, flat_topk, num_queries=num_queries)
    return {
        "db": db.name,
        "index_time_sec": index_time,
        "disk_mb": disk,
        **m,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data", type=Path)
    ap.add_argument("--output", default="results/results.csv", type=Path)
    ap.add_argument("--subset", type=int, default=None,
                    help="use only first N corpus rows (default: full 523K)")
    ap.add_argument("--num-queries", type=int, default=None,
                    help="cap queries (default: all with qrels)")
    ap.add_argument("--only", type=str, default=None,
                    help="comma-list of dbs: faiss_flat,faiss_hnsw,chroma,qdrant,pgvector")
    args = ap.parse_args()

    corpus, corpus_ids, queries, query_ids, qrels = load_data(args.data_dir, args.subset)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    only = set(args.only.split(",")) if args.only else None

    rows: List[Dict] = []

    flat_topk: Dict[str, List[str]] | None = None

    # ---- FAISS Flat (baseline + golden truth for recall_vs_flat) ----
    if only is None or "faiss_flat" in only:
        flat = FaissFlatDB(dim=DIM)
        row = benchmark_db(flat, corpus, corpus_ids, queries, query_ids, qrels, None, args.num_queries)
        rows.append(row)
        # collect flat top-K for downstream recall_vs_flat
        flat_topk = {}
        for i in range(args.num_queries or len(queries)):
            res = flat.search(queries[i], top_k=TOP_K)
            flat_topk[query_ids[i]] = [d for d, _ in res]
        flat.cleanup()
    else:
        print("[warn] faiss_flat skipped → recall_flat won't be computed")

    # ---- FAISS HNSW: build once, sweep ef ----
    if only is None or "faiss_hnsw" in only:
        hnsw = FaissHNSWDB(dim=DIM, M=32, ef_construction=200, ef_search=EF_SWEEP[0])
        print(f"\n=== {hnsw.name} (will sweep ef) ===")
        t0 = time.perf_counter()
        hnsw.index(corpus, corpus_ids)
        idx_t = round(time.perf_counter() - t0, 2)
        disk = round(hnsw.disk_size_mb(), 1)
        print(f"  index_time={idx_t}s  disk={disk} MB")
        for ef in EF_SWEEP:
            hnsw.set_ef(ef)
            m = measure_db(hnsw, queries, query_ids, qrels, flat_topk, num_queries=args.num_queries)
            rows.append({"db": hnsw.name, "index_time_sec": idx_t, "disk_mb": disk, **m})
            print(f"  ef={ef}: p50={m['latency_p50_ms']}ms p95={m['latency_p95_ms']}ms "
                  f"recall_qrels={m['recall_qrels@10']} recall_flat={m['recall_flat@10']}")
        hnsw.cleanup()

    # ---- Chroma: single config (no ef sweep without rebuild) ----
    if only is None or "chroma" in only:
        chroma = ChromaDB(dim=DIM, M=32, ef_construction=200, ef_search=64)
        row = benchmark_db(chroma, corpus, corpus_ids, queries, query_ids, qrels, flat_topk, args.num_queries)
        rows.append(row)
        chroma.cleanup()

    # ---- Qdrant: build once, sweep ef ----
    if only is None or "qdrant" in only:
        qd = QdrantDB(dim=DIM, M=32, ef_construction=200, ef_search=EF_SWEEP[0])
        print(f"\n=== {qd.name} (will sweep ef) ===")
        t0 = time.perf_counter()
        qd.index(corpus, corpus_ids)
        idx_t = round(time.perf_counter() - t0, 2)
        disk = round(qd.disk_size_mb(), 1)
        print(f"  index_time={idx_t}s  disk={disk} MB")
        for ef in EF_SWEEP:
            qd.set_ef(ef)
            m = measure_db(qd, queries, query_ids, qrels, flat_topk, num_queries=args.num_queries)
            rows.append({"db": qd.name, "index_time_sec": idx_t, "disk_mb": disk, **m})
            print(f"  ef={ef}: p50={m['latency_p50_ms']}ms p95={m['latency_p95_ms']}ms "
                  f"recall_qrels={m['recall_qrels@10']} recall_flat={m['recall_flat@10']}")
        qd.cleanup()

    # ---- pgvector: build once, sweep ef ----
    if only is None or "pgvector" in only:
        pg = PgvectorDB(dim=DIM, M=32, ef_construction=200, ef_search=EF_SWEEP[0])
        print(f"\n=== {pg.name} (will sweep ef) ===")
        t0 = time.perf_counter()
        pg.index(corpus, corpus_ids)
        idx_t = round(time.perf_counter() - t0, 2)
        disk = round(pg.disk_size_mb(), 1)
        print(f"  index_time={idx_t}s  disk={disk} MB")
        for ef in EF_SWEEP:
            pg.set_ef(ef)
            m = measure_db(pg, queries, query_ids, qrels, flat_topk, num_queries=args.num_queries)
            rows.append({"db": pg.name, "index_time_sec": idx_t, "disk_mb": disk, **m})
            print(f"  ef={ef}: p50={m['latency_p50_ms']}ms p95={m['latency_p95_ms']}ms "
                  f"recall_qrels={m['recall_qrels@10']} recall_flat={m['recall_flat@10']}")
        pg.cleanup()

    # ---- write CSV ----
    field_order = [
        "db", "num_queries", "index_time_sec", "disk_mb",
        "latency_p50_ms", "latency_p95_ms", "latency_p99_ms", "latency_mean_ms",
        "recall_qrels@10", "mrr_qrels@10", "recall_flat@10",
    ]
    with args.output.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=field_order)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in field_order})

    print(f"\n[done] {len(rows)} rows → {args.output}")


if __name__ == "__main__":
    main()
