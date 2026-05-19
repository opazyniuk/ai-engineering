"""
HNSW retriever — separate clean process. Compared against dense brute-force baseline.

Run after run_fixes.py. Appends rows to results/fixes.csv.
"""
import csv
import json
import time
from pathlib import Path

import faiss
import numpy as np
import psutil

from metrics import evaluate
from retriever import select_subset_indices


ROOT = Path(__file__).parent.parent
CACHE = ROOT / "data" / "cache"
RESULTS = ROOT / "results"

SIZES = [100_000, 300_000]
TOP_K = 10
HNSW_M = 16
HNSW_EF_CONSTRUCTION = 200
HNSW_EF_SEARCH = 64


def rss_mb() -> float:
    return psutil.Process().memory_info().rss / (1024 ** 2)


def run_hnsw_size(size, embeddings, corpus_ids, query_vecs, eval_set):
    print(f"\n==================== HNSW size={size} ====================")
    relevant_ids = {rid for e in eval_set for rid in e["relevant_ids"]}
    idx = select_subset_indices(corpus_ids, relevant_ids, size)
    sub_emb = np.ascontiguousarray(embeddings[idx]).astype(np.float32)
    sub_ids = [corpus_ids[i] for i in idx]
    print(f"  subset {sub_emb.shape}, RSS={rss_mb():.0f}MB")

    t0 = time.perf_counter()
    index = faiss.IndexHNSWFlat(sub_emb.shape[1], HNSW_M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
    index.hnsw.efSearch = HNSW_EF_SEARCH
    index.add(sub_emb)
    build_time = time.perf_counter() - t0
    print(f"  built HNSW in {build_time:.1f}s, RSS={rss_mb():.0f}MB")

    # Warmup
    _, _ = index.search(query_vecs[:1].astype(np.float32), TOP_K)

    lats = []
    retrieved: list[list[str]] = []
    for qvec in query_vecs:
        t0 = time.perf_counter()
        _, ids = index.search(qvec[None, :].astype(np.float32), TOP_K)
        lats.append((time.perf_counter() - t0) * 1000)
        retrieved.append([sub_ids[i] for i in ids[0] if i >= 0])

    metrics = evaluate(eval_set, retrieved, ks=(1, 5, 10))
    p50, p95, p99 = (float(np.percentile(lats, p)) for p in (50, 95, 99))
    qps = len(lats) / sum(lats) * 1000

    row = {
        "size": size,
        "retriever": f"hnsw_M{HNSW_M}_ef{HNSW_EF_SEARCH}",
        **metrics,
        "latency_p50_ms": round(p50, 2),
        "latency_p95_ms": round(p95, 2),
        "latency_p99_ms": round(p99, 2),
        "throughput_qps": round(qps, 1),
        "build_time_s": round(build_time, 1),
        "rss_mb": round(rss_mb(), 0),
    }
    print(f"  -> {row['retriever']}: {metrics} | p50={p50:.2f}ms p95={p95:.2f}ms | build={build_time:.1f}s")
    return row


def main():
    embeddings = np.load(CACHE / "embeddings_corpus.npy")
    corpus_ids = json.load(open(CACHE / "corpus_ids.json"))
    query_vecs = np.load(CACHE / "embeddings_queries.npy")
    eval_set = json.load(open(CACHE / "queries_meta.json"))

    new_rows = [run_hnsw_size(s, embeddings, corpus_ids, query_vecs, eval_set) for s in SIZES]

    fixes_csv = RESULTS / "fixes.csv"
    existing = []
    if fixes_csv.exists():
        with open(fixes_csv) as f:
            existing = list(csv.DictReader(f))
    all_rows = existing + [{k: str(v) for k, v in r.items()} for r in new_rows]
    with open(fixes_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(new_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nAppended HNSW rows to {fixes_csv}")


if __name__ == "__main__":
    main()
