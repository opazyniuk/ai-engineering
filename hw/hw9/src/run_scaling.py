"""
Baseline scaling experiment: Dense brute-force retriever over 1K → 300K corpus sizes.

Outputs:
  results/baseline.csv with one row per (size, retriever) — metrics + latency + RAM.
  results/screenshots/step4_scaling.log via stdout tee.

Run from hw/hw9/:
  .venv/bin/python -u src/run_scaling.py
"""
import csv
import json
import time
from pathlib import Path

import numpy as np
import psutil

from metrics import evaluate
from retriever import DenseRetriever, select_subset_indices


ROOT = Path(__file__).parent.parent
CACHE = ROOT / "data" / "cache"
RESULTS = ROOT / "results"

SUBSET_SIZES = [1_000, 10_000, 100_000, 300_000]
TOP_K = 10
LATENCY_REPEATS = 3   # per query, take min — reduces OS noise


def load_artifacts():
    embeddings = np.load(CACHE / "embeddings_corpus.npy")
    corpus_ids = json.load(open(CACHE / "corpus_ids.json"))
    query_vecs = np.load(CACHE / "embeddings_queries.npy")
    eval_set = json.load(open(CACHE / "queries_meta.json"))
    return embeddings, corpus_ids, query_vecs, eval_set


def rss_mb() -> float:
    return psutil.Process().memory_info().rss / (1024 ** 2)


def measure_query_latency(retriever, query_vec, top_k, repeats):
    """Return (best_latency_ms, retrieved_ids) — min over repeats kills OS noise."""
    best = float("inf")
    retrieved = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        ret = retriever.search(query_vec, top_k)
        dt = (time.perf_counter() - t0) * 1000
        if dt < best:
            best = dt
            retrieved = ret
    return best, retrieved


def run_size(size: int, embeddings, corpus_ids, query_vecs, eval_set):
    print(f"\n=== size={size} ===")
    relevant_ids = {rid for e in eval_set for rid in e["relevant_ids"]}
    idx = select_subset_indices(corpus_ids, relevant_ids, size)
    sub_emb = np.ascontiguousarray(embeddings[idx])    # contiguous → BLAS-friendly
    sub_ids = [corpus_ids[i] for i in idx]
    print(f"  subset: {sub_emb.shape}, RAM after slice: {rss_mb():.0f} MB")

    retriever = DenseRetriever(sub_emb, sub_ids)

    # Warm-up: one full pass, discard timings (BLAS JIT, page faults)
    _ = retriever.search(query_vecs[0], TOP_K)

    latencies_ms = []
    retrieved_per_query: list[list[str]] = []
    t_start = time.perf_counter()
    for i, qvec in enumerate(query_vecs):
        lat, ret = measure_query_latency(retriever, qvec, TOP_K, LATENCY_REPEATS)
        latencies_ms.append(lat)
        retrieved_per_query.append(ret)
    total_s = time.perf_counter() - t_start

    metrics = evaluate(eval_set, retrieved_per_query, ks=(1, 5, 10))
    lat_arr = np.array(latencies_ms)
    p50, p95, p99 = float(np.percentile(lat_arr, 50)), float(np.percentile(lat_arr, 95)), float(np.percentile(lat_arr, 99))
    throughput = len(query_vecs) / total_s

    row = {
        "size": size,
        "retriever": "dense_bruteforce",
        **metrics,
        "latency_p50_ms": round(p50, 2),
        "latency_p95_ms": round(p95, 2),
        "latency_p99_ms": round(p99, 2),
        "throughput_qps": round(throughput, 1),
        "rss_mb": round(rss_mb(), 0),
        "index_mb": round(sub_emb.nbytes / 1024 ** 2, 1),
    }
    print(f"  metrics: {metrics}")
    print(f"  latency p50={p50:.2f} ms, p95={p95:.2f} ms, p99={p99:.2f} ms, qps={throughput:.1f}")
    print(f"  index_mb={row['index_mb']}, rss_mb={row['rss_mb']}")
    return row


def main():
    print("Loading cached embeddings...")
    embeddings, corpus_ids, query_vecs, eval_set = load_artifacts()
    print(f"  corpus: {embeddings.shape}, queries: {query_vecs.shape}, eval: {len(eval_set)}")
    print(f"  RAM after load: {rss_mb():.0f} MB")

    rows = []
    for size in SUBSET_SIZES:
        rows.append(run_size(size, embeddings, corpus_ids, query_vecs, eval_set))

    out_path = RESULTS / "baseline.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
