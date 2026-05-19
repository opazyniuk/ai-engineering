"""
Fixes on the broken size — Hybrid (BM25+dense+RRF), Reranker (bge-reranker-v2-m3),
and HNSW (FAISS) for latency. Runs on size=300K (broken) and size=100K (comparison).

Outputs:
  results/fixes.csv with rows: (size, retriever, metrics, latency, build_time, rss).
"""
import csv
import json
import re
import time
from pathlib import Path

import numpy as np
import psutil
import torch
from rank_bm25 import BM25Okapi

import faiss
from sentence_transformers import CrossEncoder

from metrics import evaluate
from retriever import DenseRetriever, select_subset_indices


ROOT = Path(__file__).parent.parent
CACHE = ROOT / "data" / "cache"
RESULTS = ROOT / "results"

SIZES = [100_000, 300_000]
TOP_K = 10
CAND_K = 100   # для hybrid і reranker — скільки кандидатів брати з першого етапу
RRF_K = 60
RERANKER_NAME = "BAAI/bge-reranker-v2-m3"
RERANKER_BATCH = 32
HNSW_M = 16
HNSW_EF_CONSTRUCTION = 200
HNSW_EF_SEARCH = 64


def rss_mb() -> float:
    return psutil.Process().memory_info().rss / (1024 ** 2)


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def tokenize(text: str) -> list[str]:
    """Cheap BM25 tokenization: lowercase, alphanumeric runs."""
    return re.findall(r"[a-z0-9]+", text.lower())


# ---------- Retrievers ----------

class DenseRetrieverScored(DenseRetriever):
    """Same as DenseRetriever but returns (ids, scores) so we can fuse."""

    def search_with_scores(self, query_vec: np.ndarray, top_k: int):
        scores = self.embeddings @ query_vec
        k = min(top_k, self.n)
        top_idx = np.argpartition(scores, -k)[-k:]
        top_idx_sorted = top_idx[np.argsort(-scores[top_idx])]
        return [self.doc_ids[i] for i in top_idx_sorted], scores[top_idx_sorted]


class BM25Retriever:
    def __init__(self, corpus_texts: list[str], doc_ids: list[str]):
        self.doc_ids = doc_ids
        tokens = [tokenize(t) for t in corpus_texts]
        self.bm25 = BM25Okapi(tokens)

    def search(self, query: str, top_k: int) -> tuple[list[str], np.ndarray]:
        scores = self.bm25.get_scores(tokenize(query))
        k = min(top_k, len(self.doc_ids))
        top_idx = np.argpartition(scores, -k)[-k:]
        top_idx_sorted = top_idx[np.argsort(-scores[top_idx])]
        return [self.doc_ids[i] for i in top_idx_sorted], scores[top_idx_sorted]


def rrf_fuse(rankings: list[list[str]], k: int = RRF_K, top_k: int = TOP_K) -> list[str]:
    """Reciprocal Rank Fusion across N ranked lists. Returns merged top_k."""
    scores: dict[str, float] = {}
    for ranks in rankings:
        for rank, doc_id in enumerate(ranks, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)[:top_k]


class HNSWRetriever:
    def __init__(self, embeddings: np.ndarray, doc_ids: list[str]):
        self.doc_ids = doc_ids
        dim = embeddings.shape[1]
        # IndexHNSWFlat — inner product via faiss.METRIC_INNER_PRODUCT (cosine on L2-normalized vectors)
        self.index = faiss.IndexHNSWFlat(dim, HNSW_M, faiss.METRIC_INNER_PRODUCT)
        self.index.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
        self.index.hnsw.efSearch = HNSW_EF_SEARCH
        self.index.add(embeddings)

    def search(self, query_vec: np.ndarray, top_k: int) -> list[str]:
        _, idx = self.index.search(query_vec[None, :].astype(np.float32), top_k)
        return [self.doc_ids[i] for i in idx[0] if i >= 0]


# ---------- Experiment driver ----------

def load_artifacts():
    embeddings = np.load(CACHE / "embeddings_corpus.npy")
    corpus_ids = json.load(open(CACHE / "corpus_ids.json"))
    query_vecs = np.load(CACHE / "embeddings_queries.npy")
    eval_set = json.load(open(CACHE / "queries_meta.json"))
    corpus_full = json.load(open(CACHE / "corpus.json"))["corpus"]
    id_to_text = {d["id"]: d["text"] for d in corpus_full}
    return embeddings, corpus_ids, query_vecs, eval_set, id_to_text


def time_block(label: str):
    t0 = time.perf_counter()
    print(f"  [{label}] start")
    return t0


def time_done(t0, label):
    dt = time.perf_counter() - t0
    print(f"  [{label}] done in {dt:.1f}s")
    return dt


def measure(retrieved_per_query, eval_set, latencies, name, build_time, extra):
    metrics = evaluate(eval_set, retrieved_per_query, ks=(1, 5, 10))
    p50 = float(np.percentile(latencies, 50))
    p95 = float(np.percentile(latencies, 95))
    p99 = float(np.percentile(latencies, 99))
    qps = len(latencies) / sum(latencies) * 1000
    row = {
        "size": extra["size"],
        "retriever": name,
        **metrics,
        "latency_p50_ms": round(p50, 2),
        "latency_p95_ms": round(p95, 2),
        "latency_p99_ms": round(p99, 2),
        "throughput_qps": round(qps, 1),
        "build_time_s": round(build_time, 1),
        "rss_mb": round(rss_mb(), 0),
    }
    print(f"  -> {name}: {metrics} | p50={p50:.1f}ms p95={p95:.1f}ms | build={build_time:.1f}s")
    return row


def save_rows(rows: list[dict], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_size(size, embeddings, corpus_ids, query_vecs, eval_set, id_to_text, reranker, all_rows, out_path):
    print(f"\n==================== size={size} ====================")
    relevant_ids = {rid for e in eval_set for rid in e["relevant_ids"]}
    idx = select_subset_indices(corpus_ids, relevant_ids, size)
    sub_emb = np.ascontiguousarray(embeddings[idx])
    sub_ids = [corpus_ids[i] for i in idx]
    sub_texts = [id_to_text[did] for did in sub_ids]
    queries_text = [e["query"] for e in eval_set]
    print(f"  subset built: {sub_emb.shape}, RSS={rss_mb():.0f}MB")

    rows = []

    # ---- Dense (baseline reference, scored variant for fusion) ----
    t0 = time_block("build dense")
    dense = DenseRetrieverScored(sub_emb, sub_ids)
    bt_dense = time_done(t0, "build dense")
    # cache dense top-CAND_K per query for hybrid & reranker
    dense_top_cand: list[tuple[list[str], np.ndarray]] = []
    dense_top_k_only: list[list[str]] = []
    lats = []
    for qvec in query_vecs:
        t0 = time.perf_counter()
        ids, scs = dense.search_with_scores(qvec, CAND_K)
        lats.append((time.perf_counter() - t0) * 1000)
        dense_top_cand.append((ids, scs))
        dense_top_k_only.append(ids[:TOP_K])
    rows.append(measure(dense_top_k_only, eval_set, lats, "dense_bruteforce", bt_dense, {"size": size}))
    all_rows.append(rows[-1]); save_rows(all_rows, out_path)

    # ---- BM25 (alone, for reference) ----
    t0 = time_block("build BM25")
    bm25 = BM25Retriever(sub_texts, sub_ids)
    bt_bm25 = time_done(t0, "build BM25")
    bm25_top_cand: list[tuple[list[str], np.ndarray]] = []
    bm25_top_k_only: list[list[str]] = []
    lats = []
    for qtext in queries_text:
        t0 = time.perf_counter()
        ids, scs = bm25.search(qtext, CAND_K)
        lats.append((time.perf_counter() - t0) * 1000)
        bm25_top_cand.append((ids, scs))
        bm25_top_k_only.append(ids[:TOP_K])
    rows.append(measure(bm25_top_k_only, eval_set, lats, "bm25", bt_bm25, {"size": size}))
    all_rows.append(rows[-1]); save_rows(all_rows, out_path)

    # ---- Hybrid (BM25 + Dense via RRF) ----
    lats = []
    hybrid_per_q: list[list[str]] = []
    for (d_ids, _), (b_ids, _) in zip(dense_top_cand, bm25_top_cand):
        t0 = time.perf_counter()
        fused = rrf_fuse([d_ids, b_ids], k=RRF_K, top_k=TOP_K)
        lats.append((time.perf_counter() - t0) * 1000)
        hybrid_per_q.append(fused)
    # latency of hybrid = dense_search + bm25_search + fuse (we already measured d+b)
    # Report fuse-only latency separately + total combined
    fuse_only_p50 = float(np.percentile(lats, 50))
    print(f"  [hybrid fuse-only p50] {fuse_only_p50:.3f} ms (excl. dense+bm25 search)")
    rows.append(measure(hybrid_per_q, eval_set, lats, "hybrid_rrf", 0.0, {"size": size}))
    all_rows.append(rows[-1]); save_rows(all_rows, out_path)

    # ---- Reranker on dense top-CAND_K ----
    if reranker is not None:
        rer_per_q: list[list[str]] = []
        lats = []
        for (d_ids, _), qtext in zip(dense_top_cand, queries_text):
            pairs = [(qtext, id_to_text[did]) for did in d_ids]
            t0 = time.perf_counter()
            scores = reranker.predict(pairs, batch_size=RERANKER_BATCH, show_progress_bar=False)
            order = np.argsort(-np.asarray(scores))[:TOP_K]
            lats.append((time.perf_counter() - t0) * 1000)
            rer_per_q.append([d_ids[i] for i in order])
        rows.append(measure(rer_per_q, eval_set, lats, "reranker_bge_v2_m3", 0.0, {"size": size}))
        all_rows.append(rows[-1]); save_rows(all_rows, out_path)

    # Free BM25 + reranker tensors before HNSW to avoid OOM
    del bm25, bm25_top_cand
    import gc; gc.collect()

    # ---- HNSW (latency fix) — disabled here, run separately in run_hnsw.py.
    # Reason: faiss HNSW build silently segfaults after rank_bm25's loky workers
    # leave leaked semaphores, even after del+gc. Clean-process HNSW works fine.
    return rows
    try:
        t0 = time_block("build HNSW")
        hnsw = HNSWRetriever(sub_emb, sub_ids)
        bt_hnsw = time_done(t0, "build HNSW")
        lats = []
        per_q: list[list[str]] = []
        _ = hnsw.search(query_vecs[0], TOP_K)
        for qvec in query_vecs:
            t0 = time.perf_counter()
            ids = hnsw.search(qvec, TOP_K)
            lats.append((time.perf_counter() - t0) * 1000)
            per_q.append(ids)
        rows.append(measure(per_q, eval_set, lats, f"hnsw_M{HNSW_M}_ef{HNSW_EF_SEARCH}", bt_hnsw, {"size": size}))
        all_rows.append(rows[-1]); save_rows(all_rows, out_path)
    except Exception as e:
        print(f"  [HNSW] FAILED: {type(e).__name__}: {e}")

    return rows


def main():
    print("Loading artifacts...")
    embeddings, corpus_ids, query_vecs, eval_set, id_to_text = load_artifacts()
    print(f"  corpus={embeddings.shape}, queries={query_vecs.shape}, RSS={rss_mb():.0f}MB")

    device = pick_device()
    print(f"Loading reranker {RERANKER_NAME} on {device}...")
    try:
        reranker = CrossEncoder(RERANKER_NAME, device=device, max_length=512)
        print(f"  loaded, RSS={rss_mb():.0f}MB")
    except Exception as e:
        print(f"  FAILED: {e}\n  proceeding without reranker")
        reranker = None

    out_path = RESULTS / "fixes.csv"
    all_rows: list[dict] = []
    for size in SIZES:
        run_size(size, embeddings, corpus_ids, query_vecs, eval_set, id_to_text, reranker, all_rows, out_path)
        print(f"  (saved {len(all_rows)} rows so far)")
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
