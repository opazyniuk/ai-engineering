"""
Embed the cached corpus + eval queries once, save as .npy + ids JSON.

- Model: BAAI/bge-small-en-v1.5 (384d, MPS-accelerated on Apple Silicon)
- Queries get the official bge retrieval prefix; passages do not (asymmetric s2p).
- Output is L2-normalized → cosine(a,b) = a·b → fast brute-force with matmul.

Cache files (data/cache/):
  embeddings_corpus.npy   (N x 384, float32) — passages in pool order
  corpus_ids.json         (list[str])         — doc_id per row of corpus.npy
  embeddings_queries.npy  (100 x 384, float32)
  queries_meta.json       (list[{qid, query, relevant_ids}]) — same order as queries.npy
"""
import json
import time
from pathlib import Path

import numpy as np
import psutil
import torch
from sentence_transformers import SentenceTransformer


MODEL_NAME = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
BATCH_SIZE = 64

ROOT = Path(__file__).parent.parent
CACHE_DIR = ROOT / "data" / "cache"
CORPUS_JSON = CACHE_DIR / "corpus.json"


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def rss_mb() -> float:
    return psutil.Process().memory_info().rss / (1024 ** 2)


def embed_texts(model: SentenceTransformer, texts: list[str], desc: str) -> np.ndarray:
    t0 = time.perf_counter()
    rss_before = rss_mb()
    vecs = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    elapsed = time.perf_counter() - t0
    rss_after = rss_mb()
    throughput = len(texts) / elapsed
    print(
        f"[{desc}] {len(texts)} texts → shape={vecs.shape}, "
        f"time={elapsed:.1f}s, throughput={throughput:.0f}/s, "
        f"RSS {rss_before:.0f}→{rss_after:.0f} MB"
    )
    return vecs


def main() -> None:
    assert CORPUS_JSON.exists(), f"Run data_loader.py first to create {CORPUS_JSON}"

    print(f"Loading {CORPUS_JSON}...")
    data = json.load(open(CORPUS_JSON))
    corpus = data["corpus"]
    eval_set = data["eval_set"]
    print(f"  corpus={len(corpus)} docs, eval_set={len(eval_set)} queries")

    device = pick_device()
    print(f"Device: {device}")
    model = SentenceTransformer(MODEL_NAME, device=device)
    print(f"Model: {MODEL_NAME} (dim={model.get_sentence_embedding_dimension()})")

    # Passages — no prefix
    passages = [d["text"] for d in corpus]
    corpus_vecs = embed_texts(model, passages, "corpus")

    np.save(CACHE_DIR / "embeddings_corpus.npy", corpus_vecs)
    json.dump([d["id"] for d in corpus], open(CACHE_DIR / "corpus_ids.json", "w"))
    print(f"Saved embeddings_corpus.npy ({corpus_vecs.nbytes / 1024**2:.1f} MB)")

    # Queries — with bge retrieval prefix
    query_texts = [QUERY_PREFIX + e["query"] for e in eval_set]
    query_vecs = embed_texts(model, query_texts, "queries")

    np.save(CACHE_DIR / "embeddings_queries.npy", query_vecs)
    json.dump(eval_set, open(CACHE_DIR / "queries_meta.json", "w"))
    print(f"Saved embeddings_queries.npy ({query_vecs.nbytes / 1024:.1f} KB)")

    # Sanity: cosine between first query and its relevant doc
    q0 = query_vecs[0]
    rel_id = eval_set[0]["relevant_ids"][0]
    corpus_ids = [d["id"] for d in corpus]
    rel_idx = corpus_ids.index(rel_id)
    rel_vec = corpus_vecs[rel_idx]
    cos = float(np.dot(q0, rel_vec))
    print(f"\nSanity: cos(query[0], relevant_doc) = {cos:.3f}")
    print(f"  query: \"{eval_set[0]['query']}\"")
    print(f"  doc:   \"{corpus[rel_idx]['text'][:120]}...\"")


if __name__ == "__main__":
    main()
