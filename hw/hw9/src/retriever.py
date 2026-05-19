"""
Retrievers — pluggable interface so scaling loop can swap dense / hybrid / reranker / HNSW.

For Step 4 we only need DenseRetriever (numpy brute-force).
Hybrid/Reranker/HNSW land in Step 6 (run_fixes.py).
"""
import random
from typing import Protocol

import numpy as np


SEED = 42


class Retriever(Protocol):
    def search(self, query_vec: np.ndarray, top_k: int) -> list[str]:
        """Return ranked list of doc_ids."""
        ...


class DenseRetriever:
    """Brute-force dense retriever: scores = embeddings @ query, top-K via argpartition."""

    def __init__(self, embeddings: np.ndarray, doc_ids: list[str]):
        assert embeddings.shape[0] == len(doc_ids)
        self.embeddings = embeddings   # (N, d), L2-normalized → dot = cosine
        self.doc_ids = doc_ids
        self.n = embeddings.shape[0]

    def search(self, query_vec: np.ndarray, top_k: int) -> list[str]:
        scores = self.embeddings @ query_vec               # (N,)
        k = min(top_k, self.n)
        top_idx = np.argpartition(scores, -k)[-k:]         # unsorted top-K
        top_idx_sorted = top_idx[np.argsort(-scores[top_idx])]  # sort K only
        return [self.doc_ids[i] for i in top_idx_sorted]


def select_subset_indices(
    corpus_ids: list[str],
    relevant_ids: set[str],
    size: int,
    seed: int = SEED,
) -> list[int]:
    """
    Mirror data_loader.build_subset() on indices, so we can slice embeddings_corpus.npy.

    Returns indices into the original corpus_ids list: all relevant docs first,
    then a seed-shuffled prefix of distractor indices.
    """
    rel_idx = [i for i, did in enumerate(corpus_ids) if did in relevant_ids]
    dist_idx = [i for i, did in enumerate(corpus_ids) if did not in relevant_ids]

    rng = random.Random(seed)
    rng.shuffle(dist_idx)

    n_dist = max(size - len(rel_idx), 0)
    if n_dist > len(dist_idx):
        raise ValueError(
            f"Need {n_dist} distractors but pool has only {len(dist_idx)}."
        )
    return rel_idx + dist_idx[:n_dist]
