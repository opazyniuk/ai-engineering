"""IR metrics: recall@K, MRR@K, latency percentiles."""

from __future__ import annotations

from typing import List, Set

import numpy as np


def recall_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """|retrieved[:k] ∩ relevant| / min(k, |relevant|).

    Returns 0.0 when relevant is empty (caller should usually skip these queries).
    """
    if not relevant:
        return 0.0
    hits = len(set(retrieved[:k]) & relevant)
    return hits / min(k, len(relevant))


def mrr_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """1 / rank of first hit in retrieved[:k]. 0.0 if no hit."""
    for rank, doc_id in enumerate(retrieved[:k], start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def percentiles(latencies_ms: np.ndarray) -> dict:
    """Returns {p50, p95, p99, mean, min, max} in ms."""
    return {
        "p50": float(np.percentile(latencies_ms, 50)),
        "p95": float(np.percentile(latencies_ms, 95)),
        "p99": float(np.percentile(latencies_ms, 99)),
        "mean": float(np.mean(latencies_ms)),
        "min": float(np.min(latencies_ms)),
        "max": float(np.max(latencies_ms)),
    }
