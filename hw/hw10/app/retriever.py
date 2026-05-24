"""
RAG retrieval: embed query → top-k cosine search у Qdrant collection 'chunks'.

Викликається з /chat/stream після того, як ми згенерували один embedding запиту
(той самий embedding потім піде у semantic cache на кроці 5).
"""
from __future__ import annotations

from dataclasses import dataclass

from .config import settings
from .embedder import embed_one
from .vector_db import get_client


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    score: float
    section: str | None = None


def search(query: str, k: int = 3) -> list[RetrievedChunk]:
    """Зручний wrapper коли потрібен лише retrieval (без cache)."""
    return search_by_vector(embed_one(query), k=k)


def search_by_vector(query_vector: list[float], k: int = 3) -> list[RetrievedChunk]:
    """
    Основний шлях з /chat/stream: вектор уже згенерований для cache lookup,
    переюзаємо його тут (без другого виклику embedder).
    """
    client = get_client()
    hits = client.query_points(
        collection_name=settings.qdrant_chunks_collection,
        query=query_vector,
        limit=k,
        with_payload=True,
    ).points

    return [
        RetrievedChunk(
            chunk_id=str(h.id),
            text=h.payload.get("text", ""),
            score=float(h.score),
            section=h.payload.get("section"),
        )
        for h in hits
    ]
