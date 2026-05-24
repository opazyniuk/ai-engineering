"""
Qdrant client — спільний для retriever (читання) і indexer (запис).

Singleton, бо QdrantClient тримає HTTP connection pool. Не створювати на запит.
"""
from __future__ import annotations

from functools import lru_cache

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from .config import settings
from .embedder import EMBEDDING_DIM


@lru_cache(maxsize=1)
def get_client() -> QdrantClient:
    return QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key or None,  # None для локального Qdrant
        timeout=10,
    )


def ensure_collection(name: str) -> None:
    """Створити колекцію якщо її ще немає. Ідемпотентно."""
    client = get_client()
    existing = {c.name for c in client.get_collections().collections}
    if name in existing:
        return
    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )
    print(f"[qdrant] created collection '{name}' (dim={EMBEDDING_DIM}, cosine)")


def recreate_collection(name: str) -> None:
    """Видалити (якщо є) і створити наново. Для повної переіндексації."""
    client = get_client()
    existing = {c.name for c in client.get_collections().collections}
    if name in existing:
        client.delete_collection(name)
        print(f"[qdrant] dropped collection '{name}'")
    ensure_collection(name)
