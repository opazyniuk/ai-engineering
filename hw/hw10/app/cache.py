"""
Semantic cache на Qdrant — окрема колекція (`cache_collection`) поряд з `chunks`.

HIT criteria:
  - similarity > settings.cache_similarity_threshold (0.92)
  - expire_at > now()        # Qdrant без built-in TTL → фільтр на read

Cache scope: GLOBAL для документу (всі API-ключі ділять один кеш). Це навмисно,
бо у нас public Q&A bot про публічний документ. Для приватних даних потрібно:
  1) додати user_id у payload
  2) фільтрувати lookup за user_id (Filter(must=[FieldCondition("user_id"...))])
  3) bypass cache якщо запит містить «my/мої/I» (PII-trigger)
Див. розгорнуте пояснення приватного варіанту в PLAN.md / education.md.

Що зберігаємо у payload:
  - query:         оригінальний текст (для debug і inspect-ів)
  - response:      повний accumulated текст відповіді
  - model:         яка модель відповіла (важливо для /usage/breakdown по моделях)
  - fallback_used: чи був fallback (для observability)
  - sources:       chunk_id-и, які LLM бачила — щоб у done event прокидати ті ж
  - created_at:    Unix timestamp для віку запису
  - expire_at:     Unix timestamp після якого запис не валідний
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass

from qdrant_client.models import FieldCondition, Filter, PointStruct, Range

from .config import settings
from .vector_db import ensure_collection, get_client


@dataclass
class CachedResponse:
    query: str
    response: str
    model: str
    fallback_used: bool
    sources: list[str]
    similarity: float
    age_seconds: int


def ensure_cache_collection() -> None:
    """Викликати раз при старті (з lifespan)."""
    ensure_collection(settings.qdrant_cache_collection)


def lookup(query_vec: list[float]) -> CachedResponse | None:
    """
    Шукаємо найближчий вектор. Повертаємо лише якщо similarity > threshold
    і запис ще не протермінований.

    Зверни увагу: Qdrant-client синхронний (qdrant_client.QdrantClient). Це не
    проблема — операція займає 5-15мс, не блокує event loop помітно. Якщо колись
    стане критично — є AsyncQdrantClient.
    """
    client = get_client()
    now = int(time.time())

    # Фільтр на live-записи. gt = strictly greater than now.
    valid_filter = Filter(
        must=[FieldCondition(key="expire_at", range=Range(gt=now))]
    )

    hits = client.query_points(
        collection_name=settings.qdrant_cache_collection,
        query=query_vec,
        query_filter=valid_filter,
        limit=1,
        with_payload=True,
    ).points

    if not hits:
        return None

    top = hits[0]
    if top.score < settings.cache_similarity_threshold:
        return None

    p = top.payload or {}
    return CachedResponse(
        query=str(p.get("query", "")),
        response=str(p.get("response", "")),
        model=str(p.get("model", "unknown")),
        fallback_used=bool(p.get("fallback_used", False)),
        sources=list(p.get("sources", [])),
        similarity=float(top.score),
        age_seconds=now - int(p.get("created_at", now)),
    )


def store(
    query_vec: list[float],
    query: str,
    response: str,
    model: str,
    fallback_used: bool,
    sources: list[str],
) -> None:
    """Зберегти MISS-результат у кеш. TTL = settings.cache_ttl_seconds."""
    client = get_client()
    now = int(time.time())

    client.upsert(
        collection_name=settings.qdrant_cache_collection,
        points=[
            PointStruct(
                id=str(uuid.uuid4()),
                vector=query_vec,
                payload={
                    "query": query,
                    "response": response,
                    "model": model,
                    "fallback_used": fallback_used,
                    "sources": sources,
                    "created_at": now,
                    "expire_at": now + settings.cache_ttl_seconds,
                },
            )
        ],
    )


def stats() -> dict:
    """Helper для /health чи debug."""
    client = get_client()
    try:
        info = client.get_collection(settings.qdrant_cache_collection)
        return {"cache_entries": info.points_count}
    except Exception:
        return {"cache_entries": 0}
