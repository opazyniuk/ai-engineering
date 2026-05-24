"""
Async Redis singleton. Використовується для rate-limit (крок 4) і метрик.

Singleton тому що redis.asyncio.Redis тримає connection pool — створювати
на запит = дорого і неправильно.

Сумісно з Upstash REST URL (rediss://default:<pw>@<host>:6379). Локально —
redis://localhost:6379/0.
"""
from __future__ import annotations

from functools import lru_cache

import redis.asyncio as aioredis

from .config import settings


@lru_cache(maxsize=1)
def get_redis() -> aioredis.Redis:
    return aioredis.from_url(
        settings.redis_url,
        decode_responses=True,
        socket_timeout=2,         # fail-fast при недоступному Redis
        socket_connect_timeout=2,
    )


async def redis_health() -> dict:
    """Для /health — щоб бачити чи Redis жив (а не "fail-open мовчки")."""
    try:
        r = get_redis()
        await r.ping()
        return {"redis": "ok"}
    except Exception as e:
        return {"redis": f"down: {e.__class__.__name__}"}
