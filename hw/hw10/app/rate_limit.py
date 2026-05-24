"""
Token-bucket rate limit per API key, з фіксованим хвилинним вікном у Redis.

API:
  await check(tier)   → RateLimitResult (allowed, used, limit, retry_after_s)
  await charge(...)   → INCRBY на bucket після реального використання токенів

Fixed window (а не sliding) — простіше, race-tolerant, для домашки достатньо.
Sliding window був би точнішим (немає сплеску на межі хвилини), але потребує
двох ключів і вагового усереднення. Залишаємо як «можливе покращення».
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from .auth import TierInfo
from .redis_client import get_redis

KEY_PREFIX = "rl"
WINDOW_S = 60


@dataclass
class RateLimitResult:
    allowed: bool
    used_tokens: int
    limit: int
    retry_after_s: int   # 0 якщо allowed


def _bucket_key(api_key: str, now: int) -> str:
    return f"{KEY_PREFIX}:{api_key}:{now // WINDOW_S}"


async def check(tier: TierInfo) -> RateLimitResult:
    """Перевірити чи бакет не переповнено. Не змінює стан."""
    now = int(time.time())
    key = _bucket_key(tier.api_key, now)
    r = get_redis()

    try:
        raw = await r.get(key)
        used = int(raw) if raw else 0
    except Exception as e:
        # Fail-open: Redis недоступний → пропускаємо. Логуємо щоб операційно бачити.
        print(f"[rate_limit] redis down, failing open: {e.__class__.__name__}: {e}")
        return RateLimitResult(
            allowed=True, used_tokens=0,
            limit=tier.tokens_per_min, retry_after_s=0,
        )

    if used >= tier.tokens_per_min:
        retry_after = WINDOW_S - (now % WINDOW_S)
        return RateLimitResult(
            allowed=False, used_tokens=used,
            limit=tier.tokens_per_min, retry_after_s=retry_after,
        )

    return RateLimitResult(
        allowed=True, used_tokens=used,
        limit=tier.tokens_per_min, retry_after_s=0,
    )


async def charge(api_key: str, tokens: int) -> None:
    """Списати реально використані токени. Викликати після LLM-відповіді."""
    if tokens <= 0:
        return
    now = int(time.time())
    key = _bucket_key(api_key, now)
    r = get_redis()
    try:
        # INCRBY + EXPIRE — стандартний патерн, сумісний з Upstash REST (немає Lua).
        # TTL у 2 рази більше за вікно — щоб не побити суму при міжхвилинному сплеску.
        await r.incrby(key, tokens)
        await r.expire(key, 2 * WINDOW_S)
    except Exception as e:
        print(f"[rate_limit] charge failed (silently): {e.__class__.__name__}: {e}")


async def current_usage(api_key: str) -> tuple[int, int]:
    """Helper для /health чи debug: (used, seconds_until_reset)."""
    now = int(time.time())
    try:
        raw = await get_redis().get(_bucket_key(api_key, now))
        used = int(raw) if raw else 0
    except Exception:
        used = 0
    return used, WINDOW_S - (now % WINDOW_S)
