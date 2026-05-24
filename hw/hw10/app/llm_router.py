"""
LLM роутер: OpenRouter + fallback chain + in-memory circuit breaker.

Інтерфейс:
    result = StreamResult()
    async for token in stream_with_fallback(models, messages, result):
        ...
    # після завершення:
    #   result.model           — реально використана модель
    #   result.fallback_used   — True якщо не primary
    #   result.usage           — {"input_tokens", "output_tokens"}
    #   result.error           — None або текст помилки (всі моделі впали)

Чому такий API:
  - generator віддає тільки токени (просто інтегрується з SSE-кодом у main.py)
  - метадані (model, usage, fallback_used) лежать на result-об'єкті,
    доступні ПІСЛЯ закінчення стріму — точно тоді, коли вони відомі.
"""
from __future__ import annotations

import asyncio
import time
from collections import deque
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Final

import httpx
import openai
from openai import AsyncOpenAI

from .config import settings


# ─── OpenAI-сумісний клієнт, що b'є в OpenRouter ─────────────────────────────

_CLIENT: AsyncOpenAI | None = None


def get_client() -> AsyncOpenAI:
    """Singleton — переюзаємо HTTP connection pool."""
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=settings.openrouter_api_key,
            timeout=30,
            default_headers={
                # Опціональні OpenRouter-headers — потрапимо у їхню analytics.
                "HTTP-Referer": "https://github.com/hw10-rag-api",
                "X-Title": "HW10 RAG API",
            },
        )
    return _CLIENT


# ─── Result container ───────────────────────────────────────────────────────

@dataclass
class StreamResult:
    model: str | None = None
    fallback_used: bool = False
    usage: dict = field(default_factory=lambda: {"input_tokens": 0, "output_tokens": 0})
    error: str | None = None
    attempts: list[str] = field(default_factory=list)  # log of tried models


# ─── Класифікація помилок ───────────────────────────────────────────────────

RETRYABLE_STATUS: Final = {429, 500, 502, 503, 504}
NON_RETRYABLE_STATUS: Final = {400, 401, 403, 404, 422}


def _is_retryable(exc: Exception) -> bool:
    """Чи можна спробувати fallback на цю помилку?"""
    if isinstance(exc, asyncio.TimeoutError):
        return True
    if isinstance(exc, (httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError,
                        httpx.PoolTimeout, httpx.ReadTimeout)):
        return True
    if isinstance(exc, openai.APIConnectionError):
        return True
    if isinstance(exc, openai.RateLimitError):
        return True
    if isinstance(exc, openai.APIStatusError):
        return exc.status_code in RETRYABLE_STATUS
    if isinstance(exc, openai.InternalServerError):
        return True
    return False


# ─── Circuit Breaker (in-memory) ────────────────────────────────────────────
# State machine: CLOSED → OPEN → HALF_OPEN → CLOSED
#
# CLOSED:    normal, requests pass through
# OPEN:      too many failures; skip the model entirely
# HALF_OPEN: after cooldown, allow ONE probe to test if model recovered
#
# Виноска: in-memory означає, що при перезапуску процесу breaker reset-неться.
# Для multi-worker production треба зберігати у Redis (sorted set з timestamps).

_FAIL_THRESHOLD = 5         # failures щоб відкрити breaker
_FAIL_WINDOW_S = 60         # вікно у якому рахуємо failures
_OPEN_DURATION_S = 60       # як довго тримати breaker OPEN перед HALF_OPEN


@dataclass
class _BreakerState:
    failures: deque = field(default_factory=lambda: deque(maxlen=_FAIL_THRESHOLD * 2))
    open_until: float = 0.0
    state: str = "CLOSED"  # CLOSED | OPEN | HALF_OPEN


_breakers: dict[str, _BreakerState] = {}


def _breaker(model: str) -> _BreakerState:
    if model not in _breakers:
        _breakers[model] = _BreakerState()
    return _breakers[model]


def is_open(model: str) -> bool:
    b = _breaker(model)
    now = time.monotonic()
    if b.state == "OPEN" and now >= b.open_until:
        b.state = "HALF_OPEN"   # дозволяємо одну спробу
    return b.state == "OPEN"


def record_failure(model: str) -> None:
    b = _breaker(model)
    now = time.monotonic()
    # Викидаємо застарілі помилки за межі вікна
    while b.failures and b.failures[0] < now - _FAIL_WINDOW_S:
        b.failures.popleft()
    b.failures.append(now)
    if len(b.failures) >= _FAIL_THRESHOLD:
        b.state = "OPEN"
        b.open_until = now + _OPEN_DURATION_S
        print(f"[llm_router] CIRCUIT OPEN for {model} (cooldown {_OPEN_DURATION_S}s)",
              flush=True)


def record_success(model: str) -> None:
    b = _breaker(model)
    if b.state in ("HALF_OPEN", "OPEN"):
        print(f"[llm_router] CIRCUIT CLOSED for {model}", flush=True)
    b.failures.clear()
    b.state = "CLOSED"
    b.open_until = 0.0


def breaker_snapshot() -> dict:
    """Для /health чи debug."""
    now = time.monotonic()
    return {
        m: {
            "state": b.state,
            "recent_failures": len(b.failures),
            "open_for_s": max(0, int(b.open_until - now)) if b.state == "OPEN" else 0,
        }
        for m, b in _breakers.items()
    }


# ─── Streaming з fallback ───────────────────────────────────────────────────

async def stream_with_fallback(
    models: list[str],
    messages: list[dict],
    result: StreamResult,
    ttft_timeout_s: float | None = None,
) -> AsyncIterator[str]:
    """
    Yield-ить token strings. Метадані відписує в result.

    Логіка fallback:
      - якщо breaker OPEN → пропускаємо модель
      - timeout на створення streaming-response = ttft_timeout_s (default з settings)
      - помилка ДО першого токена → fallback на наступну модель
      - помилка ПІСЛЯ першого токена → emit_error + закінчуємо (mid-stream restart
        не робимо — клієнт уже має часткову відповідь)

    Якщо всі моделі впали — yield не повертає нічого, а result.error = причина.
    """
    if ttft_timeout_s is None:
        ttft_timeout_s = settings.llm_request_timeout_s

    client = get_client()

    for idx, model in enumerate(models):
        result.attempts.append(model)

        if is_open(model):
            print(f"[llm_router] skipping {model} (circuit OPEN)", flush=True)
            continue

        # 1) Відкрити streaming-response з timeout-ом.
        try:
            response_stream = await asyncio.wait_for(
                client.chat.completions.create(
                    model=model,
                    messages=messages,
                    stream=True,
                    stream_options={"include_usage": True},
                ),
                timeout=ttft_timeout_s,
            )
        except Exception as e:
            if _is_retryable(e):
                record_failure(model)
                result.error = f"{model}: {e.__class__.__name__}: {e}"
                print(f"[llm_router] {model} failed at connect: {e!r} → trying next",
                      flush=True)
                continue
            # Non-retryable: пробрасуємо нагору (буде 4xx клієнту).
            raise

        # 2) Стрімимо токени. Якщо помилка ДО першого токена → fallback.
        first_token_received = False
        try:
            async for chunk in response_stream:
                # Usage прилітає окремим чанком наприкінці (через stream_options).
                if chunk.usage is not None:
                    result.usage = {
                        "input_tokens": chunk.usage.prompt_tokens,
                        "output_tokens": chunk.usage.completion_tokens,
                    }

                if not chunk.choices:
                    continue

                delta = chunk.choices[0].delta
                # Деякі моделі шлють reasoning_content (reasoning-моделі) — ігноруємо
                content = getattr(delta, "content", None)
                if not content:
                    continue

                first_token_received = True
                yield content

            # Стрім успішно завершено.
            result.model = model
            result.fallback_used = idx > 0
            record_success(model)
            return

        except Exception as e:
            if not first_token_received and _is_retryable(e):
                # Ще не пізно — пробуємо fallback
                record_failure(model)
                result.error = f"{model}: mid-handshake {e.__class__.__name__}: {e}"
                print(f"[llm_router] {model} failed before first token: {e!r} → next",
                      flush=True)
                continue
            # Помилка mid-stream — клієнт уже має токени, не можемо fallback-ити.
            result.model = model
            result.fallback_used = idx > 0
            result.error = f"mid-stream error: {e.__class__.__name__}: {e}"
            print(f"[llm_router] {model} failed MID-stream: {e!r}", flush=True)
            return

    # Усі моделі впали ще до першого токена.
    result.error = result.error or "all providers failed"
    print(f"[llm_router] ALL FAILED · attempts={result.attempts}", flush=True)
