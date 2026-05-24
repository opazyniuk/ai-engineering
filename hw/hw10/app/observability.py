"""
Langfuse observability — тонка обгортка над їхнім SDK.

Дизайн:
  - get_client() повертає Langfuse instance, якщо ключі сконфігуровані;
    інакше None → решта коду graceful-у пропускає трасування.
  - Контекстні менеджери (span_ctx, generation_ctx) обертають reusable
    pattern: start span → run code → end span з output (або error).
  - Усі виклики Langfuse у try/except — observability ніколи не валить запит.

Чому окремий модуль:
  - main.py не знає деталей Langfuse API (легко замінити на OpenTelemetry).
  - Тестується ізольовано від HTTP-шару.
"""
from __future__ import annotations

import time
import traceback
from contextlib import contextmanager
from functools import lru_cache
from typing import Any

try:
    from langfuse import Langfuse
    _SDK_AVAILABLE = True
except ImportError:
    Langfuse = None  # type: ignore
    _SDK_AVAILABLE = False

from .config import settings


@lru_cache(maxsize=1)
def get_client() -> "Langfuse | None":
    """Singleton, None якщо немає ключів."""
    if not _SDK_AVAILABLE:
        return None
    if not (settings.langfuse_public_key and settings.langfuse_secret_key):
        return None
    try:
        client = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
            # Не блокуємо event loop — Langfuse SDK сам відсилає батчами.
            flush_at=10,
            flush_interval=2,
        )
        print(f"[observability] Langfuse connected · host={settings.langfuse_host}",
              flush=True)
        return client
    except Exception as e:
        print(f"[observability] Langfuse init failed: {e}", flush=True)
        return None


def start_trace(*, name: str, user_id: str, metadata: dict,
                input_data: Any = None):
    """Створити root trace. Повертає trace-handle або None."""
    client = get_client()
    if client is None:
        return None
    try:
        return client.trace(
            name=name,
            user_id=user_id,
            metadata=metadata,
            input=input_data,
        )
    except Exception:
        traceback.print_exc()
        return None


def end_trace(trace, *, output: Any = None, metadata: dict | None = None,
              level: str | None = None) -> None:
    """level аргумент приймаємо для майбутнього compat, але v2 trace.update його НЕ підтримує."""
    if trace is None:
        return
    try:
        # Тільки підтримувані поля для v2 trace.update.
        kwargs = {}
        if output is not None:
            kwargs["output"] = output
        if metadata is not None:
            kwargs["metadata"] = metadata
        trace.update(**kwargs)
    except Exception as e:
        print(f"[observability] end_trace failed: {e!r}", flush=True)
        traceback.print_exc()


@contextmanager
def span_ctx(trace, name: str, *, input_data: Any = None,
             metadata: dict | None = None):
    """
    Контекстний менеджер для звичайного span.

    Використання:
        with span_ctx(trace, "embed", input_data=query) as span:
            result = do_embed(query)
            span.set_output({"dim": len(result)})
    """
    span = None
    if trace is not None:
        try:
            span = trace.span(name=name, input=input_data, metadata=metadata)
        except Exception:
            span = None

    class _SpanHandle:
        output: Any = None
        meta: dict | None = None

        def set_output(self, value: Any) -> None:
            self.output = value

        def set_metadata(self, value: dict) -> None:
            self.meta = value

    handle = _SpanHandle()
    t0 = time.perf_counter()
    try:
        yield handle
    except Exception as e:
        if span is not None:
            try:
                span.end(output={"error": str(e)}, level="ERROR")
            except Exception:
                pass
        raise
    else:
        if span is not None:
            try:
                span.end(output=handle.output, metadata=handle.meta)
            except Exception:
                pass


def add_generation(trace, *, name: str, model: str,
                   input_messages: list[dict],
                   metadata: dict | None = None):
    """Створити generation span (спеціальний тип для LLM-викликів). Повертає handle."""
    if trace is None:
        return None
    try:
        return trace.generation(
            name=name,
            model=model,
            input=input_messages,
            metadata=metadata or {},
        )
    except Exception:
        return None


def end_generation(generation, *, output: str, usage: dict | None = None,
                   metadata: dict | None = None, level: str | None = None):
    if generation is None:
        return
    try:
        generation.end(
            output=output,
            usage_details=usage,
            metadata=metadata,
            level=level,
        )
    except Exception:
        traceback.print_exc()


def flush() -> None:
    """Викликати у shutdown, щоб усі pending events встигли відправитись."""
    client = get_client()
    if client is not None:
        try:
            client.flush()
        except Exception:
            pass
