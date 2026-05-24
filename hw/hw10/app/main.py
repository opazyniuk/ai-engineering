"""
FastAPI entrypoint — крок 2: /chat/stream з SSE-streaming, mock LLM, disconnect handling.

На наступних кроках сюди ляжуть: auth, rate limit, cache, реальний LLM,
cost tracking, security, semaphore, observability.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import cache, cost_tracker, llm_router, observability, rate_limit, security
from .auth import TierInfo, require_api_key
from .config import settings
from .embedder import embed_one, get_model
from .pricing import calculate_cost
from .prompts import build_messages
from .redis_client import redis_health
from .retriever import search_by_vector
from .sse import SSE_HEADERS, format_event


# Швидкість «replay» закешованих токенів. Свідомо повільніше за миттєве —
# щоб UX залишався «друкується», але значно швидше за MOCK_TOKEN_DELAY (50мс).
# При real LLM на кроці 7 — buy-in аналогічний: replay HIT десь у 10x швидше за MISS.
_CACHE_REPLAY_TOKEN_DELAY_S = 0.005


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm-up: тягнемо модель у RAM під час старту, не при першому запиті.
    # Інакше перший /chat/stream чекатиме ~5 секунд на завантаження.
    get_model()

    # Ідемпотентно створюємо cache_collection якщо її ще нема.
    cache.ensure_cache_collection()

    # SQLite cost log
    await cost_tracker.init_db()

    # Concurrency limit для LLM calls (settings.llm_concurrency_limit, default 20).
    # Обмежує тільки stream_with_fallback — embedding/Qdrant/Redis НЕ під семафором.
    app.state.llm_semaphore = asyncio.Semaphore(settings.llm_concurrency_limit)
    app.state.llm_in_flight = 0

    app.state.active_streams = 0
    app.state.aborted_streams = 0
    app.state.total_streams = 0
    app.state.cache_hits = 0
    app.state.cache_misses = 0

    # Langfuse — warm up singleton при старті (щоб бачити «connected» у логах).
    observability.get_client()

    print(f"[hw10] ready · embedding={settings.embedding_model} · "
          f"qdrant={settings.qdrant_url}")
    yield
    observability.flush()    # відправити pending Langfuse events перед shutdown
    print("[hw10] shutdown")


app = FastAPI(
    title="HW10 · RAG API",
    description="Production-ready RAG: streaming, semantic cache, rate limit, fallback",
    version="0.2.0",
    lifespan=lifespan,
)


# ─── Models ──────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User query")


# ─── Static demo ─────────────────────────────────────────────────────────────

_STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


@app.get("/")
async def index():
    return FileResponse(_STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


# ─── Health ──────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "active_streams": app.state.active_streams,
        "aborted_streams": app.state.aborted_streams,
        "total_streams": app.state.total_streams,
        "llm_in_flight": app.state.llm_in_flight,
        "llm_concurrency_limit": settings.llm_concurrency_limit,
        "cache_hits": app.state.cache_hits,
        "cache_misses": app.state.cache_misses,
        "circuit_breakers": llm_router.breaker_snapshot(),
        **cache.stats(),
        **(await redis_health()),
    }


# ─── Rate-limit dependency (об'єднує auth + check) ──────────────────────────

async def enforce_rate_limit(
    tier: TierInfo = Depends(require_api_key),
) -> TierInfo:
    """
    Спершу auth (require_api_key), потім check rate-limit.
    Якщо bucket переповнено — 429 з Retry-After.
    Реальне списання токенів — у /chat/stream після завершення стріму.
    """
    result = await rate_limit.check(tier)
    if not result.allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=(
                f"Rate limit exceeded: {result.used_tokens}/{result.limit} tokens "
                f"in current minute. Retry in {result.retry_after_s}s."
            ),
            headers={
                "Retry-After": str(result.retry_after_s),
                "X-RateLimit-Limit": str(result.limit),
                "X-RateLimit-Used": str(result.used_tokens),
            },
        )
    return tier


# ─── Chat (SSE) ──────────────────────────────────────────────────────────────

@app.post("/chat/stream")
async def chat_stream(
    req: ChatRequest,
    request: Request,
    tier: TierInfo = Depends(enforce_rate_limit),
):
    """
    Workflow:
      auth → rate-limit → security checks → embed → cache lookup
        → HIT: replay cached
        → MISS: vector search → real LLM stream (з fallback chain)
                → output scan → store cache
      → done event → log cost → charge rate-limit (MISS only)
    """
    # ─── Security checks (до всього) ─────────────────────────────────────────
    # Робимо ДО запуску event_stream — щоб 400 повертався як normal HTTP-error,
    # а не як SSE-event. Це коректніше для клієнтів.
    if (err := security.check_length(req.message)):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=err,
        )
    input_scan = security.scan_input(req.message)
    if input_scan.matched:
        security.log_suspicious_request(
            api_key=tier.api_key,
            message=req.message,
            scan=input_scan,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(f"Input rejected: suspicious pattern "
                    f"'{input_scan.pattern_label}' detected. "
                    f"This event has been logged."),
        )

    app.state.active_streams += 1
    app.state.total_streams += 1

    request_id = str(uuid.uuid4())
    request_start = time.perf_counter()

    # Langfuse root trace — увесь pipeline відстежимо як один trace.
    trace = observability.start_trace(
        name="chat_stream",
        user_id=tier.api_key,
        metadata={"tier": tier.tier, "request_id": request_id},
        input_data={"message": req.message},
    )

    async def event_stream():
        disconnected = False
        accumulated = ""
        usage = {"input_tokens": 0, "output_tokens": 0}
        cache_hit = False
        fallback_used = False
        model_used = "mock-llm-v0"
        ttft_ms = 0
        first_token_seen = False
        output_filtered = False

        def mark_first_token():
            nonlocal first_token_seen, ttft_ms
            if not first_token_seen:
                ttft_ms = int((time.perf_counter() - request_start) * 1000)
                first_token_seen = True

        try:
            # 1) Один embedding — і для cache, і для vector search.
            with observability.span_ctx(trace, "embed",
                                         input_data={"query": req.message}) as sp:
                query_vec = embed_one(req.message)
                sp.set_output({"dim": len(query_vec)})

            # 2) Cache lookup перед усім (потенційно — bypass LLM)
            with observability.span_ctx(trace, "cache_lookup") as sp:
                cached = cache.lookup(query_vec)
                sp.set_output({
                    "hit": cached is not None,
                    "similarity": cached.similarity if cached else None,
                })
            if cached is not None:
                cache_hit = True
                app.state.cache_hits += 1
                model_used = cached.model
                fallback_used = cached.fallback_used
                source_ids = cached.sources

                for word in cached.response.split(" "):
                    if await request.is_disconnected():
                        disconnected = True
                        break
                    mark_first_token()
                    token = word + " "
                    accumulated += token
                    yield format_event({"type": "token", "content": token})
                    await asyncio.sleep(_CACHE_REPLAY_TOKEN_DELAY_S)

                if not disconnected:
                    yield format_event({
                        "type": "done",
                        "usage": {"input_tokens": 0, "output_tokens": 0},
                        "cost_usd": 0.0,
                        "cache_hit": True,
                        "similarity": round(cached.similarity, 3),
                        "age_seconds": cached.age_seconds,
                        "fallback_used": fallback_used,
                        "output_filtered": False,
                        "model": model_used,
                        "tier": tier.tier,
                        "request_id": request_id,
                        "sources": source_ids,
                    })
                return

            # 3) MISS — vector search + REAL LLM (with fallback) + store
            app.state.cache_misses += 1

            with observability.span_ctx(trace, "retrieve",
                                         input_data={"k": 3}) as sp:
                chunks = search_by_vector(query_vec, k=3)
                source_ids = [c.chunk_id for c in chunks]
                sp.set_output({
                    "chunks_count": len(chunks),
                    "chunk_ids": source_ids,
                    "top_score": chunks[0].score if chunks else None,
                })

            messages = build_messages(req.message, chunks)

            llm_result = llm_router.StreamResult()
            # Generation span — особливий для LLM-викликів.
            generation = observability.add_generation(
                trace,
                name="llm_call",
                model=tier.models[0],   # primary; оновимо real model в end
                input_messages=messages,
                metadata={"tier": tier.tier, "available_models": tier.models},
            )

            # Semaphore: обмежуємо паралельні LLM-виклики до llm_concurrency_limit.
            async with app.state.llm_semaphore:
                app.state.llm_in_flight += 1
                try:
                    async for token in llm_router.stream_with_fallback(
                        tier.models, messages, llm_result
                    ):
                        if await request.is_disconnected():
                            disconnected = True
                            break
                        mark_first_token()
                        accumulated += token
                        yield format_event({"type": "token", "content": token})
                except Exception as e:
                    observability.end_generation(
                        generation, output=accumulated,
                        metadata={"error": str(e)}, level="ERROR",
                    )
                    yield format_event({
                        "type": "error",
                        "message": f"{e.__class__.__name__}: {e}",
                    })
                    return
                finally:
                    app.state.llm_in_flight -= 1

            # Якщо всі моделі провалились до першого токена — emit error + не зберігаємо.
            if not first_token_seen and llm_result.error:
                yield format_event({
                    "type": "error",
                    "message": llm_result.error,
                    "attempts": llm_result.attempts,
                })
                return

            if not disconnected:
                usage = llm_result.usage if llm_result.usage["input_tokens"] else {
                    # Fallback оцінка якщо модель не повернула usage (рідко але буває)
                    "input_tokens": sum(len(m["content"]) for m in messages) // 4,
                    "output_tokens": max(1, len(accumulated) // 4),
                }
                model_used = llm_result.model or "unknown"
                fallback_used = llm_result.fallback_used
                cost = calculate_cost(model_used, usage["input_tokens"], usage["output_tokens"])

                # Закриваємо generation span з реальною моделлю/usage.
                observability.end_generation(
                    generation,
                    output=accumulated,
                    usage={"input": usage["input_tokens"],
                           "output": usage["output_tokens"]},
                    metadata={
                        "actual_model": model_used,
                        "fallback_used": fallback_used,
                        "attempts": llm_result.attempts,
                        "cost_usd": cost,
                    },
                )

                # ─── Output scan: чи модель не «прохиблила» system prompt ────
                output_scan = security.scan_output(accumulated)
                if output_scan.matched:
                    output_filtered = True
                    security.log_suspicious_response(
                        request_id=request_id,
                        api_key=tier.api_key,
                        model=model_used,
                        scan=output_scan,
                        output_preview=accumulated,
                    )

                with observability.span_ctx(trace, "cache_store") as sp:
                    cache.store(
                        query_vec=query_vec,
                        query=req.message,
                        response=accumulated.strip(),
                        model=model_used,
                        fallback_used=fallback_used,
                        sources=source_ids,
                    )
                    sp.set_output({"stored": True})

                yield format_event({
                    "type": "done",
                    "usage": usage,
                    "cost_usd": round(cost, 6),
                    "cache_hit": False,
                    "fallback_used": fallback_used,
                    "output_filtered": output_filtered,
                    "model": model_used,
                    "tier": tier.tier,
                    "request_id": request_id,
                    "sources": source_ids,
                    "attempts": llm_result.attempts,
                })
        except asyncio.CancelledError:
            disconnected = True
            raise
        finally:
            app.state.active_streams -= 1
            total_latency_ms = int((time.perf_counter() - request_start) * 1000)

            if disconnected:
                app.state.aborted_streams += 1
                return

            # Чому asyncio.shield: коли SSE-стрім завершено, Starlette скасовує
            # task-овий контекст. Будь-який await у finally може бути cancelled
            # ДО завершення. shield ізолює свою корутину від cancel'у батьківського
            # task-у — отже DB-запис і Redis INCRBY все одно завершаться.
            async def _finalize():
                if not cache_hit:
                    total_tokens = usage["input_tokens"] + usage["output_tokens"]
                    await rate_limit.charge(tier.api_key, total_tokens)

                cost_usd = 0.0 if cache_hit else calculate_cost(
                    model_used, usage["input_tokens"], usage["output_tokens"]
                )
                try:
                    await cost_tracker.log_request(
                        request_id=request_id,
                        api_key=tier.api_key,
                        tier=tier.tier,
                        model=model_used,
                        input_tokens=0 if cache_hit else usage["input_tokens"],
                        output_tokens=0 if cache_hit else usage["output_tokens"],
                        cost_usd=cost_usd,
                        latency_ms=total_latency_ms,
                        ttft_ms=ttft_ms,
                        cache_hit=cache_hit,
                        fallback_used=fallback_used,
                        output_filtered=output_filtered,
                    )
                except Exception as e:
                    print(f"[cost_tracker] log failed: {e.__class__.__name__}: {e}",
                          flush=True)

            try:
                await asyncio.shield(_finalize())
            except asyncio.CancelledError:
                # Якщо все-таки cancelled — створюємо background task без shield.
                # Loop не зупинений, тож task завершиться.
                asyncio.create_task(_finalize())
                raise

            # Закриваємо Langfuse trace з фінальним output + сумарними метаданими.
            observability.end_trace(
                trace,
                output={
                    "response": accumulated if not cache_hit else accumulated.strip(),
                    "model": model_used,
                    "cache_hit": cache_hit,
                    "fallback_used": fallback_used,
                    "output_filtered": output_filtered,
                    "sources": source_ids if not cache_hit else None,
                },
                metadata={
                    "total_latency_ms": total_latency_ms,
                    "ttft_ms": ttft_ms,
                    "cost_usd": 0.0 if cache_hit else calculate_cost(
                        model_used, usage["input_tokens"], usage["output_tokens"]),
                },
            )

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


# ─── Usage ───────────────────────────────────────────────────────────────────

@app.get("/usage/today")
async def usage_today(tier: TierInfo = Depends(require_api_key)):
    """Витрати поточної доби для цього API-key."""
    return await cost_tracker.usage_today(tier.api_key)


@app.get("/usage/breakdown")
async def usage_breakdown(
    tier: TierInfo = Depends(require_api_key),
    hours: int = Query(24, ge=1, le=720, description="window size in hours"),
):
    """Розбивка per-model + cache_hit_rate, fallback_rate, latency p50/p95."""
    return await cost_tracker.usage_breakdown(tier.api_key, hours=hours)
