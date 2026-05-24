"""
Cost tracking — persistent SQLite log усіх запитів.

Чому SQLite (а не Postgres):
  - простіше: без зовнішнього сервісу, без connection pool
  - achievable: ~100 req/s з WAL mode — більше нам не треба
  - на Fly.io лежить на volume → переживає деплой
  - якщо переростемо — `asyncpg` замість `aiosqlite`, схема та сама

Схема нормалізована «1 рядок = 1 запит». Тут немає окремої таблиці моделей,
бо це б додало JOIN-и без виграшу — модель просто текстове поле.

Індекси: (api_key, created_at) для /usage/today,
         (model) для /usage/breakdown.
"""
from __future__ import annotations

import statistics
from pathlib import Path

import aiosqlite

from .config import settings


# ─── DB path: парсимо sqlite+aiosqlite:///./path або sqlite:///./path ────────

def _db_path() -> str:
    url = settings.database_url
    for prefix in ("sqlite+aiosqlite:///", "sqlite:///"):
        if url.startswith(prefix):
            return url[len(prefix):]
    # fallback: припускаємо що це уже шлях
    return url


SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    request_id      TEXT PRIMARY KEY,
    api_key         TEXT NOT NULL,
    tier            TEXT,
    model           TEXT NOT NULL,
    input_tokens    INTEGER NOT NULL DEFAULT 0,
    output_tokens   INTEGER NOT NULL DEFAULT 0,
    cost_usd        REAL    NOT NULL DEFAULT 0,
    latency_ms      INTEGER NOT NULL,
    ttft_ms         INTEGER NOT NULL DEFAULT 0,
    cache_hit       INTEGER NOT NULL DEFAULT 0,
    fallback_used   INTEGER NOT NULL DEFAULT 0,
    output_filtered INTEGER NOT NULL DEFAULT 0,
    created_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_requests_key_date ON requests(api_key, created_at);
CREATE INDEX IF NOT EXISTS idx_requests_model    ON requests(model);
"""


async def init_db() -> None:
    """Ідемпотентно. Викликається з lifespan."""
    path = _db_path()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(path) as db:
        # WAL: concurrent reads while writing. Стандарт для async SQLite.
        await db.execute("PRAGMA journal_mode = WAL")
        await db.executescript(SCHEMA)
        await db.commit()
    print(f"[cost_tracker] db ready · {path}")


async def log_request(
    request_id: str,
    api_key: str,
    tier: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    cost_usd: float,
    latency_ms: int,
    ttft_ms: int,
    cache_hit: bool,
    fallback_used: bool,
    output_filtered: bool = False,
) -> None:
    async with aiosqlite.connect(_db_path()) as db:
        await db.execute(
            """
            INSERT INTO requests (
              request_id, api_key, tier, model,
              input_tokens, output_tokens, cost_usd,
              latency_ms, ttft_ms,
              cache_hit, fallback_used, output_filtered
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                request_id, api_key, tier, model,
                int(input_tokens), int(output_tokens), float(cost_usd),
                int(latency_ms), int(ttft_ms),
                int(cache_hit), int(fallback_used), int(output_filtered),
            ),
        )
        await db.commit()


# ─── /usage/today ────────────────────────────────────────────────────────────

async def usage_today(api_key: str) -> dict:
    """
    Агрегат за сьогодні. SQLite date('now') = UTC midnight; тут не парюсь
    з таймзонами, бо домашка. У production — TIMESTAMPTZ.
    """
    async with aiosqlite.connect(_db_path()) as db:
        db.row_factory = aiosqlite.Row
        row = await (await db.execute(
            """
            SELECT
              COUNT(*) AS requests,
              COALESCE(SUM(input_tokens + output_tokens), 0) AS tokens,
              COALESCE(SUM(cost_usd), 0) AS cost_usd,
              COALESCE(SUM(cache_hit), 0) AS cache_hits,
              COALESCE(SUM(fallback_used), 0) AS fallback_used
            FROM requests
            WHERE api_key = ? AND date(created_at) = date('now')
            """,
            (api_key,),
        )).fetchone()

    requests = row["requests"]
    return {
        "api_key": api_key,
        "requests": requests,
        "tokens": row["tokens"],
        "cost_usd": round(row["cost_usd"], 6),
        "cache_hits": row["cache_hits"],
        "cache_hit_rate": round(row["cache_hits"] / requests, 3) if requests else 0.0,
        "fallback_used": row["fallback_used"],
        "fallback_rate": round(row["fallback_used"] / requests, 3) if requests else 0.0,
    }


# ─── /usage/breakdown ────────────────────────────────────────────────────────

async def usage_breakdown(api_key: str, hours: int = 24) -> dict:
    """
    Per-model breakdown за останні N годин. p95 рахуємо в Python,
    бо SQLite не має вбудованого PERCENTILE_CONT (а ставити extension
    заради демо — overkill).
    """
    async with aiosqlite.connect(_db_path()) as db:
        db.row_factory = aiosqlite.Row

        # 1. По моделях
        rows = await (await db.execute(
            f"""
            SELECT
              model,
              COUNT(*) AS requests,
              SUM(input_tokens + output_tokens) AS tokens,
              SUM(cost_usd) AS cost_usd,
              SUM(cache_hit) AS cache_hits,
              SUM(fallback_used) AS fallback_used,
              AVG(latency_ms) AS avg_latency_ms,
              AVG(ttft_ms)    AS avg_ttft_ms
            FROM requests
            WHERE api_key = ?
              AND created_at >= datetime('now', '-{int(hours)} hours')
            GROUP BY model
            ORDER BY requests DESC
            """,
            (api_key,),
        )).fetchall()

        # 2. p95 latency (Python — простіше за віконні запити в SQLite)
        latency_rows = await (await db.execute(
            f"""
            SELECT model, latency_ms FROM requests
            WHERE api_key = ?
              AND created_at >= datetime('now', '-{int(hours)} hours')
            """,
            (api_key,),
        )).fetchall()

    latencies_by_model: dict[str, list[int]] = {}
    for r in latency_rows:
        latencies_by_model.setdefault(r["model"], []).append(r["latency_ms"])

    by_model = []
    overall_requests = 0
    overall_hits = 0
    overall_fb = 0
    overall_cost = 0.0
    for r in rows:
        model = r["model"]
        n = r["requests"]
        ls = sorted(latencies_by_model.get(model, []))
        p95 = ls[int(len(ls) * 0.95)] if ls and len(ls) > 1 else (ls[0] if ls else 0)
        p50 = ls[len(ls) // 2] if ls else 0

        by_model.append({
            "model": model,
            "requests": n,
            "tokens": r["tokens"] or 0,
            "cost_usd": round(r["cost_usd"] or 0.0, 6),
            "cache_hits": r["cache_hits"] or 0,
            "cache_hit_rate": round((r["cache_hits"] or 0) / n, 3),
            "fallback_used": r["fallback_used"] or 0,
            "fallback_rate": round((r["fallback_used"] or 0) / n, 3),
            "avg_latency_ms": int(r["avg_latency_ms"] or 0),
            "p50_latency_ms": int(p50),
            "p95_latency_ms": int(p95),
            "avg_ttft_ms": int(r["avg_ttft_ms"] or 0),
        })
        overall_requests += n
        overall_hits += r["cache_hits"] or 0
        overall_fb += r["fallback_used"] or 0
        overall_cost += r["cost_usd"] or 0.0

    return {
        "api_key": api_key,
        "window_hours": hours,
        "total_requests": overall_requests,
        "total_cost_usd": round(overall_cost, 6),
        "cache_hit_rate": round(overall_hits / overall_requests, 3) if overall_requests else 0.0,
        "fallback_rate": round(overall_fb / overall_requests, 3) if overall_requests else 0.0,
        "by_model": by_model,
    }
