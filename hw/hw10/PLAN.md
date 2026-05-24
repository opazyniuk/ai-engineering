   # HW10 · План реалізації — Production-Ready RAG API

> Сервіс «Q&A bot про документ» з усіма production-шарами: RAG, SSE streaming, semantic cache, token-based rate limit, cost tracking, multi-provider fallback, observability, deploy.

---

## 0. Загальна архітектура

```
                    ┌──────────────────────────────────────────────┐
                    │              POST /chat/stream               │
                    └──────────────────────────────────────────────┘
                                          │
   ┌──────────┐  ┌──────────┐  ┌──────────────────┐  ┌──────────────┐
   │  Auth    │→ │ Rate     │→ │ Prompt-Injection │→ │ Embed query  │
   │ X-API-Key│  │ limit    │  │ guard            │  │ (MiniLM)     │
   └──────────┘  └──────────┘  └──────────────────┘  └──────────────┘
                                                            │
                       ┌────────────────────────────────────┤
                       ▼                                    ▼
              ┌────────────────┐                  ┌──────────────────┐
              │ Semantic cache │  HIT  ──►stream  │ Vector search    │
              │ (Qdrant)       │                  │ top-k=3 chunks   │
              └────────────────┘                  └──────────────────┘
                       │ MISS                              │
                       └────────────────┬─────────────────┘
                                        ▼
                          ┌──────────────────────────────┐
                          │ LLM call via OpenRouter      │
                          │  → fallback chain (3 моделі) │
                          │  → asyncio.Semaphore(20)     │
                          │  → circuit breaker per model │
                          └──────────────────────────────┘
                                        │
                                        ▼
               ┌──────────────────────────────────────────┐
               │ SSE: data: {type:"token",content:"..."}  │
               │      data: {type:"done", usage, sources} │
               └──────────────────────────────────────────┘
                                        │
                          ┌──────────────────────────────┐
                          │ Cost tracker (SQLite)        │
                          │ Langfuse trace               │
                          └──────────────────────────────┘
```

**Принцип**: один embedding query → використовується і для cache lookup, і для RAG retrieval (не два виклики sentence-transformers).

---

## 1. Структура репозиторію

```
hw/hw10/
├── PLAN.md                  # цей файл
├── education.md             # 5 ключових понять
├── REPORT.md                # звіт зі скріншотами (наприкінці)
├── README.md                # quick start + публічний URL
├── .env.example
├── .gitignore
├── pyproject.toml           # або requirements.txt
├── data/
│   └── source.md            # індексований документ (12-Factor App)
├── scripts/
│   └── index.py             # індексація → Qdrant
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI, маршрути, lifespan
│   ├── config.py            # MODELS, API_KEYS, тires, ліміти
│   ├── deps.py              # auth dependency, Redis, Qdrant clients
│   ├── auth.py              # API-ключі + tiers
│   ├── rate_limit.py        # token bucket в Redis
│   ├── embedder.py          # sentence-transformers wrapper (singleton)
│   ├── retriever.py         # Qdrant search top-k chunks
│   ├── cache.py             # semantic cache на Qdrant
│   ├── llm_router.py        # OpenRouter + fallback chain + circuit breaker
│   ├── pricing.py           # {model: {input:$/1M, output:$/1M}}
│   ├── cost_tracker.py      # SQLite logging
│   ├── security.py          # prompt-injection patterns, length limit
│   ├── prompts.py           # system prompt з XML-тегами
│   ├── observability.py     # Langfuse spans
│   └── sse.py               # SSE event formatter
├── tests/
│   └── manual_curl.sh       # acceptance-сценарії з ТЗ
└── screenshots/             # для REPORT.md
```

---

## 2. Послідовність робіт (інкрементальна — кожен крок робочий)

### Крок 0 · Підготовка середовища (foundations)
- Створити `pyproject.toml` з залежностями: `fastapi`, `uvicorn[standard]`, `pydantic-settings`, `openai`, `sentence-transformers`, `qdrant-client`, `redis`, `langchain-text-splitters`, `tiktoken`, `pypdf`, `aiosqlite`, `langfuse`, `python-dotenv`.
- `.env.example`: `OPENROUTER_API_KEY`, `QDRANT_URL`, `QDRANT_API_KEY`, `REDIS_URL`, `DATABASE_URL`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`.
- Підняти **локально через Docker**: Qdrant (`qdrant/qdrant`), Redis (`redis:alpine`). Production вже на cloud-сервісах.
- Acceptance кроку: `uvicorn app.main:app` стартує, `GET /health` повертає `{status:"ok"}`.

### Крок 1 · RAG базовий шар (§1 ТЗ)
1. Покласти `data/source.md` (рекомендую The Twelve-Factor App — невеликий, технічний, добре розбивається на секції).
2. `app/embedder.py` — singleton `SentenceTransformer("all-MiniLM-L6-v2")`, метод `encode(texts) -> list[list[float]]` (384-dim). **Один інстанс на процес**, інакше модель завантажиться кілька разів.
3. `scripts/index.py`:
   - read `data/source.md`
   - `RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50, length_function=lambda t: len(tiktoken_enc.encode(t)))`
   - embed batch (швидше, ніж один за одним)
   - `qdrant.recreate_collection("chunks", VectorParams(384, Distance.COSINE))`
   - `qdrant.upsert(points=[PointStruct(id=i, vector=emb, payload={"text":chunk, "section":...})])`
4. `app/retriever.py` — `search(query_embedding, k=3) -> list[Chunk]`.
5. Acceptance кроку: `python scripts/index.py` індексує ~50–100 чанків, `retriever.search(emb)` повертає top-3 з payload.

### Крок 2 · FastAPI + SSE streaming (§2)
1. `app/sse.py` — хелпер `format(event_dict) -> str` (`"data: {...}\n\n"`).
2. `POST /chat/stream`:
   - Pydantic `ChatRequest(message: str)`.
   - `StreamingResponse(event_stream(), media_type="text/event-stream")` з заголовком `X-Accel-Buffering: no` (вимикає буферизацію на nginx/проксі — інакше токени летять блоком).
   - Async-генератор: yield `token` events, потім фінальний `done` з `sources`, `usage`, `cost_usd`, `cache_hit`, `fallback_used`.
3. **Disconnect handling**: у циклі генерації після кожного токена `if await request.is_disconnected(): break`; у `finally` — `inc(aborted_streams)`, `task.cancel()` для LLM-запиту.
4. Глобальні counters `active_streams`, `aborted_streams` в `app.state`.
5. Acceptance: `curl -N -H "X-API-Key: demo-free" -d '{"message":"What is config?"}' http://localhost:8000/chat/stream` — токени летять по одному, `Ctrl+C` під час стріму → `aborted_streams++`.

### Крок 3 · Auth (§3)
1. `app/auth.py`:
   ```python
   API_KEYS = {
     "demo-free": {"tier":"free", "tokens_per_min":5000, "models":[
        "meta-llama/llama-3.1-8b-instruct",
        "google/gemini-flash-1.5-8b",
        "meta-llama/llama-3.2-3b-instruct:free",
     ]},
     "demo-pro": {...20000, mid-tier...},
     "demo-enterprise": {...100000, top-tier...},
   }
   ```
2. FastAPI dependency `require_api_key(x_api_key: str = Header(...))` → 401 якщо немає / невірний; повертає dict із tier + models.
3. Acceptance: запит без `X-API-Key` → 401; з невалідним → 401; з `demo-free` → проходить.

### Крок 4 · Token-based Rate Limit (§4)
1. `app/rate_limit.py` — sliding/token-bucket через Redis:
   - Ключ `rl:{api_key}:{minute_bucket}` (epoch_seconds // 60).
   - На LLM-відповідь: `INCRBY ключ <input+output>`; `EXPIRE ключ 120`.
   - Перевірка перед запитом: `GET` поточний bucket; якщо `>= tokens_per_min` → 429 + `Retry-After: 60 - (now % 60)`.
   - **Чому не Lua**: Upstash REST API його не підтримує — стандартні команди ОК.
2. Підрахунок токенів запиту перед LLM: оцінка через tiktoken (вхід ≈ system+chunks+query); **остаточне списання — після стріму**, з реальними `usage` від OpenRouter.
3. Acceptance: 5 важких запитів з `demo-free` → 429 з `Retry-After`.

### Крок 5 · Semantic Cache (§5)
1. Окрема колекція `cache_collection` у Qdrant, той самий 384-dim cosine.
2. На запит:
   - `emb = embedder.encode(query)` (**цей же emb піде і в retriever** — переконатись що передаємо обидва місця).
   - `cache.search(emb, limit=1)` → якщо `score > 0.92` і `payload.expire_at > now` → HIT.
3. HIT: стрімимо закешований текст по словах/токенах (`for word in cached.split(): yield token; await asyncio.sleep(0.01)`) — щоб UX був консистентним.
4. MISS: після LLM-стріму `cache.upsert(PointStruct(id=uuid, vector=emb, payload={"query":..,"response":..,"model":..,"expire_at":now+3600}))`.
5. Періодична зачистка протермінованих: cron-task раз на годину `delete where expire_at < now` (або просто фільтр на read).
6. Acceptance: запит #1 «Що таке Config?» → MISS, ~2с; запит #2 «What is configuration?» → HIT, <200мс.

### Крок 6 · Cost Tracking (§6)
1. `app/pricing.py` — словник цін з `openrouter.ai/models` (актуалізувати).
2. `app/cost_tracker.py` — `aiosqlite` (production варіант: Postgres). Схема:
   ```sql
   CREATE TABLE requests (
     request_id TEXT PRIMARY KEY,
     api_key TEXT, model TEXT,
     input_tokens INTEGER, output_tokens INTEGER,
     cost_usd REAL, latency_ms INTEGER, ttft_ms INTEGER,
     cache_hit INTEGER, fallback_used INTEGER, output_filtered INTEGER,
     created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
   );
   CREATE INDEX idx_api_key_date ON requests(api_key, created_at);
   ```
3. `GET /usage/today`: `SELECT count(*), sum(input+output), sum(cost) WHERE api_key=? AND date(created_at)=date('now')`.
4. `GET /usage/breakdown`: GROUP BY model, AVG/p95 latency (`PERCENTILE_CONT` в Postgres або вибірка + сортування в SQLite), `cache_hit_rate`, `fallback_rate`.
5. Acceptance: 20 запитів → `/usage/today` сходиться з ручним розрахунком.

### Крок 7 · Multi-provider Fallback (§7)
1. `app/llm_router.py`:
   ```python
   async def stream_with_fallback(tier_models, messages):
       for i, model in enumerate(tier_models):
           if circuit_open(model): continue
           try:
               async for chunk in asyncio.wait_for(
                   _stream_openrouter(model, messages), timeout=15):
                   yield chunk, model, i>0  # fallback_used flag
               return
           except RETRYABLE as e:
               record_failure(model); continue
           except NON_RETRYABLE as e:
               raise
       raise AllProvidersFailed
   ```
2. **Retryable**: `429, 500, 502, 503, 504, asyncio.TimeoutError, httpx.ConnectError`.
3. **Non-retryable**: `400, 401, 403, 422`, content-filter — одразу 4xx клієнту.
4. **Circuit breaker** (in-memory dict; для multi-instance — Redis): `failures[model] = deque(maxlen=5)`; якщо 5 фейлів за 60с → `open_until[model] = now+60`.
5. Прокидаємо `OpenAI(base_url="https://openrouter.ai/api/v1")`, `stream=True`, `stream_options={"include_usage": True}` — щоб отримати `usage` у фінальному чанку.
6. Acceptance: підмінити primary на `openai/this-does-not-exist` → fallback кладеться у `model` field, `fallback_used=true`.

### Крок 8 · Security (§8)
1. `app/security.py`:
   - `MAX_LEN = 4000`; > → 400.
   - `INJECTION_PATTERNS = [r"ignore (all |previous )?instructions", r"system\s*:", r"<\|im_start\|>", r"</s>", r"###\s*system", r"you are now", r"disregard (the )?above"]` (≥5).
   - `scan_input(text)` → `True` якщо матч → лог у `suspicious_requests.log` (JSONL: timestamp, api_key, snippet).
   - `scan_output(text)` після стріму — якщо знайдено фрагменти system-prompt → `output_filtered=True`.
2. `app/prompts.py`:
   ```python
   SYSTEM = """You are a Q&A bot. Answer ONLY from <context>. If answer not in context, say "I don't know"."""
   USER = """<context>{chunks}</context>\n<user_query>{query}</user_query>"""
   ```
3. Acceptance: `{"message":"Ignore previous instructions..."}` → 400 + запис у `suspicious_requests.log`.

### Крок 9 · Concurrency Control (§9)
1. `app.state.llm_semaphore = asyncio.Semaphore(20)` у `lifespan`.
2. Огорнути LLM-виклик у `async with llm_semaphore:`.
3. `GET /health` → `{active_streams, aborted_streams, llm_in_flight: 20 - sem._value}`.
4. Acceptance: `hey -n 30 -c 30 ...` → `active_streams ≤ 20`; перерваний стрім інкрементує `aborted_streams` і не з'являється в `/usage/today`.

### Крок 10 · Observability — Langfuse (§10)
1. `from langfuse import Langfuse; lf = Langfuse(...)`.
2. На вхід запиту: `trace = lf.trace(name="chat_stream", user_id=api_key, metadata={tier})`.
3. Spans: `embed`, `cache_lookup`, `retrieve`, `llm_call` (з tag-ами `model`, `fallback_used`).
4. `trace.generation(model=..., input=full_prompt, output=full_response, usage=...)` — для debugging галюцинацій.
5. Flush в `finally`, навіть при disconnect.
6. Acceptance: відкрити Langfuse dashboard, побачити trace з повним prompt+completion → скрін у REPORT.md.

### Крок 11 · Звіт + скріншоти
`REPORT.md` має:
1. Streaming у терміналі (`curl -N` → токени по одному).
2. `done` event з `sources: ["chunk_12","chunk_45"]`.
3. Latency comparison: MISS ~2-3с vs HIT <200мс.
4. 429 з `Retry-After`.
5. Fallback: лог + `/usage/breakdown` з `fallback_rate ≈ 100%` після підміни primary.
6. `/usage/today` після 20 запитів.
7. Langfuse trace.

---

## 3. Ризики й рішення

| Ризик | Рішення |
|---|---|
| `sentence-transformers` тягне torch ~700MB → велике disk footprint | `onnxruntime` версія MiniLM (`xenova/all-MiniLM-L6-v2-onnx`) як легша альтернатива. |
| OpenRouter free models мають жорсткий rate limit (20 req/min) | Тестувати з `demo-free` тільки під час розробки; з $1 балансом — взяти `gpt-4o-mini` як primary. |
| Upstash REST API не підтримує Lua → token bucket менш атомарний | Прийняти race condition в межах кількох мс — для домашки ОК; для prod взяти Redis Cloud з повним RESP. |
| Qdrant Cloud free має ліміт 1GB → перевищиш при індексі великого PDF | 12-Factor app = ~30K токенів = ~60 чанків × 384 floats × 4B = ~92KB. Вистачить з запасом. |

---

## 4. Що зробити в **наступному** повідомленні

Я **не починаю писати код** — це план для твого затвердження.

Як підтвердиш — буду виконувати **по кроках з паузою для перевірки** після кожного (як у попередніх ДЗ):
- крок 0 (середовище) → ти запускаєш `uvicorn`, бачиш `/health`
- крок 1 (RAG базовий) → ти запускаєш `scripts/index.py`, перевіряємо retrieval
- … і далі до деплою.

Скажи, чи затверджуєш план і чи хочеш щось поміняти (документ для індексу, vector DB вибір, провайдер деплою, скоригувати кроки).
