# HW10 · Production-Ready RAG API

> Q&A bot про документ з усіма production-шарами: RAG, SSE streaming, semantic cache,
> token-based rate limit, cost tracking, multi-provider fallback, observability, deploy.

**Статус**: крок 0 — підготовка середовища.
План реалізації: див. [PLAN.md](./PLAN.md).
5 ключових понять: див. [education.md](./education.md).

---

## Quick start (локально)

### 1. Залежності

```bash
cd hw/hw10
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .          # читає pyproject.toml
```

> Перший запуск завантажить модель `all-MiniLM-L6-v2` (~80MB) у `~/.cache/huggingface`.

### 2. Зовнішні сервіси (Qdrant + Redis локально)

```bash
docker compose up -d
docker compose ps         # qdrant на :6333, redis на :6379
```

### 3. Конфіг

```bash
cp .env.example .env
# додай OPENROUTER_API_KEY (опціонально на кроці 0)
```

### 4. Стартуємо

```bash
uvicorn app.main:app --reload --port 8000
```

Перевірка:
```bash
curl -s localhost:8000/health | jq
# {"status":"ok","active_streams":0,"aborted_streams":0}
```

---

## Структура

```
app/                 # FastAPI service
scripts/index.py     # індексація документа в Qdrant
data/source.md       # документ для RAG
tests/               # acceptance scripts (curl-based)
screenshots/         # для REPORT.md
docker-compose.yml   # Qdrant + Redis локально
fly.toml             # production deploy
```

---

## Endpoints (фінальний набір — у міру додавання кроків)

| Метод | Шлях                | Призначення                                  |
|-------|---------------------|-----------------------------------------------|
| POST  | `/chat/stream`      | RAG chat (SSE) — основний endpoint            |
| GET   | `/health`           | liveness probe + counters                     |
| GET   | `/usage/today`      | витрати за сьогодні (per API key)             |
| GET   | `/usage/breakdown`  | розбивка по моделях, hit rate, latency        |
| POST  | `/index/rebuild`    | переіндексувати документ (admin)              |
