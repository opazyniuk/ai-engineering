"""
Ціни моделей OpenRouter (станом на квартал виконання домашки).

Структура: { "<openrouter-slug>": {"input": $/1M, "output": $/1M} }

Актуалізувати з https://openrouter.ai/models — ціни змінюються.
Якщо моделі немає у словнику — calculate_cost() поверне 0 і залогує warning.
"""
from __future__ import annotations


PRICING: dict[str, dict[str, float]] = {
    # ─── :free OpenRouter моделі ────────────────────────────────────────────
    # Всі $0/$0 — це open-source ваги, хостяться різними провайдерами
    # за рахунок OpenRouter. Обмеження: жорсткі rate limits на upstream.
    "openai/gpt-oss-120b:free":     {"input": 0.0, "output": 0.0},
    "openai/gpt-oss-20b:free":      {"input": 0.0, "output": 0.0},
    "poolside/laguna-m.1:free":     {"input": 0.0, "output": 0.0},
    "poolside/laguna-xs.2:free":    {"input": 0.0, "output": 0.0},
    "meta-llama/llama-3.2-3b-instruct:free":  {"input": 0.0, "output": 0.0},
    "meta-llama/llama-3.3-70b-instruct:free": {"input": 0.0, "output": 0.0},

    # ─── Приклади платних моделей (для майбутніх tier-ів після поповнення) ──
    # Залишаємо для довідки і легкого upgrade без переписування коду:
    "openai/gpt-4o-mini":                       {"input": 0.15, "output": 0.60},
    "openai/gpt-4o":                            {"input": 2.50, "output": 10.00},
    "anthropic/claude-3.5-sonnet":              {"input": 3.00, "output": 15.00},
    "mistralai/mistral-small-3.1-24b-instruct": {"input": 0.10, "output": 0.30},
    "meta-llama/llama-3.1-8b-instruct":         {"input": 0.02, "output": 0.05},
    "google/gemini-flash-1.5-8b":               {"input": 0.04, "output": 0.15},

    # ─── Mock (для кроків 2-6 без OpenRouter) ───────────────────────────────
    "mock-llm-v0": {"input": 0.0, "output": 0.0},
}


def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """
    Ціна в USD. Якщо модель невідома — повертаємо 0 + warning, не падаємо.
    Це навмисно: краще трохи недоцінити, ніж знести production-запит.
    """
    p = PRICING.get(model)
    if p is None:
        print(f"[pricing] WARNING: unknown model {model!r}, cost=0")
        return 0.0
    return (input_tokens * p["input"] + output_tokens * p["output"]) / 1_000_000
