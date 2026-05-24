"""
Auth: hardcoded API keys + tier metadata + FastAPI dependency.

API_KEYS — єдине джерело правди. Має все, що потрібно іншим шарам:
  - tokens_per_min  → rate limit (крок 4)
  - models[]        → fallback chain (крок 7)
  - tier name       → labelling у cost log і Langfuse (крок 6, 10)

В production цей dict замінився б на запит до Postgres / Vault. Інтерфейс
require_api_key() лишився б тим самим — тому жоден інший шар не доведеться
переписувати.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

from fastapi import Header, HTTPException, Request, status


@dataclass(frozen=True)
class TierInfo:
    """Усе, що інші шари знають про автора запиту."""
    api_key: str
    tier: str
    tokens_per_min: int
    models: list[str]   # [primary, fallback_1, fallback_2]


# ─── Hardcoded keys ──────────────────────────────────────────────────────────
# Моделі обрані з openrouter.ai/models. Усі мають OpenRouter slug.
# `:free` суфікс — безкоштовні моделі (з жорстким upstream rate limit).
#
# Принцип fallback chain:
#   models[0] = primary (основна якість/ціна для tier'у)
#   models[1] = інший провайдер схожої якості (захист від OpenAI outage etc.)
#   models[2] = маленька/безкоштовна як останній рубіж (graceful degradation)
# NOTE: Усі tier-и використовують `:free` моделі OpenRouter (домашка робиться
# без поповнення балансу). Tier відрізняє лише tokens_per_min — щоб логіка
# rate-limit-у і fallback chain була показовою.
#
# Перевірено на 2026-05-25: ці 3 моделі повертають coнтент через streaming і не
# rate-limited на upstream. Більшість інших `:free` (llama-3.3-70b, gemma-4,
# deepseek-v4-flash, qwen3) — або 429 від upstream-провайдера, або reasoning
# моделі що кладуть відповідь у поле `reasoning` замість `content` (для нашого
# pipeline незручно).
#
# Якщо ці моделі теж почнуть 429-итись — fallback chain автоматично перейде
# до наступної в списку. Це і є сенс multi-provider fallback.

_FREE_FALLBACK_CHAIN = [
    "openai/gpt-oss-120b:free",    # primary  — 120B params, OpenInference
    "poolside/laguna-m.1:free",    # fallback 1 — інший провайдер (Poolside)
    "openai/gpt-oss-20b:free",     # fallback 2 — менша модель, швидша
]


API_KEYS: dict[str, TierInfo] = {
    "demo-free": TierInfo(
        api_key="demo-free",
        tier="free",
        tokens_per_min=5_000,
        models=_FREE_FALLBACK_CHAIN,
    ),
    "demo-pro": TierInfo(
        api_key="demo-pro",
        tier="pro",
        tokens_per_min=20_000,
        models=_FREE_FALLBACK_CHAIN,
    ),
    "demo-enterprise": TierInfo(
        api_key="demo-enterprise",
        tier="enterprise",
        tokens_per_min=100_000,
        models=_FREE_FALLBACK_CHAIN,
    ),
}


# ─── FastAPI dependency ──────────────────────────────────────────────────────

def require_api_key(
    request: Request,
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> TierInfo:
    """
    Витягує X-API-Key, валідує, кладе TierInfo у request.state.
    Інші ендпоінти отримують через Depends(require_api_key).

    401 при відсутності або невалідному ключі. Свідомо не розрізняємо
    "no header" і "bad key" у тілі — щоб не давати підказок зловмисникам.
    """
    if not x_api_key or x_api_key not in API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-API-Key",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    tier = API_KEYS[x_api_key]
    # Прокидаємо у request.state — щоб middleware/обробники могли логувати без re-lookup.
    request.state.tier = tier
    return tier
