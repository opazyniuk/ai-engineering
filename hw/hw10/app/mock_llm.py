"""
Mock LLM — async generator, що імітує token-by-token стрімінг.

Чому потрібен:
- На кроці 2 ми хочемо протестувати SSE pipeline без OpenRouter ключа і без витрат.
- Видає реальну відповідь, побудовану з retrieved chunks — щоб бачити, що
  retrieval працює і RAG-зв'язок «питання → джерела → відповідь» цілий.

Буде замінений на справжній OpenRouter stream на кроці 7. Інтерфейс async-generator
збережемо, щоб main.py не довелось переписувати.
"""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from .retriever import RetrievedChunk


# Latency симулюємо реалістично: ~50мс на токен ≈ ~20 tok/s
# (приблизно як gpt-4o-mini у середньому)
_TOKEN_DELAY_S = 0.05


def _build_mock_answer(query: str, chunks: list[RetrievedChunk]) -> str:
    """Скласти текст-відповідь з retrieved chunks (без LLM)."""
    if not chunks:
        return "I don't know — no relevant context was found in the indexed document."

    # Беремо короткий уривок з top-1 chunk + посилання на секції.
    top = chunks[0]
    snippet = top.text.strip().split("\n\n")[0]
    if len(snippet) > 280:
        snippet = snippet[:280].rsplit(" ", 1)[0] + "..."

    sections = ", ".join(c.section for c in chunks if c.section) or "the indexed document"
    return (
        f"[MOCK answer · top score={top.score:.2f}] Based on {sections}: "
        f"{snippet}"
    )


async def stream_mock(query: str, chunks: list[RetrievedChunk]) -> AsyncIterator[str]:
    """
    Yield-ить «токени» (тут — слова, для простоти) з паузами.
    Якщо споживач відключиться — asyncio.CancelledError автоматично пробивається
    через asyncio.sleep і генератор припиняється.
    """
    answer = _build_mock_answer(query, chunks)
    for word in answer.split(" "):
        await asyncio.sleep(_TOKEN_DELAY_S)
        yield word + " "


def estimate_usage(query: str, chunks: list[RetrievedChunk], answer: str) -> dict[str, int]:
    """Приблизний підрахунок токенів — на кроці 7 замінимо на реальний usage від OpenRouter."""
    # 4 chars ≈ 1 token (rough English heuristic)
    context = "\n".join(c.text for c in chunks)
    input_tokens = (len(query) + len(context)) // 4
    output_tokens = max(1, len(answer) // 4)
    return {"input_tokens": input_tokens, "output_tokens": output_tokens}
