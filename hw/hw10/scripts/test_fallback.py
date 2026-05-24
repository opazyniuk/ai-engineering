"""
Acceptance §7: примусово підставляємо завідомо невалідну primary-модель —
має автоматично переключитись на fallback. Без рестарту сервера: тестуємо
llm_router напряму, бо це чистий модуль.

Запуск:  python scripts/test_fallback.py
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.llm_router import (  # noqa: E402
    StreamResult, breaker_snapshot, stream_with_fallback,
)


SCENARIOS = [
    {
        "name": "ALL VALID → primary works (no fallback)",
        "models": [
            "openai/gpt-oss-120b:free",
            "poolside/laguna-m.1:free",
            "openai/gpt-oss-20b:free",
        ],
        "expect_fallback": False,
        "expect_error": False,
    },
    {
        "name": "RETRYABLE 429 PRIMARY → fallback to #2",
        # llama-3.3-70b:free зараз rate-limited upstream → 429
        "models": [
            "meta-llama/llama-3.3-70b-instruct:free",   # 429 retryable
            "openai/gpt-oss-120b:free",                  # це має спрацювати
            "openai/gpt-oss-20b:free",
        ],
        "expect_fallback": True,
        "expect_error": False,
    },
    {
        "name": "TWO RETRYABLE 429 → fallback to #3",
        "models": [
            "meta-llama/llama-3.3-70b-instruct:free",   # 429
            "meta-llama/llama-3.2-3b-instruct:free",    # 429
            "openai/gpt-oss-120b:free",                  # OK
        ],
        "expect_fallback": True,
        "expect_error": False,
    },
    {
        "name": "NON-RETRYABLE 400 (invalid model) → NO fallback, raises",
        # Це config-помилка, не service-availability. Спеціально не fallback.
        "models": [
            "openai/this-does-not-exist:free",
            "openai/gpt-oss-120b:free",
        ],
        "expect_fallback": False,
        "expect_error": True,
    },
]


def check(label: str, ok: bool) -> str:
    return f"✓ PASS · {label}" if ok else f"✗ FAIL · {label}"


async def run_scenario(name: str, models: list[str],
                       expect_fallback: bool, expect_error: bool) -> bool:
    print(f"\n{'─' * 80}")
    print(f"▸ {name}")
    print(f"  chain: {models}")
    print(f"{'─' * 80}")

    result = StreamResult()
    messages = [
        {"role": "system",
         "content": "You are a concise assistant. Answer in one short sentence."},
        {"role": "user",
         "content": "What is the capital of France?"},
    ]

    raised: Exception | None = None
    tokens: list[str] = []
    try:
        async for tok in stream_with_fallback(models, messages, result):
            tokens.append(tok)
            print(tok, end="", flush=True)
    except Exception as e:
        raised = e
        print(f"\n  RAISED: {e.__class__.__name__}: {str(e)[:120]}")

    print()
    print(f"  model         : {result.model}")
    print(f"  fallback_used : {result.fallback_used}")
    print(f"  attempts      : {result.attempts}")
    print(f"  usage         : {result.usage}")
    print(f"  tokens        : {len(tokens)}")
    if result.error:
        print(f"  error         : {result.error[:120]}")

    passed = True
    print()
    print("  " + check(f"fallback_used = {expect_fallback}",
                        result.fallback_used == expect_fallback))
    passed &= result.fallback_used == expect_fallback
    print("  " + check(f"raised exception = {expect_error}",
                        (raised is not None) == expect_error))
    passed &= (raised is not None) == expect_error
    return passed


async def main() -> None:
    results = []
    for sc in SCENARIOS:
        ok = await run_scenario(sc["name"], sc["models"],
                                sc["expect_fallback"], sc["expect_error"])
        results.append((sc["name"], ok))
        await asyncio.sleep(1)

    print(f"\n{'═' * 80}")
    print("SUMMARY")
    print(f"{'═' * 80}")
    for name, ok in results:
        mark = "✓" if ok else "✗"
        print(f"  {mark}  {name}")
    print(f"\n  {sum(1 for _, ok in results if ok)}/{len(results)} scenarios passed")

    print(f"\n{'─' * 80}")
    print("Circuit breakers після всіх тестів:")
    snap = breaker_snapshot()
    for model, info in snap.items():
        print(f"  {model:50}  state={info['state']:9}  failures={info['recent_failures']}")


if __name__ == "__main__":
    asyncio.run(main())
