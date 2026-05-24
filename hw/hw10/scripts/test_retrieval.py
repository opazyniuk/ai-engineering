"""
Швидкий smoke-test для retrieval. Дзвонимо retriever.search(query) на кількох
тестових запитах і друкуємо top-3 chunks. Це acceptance кроку 1.

Запуск:  python scripts/test_retrieval.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.retriever import search  # noqa: E402


QUERIES = [
    "How should an app store its configuration?",
    "What is a backing service?",
    "Why use environment variables for secrets?",
    "How do you handle logs in a twelve-factor app?",
    # семантично віддалене питання — щоб побачити нижчий score
    "What is the capital of France?",
]


def main() -> int:
    for q in QUERIES:
        print(f"\n▸ Query: {q!r}")
        hits = search(q, k=3)
        for i, h in enumerate(hits, 1):
            preview = h.text.replace("\n", " ")[:120]
            print(f"  [{i}] score={h.score:.3f}  section={h.section!r}")
            print(f"      {preview}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
