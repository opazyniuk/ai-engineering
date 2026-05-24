"""
Інтерактивний CLI для retrieval — без LLM, без HTTP, лише векторний пошук.

Запуск:
    python scripts/query.py "How should secrets be stored?"
    python scripts/query.py "What is a backing service?" -k 5
    python scripts/query.py        # інтерактивний режим, ENTER=exit
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.retriever import search  # noqa: E402


# ─── ANSI colors (вимикаються якщо --no-color або не tty) ────────────────────
_USE_COLOR = sys.stdout.isatty()

def c(code: str, text: str) -> str:
    if not _USE_COLOR:
        return text
    return f"\033[{code}m{text}\033[0m"

DIM = lambda s: c("2", s)
BOLD = lambda s: c("1", s)
CYAN = lambda s: c("36", s)
GREEN = lambda s: c("32", s)
YELLOW = lambda s: c("33", s)
GREY = lambda s: c("90", s)


def _score_color(score: float) -> str:
    if score >= 0.6:
        return "32"   # green
    if score >= 0.4:
        return "33"   # yellow
    return "90"       # grey (low relevance)


def _bar(score: float, width: int = 20) -> str:
    """Візуальна шкала: ▇▇▇▇▇▁▁▁▁▁  для score 0.50."""
    filled = max(0, min(width, int(round(score * width))))
    return c(_score_color(score), "▇" * filled) + DIM("▁" * (width - filled))


def render(query: str, hits, latency_ms: float) -> None:
    term_width = shutil.get_terminal_size((80, 20)).columns
    rule = DIM("─" * min(term_width, 80))

    print(rule)
    print(f"{BOLD(' Q ')}{CYAN(query)}")
    print(f"{DIM(f' top-{len(hits)} · {latency_ms:.0f} ms · embedding + vector search')}")
    print(rule)

    if not hits:
        print(YELLOW("  (no results — is the 'chunks' collection indexed?)"))
        return

    for i, h in enumerate(hits, 1):
        score_str = c(_score_color(h.score), f"{h.score:.3f}")
        section = h.section or "—"
        print(f"\n {BOLD(f'#{i}')}  score={score_str}  {_bar(h.score)}  "
              f"{DIM('section:')} {GREEN(section)}  {DIM(f'id={h.chunk_id[:8]}')}")
        # текст з відступом, обрізаний для читабельності
        snippet = h.text.strip().replace("\n", " ")
        max_chars = max(120, term_width - 6)
        if len(snippet) > max_chars:
            snippet = snippet[:max_chars].rsplit(" ", 1)[0] + DIM(" …")
        print(f"     {snippet}")

    print()
    print(rule)


def run_one(query: str, k: int) -> None:
    t0 = time.perf_counter()
    hits = search(query, k=k)
    dt_ms = (time.perf_counter() - t0) * 1000
    render(query, hits, dt_ms)


def main() -> int:
    p = argparse.ArgumentParser(description="Query the indexed RAG corpus (no LLM)")
    p.add_argument("query", nargs="?", help="If omitted — interactive mode")
    p.add_argument("-k", type=int, default=3, help="top-k chunks (default 3)")
    p.add_argument("--no-color", action="store_true")
    args = p.parse_args()

    global _USE_COLOR
    if args.no_color:
        _USE_COLOR = False

    if args.query:
        run_one(args.query, args.k)
        return 0

    # Інтерактивний режим
    print(DIM("Interactive retriever. Empty line to exit.\n"))
    while True:
        try:
            q = input(BOLD("ask> ")).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not q:
            return 0
        run_one(q, args.k)


if __name__ == "__main__":
    raise SystemExit(main())
