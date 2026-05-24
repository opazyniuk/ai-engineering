"""
Демо-клієнт для /chat/stream: показує як токени летять у часі.

Друкує:
- [+0012ms] токен            ← time-to-first-token (TTFT)
- [+0062ms] токен            ← наступний — затримка від попереднього
- ...
- DONE з summary блоком

Запуск:
    python scripts/demo_stream.py "Where do I store secrets?"
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import httpx


def color(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if sys.stdout.isatty() else text


BOLD = lambda s: color("1", s)
DIM = lambda s: color("2", s)
CYAN = lambda s: color("36", s)
GREEN = lambda s: color("32", s)
YELLOW = lambda s: color("33", s)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("query", help="The question to send")
    p.add_argument("--url", default="http://localhost:8000/chat/stream")
    p.add_argument("--key", default="demo-pro", help="X-API-Key value")
    args = p.parse_args()

    print(DIM(f"POST {args.url}"))
    print(BOLD(f"Q: {args.query}"))
    print(DIM("─" * 70))

    t_start = time.perf_counter()
    ttft_ms: float | None = None
    tokens_received = 0
    last_token_time = t_start

    # SSE через httpx streaming: client сам не парсить data:-події,
    # але прості stream-рядки нам якраз і потрібні.
    with httpx.stream(
        "POST",
        args.url,
        headers={"Content-Type": "application/json", "X-API-Key": args.key},
        json={"message": args.query},
        timeout=30.0,
    ) as resp:
        if resp.status_code != 200:
            print(f"  HTTP {resp.status_code}: {resp.read().decode()}")
            return 1

        for raw_line in resp.iter_lines():
            if not raw_line or not raw_line.startswith("data: "):
                continue
            payload = json.loads(raw_line[6:])
            now = time.perf_counter()
            elapsed_ms = (now - t_start) * 1000

            if payload["type"] == "token":
                if ttft_ms is None:
                    ttft_ms = elapsed_ms
                    print(f"{GREEN(f'[+{elapsed_ms:6.0f}ms]')} "
                          f"{BOLD('TTFT')} ← {payload['content']!r}")
                else:
                    delta = (now - last_token_time) * 1000
                    print(f"{DIM(f'[+{elapsed_ms:6.0f}ms]')} "
                          f"{DIM(f'(Δ{delta:5.0f}ms)')} {payload['content']}")
                last_token_time = now
                tokens_received += 1

            elif payload["type"] == "done":
                print(DIM("─" * 70))
                total_ms = elapsed_ms
                print(f"{BOLD('DONE')} at +{total_ms:.0f}ms · "
                      f"{CYAN(str(tokens_received))} tokens · "
                      f"{CYAN(f'{tokens_received / (total_ms / 1000):.1f} tok/s')}")
                print(f"  TTFT:          {GREEN(f'{ttft_ms:.0f} ms')}")
                print(f"  total:         {total_ms:.0f} ms")
                print(f"  model:         {payload.get('model')}")
                print(f"  cache_hit:     {payload.get('cache_hit')}")
                print(f"  fallback_used: {payload.get('fallback_used')}")
                print(f"  usage:         {payload.get('usage')}")
                print(f"  cost_usd:      {payload.get('cost_usd')}")
                print(f"  {YELLOW('sources:')}")
                for src in payload.get("sources", []):
                    print(f"    - {src}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
