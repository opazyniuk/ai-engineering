"""
Крок 1.4 (частина 3): бенчмарк latency сервера.

Викликає POST /generate для довжин 50/100/200 токенів,
заміряє server-side gen_time + total round-trip.
3 повтори на кожну довжину для усереднення.
"""
import statistics
import time

import requests


URL = "http://localhost:8000/generate"
LENGTHS = [50, 100, 200]
REPEATS = 3
TEMPERATURE = 1.0
TOP_K = 20


def warmup() -> None:
    print("warmup...", flush=True)
    requests.post(URL, json={"max_tokens": 20, "temperature": 1.0, "top_k": 20}, timeout=60)


def bench_one(max_tokens: int) -> tuple[float, float, float]:
    """Returns (server_time, total_time, tokens_per_s) for a single request."""
    t0 = time.perf_counter()
    r = requests.post(
        URL,
        json={"max_tokens": max_tokens, "temperature": TEMPERATURE, "top_k": TOP_K},
        timeout=120,
    )
    total = time.perf_counter() - t0
    r.raise_for_status()
    data = r.json()
    return data["gen_time_s"], total, data["tokens_per_s"]


def main() -> None:
    print(f"benchmarking {URL}")
    print(f"lengths={LENGTHS}, repeats={REPEATS}, temp={TEMPERATURE}, top_k={TOP_K}")
    print()

    warmup()
    print()

    print(f"{'length':>7} | {'server avg':>11} | {'server std':>11} | {'total avg':>10} | {'tok/s avg':>9}")
    print("-" * 70)

    results = []
    for n in LENGTHS:
        servers, totals, rates = [], [], []
        for _ in range(REPEATS):
            s, t, r = bench_one(n)
            servers.append(s)
            totals.append(t)
            rates.append(r)
        avg_s = statistics.mean(servers)
        std_s = statistics.stdev(servers) if len(servers) > 1 else 0.0
        avg_t = statistics.mean(totals)
        avg_r = statistics.mean(rates)
        results.append((n, avg_s, std_s, avg_t, avg_r))
        print(f"{n:>7} | {avg_s*1000:>8.1f} ms | {std_s*1000:>8.1f} ms | {avg_t*1000:>7.1f} ms | {avg_r:>7.1f}")

    print()
    print("ratio analysis (server time):")
    base = results[0][1]
    for n, avg_s, _, _, _ in results:
        print(f"  {n:>3} tokens: {avg_s/base:.2f}× of 50-token baseline")


if __name__ == "__main__":
    main()
