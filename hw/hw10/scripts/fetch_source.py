"""
Завантажує The Twelve-Factor App (heroku/12factor, MIT license) у data/source.md.

Чому окремий скрипт, а не закомічений файл:
- джерело канонічне (raw GitHub), результат відтворюваний
- легко оновити (якщо heroku оновить контент)
- не роздуваємо git-репо на 30K токенів external content
"""
import sys
import urllib.request
from pathlib import Path

REPO_RAW = "https://raw.githubusercontent.com/heroku/12factor/master/content/en"

# Канонічний порядок з 12factor.net
FACTORS = [
    "intro",
    "codebase",
    "dependencies",
    "config",
    "backing-services",
    "build-release-run",
    "processes",
    "port-binding",
    "concurrency",
    "disposability",
    "dev-prod-parity",
    "logs",
    "admin-processes",
]


def fetch(name: str) -> str:
    url = f"{REPO_RAW}/{name}.md"
    print(f"  → {url}")
    with urllib.request.urlopen(url, timeout=15) as resp:
        return resp.read().decode("utf-8")


def main() -> int:
    out_path = Path(__file__).resolve().parent.parent / "data" / "source.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    parts: list[str] = [
        "# The Twelve-Factor App\n\n",
        "_Source: https://github.com/heroku/12factor (MIT license). "
        "Fetched by `scripts/fetch_source.py`._\n\n",
        "---\n\n",
    ]

    for name in FACTORS:
        try:
            content = fetch(name)
        except Exception as e:
            print(f"  ✗ failed {name}: {e}", file=sys.stderr)
            return 1
        parts.append(content.strip())
        parts.append("\n\n---\n\n")

    out_path.write_text("".join(parts), encoding="utf-8")
    size_kb = out_path.stat().st_size / 1024
    print(f"\n✓ Wrote {out_path} ({size_kb:.1f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
