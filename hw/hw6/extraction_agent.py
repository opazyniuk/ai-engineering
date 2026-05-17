"""CLI extraction agent: meeting transcript -> structured JSON.

Same prompt, same parser, two providers:
  - openai (gpt-4o-mini via API)
  - ollama (llama3.2:3b via http://localhost:11434)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass

import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

PROMPT_TEMPLATE = """You are a meeting transcript parser. Extract structured data from the input.

Return ONLY a valid JSON object with this exact shape:
{{
  "summary": "<one sentence in Ukrainian describing what was decided>",
  "tasks": [
    {{
      "owner": "<person name who owns this task>",
      "task": "<short description in Ukrainian>",
      "deadline": "<YYYY-MM-DD or null>"
    }}
  ],
  "decisions": ["<decision in Ukrainian>"]
}}

Rules:
- Do NOT invent names that are not in the transcript.
- Do NOT invent dates. If unclear, set deadline to null.
- "tasks" are action items assigned to a specific person.
- "decisions" are policies or agreements (not action items).
- Output ONLY the JSON object. No markdown fences, no comments, no preamble.

Transcript:
{text}
"""


@dataclass
class ExtractionResult:
    provider: str
    parsed: dict | None
    raw: str
    valid_json: bool
    latency_seconds: float
    input_tokens: int
    output_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


def _parse_json(raw: str) -> dict | None:
    """Strict parse, then fallback: extract the outermost {...} block."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _call_openai(prompt: str) -> tuple[str, int, int]:
    client = OpenAI()
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0,
    )
    raw = resp.choices[0].message.content or ""
    return raw, resp.usage.prompt_tokens, resp.usage.completion_tokens


def _call_ollama(prompt: str) -> tuple[str, int, int]:
    resp = requests.post(
        f"{OLLAMA_HOST}/api/generate",
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "format": "json",
            "stream": False,
            "options": {"temperature": 0},
        },
        timeout=300,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["response"], data.get("prompt_eval_count", 0), data.get("eval_count", 0)


def extract(text: str, provider: str) -> ExtractionResult:
    prompt = PROMPT_TEMPLATE.format(text=text)

    start = time.perf_counter()
    if provider == "openai":
        raw, in_tok, out_tok = _call_openai(prompt)
    elif provider == "ollama":
        raw, in_tok, out_tok = _call_ollama(prompt)
    else:
        raise ValueError(f"unknown provider: {provider}")
    latency = time.perf_counter() - start

    parsed = _parse_json(raw)
    return ExtractionResult(
        provider=provider,
        parsed=parsed,
        raw=raw,
        valid_json=parsed is not None,
        latency_seconds=latency,
        input_tokens=in_tok,
        output_tokens=out_tok,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract structured JSON from a meeting transcript.")
    parser.add_argument("input_file", help="Path to .txt file with the transcript")
    parser.add_argument("--provider", choices=["openai", "ollama"], default="ollama")
    parser.add_argument("--save", help="Optional path to save the parsed JSON")
    args = parser.parse_args()

    with open(args.input_file, encoding="utf-8") as f:
        text = f.read()

    result = extract(text, args.provider)

    print(f"[{result.provider}] latency={result.latency_seconds:.2f}s "
          f"tokens={result.input_tokens}+{result.output_tokens}={result.total_tokens} "
          f"valid_json={result.valid_json}")

    if result.parsed is None:
        print("--- raw output (could not parse) ---", file=sys.stderr)
        print(result.raw, file=sys.stderr)
        return 1

    print(json.dumps(result.parsed, ensure_ascii=False, indent=2))

    if args.save:
        payload = {
            "meta": {k: v for k, v in asdict(result).items() if k not in {"parsed", "raw"}},
            "result": result.parsed,
        }
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
