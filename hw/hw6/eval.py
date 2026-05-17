"""Evaluation harness: runs every (dataset, provider) combination and writes eval_results.csv."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from extraction_agent import ExtractionResult, extract

ROOT = Path(__file__).parent
SAMPLES_DIR = ROOT / "samples"
RESULTS_DIR = ROOT / "results"
GOLD_PATH = SAMPLES_DIR / "_gold.json"

DATASETS = ["simple_meeting", "chaotic_standup", "technical_sync"]
PROVIDERS = ["ollama", "openai"]

# gpt-4o-mini pricing (USD per token), as of 2026.
PRICING = {
    "openai": {"input": 0.15 / 1_000_000, "output": 0.60 / 1_000_000},
    "ollama": {"input": 0.0, "output": 0.0},
}


def _normalize(value: str | None) -> str:
    return (value or "").strip().lower()


def _task_matches(model_task: dict, gold_task: dict) -> bool:
    """Owner must match exactly, plus at least one keyword must appear in the description."""
    if _normalize(model_task.get("owner")) != _normalize(gold_task["owner"]):
        return False
    description = _normalize(model_task.get("task"))
    return any(kw.lower() in description for kw in gold_task["task_keywords"])


def evaluate(parsed: dict | None, gold: dict) -> dict:
    """Compare model output against gold labels."""
    if parsed is None:
        return {
            "tasks_found": 0,
            "tasks_total": len(gold["tasks"]),
            "hallucinated_owners": 0,
            "decisions_found": 0,
            "decisions_total": gold["decisions_count"],
        }

    model_tasks = parsed.get("tasks", []) or []
    valid_owners_lower = {o.lower() for o in gold["valid_owners"]}

    # recall: how many gold tasks the model captured
    found = 0
    for gold_task in gold["tasks"]:
        if any(_task_matches(mt, gold_task) for mt in model_tasks):
            found += 1

    # hallucinated owners: model assigned a task to someone not in valid_owners
    hallucinated = sum(
        1
        for mt in model_tasks
        if _normalize(mt.get("owner")) and _normalize(mt.get("owner")) not in valid_owners_lower
    )

    # decisions: count how many gold decisions appear (any keyword from each group)
    model_decisions = parsed.get("decisions", []) or []
    decisions_blob = " ".join(model_decisions).lower()
    decisions_found = sum(
        1
        for keyword_group in gold["decision_keywords"]
        if all(kw.lower() in decisions_blob for kw in keyword_group)
    )

    return {
        "tasks_found": found,
        "tasks_total": len(gold["tasks"]),
        "hallucinated_owners": hallucinated,
        "decisions_found": decisions_found,
        "decisions_total": gold["decisions_count"],
    }


def cost_usd(provider: str, in_tokens: int, out_tokens: int) -> float:
    p = PRICING[provider]
    return in_tokens * p["input"] + out_tokens * p["output"]


def save_raw(path: Path, result: ExtractionResult, metrics: dict) -> None:
    payload = {
        "meta": {
            "provider": result.provider,
            "valid_json": result.valid_json,
            "latency_seconds": result.latency_seconds,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
        },
        "metrics": metrics,
        "result": result.parsed,
        "raw": None if result.valid_json else result.raw,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    gold_all = json.loads(GOLD_PATH.read_text(encoding="utf-8"))
    rows: list[dict] = []

    for dataset in DATASETS:
        text = (SAMPLES_DIR / f"{dataset}.txt").read_text(encoding="utf-8")
        gold = gold_all[dataset]
        for provider in PROVIDERS:
            print(f"→ {dataset} / {provider} ...", flush=True)
            result = extract(text, provider)
            metrics = evaluate(result.parsed, gold)
            cost = cost_usd(provider, result.input_tokens, result.output_tokens)

            save_raw(RESULTS_DIR / f"{dataset}_{provider}.json", result, metrics)

            rows.append({
                "dataset": dataset,
                "provider": provider,
                "valid_json": "✅" if result.valid_json else "❌",
                "tasks": f"{metrics['tasks_found']}/{metrics['tasks_total']}",
                "halluc_owners": metrics["hallucinated_owners"],
                "decisions": f"{metrics['decisions_found']}/{metrics['decisions_total']}",
                "in_tok": result.input_tokens,
                "out_tok": result.output_tokens,
                "total_tok": result.total_tokens,
                "cost_usd": round(cost, 6),
                "latency_s": round(result.latency_seconds, 2),
            })

    df = pd.DataFrame(rows)
    csv_path = ROOT / "eval_results.csv"
    df.to_csv(csv_path, index=False)
    print()
    print(df.to_string(index=False))
    print(f"\nSaved: {csv_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
