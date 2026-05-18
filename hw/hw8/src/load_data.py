"""Download BeIR/quora and dump it as corpus.jsonl / queries.jsonl / qrels.tsv.

Run once. Files land in data/ and are reused by embed.py and runner.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


def dump_corpus(out_path: Path) -> int:
    ds = load_dataset("BeIR/quora", "corpus", split="corpus")
    with out_path.open("w") as f:
        for row in tqdm(ds, desc="corpus", unit="doc"):
            f.write(json.dumps({"_id": row["_id"], "text": row["text"]}) + "\n")
    return len(ds)


def dump_queries(out_path: Path) -> int:
    ds = load_dataset("BeIR/quora", "queries", split="queries")
    with out_path.open("w") as f:
        for row in tqdm(ds, desc="queries", unit="q"):
            f.write(json.dumps({"_id": row["_id"], "text": row["text"]}) + "\n")
    return len(ds)


def dump_qrels(out_path: Path) -> tuple[int, set[str]]:
    ds = load_dataset("BeIR/quora-qrels", split="test")
    seen_queries: set[str] = set()
    with out_path.open("w") as f:
        f.write("query-id\tcorpus-id\tscore\n")
        for row in ds:
            qid = str(row["query-id"])
            did = str(row["corpus-id"])
            score = int(row["score"])
            f.write(f"{qid}\t{did}\t{score}\n")
            seen_queries.add(qid)
    return len(ds), seen_queries


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data", type=Path)
    ap.add_argument("--force", action="store_true", help="overwrite existing files")
    args = ap.parse_args()

    args.data_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = args.data_dir / "corpus.jsonl"
    queries_path = args.data_dir / "queries.jsonl"
    qrels_path = args.data_dir / "qrels.tsv"

    paths = [corpus_path, queries_path, qrels_path]
    if all(p.exists() for p in paths) and not args.force:
        print("All files already exist. Use --force to overwrite. Paths:")
        for p in paths:
            print(f"  {p}  ({p.stat().st_size / 1e6:.1f} MB)")
        return

    n_corpus = dump_corpus(corpus_path)
    n_queries = dump_queries(queries_path)
    n_qrels, qrels_qids = dump_qrels(qrels_path)

    # cross-check: every qid in qrels must exist in queries.jsonl
    with queries_path.open() as f:
        query_ids = {json.loads(line)["_id"] for line in f}
    missing = qrels_qids - query_ids
    if missing:
        print(f"WARN: {len(missing)} qrel query-ids are not in queries.jsonl "
              f"(sample: {list(missing)[:3]})")

    print()
    print(f"corpus:  {n_corpus:>8,d} docs    → {corpus_path}")
    print(f"queries: {n_queries:>8,d} queries → {queries_path}")
    print(f"qrels:   {n_qrels:>8,d} pairs   → {qrels_path}")
    print(f"unique queries with qrels: {len(qrels_qids):,d}")


if __name__ == "__main__":
    main()
