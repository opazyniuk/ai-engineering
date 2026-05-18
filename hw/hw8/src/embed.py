"""Embed corpus + queries with OpenAI text-embedding-3-small.

Outputs:
  data/corpus_embeddings.npy   shape (N_corpus, 1536) float32
  data/corpus_ids.json         list[str], parallel to rows of .npy
  data/query_embeddings.npy    shape (N_queries, 1536) float32
  data/query_ids.json          list[str]

Resume: writes in chunks to *_part_NNN.npy, then concatenates. If the script
crashes mid-way, rerun — existing chunks are skipped.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

MODEL = "text-embedding-3-small"
DIM = 1536
CHUNK_SIZE = 50_000          # rows per part-file (≈ 305 MB on disk)
BATCH_SIZE = 256              # texts per single API call
PARALLEL_REQUESTS = 8         # concurrent in-flight requests within one chunk


def read_jsonl(path: Path) -> tuple[list[str], list[str]]:
    """Returns (ids, texts) parallel lists, in file order."""
    ids, texts = [], []
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            ids.append(row["_id"])
            texts.append(row["text"])
    return ids, texts


def embed_batch(client: OpenAI, texts: list[str]) -> np.ndarray:
    """One API call. Returns (len(texts), DIM) float32 array."""
    resp = client.embeddings.create(model=MODEL, input=texts)
    arr = np.array([d.embedding for d in resp.data], dtype=np.float32)
    return arr


def embed_chunk(client: OpenAI, texts: list[str], batch_size: int, parallel: int) -> np.ndarray:
    """Embed a chunk of texts. Splits into batches, runs N in parallel."""
    n = len(texts)
    out = np.zeros((n, DIM), dtype=np.float32)
    batches = [(i, texts[i : i + batch_size]) for i in range(0, n, batch_size)]

    with ThreadPoolExecutor(max_workers=parallel) as ex:
        futs = {ex.submit(embed_batch, client, b): (start, len(b)) for start, b in batches}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="  batches", leave=False):
            start, size = futs[fut]
            out[start : start + size] = fut.result()
    return out


def embed_file(
    client: OpenAI,
    input_path: Path,
    out_npy: Path,
    out_ids: Path,
    chunk_size: int,
    batch_size: int,
    parallel: int,
    force: bool,
) -> None:
    if out_npy.exists() and out_ids.exists() and not force:
        size_gb = out_npy.stat().st_size / 1e9
        print(f"[skip] {out_npy} already exists ({size_gb:.2f} GB). Use --force to redo.")
        return

    ids, texts = read_jsonl(input_path)
    n = len(texts)
    print(f"[{input_path.name}] {n:,d} texts → embedding in chunks of {chunk_size:,d}")

    parts_dir = out_npy.parent / f".{out_npy.stem}_parts"
    parts_dir.mkdir(exist_ok=True)

    num_chunks = (n + chunk_size - 1) // chunk_size
    for chunk_idx in range(num_chunks):
        part_path = parts_dir / f"part_{chunk_idx:04d}.npy"
        if part_path.exists() and not force:
            print(f"  [chunk {chunk_idx + 1}/{num_chunks}] cached at {part_path.name}")
            continue

        start = chunk_idx * chunk_size
        end = min(start + chunk_size, n)
        print(f"  [chunk {chunk_idx + 1}/{num_chunks}] embedding rows {start:,d}-{end:,d}")
        t0 = time.perf_counter()
        chunk = embed_chunk(client, texts[start:end], batch_size, parallel)
        np.save(part_path, chunk)
        elapsed = time.perf_counter() - t0
        rate = (end - start) / elapsed
        print(f"    saved {part_path.name} in {elapsed:.1f}s ({rate:.0f} texts/sec)")

    # concatenate parts
    print(f"  concatenating {num_chunks} parts → {out_npy}")
    parts = [np.load(parts_dir / f"part_{i:04d}.npy") for i in range(num_chunks)]
    full = np.concatenate(parts, axis=0)
    assert full.shape == (n, DIM), f"got {full.shape}, expected ({n}, {DIM})"
    np.save(out_npy, full)

    with out_ids.open("w") as f:
        json.dump(ids, f)

    # cleanup parts
    for i in range(num_chunks):
        (parts_dir / f"part_{i:04d}.npy").unlink()
    parts_dir.rmdir()

    print(f"[done] {out_npy} shape={full.shape} size={out_npy.stat().st_size/1e9:.2f} GB")


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data", type=Path)
    ap.add_argument("--corpus-only", action="store_true")
    ap.add_argument("--queries-only", action="store_true")
    ap.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--parallel", type=int, default=PARALLEL_REQUESTS)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY not set (check .env)")
    client = OpenAI(api_key=api_key, max_retries=5, timeout=60.0)

    do_corpus = not args.queries_only
    do_queries = not args.corpus_only

    if do_corpus:
        embed_file(
            client,
            input_path=args.data_dir / "corpus.jsonl",
            out_npy=args.data_dir / "corpus_embeddings.npy",
            out_ids=args.data_dir / "corpus_ids.json",
            chunk_size=args.chunk_size,
            batch_size=args.batch_size,
            parallel=args.parallel,
            force=args.force,
        )

    if do_queries:
        embed_file(
            client,
            input_path=args.data_dir / "queries.jsonl",
            out_npy=args.data_dir / "query_embeddings.npy",
            out_ids=args.data_dir / "query_ids.json",
            chunk_size=args.chunk_size,
            batch_size=args.batch_size,
            parallel=args.parallel,
            force=args.force,
        )


if __name__ == "__main__":
    main()
