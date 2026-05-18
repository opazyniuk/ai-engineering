"""Finalize benchmark: fill in missing data points without redoing the slow stuff.

What this does:
  1. Parse results/runner_summary.log → recover HNSW/Qdrant/pgvector(16/32/64) rows.
  2. Re-run FAISS-Flat (cheap: index in 7s, then 1000 queries × 1 rep ~ 80s).
  3. Re-run Chroma full (re-index ~28 min + 1000 queries).
  4. REUSE existing pgvector index in Postgres → measure ef=128, ef=256
     on 500 queries × 1 rep (no rebuild).
  5. Write CSV incrementally to results/results.csv (survives crash).
"""

from __future__ import annotations

import csv
import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from benchmarks.chroma_db import ChromaDB
from benchmarks.faiss_flat import FaissFlatDB
from benchmarks.pgvector_db import PgvectorDB
from metrics import mrr_at_k, percentiles, recall_at_k
from runner import load_data

DIM = 1536
TOP_K = 10
WARMUP = 50

FIELDS = [
    "db", "num_queries", "index_time_sec", "disk_mb",
    "latency_p50_ms", "latency_p95_ms", "latency_p99_ms", "latency_mean_ms",
    "recall_qrels@10", "mrr_qrels@10", "recall_flat@10",
]


def parse_summary_log(path: Path) -> List[Dict]:
    """Recover what we can from the old log. p99/mean/mrr are NaN — we only had p50/p95/recall."""
    rows: List[Dict] = []
    current_db = None
    current_index = None
    current_disk = None

    pat_db = re.compile(r"=== (?P<name>.+?)(?:\s*\(.*\))? ===")
    pat_idx = re.compile(r"index_time=(?P<t>[\d.]+)s\s+disk=(?P<d>[\d.]+)\s*MB")
    pat_ef = re.compile(
        r"ef=(?P<ef>\d+):\s+p50=(?P<p50>[\d.]+)ms\s+p95=(?P<p95>[\d.]+)ms"
        r"\s+recall_qrels=(?P<rq>[\d.]+)\s+recall_flat=(?P<rf>[\d.]+)"
    )

    with path.open() as f:
        for line in f:
            line = line.rstrip()
            m = pat_db.match(line)
            if m:
                current_db = m.group("name").strip()
                continue
            m = pat_idx.search(line)
            if m and current_db:
                current_index = float(m.group("t"))
                current_disk = float(m.group("d"))
                continue
            m = pat_ef.search(line)
            if m and current_db:
                ef = m.group("ef")
                family = current_db.split("(")[0]
                # construct name like "FAISS-HNSW(M=32,efC=200,efS=16)"
                # current_db looks like "FAISS-HNSW(M=32,efC=200,efS=16)" already
                # but for swept ef, we replace the ef in the name
                if "efS=" in current_db:
                    db_name = re.sub(r"efS=\d+", f"efS={ef}", current_db)
                else:
                    db_name = f"{current_db}(efS={ef})"
                rows.append({
                    "db": db_name,
                    "num_queries": 10000,
                    "index_time_sec": current_index,
                    "disk_mb": current_disk,
                    "latency_p50_ms": float(m.group("p50")),
                    "latency_p95_ms": float(m.group("p95")),
                    "latency_p99_ms": "",   # not in log
                    "latency_mean_ms": "",  # not in log
                    "recall_qrels@10": float(m.group("rq")),
                    "mrr_qrels@10": "",     # not in log
                    "recall_flat@10": float(m.group("rf")),
                })
    return rows


def write_csv(out_path: Path, rows: List[Dict]) -> None:
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})


def measure(db, queries, query_ids, qrels, flat_topk, num_q):
    """One-rep measurement: returns full metrics dict."""
    q = queries[:num_q]
    qi = query_ids[:num_q]

    # warmup
    for i in range(min(WARMUP, num_q)):
        db.search(q[i], top_k=TOP_K)

    latencies = []
    retrieved = []
    for i in tqdm(range(num_q), desc=f"  {db.name}", leave=False):
        t0 = time.perf_counter()
        res = db.search(q[i], top_k=TOP_K)
        latencies.append((time.perf_counter() - t0) * 1000)
        retrieved.append([d for d, _ in res])

    pcts = percentiles(np.array(latencies))

    recalls_q, mrrs_q = [], []
    for i, qid in enumerate(qi):
        rel = qrels.get(qid, set())
        if not rel:
            continue
        recalls_q.append(recall_at_k(retrieved[i], rel, TOP_K))
        mrrs_q.append(mrr_at_k(retrieved[i], rel, TOP_K))

    recall_flat = None
    if flat_topk is not None:
        rs = []
        for i, qid in enumerate(qi):
            fs = set(flat_topk.get(qid, []))
            if not fs:
                continue
            rs.append(len(set(retrieved[i]) & fs) / min(TOP_K, len(fs)))
        recall_flat = float(np.mean(rs)) if rs else None

    return {
        "num_queries": num_q,
        "latency_p50_ms": round(pcts["p50"], 3),
        "latency_p95_ms": round(pcts["p95"], 3),
        "latency_p99_ms": round(pcts["p99"], 3),
        "latency_mean_ms": round(pcts["mean"], 3),
        "recall_qrels@10": round(float(np.mean(recalls_q)), 4) if recalls_q else 0.0,
        "mrr_qrels@10": round(float(np.mean(mrrs_q)), 4) if mrrs_q else 0.0,
        "recall_flat@10": round(recall_flat, 4) if recall_flat is not None else None,
    }


def main() -> None:
    data_dir = Path("data")
    out = Path("results/results.csv")
    out.parent.mkdir(parents=True, exist_ok=True)

    print("[1/5] Loading data + recovering existing log rows")
    corpus, corpus_ids, queries, query_ids, qrels = load_data(data_dir, subset=None)

    existing_rows = parse_summary_log(Path("results/runner_summary.log"))
    print(f"  recovered {len(existing_rows)} rows from log")

    rows: List[Dict] = list(existing_rows)
    write_csv(out, rows)  # incremental save #0

    # --- 2: FAISS-Flat (1000 queries) ---
    print("\n[2/5] FAISS-Flat (1000 queries × 1 rep)")
    flat = FaissFlatDB(dim=DIM)
    t0 = time.perf_counter()
    flat.index(corpus, corpus_ids)
    idx_t = round(time.perf_counter() - t0, 2)
    disk = round(flat.disk_size_mb(), 1)
    print(f"  index_time={idx_t}s  disk={disk} MB")

    # capture flat topk for downstream recall_flat
    flat_topk: Dict[str, List[str]] = {}
    for i in range(1000):
        flat_topk[query_ids[i]] = [d for d, _ in flat.search(queries[i], top_k=TOP_K)]

    m = measure(flat, queries, query_ids, qrels, None, num_q=1000)
    rows.append({"db": flat.name, "index_time_sec": idx_t, "disk_mb": disk, **m})
    write_csv(out, rows)
    print(f"  done. p50={m['latency_p50_ms']}ms p95={m['latency_p95_ms']}ms recall={m['recall_qrels@10']}")
    flat.cleanup()

    # --- 3: Chroma full (re-index) ---
    print("\n[3/5] Chroma (re-index ~25-30 min)")
    chroma = ChromaDB(dim=DIM, M=32, ef_construction=200, ef_search=64,
                       persist_dir="chroma_db_finalize")
    t0 = time.perf_counter()
    chroma.index(corpus, corpus_ids)
    idx_t = round(time.perf_counter() - t0, 2)
    disk = round(chroma.disk_size_mb(), 1)
    print(f"  index_time={idx_t}s  disk={disk} MB")
    m = measure(chroma, queries, query_ids, qrels, flat_topk, num_q=1000)
    rows.append({"db": chroma.name, "index_time_sec": idx_t, "disk_mb": disk, **m})
    write_csv(out, rows)
    print(f"  done. p50={m['latency_p50_ms']}ms p95={m['latency_p95_ms']}ms recall={m['recall_qrels@10']}")
    chroma.cleanup()

    # --- 4: pgvector ef=128, 256 (REUSE existing index) ---
    print("\n[4/5] pgvector ef=128 / 256 (reusing existing index)")
    pg = PgvectorDB(dim=DIM, M=32, ef_construction=200, ef_search=128)
    pg.conn = pg._connect()
    pg.int_to_id = list(corpus_ids)
    # verify the index is alive
    row = pg.conn.execute("SELECT count(*) FROM bench").fetchone()
    print(f"  attached to existing bench table ({row[0]:,d} rows)")
    disk = round(pg.disk_size_mb(), 1)

    for ef in [128, 256]:
        pg.set_ef(ef)
        print(f"  ef={ef}: measuring 500 queries × 1 rep (may be slow due to IO)")
        m = measure(pg, queries, query_ids, qrels, flat_topk, num_q=500)
        rows.append({
            "db": pg.name,
            "index_time_sec": 7559.95,  # from original log
            "disk_mb": disk,
            **m,
        })
        write_csv(out, rows)
        print(f"  ef={ef} done. p50={m['latency_p50_ms']}ms p95={m['latency_p95_ms']}ms")
    pg.conn.close()

    # --- 5: final ---
    print(f"\n[5/5] DONE. {len(rows)} rows → {out}")


if __name__ == "__main__":
    main()
