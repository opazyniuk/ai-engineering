# HW9 — RAG Systems @ Enterprise Scale

Scaling experiment on MS MARCO Passage Ranking: find the breaking point of a brute-force dense retriever as the corpus grows from 1K → 300K passages, then fix it.

## Findings (TL;DR)

- **Recall@1 breaks at 300K** (0.98 → 0.78, −20%). Recall@10 only drops 6%. Hypothesis confirmed: relevant docs slide off rank 1 but stay in top 10.
- **Latency breaks at 100K** (4 ms p50, 205× slowdown). Brute-force unusable for SLA work beyond 10K-100K.
- **Hybrid (BM25 + dense + RRF) FAILED** on MS MARCO — recall@1 dropped to 0.69 (BM25 too weak on paraphrased queries, drags dense down via RRF).
- **Reranker (`bge-reranker-v2-m3`)** wins quality: +3-4 pts recall@1/MRR. Costs 204× latency.
- **HNSW** wins speed: 56× faster, −2-3% recall. Right tool for >1M corpora.

See [REPORT.md](REPORT.md) for full analysis and [education.md](education.md) for 5 key concepts (plain Ukrainian).

## Stack
- Dataset: `BeIR/msmarco` (HF), validation qrels, 100 eval queries
- Embeddings: `BAAI/bge-small-en-v1.5` (384d, L2-normalized), MPS on Apple Silicon
- Retrievers: NumPy brute-force, `rank_bm25`, RRF hybrid, `bge-reranker-v2-m3`, FAISS HNSW
- Eval: `recall@{1,5,10}`, `MRR@10` via fixed `metrics.py`

## Quick start

```bash
cd hw/hw9
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1) one-time corpus cache (~5 min, streams MS MARCO)
python src/data_loader.py

# 2) one-time embed all passages + queries (~14 min on MPS)
python src/embed.py

# 3) baseline scaling 1K → 300K (~30 sec)
python src/run_scaling.py

# 4) fixes: dense, BM25, hybrid, reranker on 100K + 300K (~15 min)
python src/run_fixes.py

# 5) HNSW in clean process (~2 min, must run after #4)
python src/run_hnsw.py

# 6) plots
python src/viz.py
```

## Layout

```
src/
  data_loader.py   MS MARCO streaming + reproducible subsets (template)
  metrics.py       recall@k, mrr@k (template, untouched)
  embed.py         bge-small embeddings via sentence-transformers
  retriever.py     DenseRetriever + subset selection (deterministic)
  run_scaling.py   baseline experiment (Step 4)
  run_fixes.py     hybrid + reranker (Step 6, NO HNSW — see below)
  run_hnsw.py      HNSW in clean process (Step 6 cont.)
  viz.py           plots: scaling curves, baseline-vs-fix, recall/latency trade-off

data/cache/        corpus.json (105 MB), embeddings_corpus.npy (440 MB), …  [NOT in git]
results/
  baseline.csv     4 sizes × dense_bruteforce metrics
  fixes.csv        10 rows: (size, retriever) cross
  plots/*.png      8 plots
  screenshots/*.log  raw stdout from every step
```

## Hardware / cost

- All local on M-series Mac (16 GB RAM, MPS). $0.
- Total wall time: ~35 min end-to-end (most is the one-time embedding pass).

## Notes on template

- Template `split="validation"` is correct — `datasets` remaps `dev.tsv` to `validation` split (HF file names ≠ HF split names).
- `data_loader.py` cache path adjusted to `data/cache/` to fit project layout.
- Had to split HNSW into separate process: FAISS HNSW build silently segfaults after `rank_bm25`'s `loky` multiprocessing pool leaves leaked semaphores. Documented in REPORT.
