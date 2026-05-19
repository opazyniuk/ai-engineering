"""
Plots for baseline + fixes CSVs. Outputs PNGs to results/plots/.

Reads:
  results/baseline.csv          (Step 4)
  results/fixes.csv             (Step 6, optional)

Generates:
  results/plots/recall_vs_size.png
  results/plots/latency_vs_size.png
  results/plots/mrr_vs_size.png
  results/plots/ops_vs_size.png       (throughput + RAM)
  results/plots/baseline_vs_fix.png   (only if fixes.csv exists)
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results"
PLOTS = RESULTS / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)


def plot_recall(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sizes = df["size"]
    ax.plot(sizes, df["recall@1"], "o-", label="recall@1", linewidth=2)
    ax.plot(sizes, df["recall@5"], "s-", label="recall@5", linewidth=2)
    ax.plot(sizes, df["recall@10"], "^-", label="recall@10", linewidth=2)
    ax.set_xscale("log")
    ax.set_xlabel("corpus size (passages, log scale)")
    ax.set_ylabel("recall")
    ax.set_title("Recall@K vs corpus size — Dense brute-force")
    ax.set_ylim(0, 1.05)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    ax.set_xticks(sizes)
    ax.set_xticklabels([f"{s//1000}K" for s in sizes])
    fig.tight_layout()
    fig.savefig(PLOTS / "recall_vs_size.png", dpi=120)
    plt.close(fig)
    print(f"  saved recall_vs_size.png")


def plot_latency(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sizes = df["size"]
    ax.plot(sizes, df["latency_p50_ms"], "o-", label="p50", linewidth=2)
    ax.plot(sizes, df["latency_p95_ms"], "s-", label="p95", linewidth=2)
    ax.plot(sizes, df["latency_p99_ms"], "^-", label="p99", linewidth=2)
    # Reference slope=1 line for visual linearity check
    s0, l0 = sizes.iloc[0], df["latency_p50_ms"].iloc[0]
    ref = [l0 * (s / s0) for s in sizes]
    ax.plot(sizes, ref, "k--", alpha=0.4, label="O(N) reference")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("corpus size (log scale)")
    ax.set_ylabel("latency (ms, log scale)")
    ax.set_title("Search latency vs corpus size — Dense brute-force")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    ax.set_xticks(sizes)
    ax.set_xticklabels([f"{s//1000}K" for s in sizes])
    fig.tight_layout()
    fig.savefig(PLOTS / "latency_vs_size.png", dpi=120)
    plt.close(fig)
    print(f"  saved latency_vs_size.png")


def plot_mrr(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sizes = df["size"]
    ax.plot(sizes, df["mrr@10"], "o-", color="purple", linewidth=2)
    ax.set_xscale("log")
    ax.set_xlabel("corpus size (log scale)")
    ax.set_ylabel("MRR@10")
    ax.set_title("MRR@10 vs corpus size — captures rank-sliding inside top-10")
    ax.set_ylim(0, 1.05)
    ax.grid(True, which="both", alpha=0.3)
    ax.set_xticks(sizes)
    ax.set_xticklabels([f"{s//1000}K" for s in sizes])
    fig.tight_layout()
    fig.savefig(PLOTS / "mrr_vs_size.png", dpi=120)
    plt.close(fig)
    print(f"  saved mrr_vs_size.png")


def plot_ops(df: pd.DataFrame):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    sizes = df["size"]

    ax1.plot(sizes, df["throughput_qps"], "o-", color="green", linewidth=2)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("corpus size")
    ax1.set_ylabel("throughput (queries/sec, log)")
    ax1.set_title("Throughput")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.set_xticks(sizes); ax1.set_xticklabels([f"{s//1000}K" for s in sizes])

    ax2.plot(sizes, df["index_mb"], "o-", color="orange", linewidth=2, label="index (embeddings)")
    ax2.plot(sizes, df["rss_mb"], "s-", color="red", linewidth=2, label="process RSS")
    ax2.set_xscale("log")
    ax2.set_xlabel("corpus size")
    ax2.set_ylabel("RAM (MB)")
    ax2.set_title("Memory footprint")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend()
    ax2.set_xticks(sizes); ax2.set_xticklabels([f"{s//1000}K" for s in sizes])

    fig.tight_layout()
    fig.savefig(PLOTS / "ops_vs_size.png", dpi=120)
    plt.close(fig)
    print(f"  saved ops_vs_size.png")


def plot_baseline_vs_fixes(df_base: pd.DataFrame, df_fix: pd.DataFrame, size: int):
    """Bar chart: recall@1 / recall@10 / mrr@10 across retrievers at one size."""
    base_row = df_base[df_base["size"] == size].iloc[0]
    fix_rows = df_fix[df_fix["size"] == size]

    retrievers = [base_row["retriever"]] + fix_rows["retriever"].tolist()
    # The fixes.csv also includes dense_bruteforce — drop the duplicate from baseline source
    seen = set()
    uniq = []
    for r in retrievers:
        if r in seen:
            continue
        seen.add(r); uniq.append(r)
    retrievers = uniq

    # Pull metrics
    def get_row(name):
        if name == base_row["retriever"] and not (fix_rows["retriever"] == name).any():
            return base_row
        return fix_rows[fix_rows["retriever"] == name].iloc[0]

    metrics = {"recall@1": [], "recall@10": [], "mrr@10": []}
    for r in retrievers:
        row = get_row(r)
        metrics["recall@1"].append(row["recall@1"])
        metrics["recall@10"].append(row["recall@10"])
        metrics["mrr@10"].append(row["mrr@10"])

    x = np.arange(len(retrievers))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width, metrics["recall@1"], width, label="recall@1")
    ax.bar(x,         metrics["recall@10"], width, label="recall@10")
    ax.bar(x + width, metrics["mrr@10"], width, label="MRR@10")
    ax.set_xticks(x); ax.set_xticklabels(retrievers, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score")
    ax.set_title(f"Quality vs retriever — size={size:,}")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS / f"baseline_vs_fix_{size}.png", dpi=120)
    plt.close(fig)
    print(f"  saved baseline_vs_fix_{size}.png")


def plot_recall_latency_tradeoff(df_fix: pd.DataFrame, size: int):
    """Scatter of recall@10 vs latency p50 — shows the Pareto frontier of retrievers."""
    rows = df_fix[df_fix["size"] == size]
    fig, ax = plt.subplots(figsize=(8, 5))
    for _, row in rows.iterrows():
        ax.scatter(row["latency_p50_ms"], row["recall@10"], s=100)
        ax.annotate(row["retriever"], (row["latency_p50_ms"], row["recall@10"]),
                    xytext=(8, 4), textcoords="offset points", fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("latency p50 (ms, log)")
    ax.set_ylabel("recall@10")
    ax.set_title(f"Recall vs latency trade-off — size={size:,}")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS / f"tradeoff_{size}.png", dpi=120)
    plt.close(fig)
    print(f"  saved tradeoff_{size}.png")


import numpy as np  # noqa: E402


def find_knee(df: pd.DataFrame, threshold: float = 0.20):
    """First size where any recall metric or latency degrades >= threshold from 1K baseline."""
    baseline = df.iloc[0]
    knee_info = []
    for _, row in df.iloc[1:].iterrows():
        for col in ["recall@1", "recall@5", "recall@10", "mrr@10"]:
            drop = (baseline[col] - row[col]) / baseline[col]
            if drop >= threshold:
                knee_info.append((row["size"], col, baseline[col], row[col], drop))
        # Latency: relative INCREASE
        lat_growth = row["latency_p50_ms"] / max(baseline["latency_p50_ms"], 0.01)
        if lat_growth >= 100:    # 100× slowdown ~= bottleneck
            knee_info.append((row["size"], "latency_p50_ms", baseline["latency_p50_ms"], row["latency_p50_ms"], lat_growth))
    return knee_info


def main():
    base_csv = RESULTS / "baseline.csv"
    if not base_csv.exists():
        print(f"Missing {base_csv}. Run src/run_scaling.py first.")
        return
    df = pd.read_csv(base_csv)
    print(f"Loaded {len(df)} rows from {base_csv}")
    print(df.to_string(index=False))

    print("\nPlotting baseline...")
    plot_recall(df)
    plot_latency(df)
    plot_mrr(df)
    plot_ops(df)

    print("\nKnee-point analysis (>=20% drop or >=100x latency slowdown vs 1K):")
    for size, metric, base, now, mag in find_knee(df):
        if "latency" in metric:
            print(f"  size={size}: {metric} grew {mag:.1f}x ({base:.2f} -> {now:.2f} ms)")
        else:
            print(f"  size={size}: {metric} dropped {mag*100:.1f}% ({base:.3f} -> {now:.3f})")

    fix_csv = RESULTS / "fixes.csv"
    if fix_csv.exists():
        df_fix = pd.read_csv(fix_csv)
        print(f"\nPlotting baseline vs fixes...")
        for size in [100_000, 300_000]:
            plot_baseline_vs_fixes(df, df_fix, size)
            plot_recall_latency_tradeoff(df_fix, size)


if __name__ == "__main__":
    main()
