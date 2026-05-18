"""Generate Pareto / latency / disk charts from results.csv."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def db_family(name: str) -> str:
    """Group rows for coloring: FAISS-Flat / FAISS-HNSW / Chroma / Qdrant / pgvector."""
    if name.startswith("FAISS-Flat"):
        return "FAISS-Flat"
    if name.startswith("FAISS-HNSW"):
        return "FAISS-HNSW"
    if name.startswith("Chroma"):
        return "Chroma"
    if name.startswith("Qdrant"):
        return "Qdrant"
    if name.startswith("pgvector"):
        return "pgvector"
    return "other"


FAMILY_COLORS = {
    "FAISS-Flat": "#000000",
    "FAISS-HNSW": "#1f77b4",
    "Chroma": "#2ca02c",
    "Qdrant": "#d62728",
    "pgvector": "#9467bd",
}


def pareto_indices(xs: np.ndarray, ys: np.ndarray) -> list[int]:
    """Return indices of points on the upper-left Pareto frontier.
    xs = latency (minimize), ys = recall (maximize).
    """
    order = np.argsort(xs)
    pareto = []
    best_y = -np.inf
    for i in order:
        if ys[i] > best_y:
            pareto.append(int(i))
            best_y = ys[i]
    return pareto


def plot_pareto(df: pd.DataFrame, out: Path, recall_col: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))

    xs = df["latency_p50_ms"].to_numpy()
    ys = df[recall_col].to_numpy()
    families = df["family"].tolist()

    for fam in FAMILY_COLORS:
        mask = df["family"] == fam
        if not mask.any():
            continue
        ax.scatter(
            df.loc[mask, "latency_p50_ms"],
            df.loc[mask, recall_col],
            label=fam,
            color=FAMILY_COLORS[fam],
            s=90, edgecolor="black", linewidth=0.5, alpha=0.85,
        )

    pi = pareto_indices(xs, ys)
    if len(pi) >= 2:
        px = xs[pi]; py = ys[pi]
        ax.plot(px, py, "--", color="grey", linewidth=1.2, label="Pareto frontier")

    # annotate each point with its db name
    for _, row in df.iterrows():
        ax.annotate(
            row["db"], xy=(row["latency_p50_ms"], row[recall_col]),
            xytext=(5, 5), textcoords="offset points", fontsize=7, color="#444",
        )

    ax.set_xscale("log")
    ax.set_xlabel("Latency p50 (ms, log scale)")
    ax.set_ylabel(recall_col)
    ax.set_title(f"Pareto frontier: {recall_col} vs latency (lower-right = bad)")
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  → {out}")


def plot_latency_distribution(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(max(10, 0.45 * len(df)), 6))

    x = np.arange(len(df))
    width = 0.27

    ax.bar(x - width, df["latency_p50_ms"], width, label="p50", color="#4C9AFF")
    ax.bar(x,         df["latency_p95_ms"], width, label="p95", color="#FFAB00")
    ax.bar(x + width, df["latency_p99_ms"], width, label="p99", color="#DE350B")

    ax.set_yscale("log")
    ax.set_ylabel("Latency (ms, log scale)")
    ax.set_title("Latency percentiles per DB / config")
    ax.set_xticks(x)
    ax.set_xticklabels(df["db"], rotation=45, ha="right", fontsize=8)
    ax.legend()
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  → {out}")


def plot_disk(df: pd.DataFrame, out: Path) -> None:
    # one row per family (use max disk_mb if multiple configs)
    fam_disk = df.groupby("family")["disk_mb"].max().sort_values()

    fig, ax = plt.subplots(figsize=(9, 4))
    colors = [FAMILY_COLORS.get(f, "#888") for f in fam_disk.index]
    bars = ax.barh(fam_disk.index, fam_disk.values, color=colors, edgecolor="black", linewidth=0.4)
    for bar, val in zip(bars, fam_disk.values):
        ax.text(bar.get_width() * 1.01, bar.get_y() + bar.get_height() / 2,
                f"{val:,.0f} MB", va="center", fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("Index size on disk (MB, log scale)")
    ax.set_title("Disk footprint per DB (max across configs)")
    ax.grid(True, axis="x", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  → {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="results/results.csv", type=Path)
    ap.add_argument("--output", default="results/", type=Path)
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    df["family"] = df["db"].apply(db_family)
    args.output.mkdir(parents=True, exist_ok=True)

    print(f"loaded {len(df)} rows from {args.input}")

    plot_pareto(df, args.output / "pareto_frontier.png", "recall_qrels@10")
    if df["recall_flat@10"].notna().any():
        plot_pareto(df.dropna(subset=["recall_flat@10"]),
                    args.output / "pareto_frontier_vs_flat.png", "recall_flat@10")
    plot_latency_distribution(df, args.output / "latency_distribution.png")
    plot_disk(df, args.output / "disk_size_chart.png")


if __name__ == "__main__":
    main()
