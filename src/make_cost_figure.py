"""Accuracy against measured inference cost (the paper's cost figure).

Reads results/confirm_metrics.csv (mean F1 over the six categories, each a
mean over five seeds) and results/cost_benchmark_summary.csv (median
throughput over the 30 category-seed cells), and writes
figures/fig_accuracy_cost.pdf and figures/fig_accuracy_cost.png.

Run:  python src/make_cost_figure.py   (or `make figures`)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FixedLocator, NullLocator  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

LABELS = {
    "Unigram BoW + LogReg": "Unigram BoW\n+ LogReg",
    "Word 1-2-gram + LogReg": "Word 1–2-gram\n+ LogReg",
    "Word 1-3-gram + LogReg": "Word 1–3-gram\n+ LogReg",
    "Char n-gram(3-5) + LogReg": "Char $n$-gram(3–5)\n+ LogReg",
    "BoW+MLP(256)": "BoW + MLP(256)",
    "CNN": "CNN",
    "LSTM": "LSTM",
}
# label placement: (dx, dy) in points, horizontal alignment
OFFSETS = {
    "Unigram BoW + LogReg": ((8, -2), "left"),
    "Word 1-2-gram + LogReg": ((0, 10), "left"),
    "Word 1-3-gram + LogReg": ((-8, 2), "right"),
    "Char n-gram(3-5) + LogReg": ((8, -2), "left"),
    "BoW+MLP(256)": ((8, 0), "left"),
    "CNN": ((8, -6), "left"),
    "LSTM": ((0, 7), "center"),
}
DEEP = {"CNN", "LSTM"}
TICK_CANDIDATES = [10, 20, 40, 70, 120, 200, 350, 600, 1000]


def build() -> pd.DataFrame:
    conf = pd.read_csv(RESULTS / "confirm_metrics.csv")
    cost = pd.read_csv(RESULTS / "cost_benchmark_summary.csv")
    f1 = conf.groupby("model").f1_mean.mean().rename("f1")
    thr = cost.set_index("model").throughput_items_s_median.rename("thr") / 1000
    return pd.concat([f1, thr], axis=1, join="inner")


def main() -> None:
    d = build()
    plt.rcParams.update({"font.family": "serif", "font.size": 9,
                         "mathtext.fontset": "dejavuserif"})
    fig, ax = plt.subplots(figsize=(5.8, 3.5))
    for model, r in d.iterrows():
        deep = model in DEEP
        ax.scatter(r.thr, r.f1, s=42, zorder=3,
                   marker="s" if deep else "o",
                   facecolor="white" if deep else "#333333",
                   edgecolor="#333333", linewidth=1.0)
        (dx, dy), ha = OFFSETS.get(model, ((6, 0), "left"))
        ax.annotate(LABELS.get(model, model), (r.thr, r.f1),
                    xytext=(dx, dy), textcoords="offset points",
                    ha=ha, va="center", fontsize=7.5, linespacing=1.0)
    ax.set_xscale("log")
    lo, hi = d.thr.min() / 1.25, d.thr.max() * 1.35
    ax.set_xlim(lo, hi)
    ticks = [t for t in TICK_CANDIDATES if lo <= t <= hi]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels([str(t) for t in ticks])
    ylo = min(0.95, (d.f1.min() - 0.005))
    ax.set_ylim(ylo, max(1.003, d.f1.max() + 0.004))
    ax.set_xlabel("Inference throughput, thousands of items per second "
                  "(log scale; higher is cheaper)")
    ax.set_ylabel("Mean $F_1$ over six categories")
    ax.grid(True, color="#dddddd", linewidth=0.6, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.scatter([], [], marker="o", facecolor="#333333", edgecolor="#333333",
               label="Bag-of-words and $n$-gram models")
    ax.scatter([], [], marker="s", facecolor="white", edgecolor="#333333",
               label="Deep models (CNN, LSTM)")
    ax.legend(loc="lower right", frameon=False, fontsize=7.5)
    fig.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / "fig_accuracy_cost.pdf")
    fig.savefig(FIGURES / "fig_accuracy_cost.png", dpi=300)
    plt.close(fig)
    print(d.sort_values("thr").round(4).to_string())
    print("wrote figures/fig_accuracy_cost.pdf, figures/fig_accuracy_cost.png")


if __name__ == "__main__":
    main()
