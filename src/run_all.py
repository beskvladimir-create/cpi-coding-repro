"""End-to-end driver: data -> trie coverage -> confirm models -> consensus.

Writes CSVs to results/, figures to figures/, and prints Markdown tables ready
to paste into the paper. Everything is deterministic from fixed seeds.

Run with:  python src/run_all.py   (or `make all`)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"


def _fmt(mean, sd):
    return f"{mean:.3f}+/-{sd:.3f}"


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)

    import make_synth
    import trie_classifier
    import confirm_models
    import consensus_sim

    print("\n[1/4] Generating synthetic corpus ...")
    make_synth.main()

    print("\n[2/4] Trie pre-classifier coverage ...")
    cov = trie_classifier.coverage_report()
    cov.to_csv(RESULTS / "coverage.csv", index=False)
    print(cov.to_string(index=False))

    print("\n[3/4] Per-category binary confirmation (5 seeds) ...")
    conf = confirm_models.evaluate_all()
    conf.to_csv(RESULTS / "confirm_metrics.csv", index=False)
    lc = confirm_models.learning_curve()
    lc.to_csv(RESULTS / "learning_curve.csv", index=False)

    print("\n[4/4] Consensus simulation (60 runs) ...")
    cons = consensus_sim.run()
    cons.to_csv(RESULTS / "consensus.csv", index=False)

    _make_figures(cov, conf, lc, cons)
    _print_markdown(cov, conf, lc, cons)
    _write_markdown_file(cov, conf, lc, cons)
    print(f"\nDone. CSVs in {RESULTS}/  figures in {FIGURES}/  "
          f"tables in {RESULTS / 'tables.md'}")


def _make_figures(cov, conf, lc, cons) -> None:
    # per-category F1 bar (one group per model)
    fig, ax = plt.subplots(figsize=(11, 5))
    models = conf.model.unique()
    cats = conf.category.unique()
    import numpy as np
    x = np.arange(len(cats))
    w = 0.8 / len(models)
    for i, m in enumerate(models):
        sub = conf[conf.model == m].set_index("category").reindex(cats)
        ax.bar(x + i * w, sub.f1_mean.values, w, yerr=sub.f1_sd.values,
               label=m, capsize=2)
    ax.set_xticks(x + 0.4)
    ax.set_xticklabels(cats, rotation=20, ha="right")
    ax.set_ylabel("F1 (mean +/- s.d., 5 seeds)")
    ax.set_title("Per-category binary confirmation F1 by model")
    ax.legend(fontsize=8, ncol=2)
    ax.set_ylim(0, 1.02)
    fig.tight_layout()
    fig.savefig(FIGURES / "per_category_f1.png", dpi=130)
    plt.close(fig)

    # learning curve
    fig, ax = plt.subplots(figsize=(8, 5))
    for cat in lc.category.unique():
        sub = lc[lc.category == cat]
        ax.plot(sub.n_train_examples, sub.f1_mean, marker="o", label=cat)
    ax.set_xscale("log")
    ax.set_xlabel("training examples (log scale)")
    ax.set_ylabel("F1 (unigram BoW + LogReg)")
    ax.set_title("Learning curve: data efficiency of bag-of-words")
    ax.axvline(67, color="grey", ls="--", lw=1)
    ax.text(67, ax.get_ylim()[0] + 0.02, " ~67", color="grey", fontsize=8)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "learning_curve.png", dpi=130)
    plt.close(fig)

    # coverage
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(cov.category, cov.coverage)
    ax.set_ylabel("coverage (1 - unidentified share)")
    ax.set_title("Trie pre-classifier coverage by category")
    ax.set_ylim(0, 1.02)
    ax.set_xticklabels(cov.category, rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(FIGURES / "coverage.png", dpi=130)
    plt.close(fig)

    # consensus by k
    fig, ax = plt.subplots(figsize=(8, 5))
    for name in ["Majority", "Reliability-weighted", "Dawid-Skene"]:
        ax.errorbar(cons.k, cons[f"{name}_mean"], yerr=cons[f"{name}_sd"],
                    marker="o", capsize=3, label=name)
    ax.set_xlabel("votes per item (k)")
    ax.set_ylabel("label-recovery accuracy")
    ax.set_title("Consensus aggregators (60 runs, 1500 items, 12 annotators)")
    ax.set_xticks(cons.k)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "consensus_by_k.png", dpi=130)
    plt.close(fig)


def _print_markdown(cov, conf, lc, cons) -> None:
    print("\n" + "=" * 70)
    print("MARKDOWN TABLES (paste into the paper)")
    print("=" * 70)
    print(_md_tables(cov, conf, lc, cons))


def _write_markdown_file(cov, conf, lc, cons) -> None:
    (RESULTS / "tables.md").write_text(_md_tables(cov, conf, lc, cons))


def _md_tables(cov, conf, lc, cons) -> str:
    out = []

    out.append("### Per-category binary confirmation (mean +/- s.d., 5 seeds)\n")
    out.append("| Category | Model | Accuracy | F1 | Train (s) |")
    out.append("|---|---|---|---|---|")
    for _, r in conf.iterrows():
        out.append(f"| {r.category} | {r.model} | "
                   f"{_fmt(r.accuracy_mean, r.accuracy_sd)} | "
                   f"{_fmt(r.f1_mean, r.f1_sd)} | {r.train_s_mean:.2f} |")

    out.append("\n### Matched BoW vs CNN/LSTM (mean F1 over categories)\n")
    by_model = conf.groupby("model").f1_mean.mean().sort_values(ascending=False)
    out.append("| Model | mean F1 (all categories) |")
    out.append("|---|---|")
    for m, v in by_model.items():
        out.append(f"| {m} | {v:.3f} |")

    out.append("\n### Trie coverage\n")
    out.append("| Category | Items | Coverage | Positive recall (trie) | "
               "Unidentified |")
    out.append("|---|---|---|---|---|")
    for _, r in cov.iterrows():
        out.append(f"| {r.category} | {r.n_items} | {r.coverage:.3f} | "
                   f"{r.positive_recall_trie:.3f} | {r.unidentified} |")

    out.append("\n### Learning curve, unigram BoW (mean F1, 5 seeds)\n")
    fracs = sorted(lc.train_fraction.unique())
    out.append("| Category | " + " | ".join(
        f"{int(f*100)}% (~{int(lc[lc.train_fraction==f].n_train_examples.mean())})"
        for f in fracs) + " |")
    out.append("|---" * (len(fracs) + 1) + "|")
    for cat in lc.category.unique():
        sub = lc[lc.category == cat].set_index("train_fraction")
        cells = " | ".join(f"{sub.loc[f].f1_mean:.3f}" for f in fracs)
        out.append(f"| {cat} | {cells} |")

    out.append("\n### Consensus simulation: label-recovery accuracy "
               "(mean +/- s.d., 60 runs)\n")
    out.append("| k | Majority | Reliability-weighted | Dawid-Skene |")
    out.append("|---|---|---|---|")
    for _, r in cons.iterrows():
        out.append(f"| {int(r.k)} | {_fmt(r.Majority_mean, r.Majority_sd)} | "
                   f"{_fmt(r['Reliability-weighted_mean'], r['Reliability-weighted_sd'])} | "
                   f"{_fmt(r['Dawid-Skene_mean'], r['Dawid-Skene_sd'])} |")

    return "\n".join(out) + "\n"


if __name__ == "__main__":
    main()
