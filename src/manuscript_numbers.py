"""Derived figures quoted in the paper's text that are not printed verbatim in
one results file: ranges, medians and differences computed from released
results. Reads results/ only and writes results/manuscript_numbers.csv, one
row per quoted figure with the file it comes from, so that each figure in the
text can be checked against code.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RES = Path(__file__).resolve().parent.parent / "results"
rows = []


def put(name, value, source, note=""):
    rows.append(dict(name=name, value=value, source=source, note=note))


def main() -> None:
    # convergence: realised epochs of the deep models and the MLP
    c = pd.read_csv(RES / "robustness_convergence.csv")
    deep = c[c.model.isin(["CNN", "LSTM"])]
    put("deep_models_pooled_median_epochs", deep.n_iter.median(), "robustness_convergence.csv")
    for m in ["CNN", "LSTM", "BoW+MLP(256)"]:
        x = c[c.model == m].n_iter
        put(f"{m}_median_epochs", x.median(), "robustness_convergence.csv",
            f"range {x.min()}-{x.max()}")
    lr = c[c.model.str.contains("LogReg")].groupby("model").n_iter.median()
    put("logreg_median_iter_min", lr.min(), "robustness_convergence.csv")
    put("logreg_median_iter_max", lr.max(), "robustness_convergence.csv")
    put("fits_hit_cap", int(c.hit_cap.sum()), "robustness_convergence.csv")
    put("fits_not_converged", int((~c.converged).sum()), "robustness_convergence.csv")
    put("n_fits", len(c), "robustness_convergence.csv")

    # train-test overlap
    o = pd.read_csv(RES / "leakage_audit_overlap.csv")
    put("overlap_mean", o.test_overlap.mean(), "leakage_audit_overlap.csv")
    bycat = o.groupby("category")[["test_overlap_pos", "test_overlap_neg"]].mean()
    put("overlap_pos_by_category_min", bycat.test_overlap_pos.min(), "leakage_audit_overlap.csv")
    put("overlap_pos_by_category_max", bycat.test_overlap_pos.max(), "leakage_audit_overlap.csv")
    put("overlap_neg_by_category_min", bycat.test_overlap_neg.min(), "leakage_audit_overlap.csv")
    put("overlap_neg_by_category_max", bycat.test_overlap_neg.max(), "leakage_audit_overlap.csv")

    # learning curve: gain from 66 examples to the full split
    lc = pd.read_csv(RES / "learning_curve.csv")
    w = lc.pivot(index="category", columns="train_fraction", values="f1_mean")
    for cat, g in ((w[1.0] - w[0.05]) * 100).items():
        put(f"learning_gain_points_{cat}", g, "learning_curve.csv")

    # throughput by batch size
    t = pd.read_csv(RES / "throughput_by_batch_summary.csv")
    big = t[t.n_items > 330]
    put("unigram_gain_median_min", big.unigram_gain_median.min(), "throughput_by_batch_summary.csv")
    put("unigram_gain_median_max", big.unigram_gain_median.max(), "throughput_by_batch_summary.csv")

    # consensus simulation
    cs = pd.read_csv(RES / "consensus.csv").set_index("k")
    for k in cs.index:
        put(f"ds_minus_majority_points_k{k}",
            100 * (cs.loc[k, "Dawid-Skene_mean"] - cs.loc[k, "Majority_mean"]), "consensus.csv")
        put(f"ds_minus_weighted_points_k{k}",
            100 * (cs.loc[k, "Dawid-Skene_mean"] - cs.loc[k, "Reliability-weighted_mean"]),
            "consensus.csv")
        put(f"weighted_minus_majority_points_k{k}",
            100 * (cs.loc[k, "Reliability-weighted_mean"] - cs.loc[k, "Majority_mean"]),
            "consensus.csv")

    # pretrained: training time and precision / recall
    p = pd.read_csv(RES / "pretrained" / "pretrained_per_run_from_log.csv")
    for m, x in p.groupby("model").train_s.median().items():
        put(f"median_train_s_{m}", x, "pretrained/pretrained_per_run_from_log.csv")
    s = pd.read_csv(RES / "pretrained" / "pretrained_summary.csv")
    for m, g in s.groupby("model"):
        put(f"precision_mean_{m}", g.precision_mean.mean(), "pretrained/pretrained_summary.csv",
            "mean over the six categories")
        put(f"recall_mean_{m}", g.recall_mean.mean(), "pretrained/pretrained_summary.csv",
            "mean over the six categories")

    # seed s.d. of Table 1 (sample s.d., averaged over categories)
    disp = pd.read_csv(RES / "significance_dispersion.csv")
    for m, x in disp.groupby("model").f1_sd.mean().items():
        put(f"seed_sd_{m}", x, "significance_dispersion.csv", "ddof=1, mean over categories")

    out = pd.DataFrame(rows)
    out.to_csv(RES / "manuscript_numbers.csv", index=False)
    pd.set_option("display.width", 200)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
