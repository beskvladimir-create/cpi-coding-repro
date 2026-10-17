"""Summarise the Kaggle pretrained-encoder runs against the paper's seven models.

Usage:  python src/analyze_pretrained.py [PRETRAINED_DIR [REPO_RESULTS_DIR]]
Defaults: results/pretrained and results. Reads pretrained_per_run.csv from
PRETRAINED_DIR if present (raw output of pretrained_baselines.py), otherwise
pretrained_per_run_from_log.csv (F1 transcribed from the run log at 4 d.p.).
Writes PRETRAINED_DIR/pretrained_pairwise.csv.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

_ROOT = Path(__file__).resolve().parent.parent
RES = sys.argv[1] if len(sys.argv) > 1 else str(_ROOT / "results" / "pretrained")
REPO = sys.argv[2] if len(sys.argv) > 2 else str(_ROOT / "results")
_raw = Path(RES) / "pretrained_per_run.csv"
_src = _raw if _raw.exists() else Path(RES) / "pretrained_per_run_from_log.csv"
print("per-run input:", _src)
pre = pd.read_csv(_src)
base = pd.read_csv(f"{REPO}/significance_per_seed.csv")
allf = pd.concat([base, pre[["category", "seed", "model", "f1"]]])
wide = allf.pivot_table(index=["category", "seed"], columns="model", values="f1")
assert len(wide) == 30, len(wide)

print("== accuracy (mean over categories of seed-mean; seed sd averaged; range of category means)")
for m in pre.model.unique():
    d = pre[pre.model == m]
    cm = d.groupby("category").f1.mean()
    sd = d.groupby("category").f1.std().mean()
    un = d.groupby("category").f1_unseen.mean().mean()
    ep = d.n_epochs.median()
    print(f"{m:40s} F1={cm.mean():.4f} sd={sd:.4f} range={cm.min():.3f}-{cm.max():.3f} "
          f"unseen={un:.4f} prec={d.precision.mean():.4f} rec={d.recall.mean():.4f} med_epochs={ep}")

print("\n== paired comparisons over 30 cells (wins/losses/ties, mean diff, Wilcoxon p raw)")
others = list(base.model.unique())
rows = []
for m in pre.model.unique():
    for o in others + [x for x in pre.model.unique() if x != m]:
        a, b = wide[m].round(4), wide[o].round(4)
        w, l, t = int((a > b).sum()), int((a < b).sum()), int((a == b).sum())
        nz = (a - b) != 0
        p = wilcoxon(a, b, zero_method="wilcox").pvalue if nz.any() else 1.0
        rows.append((m, o, w, l, t, float((a - b).mean()), p))
pw = pd.DataFrame(rows, columns=["model", "vs", "wins", "losses", "ties", "mean_diff", "p_raw"])
print(pw.to_string())
pw.to_csv(f"{RES}/pretrained_pairwise.csv", index=False)

print("\n== cost (seed 0, one per category; medians)")
cost = pd.read_csv(f"{RES}/pretrained_cost.csv")
med = cost.groupby("model").agg(train_s=("train_s_device", "median"),
                                 cpu1=("cpu1_items_per_s", "median"),
                                 size_kib=("size_kib", "median"),
                                 n_params=("n_params", "median"),
                                 device=("device", "first"))
print(med.to_string())
ref = med.loc["Char n-gram(3-5) + LogReg", "cpu1"]
bow = med.loc["Unigram BoW + LogReg", "cpu1"]
for m in med.index:
    print(f"{m:40s} throughput ratio char/m = {ref / med.loc[m, 'cpu1']:.1f}  bow/m = {bow / med.loc[m, 'cpu1']:.1f}")
