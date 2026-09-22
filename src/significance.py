"""Dispersion and paired significance tests for the confirmation models.

Re-runs every model under the exact protocol of confirm_models.py (stratified
80/20 split, seeds 0..4, vocabulary fit on train only) and keeps the per-seed
F1 instead of only the mean. Then:
  * per category: mean, s.d. (ddof=1) and a 95% t-interval over the 5 seeds;
  * overall: paired two-sided Wilcoxon signed-rank tests over the 30
    (category, seed) observations for every model pair, Holm-corrected.

CNN/LSTM join automatically when PyTorch is installed; otherwise they are
reported as skipped, never silently dropped.

Run:  python src/significance.py   (writes results/significance_*.csv)
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as t_dist, wilcoxon
from sklearn.model_selection import train_test_split

import confirm_models as cm
from make_synth import _categories

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def per_seed_f1() -> pd.DataFrame:
    models = cm.SKLEARN_MODELS + (cm.TORCH_MODELS if cm.TORCH_OK else [])
    rows = []
    for cfg in _categories():
        df = pd.read_parquet(cm.DATA_DIR / f"{cfg.name}.parquet")
        texts, labels = df.text.tolist(), df.label.values
        for seed in cm.SEEDS:
            X_tr, X_te, y_tr, y_te = train_test_split(
                texts, labels, test_size=0.2, stratify=labels, random_state=seed)
            for model in models:
                fit = cm._fit_torch if model in cm.TORCH_MODELS else cm._fit_sklearn
                m = fit(model, X_tr, y_tr, X_te, y_te, seed)
                rows.append({"category": cfg.name, "seed": seed,
                             "model": model, "f1": m["f1"]})
    return pd.DataFrame(rows)


def dispersion(ps: pd.DataFrame) -> pd.DataFrame:
    def agg(g):
        v = g.f1.values
        n, mean, sd = len(v), v.mean(), v.std(ddof=1)
        half = t_dist.ppf(0.975, n - 1) * sd / np.sqrt(n)
        return pd.Series({"f1_mean": mean, "f1_sd": sd,
                          "ci95_low": mean - half, "ci95_high": mean + half})
    return ps.groupby(["category", "model"]).apply(agg, include_groups=False).reset_index()


def holm(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj.tolist()


def pairwise(ps: pd.DataFrame) -> pd.DataFrame:
    wide = ps.pivot_table(index=["category", "seed"], columns="model", values="f1")
    rows = []
    for a, b in combinations(wide.columns, 2):
        d = wide[a] - wide[b]
        nz = d[d != 0]
        p = wilcoxon(wide[a], wide[b], zero_method="wilcox").pvalue if len(nz) else 1.0
        rows.append({"model_a": a, "model_b": b, "n": len(d),
                     "mean_diff": d.mean(), "a_wins": int((d > 0).sum()),
                     "b_wins": int((d < 0).sum()), "ties": int((d == 0).sum()),
                     "p_raw": p})
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out.p_raw.tolist())
    return out.sort_values("p_holm")


def main() -> None:
    print("torch available:", cm.TORCH_OK,
          "" if cm.TORCH_OK else "-> CNN/LSTM SKIPPED in this run")
    ps = per_seed_f1()
    disp = dispersion(ps)
    pw = pairwise(ps)
    RESULTS_DIR.mkdir(exist_ok=True)
    ps.to_csv(RESULTS_DIR / "significance_per_seed.csv", index=False, float_format="%.6g")
    disp.to_csv(RESULTS_DIR / "significance_dispersion.csv", index=False, float_format="%.6g")
    pw.to_csv(RESULTS_DIR / "significance_pairwise.csv", index=False, float_format="%.6g")
    pd.set_option("display.width", 220)
    print(disp.round(4).to_string(index=False))
    print()
    print(pw.round(5).to_string(index=False))


if __name__ == "__main__":
    main()
