"""Noise-sensitivity check: does the model ordering depend on the generator's noise?

The released corpus applies character-level corruption (truncation, vowel
deletion, transposition) and token concatenation to 30% of strings, which
favours character n-gram features. This script regenerates the six corpora in
memory with the same corpus seed and the same random draws, varying only the
noise process (see synth_variants.py for the exact definitions):

  char_0.0   no noise at all;
  char_0.3   the released corpus (asserted equal to data/synth string for string);
  char_0.6   the same character-level noise applied to 60% of strings;
  word_0.3   word-level noise only, at the released rate: the three
             character-level operations are replaced by deletion or
             duplication of whole tokens, concatenation is kept.

All seven models of the main comparison are trained with the main training
functions (confirm_models._fit_sklearn / _fit_torch) under the matched
protocol: stratified 80/20 splits with seeds 0-4, vocabulary on the training
split only, and the matched stopping rule for the MLP, CNN and LSTM. For
char_0.3 the scikit-learn models are asserted to reproduce
results/significance_per_seed.csv; the torch models are compared and the
largest absolute difference is printed.

Nothing is written to data/synth. Run single-threaded; about 15 minutes on
one core.

Writes
  results/noise_sensitivity.csv           F1 per variant, category, seed, model
  results/noise_sensitivity_summary.csv   mean F1 per variant and model, with rank
  results/noise_sensitivity_pairs.csv     paired cells and band (bands.py) for
                                          the pairs that define the ordering
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import bands
import confirm_models as cm
import synth_variants as sv
from make_synth import _categories

RESULTS = Path(__file__).resolve().parent.parent / "results"
VARIANTS = {
    "char_0.0": dict(noise_rate=0.0, noise_kind="char"),
    "char_0.3": dict(noise_rate=0.3, noise_kind="char"),
    "char_0.6": dict(noise_rate=0.6, noise_kind="char"),
    "word_0.3": dict(noise_rate=0.3, noise_kind="word"),
}
MODELS = cm.SKLEARN_MODELS + (cm.TORCH_MODELS if cm.TORCH_OK else [])


def main() -> None:
    if cm.TORCH_OK:
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    else:
        print("torch unavailable: CNN/LSTM SKIPPED")
    sv.check_reproduces_release()
    ref = pd.read_csv(RESULTS / "significance_per_seed.csv").set_index(
        ["category", "seed", "model"]).f1
    rows = []
    t0 = time.time()
    for vname, kw in VARIANTS.items():
        for cfg in _categories():
            df = sv.generate_variant(cfg, **kw)
            texts, y = df.text.tolist(), df.label.values
            distinct = df.text.nunique() / len(df)
            for seed in cm.SEEDS:
                Xtr, Xte, ytr, yte = train_test_split(
                    texts, y, test_size=0.2, stratify=y, random_state=seed)
                for model in MODELS:
                    fit = cm._fit_torch if model in cm.TORCH_MODELS else cm._fit_sklearn
                    f1 = fit(model, Xtr, ytr, Xte, yte, seed)["f1"]
                    rows.append(dict(variant=vname, category=cfg.name, seed=seed,
                                     model=model, f1=f1, distinct_share=distinct))
                    if vname == "char_0.3" and model in cm.SKLEARN_MODELS:
                        r = ref[(cfg.name, seed, model)]
                        assert abs(r - f1) < 5e-6, (cfg.name, seed, model, r, f1)
            print(f"  {vname} {cfg.name} done ({time.time() - t0:.0f}s)", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(RESULTS / "noise_sensitivity.csv", index=False)

    if cm.TORCH_OK:
        c3 = d[(d.variant == "char_0.3") & d.model.isin(cm.TORCH_MODELS)]
        dev = max(abs(ref[(r.category, r.seed, r.model)] - r.f1) for r in c3.itertuples())
        print(f"char_0.3: largest |F1 - released| for CNN/LSTM = {dev:.2e}")

    summ = (d.groupby(["variant", "model"]).agg(
        f1_mean=("f1", "mean"), distinct_share=("distinct_share", "mean"))
        .reset_index())
    summ["rank"] = summ.groupby("variant").f1_mean.rank(ascending=False).astype(int)
    summ = summ.sort_values(["variant", "rank"])
    summ.to_csv(RESULTS / "noise_sensitivity_summary.csv", index=False)

    prs = []
    for vname in VARIANTS:
        w = d[d.variant == vname].pivot_table(index=["category", "seed"],
                                              columns="model", values="f1")
        for a, b in combinations(MODELS, 2):
            diff = w[a] - w[b]
            aw, bw, t = int((diff > 0).sum()), int((diff < 0).sum()), int((diff == 0).sum())
            desc = bands.describe(aw, bw, float(diff.mean()))
            prs.append(dict(variant=vname, model_a=a, model_b=b, a_wins=aw,
                            b_wins=bw, ties=t, mean_diff_points=float(diff.mean()) * 100,
                            leader=a if desc["leader"] == "a" else b,
                            leader_share_non_tied=desc["leader_share_non_tied"],
                            band=desc["band"]))
    pr = pd.DataFrame(prs)
    pr.to_csv(RESULTS / "noise_sensitivity_pairs.csv", index=False)

    pd.set_option("display.width", 220)
    print(summ.pivot(index="model", columns="variant", values="f1_mean").round(4)
          .sort_values("char_0.3", ascending=False).to_string())
    print(summ.groupby("variant").distinct_share.first().round(4).to_string())
    key = pr[(pr.model_a.str.startswith("Char") | pr.model_b.str.startswith("Char")
              | (pr.model_a == "BoW+MLP(256)") | (pr.model_b == "BoW+MLP(256)"))]
    print(key.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
