"""Grade-token shortcut check (Appendix "Robustness of the model comparison").

The generator adds a grade block to positive strings with probability 0.7 and
never to negative ones, so a grade word is a cue a model can use without
reading the product core. This script measures how much the models rely on it.

How the cue is removed. The corpus is regenerated in memory by
synth_variants.generate_variant with the same seed and the same random draws
as data/synth, and the grade block is then dropped wherever it ended up after
noise: as a token of its own, in corrupted form, or as the leading part of a
concatenated token. Key-phrase tokens are never removed, so product words that
are also grade words ("white sugar", "whole milk", "pale ale") are kept; an
earlier version of this script removed every whitespace token equal to a
grade word, which also deleted those product words and overstated the effect.
Nothing is written to data/synth; synth_variants first asserts that its
default variant reproduces the released files exactly.

The four logistic models and the MLP are re-run with the training function of
the main results (confirm_models._fit_sklearn) under the matched protocol
(stratified 80/20, seeds 0-4, vocabulary on the training split only), on the
original and on the grade-removed corpus. The original arm is asserted to
reproduce results/significance_per_seed.csv.

Writes
  results/grade_token_ablation.csv          per category, seed, variant, model
  results/grade_token_ablation_summary.csv  mean F1 over the 30 cells and change
  results/grade_word_shares.csv             per category: share of positive
      strings with a grade block; with an intact grade word outside the key
      phrase; with an intact grade word anywhere (including key-phrase tokens
      such as "white" in "white sugar"); and the same for negatives.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import confirm_models as cm
import synth_variants as sv
from make_synth import _categories

RESULTS = Path(__file__).resolve().parent.parent / "results"
MODELS = ["Char n-gram(3-5) + LogReg", "BoW+MLP(256)", "Word 1-2-gram + LogReg",
          "Unigram BoW + LogReg", "Word 1-3-gram + LogReg"]


def grade_shares(cfg) -> dict:
    df, toks = sv.generate_variant(cfg, return_tokens=True)
    grades = {g.lower() for g in cfg.grades}
    rec = {"category": cfg.name}
    for lab, name in ((1, "pos"), (0, "neg")):
        idx = np.where(df.label.values == lab)[0]
        block = [any(b == "grade" for tok in toks[i] for _, b in tok) for i in idx]
        outside = [any(len(tok) == 1 and tok[0][1] != "core"
                       and tok[0][0].lower() in grades for tok in toks[i]) for i in idx]
        anywhere = [any(t.lower() in grades for t in df.text.iloc[i].split()) for i in idx]
        rec[f"{name}_grade_block"] = float(np.mean(block))
        rec[f"{name}_grade_word_outside_key_phrase"] = float(np.mean(outside))
        rec[f"{name}_grade_word_anywhere"] = float(np.mean(anywhere))
    return rec


def main() -> None:
    sv.check_reproduces_release()
    ref = pd.read_csv(RESULTS / "significance_per_seed.csv")
    rows, shares = [], []
    for cfg in _categories():
        shares.append(grade_shares(cfg))
        variants = {
            "original": sv.generate_variant(cfg),
            "grade block removed": sv.generate_variant(cfg, drop_grade=True),
        }
        assert (variants["original"].label.values
                == variants["grade block removed"].label.values).all()
        for vname, df in variants.items():
            texts, y = df.text.tolist(), df.label.values
            for seed in cm.SEEDS:
                Xtr, Xte, ytr, yte = train_test_split(
                    texts, y, test_size=0.2, stratify=y, random_state=seed)
                for model in MODELS:
                    f1 = cm._fit_sklearn(model, Xtr, ytr, Xte, yte, seed)["f1"]
                    rows.append(dict(category=cfg.name, seed=seed,
                                     variant=vname, model=model, f1=f1))
                    if vname == "original":
                        r = ref[(ref.category == cfg.name) & (ref.seed == seed)
                                & (ref.model == model)].f1.iloc[0]
                        assert abs(r - f1) < 5e-6, (cfg.name, seed, model, r, f1)
        print("  done", cfg.name, flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(RESULTS / "grade_token_ablation.csv", index=False)
    s = d.groupby(["model", "variant"]).f1.mean().unstack()
    s = s[["original", "grade block removed"]]
    s["change_points"] = (s["grade block removed"] - s["original"]) * 100
    # paired cell counts: original against grade-removed
    w = d.pivot_table(index=["category", "seed", "model"], columns="variant",
                      values="f1").reset_index()
    diff = w["grade block removed"] - w["original"]
    cnt = w.assign(lower=diff < 0, higher=diff > 0, same=diff == 0).groupby(
        "model")[["lower", "higher", "same"]].sum()
    s = s.join(cnt.rename(columns=lambda c: f"cells_{c}"))
    s = s.sort_values("original", ascending=False)
    s.to_csv(RESULTS / "grade_token_ablation_summary.csv")
    sh = pd.DataFrame(shares)
    sh.to_csv(RESULTS / "grade_word_shares.csv", index=False)
    pd.set_option("display.width", 200)
    print(s.round(4).to_string())
    print(sh.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
