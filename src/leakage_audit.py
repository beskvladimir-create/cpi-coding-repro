"""Leakage and trie audit for the synthetic benchmark (arXiv:2606.02004).

Answers questions a reviewer cannot check from the manuscript alone:
  A. corpus duplication   -- how many generated strings are distinct;
  B. train/test overlap   -- share of test items whose exact string also occurs
                             in the training split, per seed (same split
                             protocol as confirm_models.py);
  C. honest re-evaluation -- F1 on the full test split vs F1 restricted to test
                             items unseen in training, and F1 under a
                             deduplicate-then-split protocol;
  D. trie audit           -- coverage against its construction ceiling,
                             positives lost by the trie, trie precision.

Run:  python src/leakage_audit.py   (writes results/leakage_audit_*.csv)
All data are synthetic; nothing here touches production data.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from make_synth import _categories
from trie_classifier import build_classifier, classify, tokenize

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "synth"
RESULTS_DIR = ROOT / "results"
SEEDS = [0, 1, 2, 3, 4]

VECTORIZERS = {
    "Unigram BoW + LogReg": dict(binary=True),
    "Word 1-2-gram + LogReg": dict(binary=True, ngram_range=(1, 2)),
    "Word 1-3-gram + LogReg": dict(binary=True, ngram_range=(1, 3)),
    "Char n-gram(3-5) + LogReg": dict(binary=True, analyzer="char_wb",
                                      ngram_range=(3, 5)),
}


def _load(name: str) -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / f"{name}.parquet")


def duplication_report() -> pd.DataFrame:
    rows = []
    for cfg in _categories():
        df = _load(cfg.name)
        conflicting = df.groupby("text").label.nunique()
        rows.append({
            "category": cfg.name,
            "n_items": len(df),
            "n_distinct": int(df.text.nunique()),
            "distinct_share": round(df.text.nunique() / len(df), 4),
            "n_distinct_pos": int(df[df.label == 1].text.nunique()),
            "n_distinct_neg": int(df[df.label == 0].text.nunique()),
            "strings_with_both_labels": int((conflicting > 1).sum()),
        })
    return pd.DataFrame(rows)


def _fit_predict(kw: dict, X_tr, y_tr, X_te, seed: int):
    vec = CountVectorizer(**kw)
    clf = LogisticRegression(max_iter=1000, random_state=seed)
    clf.fit(vec.fit_transform(X_tr), y_tr)
    return clf.predict(vec.transform(X_te))


def overlap_and_reeval() -> tuple[pd.DataFrame, pd.DataFrame]:
    ov_rows, ev_rows = [], []
    for cfg in _categories():
        df = _load(cfg.name)
        texts, labels = df.text.tolist(), df.label.values
        dedup = df.drop_duplicates("text")
        for seed in SEEDS:
            X_tr, X_te, y_tr, y_te = train_test_split(
                texts, labels, test_size=0.2, stratify=labels, random_state=seed)
            seen = set(X_tr)
            in_train = np.array([t in seen for t in X_te])
            y_te = np.asarray(y_te)
            ov_rows.append({
                "category": cfg.name, "seed": seed,
                "test_overlap": in_train.mean(),
                "test_overlap_pos": in_train[y_te == 1].mean(),
                "test_overlap_neg": in_train[y_te == 0].mean(),
            })
            Xd_tr, Xd_te, yd_tr, yd_te = train_test_split(
                dedup.text.tolist(), dedup.label.values, test_size=0.2,
                stratify=dedup.label.values, random_state=seed)
            for model, kw in VECTORIZERS.items():
                pred = _fit_predict(kw, X_tr, y_tr, X_te, seed)
                pred_d = _fit_predict(kw, Xd_tr, yd_tr, Xd_te, seed)
                ev_rows.append({
                    "category": cfg.name, "seed": seed, "model": model,
                    "f1_full": f1_score(y_te, pred),
                    "f1_unseen": f1_score(y_te[~in_train], pred[~in_train]),
                    "f1_dedup_split": f1_score(yd_te, pred_d),
                })
    return pd.DataFrame(ov_rows), pd.DataFrame(ev_rows)


def trie_audit() -> pd.DataFrame:
    clf = build_classifier()
    rows = []
    for cfg in _categories():
        df = _load(cfg.name)
        admitted = np.array([classify(tokenize(t), clf) == cfg.name for t in df.text])
        pos = df.label.values == 1
        n = len(df)
        rows.append({
            "category": cfg.name,
            "n_items": n,
            "n_pos": int(pos.sum()),
            "n_neg": int((~pos).sum()),
            "coverage": admitted.mean(),
            "coverage_ceiling": pos.mean(),
            "pos_admitted": int((admitted & pos).sum()),
            "pos_lost": int((~admitted & pos).sum()),
            "pos_lost_share_of_pos": (~admitted & pos).sum() / pos.sum(),
            "pos_lost_share_of_corpus": (~admitted & pos).sum() / n,
            "neg_correctly_rejected": int((~admitted & ~pos).sum()),
            "neg_falsely_admitted": int((admitted & ~pos).sum()),
            "trie_precision": (admitted & pos).sum() / max(1, admitted.sum()),
            "trie_recall": (admitted & pos).sum() / pos.sum(),
        })
    return pd.DataFrame(rows)


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    dup = duplication_report()
    ov, ev = overlap_and_reeval()
    tr = trie_audit()
    dup.to_csv(RESULTS_DIR / "leakage_audit_duplication.csv", index=False, float_format="%.6g")
    ov.to_csv(RESULTS_DIR / "leakage_audit_overlap.csv", index=False, float_format="%.6g")
    ev.to_csv(RESULTS_DIR / "leakage_audit_reeval.csv", index=False, float_format="%.6g")
    tr.to_csv(RESULTS_DIR / "trie_audit.csv", index=False, float_format="%.6g")
    pd.set_option("display.width", 200)
    print("A. duplication\n", dup.to_string(index=False))
    print("\nB. overlap (mean over seeds)\n",
          ov.groupby("category")[["test_overlap", "test_overlap_pos",
                                  "test_overlap_neg"]].mean().round(3))
    print("overall test overlap:", round(ov.test_overlap.mean(), 4))
    print("\nC. re-evaluation (mean over categories x seeds)\n",
          ev.groupby("model")[["f1_full", "f1_unseen", "f1_dedup_split"]]
          .mean().round(4))
    print("\nD. trie audit\n", tr.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
