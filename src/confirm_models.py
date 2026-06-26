"""Per-category binary confirmation models (Section 5 / Table 'controlled').

One protocol for every model:
  * stratified 80/20 train/test split per seed (seeds 0..4);
  * the vocabulary is fit on the TRAIN split only (anti-leakage -- building it
    over the full corpus would leak test information, as the paper stresses);
  * we report mean +/- s.d. of accuracy, precision, recall, F1 over seeds.

Models (matched comparison under one protocol):
  - Unigram BoW + Logistic Regression
  - Word 1-2-gram + Logistic Regression
  - Word 1-3-gram + Logistic Regression
  - Char n-gram (3-5) + Logistic Regression
  - BoW + MLP (1 hidden layer, 256)
  - CNN  (1D conv over learned embeddings)   <- NOT run in the preprint
  - LSTM (recurrent over learned embeddings) <- NOT run in the preprint

Plus a learning curve for the unigram model (F1 vs train fraction) to probe the
preprint's "~67 examples" observation.

CNN/LSTM require PyTorch. If torch is unavailable they are skipped and clearly
reported as skipped -- never silently dropped (honesty requirement).
"""
from __future__ import annotations

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from make_synth import _categories

warnings.filterwarnings("ignore")

SEEDS = [0, 1, 2, 3, 4]
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "synth"

try:
    import torch
    import torch.nn as nn
    TORCH_OK = True
except Exception:  # pragma: no cover - environment dependent
    TORCH_OK = False


# ---- sklearn linear / MLP models -------------------------------------------

def _vectorizer(kind: str) -> CountVectorizer:
    if kind == "bow":
        return CountVectorizer(binary=True)
    if kind == "1-2gram":
        return CountVectorizer(binary=True, ngram_range=(1, 2))
    if kind == "1-3gram":
        return CountVectorizer(binary=True, ngram_range=(1, 3))
    if kind == "char3-5":
        return CountVectorizer(binary=True, analyzer="char_wb", ngram_range=(3, 5))
    raise ValueError(kind)


def _metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def _fit_sklearn(model_name: str, X_tr, y_tr, X_te, y_te, seed: int) -> dict:
    t0 = time.time()
    if model_name == "BoW+MLP(256)":
        vec = _vectorizer("bow")
        Xtr = vec.fit_transform(X_tr)
        Xte = vec.transform(X_te)
        clf = MLPClassifier(hidden_layer_sizes=(256,), max_iter=80,
                            random_state=seed)
    else:
        kind = {
            "Unigram BoW + LogReg": "bow",
            "Word 1-2-gram + LogReg": "1-2gram",
            "Word 1-3-gram + LogReg": "1-3gram",
            "Char n-gram(3-5) + LogReg": "char3-5",
        }[model_name]
        vec = _vectorizer(kind)
        Xtr = vec.fit_transform(X_tr)          # vocab on TRAIN only
        Xte = vec.transform(X_te)
        clf = LogisticRegression(max_iter=1000, random_state=seed)
    clf.fit(Xtr, y_tr)
    pred = clf.predict(Xte)
    m = _metrics(y_te, pred)
    m["train_s"] = time.time() - t0
    return m


# ---- torch CNN / LSTM ------------------------------------------------------

def _build_vocab(texts, max_vocab=4000):
    from collections import Counter
    cnt = Counter(tok for t in texts for tok in t.split())
    itos = ["<pad>", "<unk>"] + [w for w, _ in cnt.most_common(max_vocab)]
    stoi = {w: i for i, w in enumerate(itos)}
    return stoi


def _encode(texts, stoi, maxlen=12):
    out = np.zeros((len(texts), maxlen), dtype=np.int64)
    for i, t in enumerate(texts):
        toks = t.split()[:maxlen]
        for j, tok in enumerate(toks):
            out[i, j] = stoi.get(tok, 1)  # 1 = <unk>
    return out


if TORCH_OK:
    class _CNN(nn.Module):
        def __init__(self, vocab, emb=32, ch=64):
            super().__init__()
            self.emb = nn.Embedding(vocab, emb, padding_idx=0)
            self.conv = nn.Conv1d(emb, ch, kernel_size=3, padding=1)
            self.fc = nn.Linear(ch, 1)

        def forward(self, x):
            e = self.emb(x).transpose(1, 2)        # (B, emb, L)
            h = torch.relu(self.conv(e))
            h = torch.max(h, dim=2).values         # global max pool
            return self.fc(h).squeeze(1)

    class _LSTM(nn.Module):
        def __init__(self, vocab, emb=32, hid=64):
            super().__init__()
            self.emb = nn.Embedding(vocab, emb, padding_idx=0)
            self.lstm = nn.LSTM(emb, hid, batch_first=True)
            self.fc = nn.Linear(hid, 1)

        def forward(self, x):
            e = self.emb(x)
            _, (h, _) = self.lstm(e)
            return self.fc(h[-1]).squeeze(1)


def _fit_torch(kind: str, X_tr, y_tr, X_te, y_te, seed: int) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    t0 = time.time()
    stoi = _build_vocab(X_tr)                       # vocab on TRAIN only
    Xtr = torch.tensor(_encode(X_tr, stoi))
    Xte = torch.tensor(_encode(X_te, stoi))
    ytr = torch.tensor(np.asarray(y_tr), dtype=torch.float32)
    model = _CNN(len(stoi)) if kind == "CNN" else _LSTM(len(stoi))
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    model.train()
    bs = 128
    for _epoch in range(8):
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            out = model(Xtr[idx])
            loss = loss_fn(out, ytr[idx])
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        pred = (torch.sigmoid(model(Xte)) > 0.5).long().numpy()
    m = _metrics(y_te, pred)
    m["train_s"] = time.time() - t0
    return m


# ---- orchestration ---------------------------------------------------------

SKLEARN_MODELS = [
    "Unigram BoW + LogReg",
    "Word 1-2-gram + LogReg",
    "Word 1-3-gram + LogReg",
    "Char n-gram(3-5) + LogReg",
    "BoW+MLP(256)",
]
TORCH_MODELS = ["CNN", "LSTM"]


def _aggregate(per_seed: list[dict]) -> dict:
    keys = ["accuracy", "precision", "recall", "f1", "train_s"]
    out = {}
    for k in keys:
        vals = np.array([d[k] for d in per_seed])
        out[f"{k}_mean"] = float(vals.mean())
        out[f"{k}_sd"] = float(vals.std(ddof=0))
    return out


def evaluate_all() -> pd.DataFrame:
    rows = []
    cats = _categories()
    for cfg in cats:
        df = pd.read_parquet(DATA_DIR / f"{cfg.name}.parquet")
        texts = df.text.tolist()
        labels = df.label.values
        for model in SKLEARN_MODELS + (TORCH_MODELS if TORCH_OK else []):
            per_seed = []
            for seed in SEEDS:
                X_tr, X_te, y_tr, y_te = train_test_split(
                    texts, labels, test_size=0.2, stratify=labels,
                    random_state=seed)
                if model in TORCH_MODELS:
                    per_seed.append(_fit_torch(model, X_tr, y_tr, X_te, y_te, seed))
                else:
                    per_seed.append(_fit_sklearn(model, X_tr, y_tr, X_te, y_te, seed))
            agg = _aggregate(per_seed)
            agg.update({"category": cfg.name, "model": model})
            rows.append(agg)
            print(f"  {cfg.name:18s} {model:26s} "
                  f"F1={agg['f1_mean']:.3f}+/-{agg['f1_sd']:.3f}")
        if not TORCH_OK:
            print(f"  {cfg.name:18s} CNN/LSTM ............ SKIPPED (torch unavailable)")
    cols = ["category", "model", "accuracy_mean", "accuracy_sd",
            "precision_mean", "precision_sd", "recall_mean", "recall_sd",
            "f1_mean", "f1_sd", "train_s_mean", "train_s_sd"]
    return pd.DataFrame(rows)[cols]


def learning_curve() -> pd.DataFrame:
    """Unigram BoW F1 vs train fraction, to probe the '~67 examples' claim."""
    fractions = [0.05, 0.10, 0.20, 0.40, 1.00]
    rows = []
    for cfg in _categories():
        df = pd.read_parquet(DATA_DIR / f"{cfg.name}.parquet")
        texts = df.text.tolist()
        labels = df.label.values
        for frac in fractions:
            f1s, ns = [], []
            for seed in SEEDS:
                X_tr, X_te, y_tr, y_te = train_test_split(
                    texts, labels, test_size=0.2, stratify=labels,
                    random_state=seed)
                if frac < 1.0:
                    X_sub, _, y_sub, _ = train_test_split(
                        X_tr, y_tr, train_size=frac, stratify=y_tr,
                        random_state=seed)
                else:
                    X_sub, y_sub = X_tr, y_tr
                vec = _vectorizer("bow")
                Xtr = vec.fit_transform(X_sub)
                Xte = vec.transform(X_te)
                clf = LogisticRegression(max_iter=1000, random_state=seed)
                clf.fit(Xtr, y_sub)
                f1s.append(f1_score(y_te, clf.predict(Xte), zero_division=0))
                ns.append(len(X_sub))
            rows.append({
                "category": cfg.name,
                "train_fraction": frac,
                "n_train_examples": int(np.mean(ns)),
                "f1_mean": float(np.mean(f1s)),
                "f1_sd": float(np.std(f1s, ddof=0)),
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("torch available:", TORCH_OK)
    print(evaluate_all().to_string(index=False))
