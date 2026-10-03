"""Robustness checks behind Appendix "Robustness of the model comparison".

Each check tests whether a result in the model comparison depends on a choice
made for convenience rather than for a reason.

A. Did any model hit its iteration limit without converging?
B. Does the deep-model result depend on the training budget?
C. Does the ranking survive a threshold-free metric (average precision)?
D. Is throughput measured on a 330-item split representative of a real batch?
"""
from __future__ import annotations
import sys, time, warnings
import numpy as np, pandas as pd
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import confirm_models as cm
from make_synth import _categories

SEEDS = cm.SEEDS
CATS = _categories()


def load(cat):
    df = pd.read_parquet(cm.DATA_DIR / f"{cat.name}.parquet")
    return df.text.tolist(), df.label.values


# ---- A. convergence --------------------------------------------------------
def check_convergence():
    print("=" * 78)
    print("A. CONVERGENCE")
    rows = []
    for cfg in CATS:
        texts, labels = load(cfg)
        for seed in SEEDS:
            Xtr, Xte, ytr, yte = train_test_split(
                texts, labels, test_size=0.2, stratify=labels, random_state=seed)
            for name, kind in [("Unigram BoW + LogReg", "bow"),
                               ("Word 1-2-gram + LogReg", "1-2gram"),
                               ("Word 1-3-gram + LogReg", "1-3gram"),
                               ("Char n-gram(3-5) + LogReg", "char3-5")]:
                vec = cm._vectorizer(kind)
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    clf = LogisticRegression(max_iter=1000, random_state=seed)
                    clf.fit(vec.fit_transform(Xtr), ytr)
                    conv = not any("converge" in str(x.message).lower() for x in w)
                rows.append({"model": name, "n_iter": int(np.max(clf.n_iter_)),
                             "max_iter": 1000, "converged": conv})
            vec = cm._vectorizer("bow")
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                mlp = MLPClassifier(hidden_layer_sizes=(256,), max_iter=80,
                                    random_state=seed)
                mlp.fit(vec.fit_transform(Xtr), ytr)
                conv = not any("converge" in str(x.message).lower() for x in w)
            rows.append({"model": "BoW+MLP(256)", "n_iter": int(mlp.n_iter_),
                         "max_iter": 80, "converged": conv})
    d = pd.DataFrame(rows)
    g = d.groupby("model").agg(runs=("converged", "size"),
                               converged=("converged", "sum"),
                               median_iter=("n_iter", "median"),
                               max_seen=("n_iter", "max"),
                               budget=("max_iter", "first"))
    print(g.to_string())
    bad = g[g.converged < g.runs]
    print("\nVERDICT:", "all models converged" if bad.empty else
          f"NOT converged in some runs -> {list(bad.index)}")
    return d


# ---- B. deep-model epoch budget -------------------------------------------
def _fit_torch_epochs(kind, X_tr, y_tr, X_te, seed, epochs, early_stop=False):
    import torch, torch.nn as nn
    torch.manual_seed(seed); np.random.seed(seed); torch.set_num_threads(1)
    stoi = cm._build_vocab(X_tr)
    # carve a validation split out of TRAIN only, for honest early stopping
    if early_stop:
        Xa, Xv, ya, yv = train_test_split(X_tr, y_tr, test_size=0.2,
                                          stratify=y_tr, random_state=seed)
    else:
        Xa, ya, Xv, yv = X_tr, y_tr, None, None
    Xt = torch.tensor(cm._encode(Xa, stoi)); yt = torch.tensor(np.asarray(ya), dtype=torch.float32)
    model = cm._CNN(len(stoi)) if kind == "CNN" else cm._LSTM(len(stoi))
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    best_f1, best_state, bs = -1.0, None, 128
    for _ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), bs):
            idx = perm[i:i + bs]
            opt.zero_grad(); loss_fn(model(Xt[idx]), yt[idx]).backward(); opt.step()
        if early_stop:
            model.eval()
            with torch.no_grad():
                pv = (torch.sigmoid(model(torch.tensor(cm._encode(Xv, stoi)))) > 0.5).long().numpy()
            f1v = f1_score(yv, pv, zero_division=0)
            if f1v > best_f1:
                best_f1 = f1v
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
    if early_stop and best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        prob = torch.sigmoid(model(torch.tensor(cm._encode(X_te, stoi)))).numpy()
    return (prob > 0.5).astype(int), prob


def check_epoch_budget():
    print("=" * 78)
    print("B. TRAINING BUDGET: fixed 8 epochs vs fixed 40 vs matched stopping rule")
    rows = []
    for cfg in CATS:
        texts, labels = load(cfg)
        for seed in SEEDS:
            Xtr, Xte, ytr, yte = train_test_split(
                texts, labels, test_size=0.2, stratify=labels, random_state=seed)
            for kind in ["CNN", "LSTM"]:
                for tag, ep, es in [("8 epochs, fixed", 8, False),
                                    ("40 epochs, fixed", 40, False),
                                    ("40 epochs + early stop", 40, True)]:
                    pred, _ = _fit_torch_epochs(kind, Xtr, ytr, Xte, seed, ep, es)
                    rows.append({"model": kind, "setting": tag, "category": cfg.name,
                                 "seed": seed, "f1": f1_score(yte, pred, zero_division=0)})
        print(f"  {cfg.name} done")
    d = pd.DataFrame(rows)
    print()
    print(d.pivot_table(index="model", columns="setting", values="f1").round(4).to_string())
    return d


# ---- C. threshold-free ranking --------------------------------------------
def check_threshold_free():
    print("=" * 78)
    print("C. DOES THE RANKING SURVIVE A THRESHOLD-FREE METRIC?")
    rows = []
    for cfg in CATS:
        texts, labels = load(cfg)
        for seed in SEEDS:
            Xtr, Xte, ytr, yte = train_test_split(
                texts, labels, test_size=0.2, stratify=labels, random_state=seed)
            for name, kind in [("Unigram BoW + LogReg", "bow"),
                               ("Word 1-2-gram + LogReg", "1-2gram"),
                               ("Word 1-3-gram + LogReg", "1-3gram"),
                               ("Char n-gram(3-5) + LogReg", "char3-5")]:
                vec = cm._vectorizer(kind)
                clf = LogisticRegression(max_iter=1000, random_state=seed)
                clf.fit(vec.fit_transform(Xtr), ytr)
                p = clf.predict_proba(vec.transform(Xte))[:, 1]
                rows.append({"model": name, "f1": f1_score(yte, (p > .5).astype(int), zero_division=0),
                             "ap": average_precision_score(yte, p)})
            vec = cm._vectorizer("bow")
            mlp = MLPClassifier(hidden_layer_sizes=(256,), max_iter=80, random_state=seed)
            mlp.fit(vec.fit_transform(Xtr), ytr)
            p = mlp.predict_proba(vec.transform(Xte))[:, 1]
            rows.append({"model": "BoW+MLP(256)", "f1": f1_score(yte, (p > .5).astype(int), zero_division=0),
                         "ap": average_precision_score(yte, p)})
            for kind in ["CNN", "LSTM"]:
                pred, prob = _fit_torch_epochs(kind, Xtr, ytr, Xte, seed, 8, False)
                rows.append({"model": kind, "f1": f1_score(yte, pred, zero_division=0),
                             "ap": average_precision_score(yte, prob)})
    d = pd.DataFrame(rows)
    g = d.groupby("model")[["f1", "ap"]].mean().sort_values("f1", ascending=False)
    g["rank_f1"] = g.f1.rank(ascending=False).astype(int)
    g["rank_ap"] = g.ap.rank(ascending=False).astype(int)
    print(g.round(4).to_string())
    print("\nVERDICT:", "ranking identical under both metrics"
          if (g.rank_f1 == g.rank_ap).all() else "RANKING MOVES between F1 and average precision")
    return d


# ---- D. throughput at realistic batch size ---------------------------------
def check_throughput_scale():
    print("=" * 78)
    print("D. THROUGHPUT AGAINST BATCH SIZE")
    texts, labels = load(CATS[0])
    Xtr, Xte, ytr, yte = train_test_split(texts, labels, test_size=0.2,
                                          stratify=labels, random_state=0)
    rows = []
    for name, kind in [("Unigram BoW + LogReg", "bow"),
                       ("Char n-gram(3-5) + LogReg", "char3-5")]:
        vec = cm._vectorizer(kind)
        clf = LogisticRegression(max_iter=1000, random_state=0)
        clf.fit(vec.fit_transform(Xtr), ytr)
        for n in [330, 3_300, 33_000, 330_000]:
            batch = (Xte * (n // len(Xte) + 1))[:n]
            best = min(_time_predict(vec, clf, batch) for _ in range(3))
            rows.append({"model": name, "n_items": n, "items_s": n / best})
    d = pd.DataFrame(rows)
    p = d.pivot_table(index="n_items", columns="model", values="items_s")
    print(p.round(0).to_string())
    base = p.loc[330]; big = p.loc[330_000]
    print("\nratio at 330 items : %.2fx" % (base["Unigram BoW + LogReg"] / base["Char n-gram(3-5) + LogReg"]))
    print("ratio at 330k items: %.2fx" % (big["Unigram BoW + LogReg"] / big["Char n-gram(3-5) + LogReg"]))
    return d


def _time_predict(vec, clf, batch):
    t0 = time.perf_counter()
    clf.predict(vec.transform(batch))
    return time.perf_counter() - t0


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "a"): check_convergence()
    if which in ("all", "d"): check_throughput_scale()
    if which in ("all", "c"): check_threshold_free()
    if which in ("all", "b"): check_epoch_budget()
