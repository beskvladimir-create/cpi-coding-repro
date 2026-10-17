"""Robustness checks behind Appendix "Robustness of the model comparison".

Each check tests whether a result in the model comparison depends on a choice
made for convenience rather than for a reason. Every check writes its
per-cell results and a summary to results/:

A. Did any model hit its iteration limit without converging?
     results/robustness_convergence.csv          (one row per category/seed/model)
     results/robustness_convergence_summary.csv
B. Does the deep-model result depend on the training budget?
     results/robustness_training_budget.csv      (one row per category/seed/model/arm)
     results/robustness_training_budget_summary.csv
C. Does the ranking survive a threshold-free metric (average precision)?
     results/robustness_average_precision.csv    (one row per category/seed/model)
     results/robustness_average_precision_summary.csv
D. Is throughput measured on a 330-item split representative of a real batch?
     results/throughput_by_batch.csv             (one row per category/model/size)
     results/throughput_by_batch_summary.csv

The four arms of check B
------------------------
All four use the same architecture, optimiser (Adam, lr 2e-3), batch size
(128), seed handling and vocabulary (built on the full training split) as
confirm_models.train_torch. They differ only in when training stops:

  "8 epochs, fixed"    exactly 8 passes over the training split.
  "40 epochs, fixed"   exactly 40 passes over the training split.
  "matched rule"       confirm_models.train_torch itself, i.e. the rule used
                       for every number in the paper: stop when the mean epoch
                       TRAINING loss fails to improve on the best so far by
                       more than 1e-4 for 10 consecutive epochs, cap 200; the
                       final weights are used (no checkpoint restore).
  "validation early stop"
                       a stratified 20% validation split is carved out of the
                       TRAINING split (random_state = seed), the model is
                       trained on the remaining 80% for a cap of 40 epochs with
                       no patience (all 40 epochs always run), validation F1
                       at threshold 0.5 is computed after every epoch, and the
                       weights of the epoch with the highest validation F1
                       (first occurrence on ties) are restored before scoring
                       the test split. The vocabulary is built on the full
                       training split, as in the other arms.

C and the matched-rule arm of B call confirm_models.train_torch, so they
evaluate exactly the trained models the accuracy tables come from.
"""
from __future__ import annotations
import sys, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from threadpoolctl import threadpool_limits
import confirm_models as cm
from make_synth import _categories

SEEDS = cm.SEEDS
CATS = _categories()
RESULTS = Path(__file__).resolve().parent.parent / "results"
LOGREG = [("Unigram BoW + LogReg", "bow"),
          ("Word 1-2-gram + LogReg", "1-2gram"),
          ("Word 1-3-gram + LogReg", "1-3gram"),
          ("Char n-gram(3-5) + LogReg", "char3-5")]
BATCH_SIZES = [330, 3_300, 33_000, 330_000]


def load(cat):
    df = pd.read_parquet(cm.DATA_DIR / f"{cat.name}.parquet")
    return df.text.tolist(), df.label.values


def split(texts, labels, seed):
    return train_test_split(texts, labels, test_size=0.2, stratify=labels,
                            random_state=seed)


def _save(df: pd.DataFrame, name: str) -> None:
    RESULTS.mkdir(exist_ok=True)
    df.to_csv(RESULTS / name, index=False, float_format="%.6g")
    print(f"  wrote results/{name}")


# ---- matched-rule torch models, trained once and shared by A, B and C -------
_MATCHED: dict = {}


def matched_torch(kind, cat_name, seed, Xtr, ytr, Xte):
    """confirm_models.train_torch on this cell; returns (pred, prob, n_epochs).

    Memoised so that checks A, B and C evaluate one and the same fitted model
    per cell (training is seeded and deterministic in any case)."""
    key = (kind, cat_name, seed)
    if key not in _MATCHED:
        import torch
        model, stoi, n_epochs = cm.train_torch(kind, Xtr, ytr, seed)
        model.eval()
        with torch.no_grad():
            prob = torch.sigmoid(model(torch.tensor(cm._encode(Xte, stoi)))).numpy()
        pred = cm.predict_torch(model, stoi, Xte)
        _MATCHED[key] = (pred, prob, n_epochs)
    return _MATCHED[key]


# ---- A. convergence --------------------------------------------------------
def check_convergence():
    print("=" * 78)
    print("A. CONVERGENCE")
    rows = []
    for cfg in CATS:
        texts, labels = load(cfg)
        for seed in SEEDS:
            Xtr, Xte, ytr, yte = split(texts, labels, seed)
            base = {"category": cfg.name, "seed": seed}
            for name, kind in LOGREG:
                vec = cm._vectorizer(kind)
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    clf = LogisticRegression(max_iter=1000, random_state=seed)
                    clf.fit(vec.fit_transform(Xtr), ytr)
                    conv = not any("converge" in str(x.message).lower() for x in w)
                n_iter = int(np.max(clf.n_iter_))
                rows.append({**base, "model": name, "n_iter": n_iter,
                             "max_iter": 1000, "hit_cap": n_iter >= 1000,
                             "converged": conv})
            vec = cm._vectorizer("bow")
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                mlp = MLPClassifier(hidden_layer_sizes=(256,), max_iter=80,
                                    random_state=seed)
                mlp.fit(vec.fit_transform(Xtr), ytr)
                conv = not any("converge" in str(x.message).lower() for x in w)
            rows.append({**base, "model": "BoW+MLP(256)", "n_iter": int(mlp.n_iter_),
                         "max_iter": 80, "hit_cap": int(mlp.n_iter_) >= 80,
                         "converged": conv})
            if cm.TORCH_OK:
                for kind in cm.TORCH_MODELS:
                    _p, _pr, n_ep = matched_torch(kind, cfg.name, seed, Xtr, ytr, Xte)
                    hit = n_ep >= cm.TORCH_MAX_EPOCHS
                    rows.append({**base, "model": kind, "n_iter": n_ep,
                                 "max_iter": cm.TORCH_MAX_EPOCHS, "hit_cap": hit,
                                 "converged": not hit})
    d = pd.DataFrame(rows)
    g = d.groupby("model", sort=False).agg(
        runs=("converged", "size"), converged=("converged", "sum"),
        hit_cap=("hit_cap", "sum"), median_iter=("n_iter", "median"),
        min_seen=("n_iter", "min"), max_seen=("n_iter", "max"),
        budget=("max_iter", "first")).reset_index()
    print(g.to_string(index=False))
    if cm.TORCH_OK:
        pooled = d[d.model.isin(cm.TORCH_MODELS)].n_iter.median()
        print(f"\nCNN+LSTM pooled median epochs: {pooled}")
    bad = g[(g.converged < g.runs) | (g.hit_cap > 0)]
    print("\nVERDICT:", "all models converged, none hit its cap" if bad.empty else
          f"NOT converged / hit cap in some runs -> {list(bad.model)}")
    _save(d, "robustness_convergence.csv")
    _save(g, "robustness_convergence_summary.csv")
    return d


# ---- B. deep-model training budget ----------------------------------------
def _fit_torch_epochs(kind, X_tr, y_tr, X_te, seed, epochs, early_stop=False):
    """Fixed-epoch training, optionally with validation-F1 checkpoint selection.

    See the module docstring for the exact definition of each arm. Returns
    (pred, prob, epochs_run, best_epoch) where best_epoch is the 1-based epoch
    whose weights were restored (validation arm) or None.
    """
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
    best_f1, best_state, best_epoch, bs = -1.0, None, None, 128
    for ep in range(epochs):
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
                best_f1, best_epoch = f1v, ep + 1
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
    if early_stop and best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        prob = torch.sigmoid(model(torch.tensor(cm._encode(X_te, stoi)))).numpy()
    return (prob > 0.5).astype(int), prob, epochs, best_epoch


ARM_8 = "8 epochs, fixed"
ARM_40 = "40 epochs, fixed"
ARM_MATCHED = "matched rule (train loss, tol 1e-4, patience 10, cap 200)"
ARM_VAL = "validation early stop (20% of train, cap 40, best val F1)"
ARMS = [ARM_8, ARM_40, ARM_MATCHED, ARM_VAL]


def check_epoch_budget():
    print("=" * 78)
    print("B. TRAINING BUDGET: fixed 8 / fixed 40 / matched rule / validation early stop")
    rows = []
    for cfg in CATS:
        texts, labels = load(cfg)
        for seed in SEEDS:
            Xtr, Xte, ytr, yte = split(texts, labels, seed)
            for kind in cm.TORCH_MODELS:
                for arm in ARMS:
                    best_epoch = None
                    if arm == ARM_MATCHED:
                        pred, _prob, n_ep = matched_torch(kind, cfg.name, seed, Xtr, ytr, Xte)
                    elif arm == ARM_VAL:
                        pred, _prob, n_ep, best_epoch = _fit_torch_epochs(
                            kind, Xtr, ytr, Xte, seed, 40, True)
                    else:
                        ep = 8 if arm == ARM_8 else 40
                        pred, _prob, n_ep, _ = _fit_torch_epochs(kind, Xtr, ytr, Xte, seed, ep, False)
                    rows.append({"category": cfg.name, "seed": seed, "model": kind,
                                 "setting": arm, "f1": f1_score(yte, pred, zero_division=0),
                                 "epochs_run": n_ep,
                                 "restored_epoch": np.nan if best_epoch is None else best_epoch})
        print(f"  {cfg.name} done")
    d = pd.DataFrame(rows)
    s = (d.groupby(["model", "setting"], sort=False)
         .agg(n=("f1", "size"), f1_mean=("f1", "mean"), f1_sd=("f1", "std"),
              epochs_run_median=("epochs_run", "median"),
              restored_epoch_median=("restored_epoch", "median"))
         .reset_index())
    print()
    print(d.pivot_table(index="model", columns="setting", values="f1")[ARMS].round(4).to_string())
    _save(d, "robustness_training_budget.csv")
    _save(s, "robustness_training_budget_summary.csv")
    return d


# ---- C. threshold-free ranking --------------------------------------------
def check_threshold_free():
    print("=" * 78)
    print("C. DOES THE RANKING SURVIVE A THRESHOLD-FREE METRIC?")
    rows = []
    for cfg in CATS:
        texts, labels = load(cfg)
        for seed in SEEDS:
            Xtr, Xte, ytr, yte = split(texts, labels, seed)
            base = {"category": cfg.name, "seed": seed}
            for name, kind in LOGREG:
                vec = cm._vectorizer(kind)
                clf = LogisticRegression(max_iter=1000, random_state=seed)
                clf.fit(vec.fit_transform(Xtr), ytr)
                p = clf.predict_proba(vec.transform(Xte))[:, 1]
                rows.append({**base, "model": name,
                             "f1": f1_score(yte, (p > .5).astype(int), zero_division=0),
                             "ap": average_precision_score(yte, p)})
            vec = cm._vectorizer("bow")
            mlp = MLPClassifier(hidden_layer_sizes=(256,), max_iter=80, random_state=seed)
            mlp.fit(vec.fit_transform(Xtr), ytr)
            p = mlp.predict_proba(vec.transform(Xte))[:, 1]
            rows.append({**base, "model": "BoW+MLP(256)",
                         "f1": f1_score(yte, (p > .5).astype(int), zero_division=0),
                         "ap": average_precision_score(yte, p)})
            if cm.TORCH_OK:
                for kind in cm.TORCH_MODELS:   # matched stopping rule, as in the paper
                    pred, prob, _n = matched_torch(kind, cfg.name, seed, Xtr, ytr, Xte)
                    rows.append({**base, "model": kind,
                                 "f1": f1_score(yte, pred, zero_division=0),
                                 "ap": average_precision_score(yte, prob)})
    d = pd.DataFrame(rows)
    g = d.groupby("model").agg(n=("ap", "size"), f1_mean=("f1", "mean"),
                               ap_mean=("ap", "mean"), ap_sd=("ap", "std"),
                               ap_min=("ap", "min"),
                               splits_ap_eq_1=("ap", lambda s: int((s == 1.0).sum())))
    g = g.sort_values("f1_mean", ascending=False)
    g["rank_f1"] = g.f1_mean.rank(ascending=False, method="min").astype(int)
    g["rank_ap"] = g.ap_mean.rank(ascending=False, method="min").astype(int)
    g = g.reset_index()
    pd.set_option("display.width", 200)
    print(g.round(4).to_string(index=False))
    print("\nVERDICT:", "ranking identical under both metrics"
          if (g.rank_f1 == g.rank_ap).all() else "RANKING MOVES between F1 and average precision")
    _save(d, "robustness_average_precision.csv")
    _save(g, "robustness_average_precision_summary.csv")
    return d


# ---- D. throughput at realistic batch size ---------------------------------
def check_throughput_scale():
    """Seed-0 split of every category; the 330-item test split is tiled to
    reach each batch size; best of 3 passes; single-threaded."""
    print("=" * 78)
    print("D. THROUGHPUT AGAINST BATCH SIZE (all categories)")
    rows = []
    with threadpool_limits(limits=1):
        for cfg in CATS:
            texts, labels = load(cfg)
            Xtr, Xte, ytr, yte = split(texts, labels, 0)
            for name, kind in [("Unigram BoW + LogReg", "bow"),
                               ("Char n-gram(3-5) + LogReg", "char3-5")]:
                vec = cm._vectorizer(kind)
                clf = LogisticRegression(max_iter=1000, random_state=0)
                clf.fit(vec.fit_transform(Xtr), ytr)
                _time_predict(vec, clf, Xte)          # warm-up
                for n in BATCH_SIZES:
                    batch = (Xte * (n // len(Xte) + 1))[:n]
                    best = min(_time_predict(vec, clf, batch) for _ in range(3))
                    rows.append({"category": cfg.name, "model": name, "n_items": n,
                                 "best_s": best, "items_s": n / best})
            print(f"  {cfg.name} done")
    d = pd.DataFrame(rows)
    w = d.pivot_table(index=["category", "n_items"], columns="model",
                      values="items_s").reset_index()
    w["ratio_unigram_over_char"] = w["Unigram BoW + LogReg"] / w["Char n-gram(3-5) + LogReg"]
    base = w[w.n_items == 330].set_index("category")
    w["unigram_gain_vs_330"] = w.apply(
        lambda r: r["Unigram BoW + LogReg"] / base.loc[r.category, "Unigram BoW + LogReg"], axis=1)
    w["char_gain_vs_330"] = w.apply(
        lambda r: r["Char n-gram(3-5) + LogReg"] / base.loc[r.category, "Char n-gram(3-5) + LogReg"], axis=1)
    s = w.groupby("n_items").agg(
        unigram_items_s_median=("Unigram BoW + LogReg", "median"),
        char_items_s_median=("Char n-gram(3-5) + LogReg", "median"),
        ratio_median=("ratio_unigram_over_char", "median"),
        ratio_min=("ratio_unigram_over_char", "min"),
        ratio_max=("ratio_unigram_over_char", "max"),
        unigram_gain_median=("unigram_gain_vs_330", "median"),
        char_gain_median=("char_gain_vs_330", "median")).reset_index()
    pd.set_option("display.width", 200)
    print(w.round(3).to_string(index=False))
    print()
    print(s.round(3).to_string(index=False))
    d = d.merge(w[["category", "n_items", "ratio_unigram_over_char"]],
                on=["category", "n_items"])
    _save(d, "throughput_by_batch.csv")
    _save(s, "throughput_by_batch_summary.csv")
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
