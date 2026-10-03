"""Computational cost of each confirmation model, under the evaluation protocol.

The manuscript previously called the character n-gram logistic regression the
"least expensive" model without measuring anything. This script supplies the
measurement: for every (category, seed) cell of the same stratified 80/20
protocol used by confirm_models.py it records

  * fit_s         -- wall-clock seconds to build the representation and fit
                     the classifier on the training split;
  * predict_s     -- wall-clock seconds to encode and score the test split;
  * throughput    -- test items scored per second (encoding included, because
                     that is what an operator actually pays);
  * size_bytes    -- serialized size of everything needed at inference time
                     (vectorizer plus classifier, or the torch state dict);
  * peak_mib      -- peak Python heap allocation during the fit, via
                     tracemalloc. For the torch models this undercounts tensor
                     memory held outside the Python allocator, so the
                     parameter count is reported alongside it.

Everything runs single-threaded on CPU so the numbers are comparable; the
thread caps are set before numpy/torch are imported by the caller where
possible, and again on the torch side here.

Run:  python src/cost_benchmark.py   (writes results/cost_benchmark*.csv)
"""
from __future__ import annotations

import io
import json
import os
import pickle
import platform
import time
import tracemalloc
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

import confirm_models as cm
from make_synth import _categories

# Same narrow filter as confirm_models: a blanket "ignore" here would hide a
# ConvergenceWarning raised while timing a fit.
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
REPEATS = 3  # inference is fast; take the best of a few passes to cut jitter


def _warm_up() -> None:
    """Absorb one-off framework initialisation before any timing is taken.

    The first call into PyTorch in a process pays lazy initialisation that has
    nothing to do with the model: measured cold it was 4.4 s against a 0.29 s
    median for the same architecture, and it inflated the CNN's mean training
    time by half. The same applies in miniature to the first scikit-learn fit,
    which pays BLAS set-up. Both are paid here, outside the measured region.
    """
    texts = [f"warm up token {i} alpha beta" for i in range(64)]
    labels = np.array([i % 2 for i in range(64)])
    vec = CountVectorizer(binary=True)
    LogisticRegression(max_iter=50).fit(vec.fit_transform(texts), labels)
    MLPClassifier(hidden_layer_sizes=(8,), max_iter=5,
                  random_state=0).fit(vec.transform(texts), labels)
    if cm.TORCH_OK:
        import torch
        torch.set_num_threads(1)
        stoi = cm._build_vocab(texts)
        x = torch.tensor(cm._encode(texts, stoi))
        y = torch.tensor(labels, dtype=torch.float32)
        for model in (cm._CNN(len(stoi)), cm._LSTM(len(stoi))):
            opt = torch.optim.Adam(model.parameters(), lr=2e-3)
            loss = torch.nn.BCEWithLogitsLoss()(model(x), y)
            loss.backward()
            opt.step()
            model.eval()
            with torch.no_grad():
                model(x)


# ---- helpers ---------------------------------------------------------------

def _sklearn_spec(model_name: str, seed: int):
    """Return (vectorizer, classifier) exactly as confirm_models builds them."""
    if model_name == "BoW+MLP(256)":
        return (CountVectorizer(binary=True),
                MLPClassifier(hidden_layer_sizes=(256,), max_iter=80,
                              random_state=seed))
    kind = {
        "Unigram BoW + LogReg": "bow",
        "Word 1-2-gram + LogReg": "1-2gram",
        "Word 1-3-gram + LogReg": "1-3gram",
        "Char n-gram(3-5) + LogReg": "char3-5",
    }[model_name]
    return (cm._vectorizer(kind),
            LogisticRegression(max_iter=1000, random_state=seed))


def _pickle_size(obj) -> int:
    buf = io.BytesIO()
    pickle.dump(obj, buf, protocol=pickle.HIGHEST_PROTOCOL)
    return buf.tell()


def _measure_sklearn(model_name: str, X_tr, y_tr, X_te, seed: int) -> dict:
    vec, clf = _sklearn_spec(model_name, seed)

    tracemalloc.start()
    t0 = time.perf_counter()
    Xtr = vec.fit_transform(X_tr)      # vocabulary from the train split only
    clf.fit(Xtr, y_tr)
    fit_s = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    best = None
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        clf.predict(vec.transform(X_te))
        dt = time.perf_counter() - t0
        best = dt if best is None else min(best, dt)

    n_params = int(np.prod(clf.coef_.shape)) + 1 if hasattr(clf, "coef_") else \
        int(sum(np.prod(w.shape) for w in clf.coefs_) +
            sum(np.prod(b.shape) for b in clf.intercepts_))
    return {
        "fit_s": fit_s,
        "predict_s": best,
        "throughput_items_s": len(X_te) / best,
        "size_bytes": _pickle_size((vec, clf)),
        "peak_mib": peak / 2 ** 20,
        "n_params": n_params,
        "vocab_size": len(vec.vocabulary_),
    }


def _measure_torch(kind: str, X_tr, y_tr, X_te, seed: int) -> dict:
    import torch
    import torch.nn as nn
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)

    tracemalloc.start()
    t0 = time.perf_counter()
    stoi = cm._build_vocab(X_tr)
    Xtr = torch.tensor(cm._encode(X_tr, stoi))
    ytr = torch.tensor(np.asarray(y_tr), dtype=torch.float32)
    model = cm._CNN(len(stoi)) if kind == "CNN" else cm._LSTM(len(stoi))
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    model.train()
    bs = 128
    for _epoch in range(8):
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = loss_fn(model(Xtr[idx]), ytr[idx])
            loss.backward()
            opt.step()
    fit_s = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    model.eval()
    best = None
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        with torch.no_grad():
            Xte = torch.tensor(cm._encode(X_te, stoi))
            (torch.sigmoid(model(Xte)) > 0.5).long().numpy()
        dt = time.perf_counter() - t0
        best = dt if best is None else min(best, dt)

    buf = io.BytesIO()
    torch.save({"state_dict": model.state_dict(), "stoi": stoi}, buf)
    return {
        "fit_s": fit_s,
        "predict_s": best,
        "throughput_items_s": len(X_te) / best,
        "size_bytes": buf.tell(),
        "peak_mib": peak / 2 ** 20,
        "n_params": int(sum(p.numel() for p in model.parameters())),
        "vocab_size": len(stoi),
    }


# ---- orchestration ---------------------------------------------------------

def run() -> pd.DataFrame:
    models = cm.SKLEARN_MODELS + (cm.TORCH_MODELS if cm.TORCH_OK else [])
    rows = []
    for cfg in _categories():
        df = pd.read_parquet(cm.DATA_DIR / f"{cfg.name}.parquet")
        texts, labels = df.text.tolist(), df.label.values
        for seed in cm.SEEDS:
            X_tr, X_te, y_tr, _y_te = train_test_split(
                texts, labels, test_size=0.2, stratify=labels,
                random_state=seed)
            for model in models:
                if model in cm.TORCH_MODELS:
                    m = _measure_torch(model, X_tr, y_tr, X_te, seed)
                else:
                    m = _measure_sklearn(model, X_tr, y_tr, X_te, seed)
                m.update({"category": cfg.name, "seed": seed, "model": model,
                          "n_train": len(X_tr), "n_test": len(X_te)})
                rows.append(m)
        print(f"  {cfg.name} done")
    cols = ["category", "seed", "model", "n_train", "n_test", "fit_s",
            "predict_s", "throughput_items_s", "size_bytes", "peak_mib",
            "n_params", "vocab_size"]
    return pd.DataFrame(rows)[cols]


def summarize(per_run: pd.DataFrame) -> pd.DataFrame:
    """Median alongside mean: cost is right-skewed and the median is what the
    manuscript quotes, so both are published rather than only the mean."""
    g = per_run.groupby("model")
    out = pd.DataFrame({
        "fit_s_median": g.fit_s.median(),
        "fit_s_mean": g.fit_s.mean(),
        "fit_s_sd": g.fit_s.std(ddof=1),
        "throughput_items_s_median": g.throughput_items_s.median(),
        "throughput_items_s_mean": g.throughput_items_s.mean(),
        "throughput_items_s_sd": g.throughput_items_s.std(ddof=1),
        "size_kib_mean": g.size_bytes.mean() / 1024,
        "peak_mib_median": g.peak_mib.median(),
        "n_params_mean": g.n_params.mean(),
    }).reset_index().sort_values("fit_s_median")
    return out


def environment() -> dict:
    env = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "numpy": np.__version__,
    }
    for mod, key in (("sklearn", "scikit_learn"), ("scipy", "scipy"),
                     ("pandas", "pandas")):
        try:
            env[key] = __import__(mod).__version__
        except Exception:
            pass
    if cm.TORCH_OK:
        import torch
        env["torch"] = torch.__version__
        env["torch_threads"] = torch.get_num_threads()
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    env["cpu_model"] = line.split(":", 1)[1].strip()
                    break
    except Exception:
        pass
    return env


def main() -> None:
    print("torch available:", cm.TORCH_OK)
    _warm_up()
    per_run = run()
    summary = summarize(per_run)
    RESULTS_DIR.mkdir(exist_ok=True)
    per_run.to_csv(RESULTS_DIR / "cost_benchmark_per_run.csv",
                   index=False, float_format="%.6g")
    summary.to_csv(RESULTS_DIR / "cost_benchmark_summary.csv",
                   index=False, float_format="%.6g")
    env = environment()
    (RESULTS_DIR / "cost_benchmark_env.json").write_text(
        json.dumps(env, indent=2) + "\n")
    pd.set_option("display.width", 200)
    print()
    print(summary.round(4).to_string(index=False))
    print()
    print(json.dumps(env, indent=2))


if __name__ == "__main__":
    main()
