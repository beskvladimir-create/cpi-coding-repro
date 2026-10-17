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
  * peak_mib      -- peak Python heap allocation during a fit, via
                     tracemalloc. For the torch models this undercounts tensor
                     memory held outside the Python allocator, so the
                     parameter count is reported alongside it.

Timing and memory are measured in SEPARATE fits. tracemalloc hooks every
Python allocation and slows the interpreter, so a fit timed while it runs is
not the fit an operator pays for. Release 2.0.1 timed training with
tracemalloc active, which inflated the scikit-learn fit times in particular.
Here the timed fit runs with tracemalloc off; peak memory comes from a second,
identical fit (same spec, same seed) that is not timed.

Everything runs single-threaded on CPU so the numbers are comparable:
OMP/MKL/OpenBLAS thread counts are set to one below, before numpy is
imported; every measurement runs inside threadpoolctl.threadpool_limits(1);
and torch is limited to one intra-op and one inter-op thread. The thread-pool
state actually in force is recorded in results/cost_benchmark_env.json.

Run:  python src/cost_benchmark.py   (writes results/cost_benchmark*.csv)
"""
from __future__ import annotations

import os

# Must precede the numpy / scipy / torch imports: the BLAS and OpenMP runtimes
# read these once, when they are loaded.
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_var] = "1"

import io  # noqa: E402
import json  # noqa: E402
import pickle  # noqa: E402
import platform  # noqa: E402
import time  # noqa: E402
import tracemalloc  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.feature_extraction.text import CountVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.neural_network import MLPClassifier  # noqa: E402
from threadpoolctl import threadpool_info, threadpool_limits  # noqa: E402

import confirm_models as cm  # noqa: E402
from make_synth import _categories  # noqa: E402

_INTEROP_SET = None
if cm.TORCH_OK:
    import torch
    torch.set_num_threads(1)
    try:  # only allowed before any inter-op parallel work has started
        torch.set_num_interop_threads(1)
        _INTEROP_SET = True
    except RuntimeError:
        _INTEROP_SET = False

# Same narrow filter as confirm_models: a blanket "ignore" here would hide a
# ConvergenceWarning raised while timing a fit.
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
REPEATS = 3  # inference is fast; take the best of a few passes to cut jitter


def _cold_start_probe() -> dict:
    """Time the very first torch fit in this process, before any warm-up.

    Reported in the results so the appendix can quote a current figure for
    framework initialisation rather than one carried over from an earlier run.
    """
    if not cm.TORCH_OK:
        return {}
    df = pd.read_parquet(cm.DATA_DIR / "granulated_sugar.parquet")
    X_tr, _X_te, y_tr, _y_te = train_test_split(
        df.text.tolist(), df.label.values, test_size=0.2,
        stratify=df.label.values, random_state=0)
    t0 = time.perf_counter()
    _m, _s, n_epochs = cm.train_torch("CNN", X_tr, y_tr, 0)
    cold = time.perf_counter() - t0
    t0 = time.perf_counter()
    _m, _s, _n = cm.train_torch("CNN", X_tr, y_tr, 0)
    warm = time.perf_counter() - t0
    print(f"  cold first fit {cold:.2f}s vs warm {warm:.2f}s ({n_epochs} epochs)")
    return {"cold_s": round(cold, 3), "warm_s": round(warm, 3),
            "n_epochs": n_epochs, "model": "CNN",
            "category": "granulated_sugar", "seed": 0}


def _warm_up() -> None:
    """Absorb one-off framework initialisation before any timing is taken.

    The first call into PyTorch in a process pays lazy initialisation that has
    nothing to do with the model, and left in the measured region it inflates
    the first CNN fit and with it the mean and spread of the CNN's training
    time. The size of the effect on the measuring host is recorded by
    _cold_start_probe (cold_start_probe_s in results/cost_benchmark_env.json). The
    same applies in miniature to the first scikit-learn fit, which pays BLAS
    set-up. Both are paid here, outside the measured region.
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
    # 1) timed fit, tracemalloc OFF
    assert not tracemalloc.is_tracing()
    vec, clf = _sklearn_spec(model_name, seed)
    t0 = time.perf_counter()
    Xtr = vec.fit_transform(X_tr)      # vocabulary from the train split only
    clf.fit(Xtr, y_tr)
    fit_s = time.perf_counter() - t0

    # 2) untimed, identical second fit, used only for peak heap allocation
    vec_m, clf_m = _sklearn_spec(model_name, seed)
    tracemalloc.start()
    clf_m.fit(vec_m.fit_transform(X_tr), y_tr)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del vec_m, clf_m

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
    """Time the SAME training function the accuracy results come from.

    This deliberately calls cm.train_torch instead of carrying its own loop,
    so the budget measured here cannot drift from the budget evaluated there.
    """
    import torch
    torch.set_num_threads(1)

    # 1) timed fit, tracemalloc OFF
    assert not tracemalloc.is_tracing()
    t0 = time.perf_counter()
    model, stoi, n_epochs = cm.train_torch(kind, X_tr, y_tr, seed)
    fit_s = time.perf_counter() - t0

    # 2) untimed, identical second fit (same seed), used only for peak memory
    tracemalloc.start()
    _m2, _s2, n_epochs_mem = cm.train_torch(kind, X_tr, y_tr, seed)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del _m2, _s2
    if n_epochs_mem != n_epochs:  # training is seeded; this should not happen
        print(f"  WARNING: memory fit of {kind} ran {n_epochs_mem} epochs "
              f"against {n_epochs} in the timed fit")

    best = None
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        cm.predict_torch(model, stoi, X_te)
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
        "n_epochs": n_epochs,
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
    df = pd.DataFrame(rows)
    if "n_epochs" in df.columns:          # torch models only; audits the budget
        cols = cols + ["n_epochs"]
    return df[cols]


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
    })
    if "n_epochs" in per_run.columns:     # realised epochs, torch models only
        out["n_epochs_median"] = g.n_epochs.median()
        out["n_epochs_max"] = g.n_epochs.max()
    return out.reset_index().sort_values("fit_s_median")


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
        env["torch_interop_threads"] = torch.get_num_interop_threads()
        env["torch_set_num_interop_threads_1_applied"] = _INTEROP_SET
    try:
        import threadpoolctl
        env["threadpoolctl"] = threadpoolctl.__version__
    except Exception:
        pass
    env["thread_env_vars"] = {v: os.environ.get(v) for v in
                              ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                               "OPENBLAS_NUM_THREADS")}
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
    with threadpool_limits(limits=1):
        pools = threadpool_info()          # state in force while measuring
        for pool in pools:                 # library file name only, no local path
            if pool.get("filepath"):
                pool["filepath"] = os.path.basename(pool["filepath"])
        cold = _cold_start_probe()
        _warm_up()
        per_run = run()
    summary = summarize(per_run)
    RESULTS_DIR.mkdir(exist_ok=True)
    per_run.to_csv(RESULTS_DIR / "cost_benchmark_per_run.csv",
                   index=False, float_format="%.6g")
    summary.to_csv(RESULTS_DIR / "cost_benchmark_summary.csv",
                   index=False, float_format="%.6g")
    env = environment()
    env["threadpool_limits"] = 1
    env["threadpool_info_during_measurement"] = pools
    env["measurement"] = {
        "tracemalloc_during_timing": False,
        "fit_timing": "wall clock (time.perf_counter) of one fit with "
                      "tracemalloc off",
        "peak_memory": "tracemalloc peak over a second, identical, untimed fit "
                       "(same spec and seed)",
        "inference": f"fastest of {REPEATS} passes over the test split, "
                     "feature extraction included",
        "warm_up": "both families exercised once on a throwaway problem "
                   "before any timing",
    }
    env["cold_start_probe_s"] = cold
    (RESULTS_DIR / "cost_benchmark_env.json").write_text(
        json.dumps(env, indent=2) + "\n")
    pd.set_option("display.width", 200)
    print()
    print(summary.round(4).to_string(index=False))
    print()
    print(json.dumps(env, indent=2))


if __name__ == "__main__":
    main()
