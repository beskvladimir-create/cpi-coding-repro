"""Pretrained-encoder baselines for the CPI product-coding benchmark.

Adds the comparison every reviewer of arXiv:2606.02004 asked for: models that
start from pretrained weights, run under the SAME matched protocol as the
seven models already in the paper (cpi-coding-repro, release 2.0.1):

  * same six synthetic categories, read from the committed parquet files;
  * same stratified 80/20 split, seeds 0..4 (sklearn train_test_split);
  * same stopping rule as the CNN/LSTM: stop when the epoch training loss
    fails to improve by more than 1e-4 for 10 consecutive epochs, max 200;
  * same metrics (accuracy, precision, recall, F1), plus F1 on test items
    whose exact string does not occur in training (as in leakage_audit.py).

Models:
  1. DistilBERT fine-tuned end to end (distilbert-base-uncased).
  2. DistilBERT fine-tuned for a fixed 3 epochs (the common recipe), reported
     separately so the stopping rule cannot be blamed for the result.
  3. Frozen MiniLM sentence embeddings (all-MiniLM-L6-v2, mean pooling)
     + logistic regression, the cheap way to use pretrained knowledge.

Cost: training time is measured on whatever device runs the script (GPU on
Kaggle) and is NOT comparable to the CPU timings in the paper; it is reported
with the device name. Inference throughput is measured on CPU with one thread,
tokenization included, after a warm-up, which IS comparable to the paper's
cost table.

Run inside the repository (needs `pip install transformers`, see
requirements.txt):  python src/pretrained_baselines.py
  The repository is detected from this file's location and used as is: no
  clone and no checkout. Outputs go to results/pretrained/rerun/ (override
  with OUT=...), so the Kaggle outputs committed in results/pretrained/ are
  not overwritten.

Run standalone (e.g. on Kaggle: Settings -> Internet ON, Accelerator GPU,
T4 or P100): copy this single file into the notebook. When it is not inside
the repository it clones cpi-coding-repro and checks out REPO_COMMIT, and
writes to /kaggle/working/cpi_pretrained/ (or ./cpi_pretrained/). The paper's
pretrained results were produced this way, at commit 28c733a (release 2.0.1);
the data files it reads are unchanged in later releases.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

REPO_URL = "https://github.com/beskvladimir-create/cpi-coding-repro.git"
REPO_COMMIT = "28c733a"  # release 2.0.1, the snapshot cited in the paper
_HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
_IN_REPO = (_HERE.parent / "data" / "synth").is_dir() and (_HERE / "make_synth.py").is_file()
if _IN_REPO:
    REPO = _HERE.parent
    OUT = Path(os.environ.get("OUT", REPO / "results" / "pretrained" / "rerun"))
else:
    WORK = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd()
    REPO = WORK / "cpi-coding-repro"
    OUT = Path(os.environ.get("OUT", WORK / "cpi_pretrained"))

FT_MODEL = os.environ.get("FT_MODEL", "distilbert-base-uncased")
EMB_MODEL = os.environ.get("EMB_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
SEEDS = [0, 1, 2, 3, 4]
MAX_LEN = 32
BATCH = 32
LR = 2e-5
MAX_EPOCHS = int(os.environ.get("MAX_EPOCHS", "200"))
TOL = 1e-4
N_ITER_NO_CHANGE = 10
FIXED_EPOCHS = 3
QUICK = os.environ.get("QUICK") == "1"  # smoke test: 1 category, 1 seed


def sh(cmd: str) -> None:
    print("$", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


def setup() -> None:
    if not _IN_REPO:  # standalone run: fetch the pinned snapshot
        if not REPO.exists():
            sh(f"git clone -q {REPO_URL} {REPO}")
        sh(f"cd {REPO} && git checkout -q {REPO_COMMIT}")
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        sh(f"{sys.executable} -m pip install -q pyarrow")
        import pyarrow  # noqa: F401  (fail loudly if the install did not work)


setup()
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (accuracy_score, f1_score, precision_score,  # noqa: E402
                             recall_score)
from sklearn.model_selection import train_test_split  # noqa: E402
from transformers import (AutoModel, AutoModelForSequenceClassification,  # noqa: E402
                          AutoTokenizer)
from transformers import logging as hf_logging  # noqa: E402

from make_synth import _categories  # noqa: E402

hf_logging.set_verbosity_error()
DATA_DIR = REPO / "data" / "synth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT.mkdir(parents=True, exist_ok=True)


def metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def seed_all(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---- fine-tuned encoder -----------------------------------------------------

def finetune(X_tr, y_tr, seed: int, fixed_epochs: int | None):
    seed_all(seed)
    tok = AutoTokenizer.from_pretrained(FT_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(FT_MODEL, num_labels=2).to(DEVICE)
    enc = tok(list(X_tr), padding="max_length", truncation=True,
              max_length=MAX_LEN, return_tensors="pt")
    ids, att = enc["input_ids"], enc["attention_mask"]
    y = torch.tensor(np.asarray(y_tr), dtype=torch.long)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    gen = torch.Generator().manual_seed(seed)
    best, no_imp, n_ep = np.inf, 0, 0
    t0 = time.time()
    model.train()
    limit = fixed_epochs if fixed_epochs else MAX_EPOCHS
    for _ in range(limit):
        perm = torch.randperm(len(y), generator=gen)
        total, nb = 0.0, 0
        for i in range(0, len(y), BATCH):
            idx = perm[i:i + BATCH]
            out = model(input_ids=ids[idx].to(DEVICE), attention_mask=att[idx].to(DEVICE),
                        labels=y[idx].to(DEVICE))
            opt.zero_grad()
            out.loss.backward()
            opt.step()
            total += float(out.loss.detach())
            nb += 1
        n_ep += 1
        ep_loss = total / nb
        if fixed_epochs is None:
            no_imp = no_imp + 1 if ep_loss > best - TOL else 0
            best = min(best, ep_loss)
            if no_imp >= N_ITER_NO_CHANGE:
                break
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    train_s = time.time() - t0
    return model, tok, n_ep, train_s


@torch.no_grad()
def predict_ft(model, tok, X, device) -> np.ndarray:
    model.eval()
    preds = []
    for i in range(0, len(X), 256):
        enc = tok(list(X[i:i + 256]), padding=True, truncation=True,
                  max_length=MAX_LEN, return_tensors="pt").to(device)
        preds.append(model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits.argmax(-1).cpu().numpy())
    return np.concatenate(preds)


# ---- frozen embeddings + logistic regression -------------------------------

_EMB = {}


def emb_model():
    if "m" not in _EMB:
        _EMB["tok"] = AutoTokenizer.from_pretrained(EMB_MODEL)
        _EMB["m"] = AutoModel.from_pretrained(EMB_MODEL).to(DEVICE).eval()
    return _EMB["tok"], _EMB["m"]


@torch.no_grad()
def embed(X, tok, m, device) -> np.ndarray:
    outs = []
    for i in range(0, len(X), 256):
        enc = tok(list(X[i:i + 256]), padding=True, truncation=True,
                  max_length=MAX_LEN, return_tensors="pt").to(device)
        h = m(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).float()
        v = (h * mask).sum(1) / mask.sum(1)
        outs.append(torch.nn.functional.normalize(v, dim=-1).cpu().numpy())
    return np.concatenate(outs)


# ---- CPU single-thread inference throughput (comparable to the paper) -------

def cpu_throughput(predict_fn, X, repeats: int = 3) -> float:
    torch.set_num_threads(1)
    predict_fn(X[:32])  # warm-up
    best = np.inf
    for _ in range(repeats):
        t0 = time.perf_counter()
        predict_fn(X)
        best = min(best, time.perf_counter() - t0)
    torch.set_num_threads(os.cpu_count() or 1)
    return len(X) / best


def n_params(m) -> int:
    return sum(p.numel() for p in m.parameters())


def size_kib(obj) -> float:
    import io
    import pickle
    buf = io.BytesIO()
    if isinstance(obj, torch.nn.Module):
        torch.save(obj.state_dict(), buf)
    else:
        pickle.dump(obj, buf, protocol=pickle.HIGHEST_PROTOCOL)
    return buf.tell() / 1024


def reference_cpu_cost(X_tr, y_tr, X_te, seed: int) -> list:
    """Re-time two models from the paper on THIS machine, single-threaded, with
    the repo's own vectorizer spec, so pretrained throughput can be read as a
    ratio against them (absolute rates differ between hosts)."""
    import confirm_models as cm
    from threadpoolctl import threadpool_limits
    out = []
    with threadpool_limits(limits=1):
        for name, kind in [("Unigram BoW + LogReg", "bow"),
                           ("Char n-gram(3-5) + LogReg", "char3-5")]:
            vec = cm._vectorizer(kind)
            clf = LogisticRegression(max_iter=1000, random_state=seed)
            t0 = time.perf_counter()
            clf.fit(vec.fit_transform(X_tr), y_tr)
            fit_s = time.perf_counter() - t0
            best = np.inf
            for _ in range(3):
                t0 = time.perf_counter()
                clf.predict(vec.transform(X_te))
                best = min(best, time.perf_counter() - t0)
            out.append({"model": name, "train_s_device": fit_s, "device": "cpu1",
                        "cpu1_items_per_s": len(X_te) / best,
                        "n_params": int(np.prod(clf.coef_.shape)) + 1,
                        "size_kib": size_kib((vec, clf))})
    return out


def main() -> None:
    cats = _categories()[:1] if QUICK else _categories()
    seeds = SEEDS[:1] if QUICK else SEEDS
    rows, cost_rows = [], []
    for cfg in cats:
        df = pd.read_parquet(DATA_DIR / f"{cfg.name}.parquet")
        texts, labels = df.text.tolist(), df.label.values
        for seed in seeds:
            X_tr, X_te, y_tr, y_te = train_test_split(
                texts, labels, test_size=0.2, stratify=labels, random_state=seed)
            y_te = np.asarray(y_te)
            unseen = ~np.array([t in set(X_tr) for t in X_te])

            def record(name, pred, extra):
                m = metrics(y_te, pred)
                m["f1_unseen"] = f1_score(y_te[unseen], pred[unseen], zero_division=0)
                m.update({"category": cfg.name, "seed": seed, "model": name, **extra})
                rows.append(m)
                print(f"{cfg.name:18s} seed={seed} {name:40s} F1={m['f1']:.4f} "
                      f"unseen={m['f1_unseen']:.4f} {extra}", flush=True)

            for fixed, name in [(None, "DistilBERT fine-tuned (matched stop)"),
                                (FIXED_EPOCHS, f"DistilBERT fine-tuned ({FIXED_EPOCHS} epochs)")]:
                model, tok, n_ep, train_s = finetune(X_tr, y_tr, seed, fixed)
                pred = predict_ft(model, tok, X_te, DEVICE)
                record(name, pred, {"n_epochs": n_ep, "train_s": train_s})
                if seed == seeds[0]:
                    cpu_model = model.to("cpu")
                    thr = cpu_throughput(lambda X: predict_ft(cpu_model, tok, X, torch.device("cpu")), X_te)
                    cost_rows.append({"category": cfg.name, "model": name,
                                      "train_s_device": train_s, "device": str(DEVICE),
                                      "cpu1_items_per_s": thr, "n_params": n_params(model),
                                      "size_kib": size_kib(model)})
                del model
                if DEVICE.type == "cuda":
                    torch.cuda.empty_cache()

            tok, m = emb_model()
            t0 = time.time()
            E_tr = embed(X_tr, tok, m, DEVICE)
            clf = LogisticRegression(max_iter=1000, random_state=seed).fit(E_tr, y_tr)
            train_s = time.time() - t0
            pred = clf.predict(embed(X_te, tok, m, DEVICE))
            name = "MiniLM embeddings + LogReg"
            record(name, pred, {"n_epochs": np.nan, "train_s": train_s})
            if seed == seeds[0]:
                m_cpu = m.to("cpu")
                thr = cpu_throughput(lambda X: clf.predict(embed(X, tok, m_cpu, torch.device("cpu"))), X_te)
                cost_rows.append({"category": cfg.name, "model": name,
                                  "train_s_device": train_s, "device": str(DEVICE),
                                  "cpu1_items_per_s": thr, "n_params": n_params(m),
                                  "size_kib": size_kib(m) + size_kib(clf)})
                for r in reference_cpu_cost(X_tr, y_tr, X_te, seed):
                    cost_rows.append({"category": cfg.name, **r})
                _EMB["m"] = m_cpu.to(DEVICE)

    per_run = pd.DataFrame(rows)
    per_run.to_csv(OUT / "pretrained_per_run.csv", index=False)
    summ = (per_run.groupby(["category", "model"])
            .agg(f1_mean=("f1", "mean"), f1_sd=("f1", "std"),
                 f1_unseen_mean=("f1_unseen", "mean"),
                 precision_mean=("precision", "mean"), recall_mean=("recall", "mean"),
                 n_epochs_mean=("n_epochs", "mean"), train_s_mean=("train_s", "mean"))
            .reset_index())
    summ.to_csv(OUT / "pretrained_summary.csv", index=False)
    pd.DataFrame(cost_rows).to_csv(OUT / "pretrained_cost.csv", index=False)

    def rev(name):
        try:
            return AutoModel.from_pretrained(name).config._commit_hash
        except Exception as e:  # pragma: no cover
            return f"unknown ({e})"
    def repo_commit():
        if not _IN_REPO:
            return REPO_COMMIT
        try:
            return subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                                  capture_output=True, text=True, check=True).stdout.strip()
        except Exception:
            return "unknown (not a git checkout)"
    env = {
        "repo_commit": repo_commit(), "run_from_repo": _IN_REPO, "ft_model": FT_MODEL, "ft_model_revision": rev(FT_MODEL),
        "emb_model": EMB_MODEL, "emb_model_revision": rev(EMB_MODEL),
        "device": str(DEVICE),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch": torch.__version__, "transformers": __import__("transformers").__version__,
        "python": platform.python_version(), "cpu": platform.processor() or platform.machine(),
        "hyperparams": {"max_len": MAX_LEN, "batch": BATCH, "lr": LR, "max_epochs": MAX_EPOCHS,
                        "tol": TOL, "n_iter_no_change": N_ITER_NO_CHANGE,
                        "fixed_epochs": FIXED_EPOCHS},
        "quick": QUICK,
    }
    (OUT / "pretrained_env.json").write_text(json.dumps(env, indent=2))
    print("\n=== summary ===")
    print(summ.groupby("model")[["f1_mean", "f1_unseen_mean", "n_epochs_mean"]].mean().to_string())
    print("\nFiles written to", OUT)


if __name__ == "__main__":
    main()
