# cpi-coding-repro

Reproducibility package for the methodology of **arXiv:2606.02004**
("Machine Learning for Coding Retail Product Names to Consumer-Price
Categories").

> **All data in this repository are synthetically generated.** They are
> produced in code from fixed seeds. This repository reproduces the
> *methodology* of arXiv:2606.02004 and contains **no proprietary or
> production data** of any kind.

It empirically strengthens the claims of the preprint's "Strengthening the
evidence" section: multi-category precision/recall/F1, trie coverage, a matched
bag-of-words vs CNN/LSTM comparison under one protocol, and a Monte-Carlo
validation of the weighted-consensus labeling rule.

## Quick start

```bash
pip install -r requirements.txt
make all          # runs everything end-to-end (a few minutes on CPU)
```

Outputs:
- `results/*.csv`   raw metrics (coverage, confirm_metrics, learning_curve, consensus)
- `results/tables.md`  Markdown tables ready to paste into the paper
- `figures/*.png`   per-category F1, learning curve, coverage, consensus-by-k

Python 3.11+ (developed/tested on 3.12). CNN/LSTM use PyTorch (CPU). If PyTorch
is not installed the neural models are **skipped and reported as skipped** --
never silently dropped.

## What each script does, and which claim it closes

| Script | Paper claim it addresses |
|---|---|
| `src/make_synth.py` | Generates receipt-like names for 6 COICOP-like categories (positives + hard negatives), the synthetic analogue of "icing sugar" / "sugar-free yogurt". Writes `data/synth/*.parquet` + `manifest.json`. |
| `src/trie_classifier.py` | Trie pre-classifier, Eq. (1): admit *x* to *c* iff a key-phrase matches and no stop-phrase matches. Produces **coverage** (admitted vs unidentified). |
| `src/confirm_models.py` | Per-category binary confirm/reject. Matched protocol (stratified 80/20, **vocab fit on train only**) across unigram/1-2/1-3-gram/char-n-gram BoW + LogReg, BoW+MLP, and **CNN/LSTM** (not run in the preprint). Plus a learning curve (F1 vs train size) probing the "~67 examples" observation. |
| `src/consensus_sim.py` | Monte-Carlo (Section 9): 1500 items, 12 mixed annotators, k in {3,5,7}, 60 runs. Compares majority vote, the paper's reliability-weighted rule (Eq. 2, additive update with cap), and Dawid-Skene EM. |
| `src/run_all.py` | Runs all of the above, writes CSVs/figures, prints Markdown tables. |

## Honesty requirements (also enforced in code comments)

- No production figures are mixed in. Every number is the output of these
  scripts on synthetic data.
- Synthetic results are reported **as is**. If a bag-of-words model does not
  saturate some category, that stands -- the generator is not tuned to hit a
  target F1.
- All seeds (`[0, 1, 2, 3, 4]` for splits; per-run seeds for the simulation)
  are fixed; runs are fully deterministic.

## Reproducing / extending

```bash
make data        # just regenerate the synthetic corpus
make trie        # coverage report
make confirm     # confirmation models
make consensus   # consensus simulation
make clean       # remove generated data/results/figures
```

## External validation (planned, not included)

All results here are on synthetic data. The single most valuable extension is
to run the **same matched protocol** on a *public, real* product-name corpus
(e.g. open scanner or web-scraped catalogues coded to COICOP). The code is
structured so this is a drop-in: provide a dataframe with `text` and `label`
columns per category and reuse `src/confirm_models.py` unchanged. This is
flagged in the paper as the priority next step; real product text is harder
than any synthetic corpus (cf. ONS's ~0.79 macro-precision on real
web-scraped clothing), so the high synthetic F1 reflects the benchmark, not a
solved problem.
