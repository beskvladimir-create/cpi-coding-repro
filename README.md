# cpi-coding-repro

Reproducibility package for the paper **"Accuracy and Cost in Product Coding
for Consumer Price Statistics: A Reproducible Benchmark"** (Vladimir
Beskorovainyi, Besk Tech), which develops the methodology of the preprint
arXiv:2606.02004.

> **All data in this repository are synthetically generated.** They are
> produced in code from fixed seeds. The repository contains **no proprietary
> or production data** of any kind.

It provides every number in the paper's evaluation: multi-category
precision/recall/F1 for seven models under one matched protocol, the
computational cost of each model, trie coverage, train-test overlap, pretrained
encoder baselines, robustness checks, and a Monte-Carlo validation of the
reliability-weighted consensus labeling rule.

Release 2.1.0 re-measures the cost benchmark (training is no longer timed with
tracemalloc running; all numerical libraries are held to one thread), saves
the output of every robustness check, adds the matched stopping rule as an arm
of the training-budget check, runs the throughput-by-batch check on all six
categories, and adds the pretrained-encoder baseline script with its outputs.
The accuracy results (`confirm_metrics.csv`, `significance_*.csv`) are
unchanged from release 2.0.1.

## Quick start

```bash
pip install -r requirements.txt
make all          # runs everything end-to-end (a few minutes on CPU)
```

Outputs:
- `results/*.csv`   raw metrics (coverage, confirm_metrics, learning_curve, consensus, cost, robustness)
- `results/pretrained/`  pretrained-encoder baseline outputs (see its README)
- `results/tables.md`  Markdown tables
- `figures/*.png`   per-category F1, learning curve, coverage, consensus-by-k
- `figures/fig_accuracy_cost.pdf`  accuracy against inference cost (`make figures`)

Tested on Python 3.13 (3.13.16); `requirements.txt` pins the exact package
versions of the run that produced the 2.1.0 cost and robustness results, and
notes the substitution needed on Intel macOS. CNN/LSTM use PyTorch (CPU). If
PyTorch is not installed the neural models are **skipped and reported as
skipped** -- never silently dropped.

## What each script does, and which claim it closes

| Script | Paper claim it addresses |
|---|---|
| `src/make_synth.py` | Generates receipt-like names for 6 COICOP-like categories (positives + hard negatives), the synthetic analogue of "icing sugar" / "sugar-free yogurt". Writes `data/synth/*.parquet` + `manifest.json`. |
| `src/trie_classifier.py` | Trie pre-classifier, Eq. 1: admit *x* to *c* iff a key-phrase matches and no stop-phrase matches. Produces **coverage** (admitted vs unidentified). |
| `src/confirm_models.py` | Per-category binary confirm/reject. Matched protocol (stratified 80/20, **vocab fit on train only**) across unigram/1-2/1-3-gram/char-n-gram BoW + LogReg, BoW+MLP, and **CNN/LSTM** (not run in the preprint). Plus a learning curve (F1 vs train size) probing the "~67 examples" observation. |
| `src/consensus_sim.py` | Monte-Carlo (Results, "The aggregation rule: recovering labels"): 1500 items, 12 mixed annotators, k in {3,5,7}, 60 runs. Compares majority vote, the paper's reliability-weighted rule (Eq. 2, with the additive update and cap of Eq. 3), and Dawid-Skene EM. |
| `src/run_all.py` | Runs all of the above, writes CSVs/figures, prints Markdown tables. |
| `src/make_cost_figure.py` | The accuracy-against-cost figure of Results, "Computational cost": mean F1 over the six categories from `confirm_metrics.csv` against median throughput from `cost_benchmark_summary.csv`. Writes `figures/fig_accuracy_cost.pdf` and `.png`. |
| `src/leakage_audit.py` | Train/test overlap, re-evaluation on unseen test items and under dedupe-then-split, and a trie audit (coverage against its 42.4% construction ceiling, positives lost, trie precision). |
| `src/significance.py` | Per-seed F1, s.d. and 95% t-intervals per category; paired Wilcoxon signed-rank tests over the 30 (category, seed) pairs, Holm-corrected. CNN/LSTM join when PyTorch is installed. |
| `src/cost_benchmark.py` | Computational cost of every confirmation model under the same matched protocol (Results, "Computational cost"; Appendix "Cost measurement"): training time, inference throughput with feature extraction included, serialized size, parameter count and peak heap allocation, for all 30 (category, seed) cells. Training is timed with tracemalloc off; peak memory comes from a second, identical, untimed fit. All measurement runs inside `threadpoolctl.threadpool_limits(1)` with OMP/MKL/OpenBLAS and torch (intra- and inter-op) held to one thread. Both families are warmed up once before any timing is taken, because the first call into a deep-learning framework in a process pays one-off initialisation unrelated to the model. Medians and standard deviations in `results/cost_benchmark_summary.csv`, per-cell figures in `results/cost_benchmark_per_run.csv`, and the measured environment, thread-pool state and measurement protocol in `results/cost_benchmark_env.json`. |
| `src/robustness_checks.py` | Checks behind Appendix "Robustness of the model comparison", each saved per cell and as a summary: (A) convergence of all seven models against their iteration limits (`robustness_convergence*.csv`); (B) the deep models under a fixed 8-epoch budget, a fixed 40-epoch budget, the matched stopping rule of the main results and validation-split early stopping, each defined exactly in the script's docstring (`robustness_training_budget*.csv`); (C) the ranking under average precision as well as F1, with the deep models trained under the matched rule (`robustness_average_precision*.csv`); (D) unigram and character n-gram throughput at 330, 3,300, 33,000 and 330,000 items in all six categories (`throughput_by_batch*.csv`). |
| `src/aggregation_passes.py` | Whether the Dawid-Skene advantage is an artefact of batch against online processing: the reliability-weighted rule re-run with 1, 2, 5 and 20 passes over the items (`results/aggregation_passes.csv`, per run in `aggregation_passes_per_run.csv`). Its ground truth is drawn i.i.d. rather than exactly balanced as in `consensus_sim.py`; see the script's docstring. |
| `src/grade_token_ablation.py` | Shortcut check (Appendix "Robustness of the model comparison", grade-token ablation): removes every token equal to one of a category's grade words from training and test strings and re-runs the four logistic models and the MLP under the matched protocol. Writes `results/grade_token_ablation.csv`. |
| `src/pretrained_baselines.py` | Pretrained-encoder baselines (Methods and Results, "Pretrained encoder baselines" / "Pretrained encoders"): DistilBERT fine-tuned under the matched stopping rule and for a fixed 3 epochs, and frozen MiniLM embeddings + logistic regression, under the same protocol. Needs `transformers` (optional, see `requirements.txt`) and in practice a GPU. Run from the repository it uses the checkout as is; run standalone (as on Kaggle) it clones release 2.0.1. Released outputs in `results/pretrained/`. |
| `src/analyze_pretrained.py` | Paired wins/losses/ties of the pretrained baselines against the seven models over the 30 cells; writes `results/pretrained/pretrained_pairwise.csv`. |
| `src/aggregation_to_training.py` | End-to-end link between labeling and modeling: training labels replaced by simulated crowd votes, aggregated by majority, the reliability-weighted rule and Dawid-Skene, then used to train the classifiers, scored on clean test labels. Summary in `results/aggregation_to_training_summary.csv`; per-cell figures in `results/aggregation_to_training.csv`. |

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
make audit       # leakage audit, trie audit, significance tests, aggregation-to-training
make cost        # computational cost of every model (timings; run on an idle machine)
make robust      # robustness checks: convergence, training budget, decision threshold,
                 # batch size, aggregation passes (per-cell and summary CSVs)
make figures     # accuracy-cost figure from the committed results
make pretrained  # pretrained-encoder baselines (needs transformers; GPU advisable)
make pretrained-analysis  # paired comparisons for the pretrained baselines
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
