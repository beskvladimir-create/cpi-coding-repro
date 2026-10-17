# Pretrained-encoder baselines

Outputs of `src/pretrained_baselines.py`, run as the Kaggle notebook
`cpi_pretrained_baselines_gpu` on one NVIDIA Tesla T4 (Python 3.13.15, torch
2.11.0+cu128, transformers 5.16.1) against release 2.0.1 of this package
(commit 28c733a; its data are identical to those of release 2.1.0), with
`distilbert-base-uncased` at revision 12040accade4e8a0f71eabdb258fecc2e7e948be
and `sentence-transformers/all-MiniLM-L6-v2` at revision
1110a243fdf4706b3f48f1d95db1a4f5529b4d41.

`pretrained_cost.csv` is the file written by the script, retrieved from the
notebook's output. The other run outputs below were **transcribed by hand** from what the Kaggle notebook
displayed (its run log and its output preview), because the raw output files
were not retrieved. Transcribed values carry the precision at which they were
displayed. GPU training is not bit-deterministic, so a rerun reproduces these
values closely but not exactly.

| File | Provenance | Contents |
|---|---|---|
| `pretrained_cost.csv` | Written by the script, retrieved from the notebook output. | Single-core CPU inference throughput (PyTorch, or the numerical libraries for the logistic models, limited to one thread; the tokenizer's own thread pool was not limited), serialized size, parameter count and training time on the device (GPU for the pretrained models), seed 0 in each category; the unigram and character n-gram logistic models re-timed in the same session for ratios. |
| `run_log_excerpt.txt` | **Transcribed** from the run log. | The per-run lines the script printed: category, seed, configuration (`M` = DistilBERT fine-tuned under the matched stopping rule, `F` = DistilBERT fine-tuned for a fixed 3 epochs, `E` = frozen MiniLM embeddings + logistic regression), F1, F1 on unseen test items, realised epochs (`-` for MiniLM) and training seconds, at the 4 decimal places printed. 90 lines: 6 categories x 5 seeds x 3 configurations. |
| `pretrained_per_run_from_log.csv` | **Transcribed** (the same lines as `run_log_excerpt.txt`, as CSV). | F1 and unseen-item F1 per category, seed and model, with realised epochs and training seconds; precision and recall are blank because the log did not print them. |
| `pretrained_summary.csv` | **Transcribed** from the Kaggle output preview of the script's `pretrained_summary.csv`. | Per category and model: mean and s.d. of F1 over the five seeds, mean unseen-item F1, mean precision and recall, mean realised epochs and mean training seconds. The only source for precision and recall. Its F1 means agree with the per-run file to within the 4-decimal rounding of the latter (largest difference 2.1e-5). |
| `pretrained_env.json` | **Transcribed** from the `pretrained_env.json` written by the run. | Repository commit, model names and revisions, device, library versions and hyperparameters. |
| `pretrained_pairwise.csv` | Output of `src/analyze_pretrained.py` in this repository. | Wins, losses and ties over the 30 category-seed cells against the seven models of the paper (`results/significance_per_seed.csv`) and between the pretrained configurations, mean difference and raw Wilcoxon p. Computed from the 4-d.p. per-run file, so comparisons are made after rounding to four places. `src/pairwise_bands.py` adds the descriptive bands and Holm-adjusted p-values (`results/pairwise_bands.csv`). |

The raw per-run CSV written by the script (`pretrained_per_run.csv`) was not
retrieved. If it is added here, `src/analyze_pretrained.py` reads it in
preference to the transcribed file.

Re-running `python src/pretrained_baselines.py` from the repository writes to
`results/pretrained/rerun/` so these files are not overwritten. It requires
`transformers` (see `requirements.txt`) and, in practice, a GPU.
