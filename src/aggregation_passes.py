"""Is 'Dawid-Skene beats the reliability weight' partly 'batch beats online'?

The released comparison runs the reliability-weighted rule in ONE online pass
over the items, updating weights as it goes, while Dawid-Skene EM sees every
item at once and iterates 30 times. That is an asymmetry in information, not
only in estimator. This re-runs the weighted rule with extra passes (weights
carried over, labels re-decided with the converged weights) to separate the
two explanations.

Outputs: results/aggregation_passes.csv (mean accuracy per k for majority,
the weighted rule with 1/2/5/20 passes and Dawid-Skene, plus the gaps in
points) and results/aggregation_passes_per_run.csv (one row per k and run).

How this simulation differs from consensus_sim.py (results/consensus.csv):
it reuses that module's annotator population, vote generator and the three
aggregators, with the same 60 runs per k and the same per-(k, run) seed
RandomState(1000*k + run), but
  * ground truth is drawn i.i.d. as rng.randint(0, 2, N_ITEMS), so the class
    balance varies from run to run, whereas consensus_sim.py uses exactly 750
    positives and 750 negatives, shuffled; because the truth draw consumes the
    random stream differently, annotator accuracies and votes also differ
    from consensus_sim.py for the same (k, run);
  * the reliability-weighted rule breaks ties with its own stream,
    RandomState(7*run + k), rather than sharing the run's stream.
The figures here are therefore close to, but not identical with, those in
consensus.csv; the comparison that matters (passes against Dawid-Skene) is
made within this script on identical votes.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import consensus_sim as cs

RESULTS = Path(__file__).resolve().parent.parent / "results"

N_RUNS = 60


def weighted_multipass(votes, voters, rng, passes):
    r = np.full(cs.N_ANNOTATORS, cs.R_INIT)
    out = None
    for p in range(passes):
        out = []
        for i in range(len(votes)):
            vs, ws = votes[i], r[voters[i]]
            O = float((ws * vs).sum())
            yhat = 1 if O > 0 else (0 if O < 0 else rng.randint(2))
            out.append(yhat)
            yvote = 1 if yhat == 1 else -1
            for j, a in enumerate(voters[i]):
                if vs[j] == yvote:
                    r[a] = min(cs.R_CAP, r[a] + cs.R_DELTA)
                else:
                    r[a] = max(0.0, r[a] - cs.R_DELTA)
    return np.array(out), r


def main() -> None:
    rows = []
    for k in cs.K_VALUES:
        for run in range(N_RUNS):
            rng = np.random.RandomState(1000 * k + run)
            truth = rng.randint(0, 2, cs.N_ITEMS)
            acc = cs._sample_accuracies(rng)
            votes, voters = cs._generate_votes(rng, truth, acc, k)
            maj = cs._majority(votes, voters, rng)
            ds = cs._dawid_skene(votes, voters, rng)
            rec = {"k": k, "run": run, "majority": (maj == truth).mean(),
                   "dawid_skene": (ds == truth).mean()}
            for passes in (1, 2, 5, 20):
                pred, _r = weighted_multipass(votes, voters,
                                              np.random.RandomState(7 * run + k), passes)
                rec[f"weighted_{passes}pass"] = (pred == truth).mean()
            rows.append(rec)

    d = pd.DataFrame(rows)
    cols = ["majority", "weighted_1pass", "weighted_2pass", "weighted_5pass",
            "weighted_20pass", "dawid_skene"]
    g = d.groupby("k")[cols].mean()
    print(g.round(4).to_string())
    print()
    for p in (1, 2, 5, 20):
        g[f"gap_ds_minus_weighted_{p}pass_pts"] = (
            g["dawid_skene"] - g[f"weighted_{p}pass"]) * 100
    g["max_abs_change_vs_1pass_pts"] = (
        g[["weighted_2pass", "weighted_5pass", "weighted_20pass"]]
        .sub(g["weighted_1pass"], axis=0).abs().max(axis=1) * 100)
    for k in cs.K_VALUES:
        print(f"k={k}: DS over weighted  1 pass "
              f"{g.loc[k, 'gap_ds_minus_weighted_1pass_pts']:+.1f} pts   20 passes "
              f"{g.loc[k, 'gap_ds_minus_weighted_20pass_pts']:+.1f} pts")
    RESULTS.mkdir(exist_ok=True)
    g.reset_index().to_csv(RESULTS / "aggregation_passes.csv", index=False,
                           float_format="%.6g")
    d.to_csv(RESULTS / "aggregation_passes_per_run.csv", index=False,
             float_format="%.6g")
    print("  wrote results/aggregation_passes.csv, results/aggregation_passes_per_run.csv")


if __name__ == "__main__":
    main()
