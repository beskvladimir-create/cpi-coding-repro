"""Sensitivity of the aggregation comparison to the simulated annotator population.

consensus_sim.py compares majority vote, the reliability-weighted rule and
Dawid-Skene on one annotator mixture (40% expert, 40% mediocre, 20%
adversarial) with symmetric errors. Two departures are tested here, with
everything else (1500 balanced items, 12 annotators, k in {3, 5, 7}, 60 runs,
seeds 1000*k + run, the three aggregators of consensus_sim) unchanged:

  no_adversarial   the two adversarial annotators are replaced by mediocre
                   ones: 5 expert (0.85-0.97) and 7 mediocre (0.58-0.72);
  class_dependent  the released mixture, but errors depend on the true class:
                   an annotator of accuracy a reports a true positive
                   correctly with probability a - 0.10 and a true negative
                   with probability a + 0.10 (clipped to [0.01, 0.99]), so
                   positives are missed more often than false positives are
                   raised while accuracy on the balanced item set stays a.

The baseline row reproduces results/consensus.csv (asserted).

Writes results/consensus_sensitivity.csv: mean and s.d. of label-recovery
accuracy per scenario, k and rule, and the gaps in points of Dawid-Skene over
the other two rules; and results/consensus_sensitivity_weights.csv: median
final weight of the weighted rule per scenario, k and annotator type.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import consensus_sim as cs
from consensus_weights import annotator_type, weighted_with_weights

RESULTS = Path(__file__).resolve().parent.parent / "results"
SHIFT = 0.10


def acc_no_adversarial(rng):
    acc = np.concatenate([rng.uniform(0.85, 0.97, 5), rng.uniform(0.58, 0.72, 7)])
    rng.shuffle(acc)
    return acc


def votes_class_dependent(rng, truth, acc, k):
    """consensus_sim._generate_votes with class-dependent correctness."""
    n = len(truth)
    voters = np.zeros((n, k), dtype=int)
    votes = np.zeros((n, k), dtype=int)
    sens = np.clip(acc - SHIFT, 0.01, 0.99)
    spec = np.clip(acc + SHIFT, 0.01, 0.99)
    for i in range(n):
        chosen = rng.choice(cs.N_ANNOTATORS, size=k, replace=False)
        voters[i] = chosen
        for j, a in enumerate(chosen):
            p = sens[a] if truth[i] == 1 else spec[a]
            correct = rng.rand() < p
            label = truth[i] if correct else 1 - truth[i]
            votes[i, j] = 1 if label == 1 else -1
    return votes, voters


SCENARIOS = {
    "baseline": (cs._sample_accuracies, cs._generate_votes),
    "no_adversarial": (acc_no_adversarial, cs._generate_votes),
    "class_dependent": (cs._sample_accuracies, votes_class_dependent),
}


def main() -> None:
    released = pd.read_csv(RESULTS / "consensus.csv").set_index("k")
    rows, wrows = [], []
    for scen, (sample_acc, gen_votes) in SCENARIOS.items():
        for k in cs.K_VALUES:
            res = {"Majority": [], "Reliability-weighted": [], "Dawid-Skene": []}
            for run_idx in range(cs.N_RUNS):
                rng = np.random.RandomState(1000 * k + run_idx)
                truth = np.array([0, 1] * (cs.N_ITEMS // 2))
                rng.shuffle(truth)
                acc = sample_acc(rng)
                votes, voters = gen_votes(rng, truth, acc, k)
                res["Majority"].append((cs._majority(votes, voters, rng) == truth).mean())
                pred, r = weighted_with_weights(votes, voters, rng)
                res["Reliability-weighted"].append((pred == truth).mean())
                res["Dawid-Skene"].append(
                    (cs._dawid_skene(votes, voters, rng) == truth).mean())
                for a in range(cs.N_ANNOTATORS):
                    wrows.append(dict(scenario=scen, k=k, type=annotator_type(acc[a]),
                                      final_weight=r[a]))
            row = {"scenario": scen, "k": k}
            for name, v in res.items():
                row[f"{name}_mean"] = float(np.mean(v))
                row[f"{name}_sd"] = float(np.std(v, ddof=0))
            row["ds_minus_majority_pts"] = 100 * (row["Dawid-Skene_mean"] - row["Majority_mean"])
            row["ds_minus_weighted_pts"] = 100 * (row["Dawid-Skene_mean"]
                                                  - row["Reliability-weighted_mean"])
            row["weighted_minus_majority_pts"] = 100 * (row["Reliability-weighted_mean"]
                                                        - row["Majority_mean"])
            rows.append(row)
            if scen == "baseline":
                for name in res:
                    assert abs(row[f"{name}_mean"] - released.loc[k, f"{name}_mean"]) < 1e-9
            print(f"  {scen:16s} k={k}  maj={row['Majority_mean']:.4f} "
                  f"rel={row['Reliability-weighted_mean']:.4f} "
                  f"ds={row['Dawid-Skene_mean']:.4f}", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "consensus_sensitivity.csv", index=False)
    w = (pd.DataFrame(wrows).groupby(["scenario", "k", "type"]).final_weight
         .median().rename("median_final_weight").reset_index())
    w.to_csv(RESULTS / "consensus_sensitivity_weights.csv", index=False)
    pd.set_option("display.width", 220)
    print(out.round(4).to_string(index=False))
    print(w.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
