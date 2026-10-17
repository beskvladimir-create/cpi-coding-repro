"""Final reliability weights in the consensus simulation, by annotator type and k.

The paper explains why the reliability-weighted rule (Eqs. 2-3) barely improves
on majority vote by the saturation of the additive update at the cap. This
script records the weights the rule actually ends with, so that explanation
can be checked rather than asserted.

It replays consensus_sim.run() exactly: the same seeds (1000*k + run), the
same balanced ground truth, the same annotator accuracies and votes, and the
same order of random draws (majority vote's tie-breaks are drawn before the
weighted rule's, as in consensus_sim). The weighted rule is re-implemented
only to return its final weight vector; its label-recovery accuracy is
asserted equal to that of consensus_sim._reliability_weighted on every run,
and its mean per k is checked against results/consensus.csv.

Annotator type is read off the accuracy range it was drawn from: expert
(0.85-0.97), mediocre (0.58-0.72) or adversarial (0.35-0.50).

Writes
  results/consensus_final_weights.csv          summary per k and type
  results/consensus_final_weights_per_annotator.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import consensus_sim as cs

RESULTS = Path(__file__).resolve().parent.parent / "results"


def weighted_with_weights(votes, voters, rng, n_annotators=cs.N_ANNOTATORS):
    """consensus_sim._reliability_weighted, also returning the final weights."""
    r = np.full(n_annotators, cs.R_INIT)
    out = []
    for i in range(len(votes)):
        vs = votes[i]
        O = float((r[voters[i]] * vs).sum())
        yhat = 1 if O > 0 else 0 if O < 0 else rng.randint(2)
        out.append(yhat)
        yvote = 1 if yhat == 1 else -1
        for j, a in enumerate(voters[i]):
            if vs[j] == yvote:
                r[a] = min(cs.R_CAP, r[a] + cs.R_DELTA)
            else:
                r[a] = max(0.0, r[a] - cs.R_DELTA)
    return np.array(out), r


def annotator_type(acc: float) -> str:
    if acc >= 0.85:
        return "expert"
    if acc >= 0.58:
        return "mediocre"
    return "adversarial"


def main() -> None:
    released = pd.read_csv(RESULTS / "consensus.csv").set_index("k")
    rows, acc_check = [], []
    for k in cs.K_VALUES:
        accs = []
        for run_idx in range(cs.N_RUNS):
            rng = np.random.RandomState(1000 * k + run_idx)
            truth = np.array([0, 1] * (cs.N_ITEMS // 2))
            rng.shuffle(truth)
            acc = cs._sample_accuracies(rng)
            votes, voters = cs._generate_votes(rng, truth, acc, k)
            cs._majority(votes, voters, rng)          # consumes tie-break draws
            state = rng.get_state()
            pred, r = weighted_with_weights(votes, voters, rng)
            rng2 = np.random.RandomState()
            rng2.set_state(state)
            ref = cs._reliability_weighted(votes, voters, rng2)
            assert (ref == pred).all()
            accs.append((pred == truth).mean())
            for a in range(cs.N_ANNOTATORS):
                rows.append(dict(k=k, run=run_idx, annotator=a, accuracy=acc[a],
                                 type=annotator_type(acc[a]), final_weight=r[a]))
        m = float(np.mean(accs))
        assert abs(m - released.loc[k, "Reliability-weighted_mean"]) < 1e-9, (k, m)
        print(f"k={k}: weighted-rule accuracy {m:.6f} matches results/consensus.csv")
    d = pd.DataFrame(rows)
    d.to_csv(RESULTS / "consensus_final_weights_per_annotator.csv", index=False)
    s = d.groupby(["k", "type"]).final_weight.agg(
        n="size", median="median", q25=lambda x: x.quantile(0.25),
        q75=lambda x: x.quantile(0.75), mean="mean",
        share_at_cap=lambda x: (x >= cs.R_CAP - 1e-9).mean(),
        share_at_zero=lambda x: (x <= 1e-9).mean()).reset_index()
    s.to_csv(RESULTS / "consensus_final_weights.csv", index=False)
    print(s.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
