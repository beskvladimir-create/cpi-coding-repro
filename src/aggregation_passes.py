"""Is 'Dawid-Skene beats the reliability weight' partly 'batch beats online'?

The released comparison runs the reliability-weighted rule in ONE online pass
over the items, updating weights as it goes, while Dawid-Skene EM sees every
item at once and iterates 30 times. That is an asymmetry in information, not
only in estimator. This re-runs the weighted rule with extra passes (weights
carried over, labels re-decided with the converged weights) to separate the
two explanations.
"""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import consensus_sim as cs

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
            rec = {"k": k, "majority": (maj == truth).mean(),
                   "dawid_skene": (ds == truth).mean()}
            for passes in (1, 2, 5, 20):
                pred, _r = weighted_multipass(votes, voters,
                                              np.random.RandomState(7 * run + k), passes)
                rec[f"weighted_{passes}pass"] = (pred == truth).mean()
            rows.append(rec)

    d = pd.DataFrame(rows)
    g = d.groupby("k").mean().drop(columns=[]) if False else d.groupby("k").mean()
    cols = ["majority", "weighted_1pass", "weighted_2pass", "weighted_5pass",
            "weighted_20pass", "dawid_skene"]
    print(g[cols].round(4).to_string())
    print()
    for k in cs.K_VALUES:
        gap1 = (g.loc[k, "dawid_skene"] - g.loc[k, "weighted_1pass"]) * 100
        gap20 = (g.loc[k, "dawid_skene"] - g.loc[k, "weighted_20pass"]) * 100
        print(f"k={k}: DS over weighted  1 pass {gap1:+.1f} pts   20 passes {gap20:+.1f} pts")


if __name__ == "__main__":
    main()
