"""Consensus-rule Monte-Carlo (Section 9 'Validating the consensus rule').

Reproduces the simulation that validates the human-in-the-loop labeling
protocol of arXiv:2606.02004, independent of any real data:

  * 1500 items, balanced binary ground truth;
  * 12 annotators from a mixed population --
        40% expert       (accuracy 0.85-0.97),
        40% mediocre     (accuracy 0.58-0.72),
        20% adversarial  (accuracy 0.35-0.50);
  * each item receives k in {3, 5, 7} independent votes;
  * 60 runs.

Three aggregators recover the true label:
  * majority vote;
  * reliability-weighted vote (Eq. 2): O = sum_a r_a v_a, with an additive
    online weight update and a fixed cap;
  * Dawid-Skene EM (latent per-annotator confusion matrices).

Expected qualitative result (matches the paper): Dawid-Skene dominates;
reliability-weighted barely beats majority because additive weights saturate
to the cap and the vote collapses toward majority.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

N_ITEMS = 1500
N_ANNOTATORS = 12
K_VALUES = [3, 5, 7]
N_RUNS = 60

# reliability-weighted update hyper-parameters (additive rule with cap, Eq. 2)
R_INIT = 1.0
R_DELTA = 0.10
R_CAP = 2.0


def _sample_accuracies(rng: np.random.RandomState) -> np.ndarray:
    """12 annotators: 40% expert, 40% mediocre, 20% adversarial."""
    n_exp = int(round(0.40 * N_ANNOTATORS))
    n_med = int(round(0.40 * N_ANNOTATORS))
    n_adv = N_ANNOTATORS - n_exp - n_med
    acc = np.concatenate([
        rng.uniform(0.85, 0.97, n_exp),
        rng.uniform(0.58, 0.72, n_med),
        rng.uniform(0.35, 0.50, n_adv),
    ])
    rng.shuffle(acc)
    return acc


def _generate_votes(rng, truth, acc, k):
    """For each item, pick k annotators; each votes truth w.p. acc else flips.

    Returns (votes, voters): votes in {+1,-1} ground-truth-encoded, voters is
    the annotator index per vote. Items use a fixed k voters each.
    """
    n = len(truth)
    voters = np.zeros((n, k), dtype=int)
    votes = np.zeros((n, k), dtype=int)
    for i in range(n):
        chosen = rng.choice(N_ANNOTATORS, size=k, replace=False)
        voters[i] = chosen
        for j, a in enumerate(chosen):
            correct = rng.rand() < acc[a]
            label = truth[i] if correct else 1 - truth[i]
            votes[i, j] = 1 if label == 1 else -1
    return votes, voters


def _majority(votes, voters, rng) -> np.ndarray:
    out = []
    for row in votes:
        s = row.sum()
        if s > 0:
            out.append(1)
        elif s < 0:
            out.append(0)
        else:
            out.append(rng.randint(2))  # tie -> coin flip (deferred in practice)
    return np.array(out)


def _reliability_weighted(votes, voters, rng) -> np.ndarray:
    """Online additive weight update with a fixed cap (Eq. 2)."""
    r = np.full(N_ANNOTATORS, R_INIT)
    out = []
    for i in range(len(votes)):
        vs = votes[i]
        ws = r[voters[i]]
        O = float((ws * vs).sum())
        if O > 0:
            yhat = 1
        elif O < 0:
            yhat = 0
        else:
            yhat = rng.randint(2)
        out.append(yhat)
        yvote = 1 if yhat == 1 else -1
        for j, a in enumerate(voters[i]):
            if vs[j] == yvote:
                r[a] = min(R_CAP, r[a] + R_DELTA)
            else:
                r[a] = max(0.0, r[a] - R_DELTA)
    return np.array(out)


def _dawid_skene(votes, voters, rng, n_iter=30) -> np.ndarray:
    """Binary Dawid-Skene EM. votes in {+1,-1} -> map to {1,0}."""
    n = len(votes)
    k = votes.shape[1]
    obs = (votes == 1).astype(int)               # 1 = 'valid', 0 = 'reject'
    # init E-step from majority
    T = np.zeros((n, 2))
    for i in range(n):
        ones = obs[i].sum()
        p1 = ones / k
        T[i] = [1 - p1, p1]
    for _ in range(n_iter):
        # M-step: per-annotator confusion pi[a, true, obs], class prior
        pi = np.ones((N_ANNOTATORS, 2, 2))       # Laplace smoothing
        prior = T.sum(axis=0) + 1.0
        prior /= prior.sum()
        for i in range(n):
            for j in range(k):
                a = voters[i, j]
                o = obs[i, j]
                pi[a, 0, o] += T[i, 0]
                pi[a, 1, o] += T[i, 1]
        pi /= pi.sum(axis=2, keepdims=True)
        # E-step
        newT = np.zeros((n, 2))
        for i in range(n):
            logp = np.log(prior + 1e-12).copy()
            for j in range(k):
                a = voters[i, j]
                o = obs[i, j]
                logp[0] += np.log(pi[a, 0, o] + 1e-12)
                logp[1] += np.log(pi[a, 1, o] + 1e-12)
            logp -= logp.max()
            p = np.exp(logp)
            newT[i] = p / p.sum()
        if np.abs(newT - T).max() < 1e-6:
            T = newT
            break
        T = newT
    return (T[:, 1] > 0.5).astype(int)


def run() -> pd.DataFrame:
    rows = []
    for k in K_VALUES:
        accs = {"Majority": [], "Reliability-weighted": [], "Dawid-Skene": []}
        for run_idx in range(N_RUNS):
            rng = np.random.RandomState(1000 * k + run_idx)
            truth = np.array([0, 1] * (N_ITEMS // 2))
            rng.shuffle(truth)
            acc = _sample_accuracies(rng)
            votes, voters = _generate_votes(rng, truth, acc, k)
            accs["Majority"].append(
                (_majority(votes, voters, rng) == truth).mean())
            accs["Reliability-weighted"].append(
                (_reliability_weighted(votes, voters, rng) == truth).mean())
            accs["Dawid-Skene"].append(
                (_dawid_skene(votes, voters, rng) == truth).mean())
        row = {"k": k}
        for name, vals in accs.items():
            v = np.array(vals)
            row[f"{name}_mean"] = float(v.mean())
            row[f"{name}_sd"] = float(v.std(ddof=0))
        rows.append(row)
        print(f"  k={k}  maj={row['Majority_mean']:.3f}  "
              f"rel={row['Reliability-weighted_mean']:.3f}  "
              f"ds={row['Dawid-Skene_mean']:.3f}")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print(run().to_string(index=False))
