"""Supplementary table of every paired model comparison, with its band.

Reads the released paired comparisons and adds the descriptive band of
bands.py (share of non-tied cells won by the model with the higher mean):

  results/significance_pairwise.csv           the seven models trained from
                                              scratch (Wilcoxon p, Holm p over
                                              their 21 pairs)
  results/pretrained/pretrained_pairwise.csv  pretrained encoders against the
                                              seven and against each other
                                              (raw Wilcoxon p; Holm is applied
                                              here over those 24 distinct pairs
                                              as a separate family)
  results/aggregation_to_training.csv         classifiers trained on labels
                                              from each aggregation rule

Nothing upstream is recomputed or modified. The p-values are descriptive only:
the 30 category-seed cells are not independent replications (see the paper,
Section "Protocol").

Writes
  results/pairwise_bands.csv                  one row per model pair
  results/aggregation_to_training_pairs.csv   one row per model, k and rule pair
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

import bands
from significance import holm

RESULTS = Path(__file__).resolve().parent.parent / "results"


def _row(source, a, b, aw, bw, t, md, p, ph):
    d = bands.describe(aw, bw, md)
    lead, other = (a, b) if d["leader"] == "a" else (b, a)
    wl, wo = (aw, bw) if d["leader"] == "a" else (bw, aw)
    return dict(source=source, leader=lead, other=other, leader_wins=wl,
                other_wins=wo, ties=t, mean_diff_points=abs(md) * 100,
                leader_share_non_tied=d["leader_share_non_tied"],
                band=d["band"], wilcoxon_p=p, holm_p=ph)


def main() -> None:
    rows = []
    sp = pd.read_csv(RESULTS / "significance_pairwise.csv")
    for r in sp.itertuples():
        rows.append(_row("from scratch", r.model_a, r.model_b, r.a_wins, r.b_wins,
                         r.ties, r.mean_diff, r.p_raw, r.p_holm))

    pp = pd.read_csv(RESULTS / "pretrained" / "pretrained_pairwise.csv")
    seen, keep = set(), []
    for r in pp.itertuples():
        key = frozenset((r.model, r.vs))
        if key in seen:
            continue
        seen.add(key)
        keep.append(r)
    ph = holm([r.p_raw for r in keep])
    for r, h in zip(keep, ph):
        rows.append(_row("pretrained", r.model, r.vs, r.wins, r.losses, r.ties,
                         r.mean_diff, r.p_raw, h))
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "pairwise_bands.csv", index=False)

    ag = pd.read_csv(RESULTS / "aggregation_to_training.csv")
    rules = ["Majority", "Reliability-weighted", "Dawid-Skene"]
    arows = []
    for (model, k), g in ag.groupby(["model", "k"]):
        w = g.pivot_table(index=["category", "seed"], columns="aggregator",
                          values="test_f1")
        for a, b in combinations(rules, 2):
            diff = w[b] - w[a]
            bw, aw, t = int((diff > 0).sum()), int((diff < 0).sum()), int((diff == 0).sum())
            d = bands.describe(bw, aw, float(diff.mean()))
            arows.append(dict(model=model, k=k, rule=b, against=a, rule_wins=bw,
                              against_wins=aw, ties=t,
                              mean_diff_points=float(diff.mean()) * 100,
                              leader=b if d["leader"] == "a" else a,
                              leader_share_non_tied=d["leader_share_non_tied"],
                              band=d["band"]))
    ao = pd.DataFrame(arows)
    ao.to_csv(RESULTS / "aggregation_to_training_pairs.csv", index=False)

    pd.set_option("display.width", 250)
    print(out.round(4).to_string(index=False))
    print(ao.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
