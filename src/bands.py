"""Descriptive bands for paired comparisons over category-seed cells.

The paper does not quote p-values for model comparisons, because the five
seeds within a category are repeated splits of one synthetic corpus and the
six categories come from one generator, so the cells are not independent
replications. Differences are instead described by the share of NON-TIED
cells that the model with the higher mean F1 wins:

  * reliable                 the leading model wins >= 90% of non-tied cells;
  * small but consistent     it wins >= 70% and < 90% of non-tied cells;
  * indistinguishable        it wins <  70% of non-tied cells, or fewer than
                             10 of the cells are non-tied, too few to
                             establish a direction.

Ties are excluded from the denominator because near the ceiling many cells are
tied at the same F1 (often 1.0) and carry no information about direction. A
pair with no non-tied cell is reported as "identical". The bands describe the
consistency of the direction of a difference, not its size; the mean
difference is always reported beside them.
"""
from __future__ import annotations

RELIABLE = 0.90
CONSISTENT = 0.70
MIN_NON_TIED = 10


def band(wins_leader: int, wins_other: int) -> str:
    n = wins_leader + wins_other
    if n == 0:
        return "identical"
    share = wins_leader / n
    if n < MIN_NON_TIED:
        return "indistinguishable"
    if share >= RELIABLE:
        return "reliable"
    if share >= CONSISTENT:
        return "small but consistent"
    return "indistinguishable"


def describe(a_wins: int, b_wins: int, mean_diff_a_minus_b: float) -> dict:
    """Band of a pair, oriented on the model with the higher mean."""
    lead_is_a = mean_diff_a_minus_b >= 0
    wl, wo = (a_wins, b_wins) if lead_is_a else (b_wins, a_wins)
    # if the higher-mean model wins fewer non-tied cells, the pair is
    # indistinguishable by construction (share < 0.5)
    n = wl + wo
    return {"leader": "a" if lead_is_a else "b",
            "leader_share_non_tied": (wl / n) if n else float("nan"),
            "band": band(wl, wo)}
