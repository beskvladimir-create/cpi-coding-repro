"""Trie pre-classifier -- Equation (1) of arXiv:2606.02004.

A category c admits an item x iff
    (exists key-phrase p in K_c : p is a contiguous token subsequence of x)
  AND
    (no stop-phrase q in S_c : q is a contiguous token subsequence of x).

The phrase sets are taken verbatim from make_synth's CategoryConfig so the
rule-based stage and the data generator share one source of truth.

Coverage = admitted / total over a stream of items. An item admitted by more
than one category is reported as 'ambiguous' (passed to the neural stage in the
full system); an item admitted by none is 'unidentified'.
"""
from __future__ import annotations

from dataclasses import dataclass

from make_synth import _categories


class PhraseTrie:
    """Prefix tree over token phrases; supports contiguous-subsequence search."""

    def __init__(self, phrases: list[list[str]]):
        self.root: dict = {}
        for phrase in phrases:
            node = self.root
            for tok in phrase:
                node = node.setdefault(tok, {})
            node["$"] = True  # end-of-phrase marker

    def matches(self, tokens: list[str]) -> bool:
        """True iff any stored phrase occurs as a contiguous token subsequence."""
        n = len(tokens)
        for start in range(n):
            node = self.root
            for i in range(start, n):
                tok = tokens[i]
                if tok not in node:
                    break
                node = node[tok]
                if node.get("$"):
                    return True
        return False


@dataclass
class CategoryTrie:
    name: str
    key_trie: PhraseTrie
    stop_trie: PhraseTrie

    def admits(self, tokens: list[str]) -> bool:
        return self.key_trie.matches(tokens) and not self.stop_trie.matches(tokens)


def build_classifier() -> list[CategoryTrie]:
    out = []
    for cfg in _categories():
        out.append(CategoryTrie(
            name=cfg.name,
            key_trie=PhraseTrie(cfg.key_phrases),
            stop_trie=PhraseTrie(cfg.stop_phrases),
        ))
    return out


def tokenize(text: str) -> list[str]:
    return text.lower().split()


def classify(tokens: list[str], clf: list[CategoryTrie]) -> str:
    """Return the admitting category, or 'ambiguous'/'unidentified'."""
    admitted = [ct.name for ct in clf if ct.admits(tokens)]
    if len(admitted) == 1:
        return admitted[0]
    if len(admitted) > 1:
        return "ambiguous"
    return "unidentified"


def coverage_report() -> "pd.DataFrame":
    """Run the trie over the whole synthetic corpus and tabulate coverage."""
    import pandas as pd
    from pathlib import Path

    data_dir = Path(__file__).resolve().parent.parent / "data" / "synth"
    clf = build_classifier()
    rows = []
    for cfg in _categories():
        df = pd.read_parquet(data_dir / f"{cfg.name}.parquet")
        verdicts = [classify(tokenize(t), clf) for t in df.text]
        admitted_here = sum(v == cfg.name for v in verdicts)
        ambiguous = sum(v == "ambiguous" for v in verdicts)
        unidentified = sum(v == "unidentified" for v in verdicts)
        other = len(df) - admitted_here - ambiguous - unidentified
        # how many of the true positives were correctly admitted to THIS category
        pos_mask = df.label.values == 1
        pos_admitted = sum(
            (v == cfg.name) for v, p in zip(verdicts, pos_mask) if p
        )
        rows.append({
            "category": cfg.name,
            "n_items": len(df),
            "n_positives": int(pos_mask.sum()),
            "admitted_to_category": admitted_here,
            "ambiguous": ambiguous,
            "unidentified": unidentified,
            "admitted_other_category": other,
            "coverage": round(1 - unidentified / len(df), 4),
            "positive_recall_trie": round(pos_admitted / max(1, pos_mask.sum()), 4),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print(coverage_report().to_string(index=False))
