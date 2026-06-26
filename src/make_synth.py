"""Synthetic 'receipt-like' item-name generator.

Reproducibility package for arXiv:2606.02004. ALL data here are synthetic and
generated deterministically from fixed seeds. No real, proprietary, or
production data are used anywhere in this repository.

For each COICOP-like category we generate:
  * positives      -- strings that name the target product;
  * hard negatives -- strings that share tokens with the target but name a
                      different product (the synthetic analogue of
                      "icing sugar" / "sugar-free yogurt" from the paper).

Output: one parquet per category in data/synth/ plus a manifest.json recording
the seed and generation parameters so the corpus is fully reproducible.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "synth"

# Generation seed for the corpus itself. Model-evaluation seeds (the [0..4]
# split seeds) are applied later in confirm_models.py; the corpus is fixed.
CORPUS_SEED = 0


@dataclass
class CategoryConfig:
    """Token vocabulary and assembly rules for one category.

    key_phrases / stop_phrases are ALSO consumed by the trie pre-classifier
    (trie_classifier.py), so the generator and the rule-based stage stay in
    sync by construction.
    """

    name: str
    key_phrases: list[list[str]]          # positive triggers (token lists)
    stop_phrases: list[list[str]]         # exclusions used by the trie
    brands: list[str]
    grades: list[str]
    units: list[str]
    fillers: list[str]                    # generic receipt noise tokens
    # hard-negative product cores: share tokens with the target, different item.
    negative_cores: list[list[str]]
    n_pos: int = 700
    n_neg: int = 950


def _categories() -> list[CategoryConfig]:
    return [
        CategoryConfig(
            name="granulated_sugar",
            key_phrases=[["sugar"], ["granulated", "sugar"], ["white", "sugar"]],
            stop_phrases=[["sugar", "free"], ["icing", "sugar"], ["brown", "sugar"],
                          ["cotton", "candy"], ["sugary"]],
            brands=["dansukker", "silver", "spoon", "tate", "domino", "store"],
            grades=["granulated", "white", "fine", "refined", "caster"],
            units=["1kg", "500g", "2kg", "1000g", "900g"],
            fillers=["pack", "bag", "premium", "value"],
            negative_cores=[["sugar", "free", "yogurt"], ["icing", "sugar"],
                            ["brown", "sugar"], ["sugary", "bun"],
                            ["cotton", "candy"], ["sugar", "free", "gum"],
                            ["sweetener"]],
        ),
        CategoryConfig(
            name="milk",
            key_phrases=[["milk"], ["whole", "milk"], ["semi", "skimmed", "milk"]],
            stop_phrases=[["milk", "chocolate"], ["coconut", "milk"], ["soy", "milk"],
                          ["oat", "milk"], ["milkshake"], ["condensed", "milk"]],
            brands=["arla", "cravendale", "dairy", "store", "valio", "farm"],
            grades=["whole", "semi", "skimmed", "fresh", "uht"],
            units=["1l", "2l", "500ml", "1000ml", "750ml"],
            fillers=["pack", "carton", "bottle", "value"],
            negative_cores=[["milk", "chocolate", "bar"], ["coconut", "milk"],
                            ["soy", "milk"], ["oat", "milk"], ["milkshake"],
                            ["condensed", "milk"], ["milk", "powder"]],
        ),
        CategoryConfig(
            name="bread",
            key_phrases=[["bread"], ["white", "bread"], ["wholemeal", "bread"]],
            stop_phrases=[["bread", "crumbs"], ["breadsticks"], ["gingerbread"],
                          ["banana", "bread"], ["bread", "maker"]],
            brands=["hovis", "warburtons", "store", "kingsmill", "bakery"],
            grades=["white", "wholemeal", "sliced", "seeded", "sourdough"],
            units=["800g", "400g", "loaf", "750g"],
            fillers=["pack", "fresh", "value", "thick"],
            negative_cores=[["bread", "crumbs"], ["breadsticks"], ["gingerbread", "man"],
                            ["banana", "bread"], ["bread", "maker"], ["garlic", "bread"]],
        ),
        CategoryConfig(
            name="beer",
            key_phrases=[["beer"], ["lager", "beer"], ["pale", "ale"]],
            stop_phrases=[["beer", "free"], ["alcohol", "free", "beer"],
                          ["ginger", "beer"], ["root", "beer"], ["beer", "battered"]],
            brands=["heineken", "carlsberg", "store", "stella", "brewdog"],
            grades=["lager", "pale", "ipa", "premium", "craft"],
            units=["330ml", "500ml", "440ml", "4x440ml", "6x330ml"],
            fillers=["pack", "can", "bottle", "value"],
            negative_cores=[["ginger", "beer"], ["root", "beer"],
                            ["alcohol", "free", "beer"], ["beer", "battered", "fish"],
                            ["beer", "free"], ["beer", "glass"]],
        ),
        CategoryConfig(
            name="laundry_detergent",
            key_phrases=[["laundry", "detergent"], ["washing", "powder"],
                         ["laundry", "liquid"]],
            stop_phrases=[["dish", "detergent"], ["dishwasher"], ["hand", "wash"],
                          ["washing", "up"], ["fabric", "softener"]],
            brands=["ariel", "persil", "store", "surf", "bold"],
            grades=["bio", "non", "color", "gel", "powder"],
            units=["1.5l", "40wash", "2kg", "30wash", "1l"],
            fillers=["pack", "value", "concentrated", "fresh"],
            negative_cores=[["dishwasher", "tablets"], ["washing", "up", "liquid"],
                            ["fabric", "softener"], ["hand", "wash", "soap"],
                            ["dish", "detergent"], ["surface", "cleaner"]],
        ),
        CategoryConfig(
            name="fresh_apples",
            key_phrases=[["apples"], ["fresh", "apples"], ["gala", "apples"]],
            stop_phrases=[["apple", "juice"], ["apple", "sauce"], ["apple", "pie"],
                          ["pineapple"], ["apple", "cider"], ["dried", "apple"]],
            brands=["store", "farm", "orchard", "value", "organic"],
            grades=["gala", "fresh", "braeburn", "pink", "lady"],
            units=["1kg", "6pack", "500g", "loose", "4pack"],
            fillers=["pack", "premium", "class", "value"],
            negative_cores=[["apple", "juice"], ["apple", "sauce"], ["apple", "pie"],
                            ["pineapple", "chunks"], ["apple", "cider", "vinegar"],
                            ["dried", "apple", "rings"]],
        ),
    ]


# ---- string assembly with controllable noise -------------------------------

NOISE_RATE = 0.30  # fraction of generated strings that receive token-level noise


def _maybe_abbrev(token: str, rng: np.random.RandomState) -> str:
    """Occasionally abbreviate or typo a token (receipt OCR-style noise)."""
    if len(token) <= 3 or rng.rand() > 0.5:
        return token
    mode = rng.randint(3)
    if mode == 0:                                   # truncate
        return token[: max(3, len(token) - rng.randint(1, 3))]
    if mode == 1:                                   # drop a vowel
        no_vowels = [c for c in token if c not in "aeiou"]
        return "".join(no_vowels) if len(no_vowels) >= 3 else token
    idx = rng.randint(len(token) - 1)               # swap two adjacent chars
    return token[:idx] + token[idx + 1] + token[idx] + token[idx + 2:]


def _assemble(core: list[str], cfg: CategoryConfig, rng: np.random.RandomState,
              positive: bool) -> str:
    """Glue tokens: brand/grade + core + unit + filler, with shuffling/noise."""
    parts: list[list[str]] = []
    if positive and rng.rand() < 0.7:
        parts.append([rng.choice(cfg.grades)])
    if rng.rand() < 0.6:
        parts.append([rng.choice(cfg.brands)])
    parts.append(list(core))
    if rng.rand() < 0.7:
        parts.append([rng.choice(cfg.units)])
    if rng.rand() < 0.4:
        parts.append([rng.choice(cfg.fillers)])

    # light reordering of the non-core blocks around the core
    core_idx = parts.index(list(core)) if list(core) in parts else None
    if core_idx is None:
        parts.insert(rng.randint(len(parts) + 1), list(core))

    tokens: list[str] = [t for block in parts for t in block]

    if rng.rand() < NOISE_RATE:
        # apply token-level noise + occasional concatenation
        tokens = [_maybe_abbrev(t, rng) for t in tokens]
        if len(tokens) >= 2 and rng.rand() < 0.3:
            j = rng.randint(len(tokens) - 1)
            tokens = tokens[:j] + [tokens[j] + tokens[j + 1]] + tokens[j + 2:]
    return " ".join(tokens)


def generate_category(cfg: CategoryConfig, seed: int) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    rows = []
    for _ in range(cfg.n_pos):
        core = list(rng.permutation(cfg.key_phrases[rng.randint(len(cfg.key_phrases))]))
        # keep key-phrase order intact most of the time so the trie can match
        core = cfg.key_phrases[rng.randint(len(cfg.key_phrases))]
        rows.append((_assemble(core, cfg, rng, positive=True), 1))
    for _ in range(cfg.n_neg):
        core = cfg.negative_cores[rng.randint(len(cfg.negative_cores))]
        rows.append((_assemble(core, cfg, rng, positive=False), 0))
    df = pd.DataFrame(rows, columns=["text", "label"])
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cats = _categories()
    manifest = {
        "corpus_seed": CORPUS_SEED,
        "noise_rate": NOISE_RATE,
        "disclaimer": ("All data are synthetically generated; this corpus "
                       "reproduces the methodology of arXiv:2606.02004 and "
                       "contains no proprietary or production data."),
        "categories": {},
    }
    for cfg in cats:
        df = generate_category(cfg, CORPUS_SEED)
        out = DATA_DIR / f"{cfg.name}.parquet"
        df.to_parquet(out, index=False)
        manifest["categories"][cfg.name] = {
            "n_pos": int((df.label == 1).sum()),
            "n_neg": int((df.label == 0).sum()),
            "key_phrases": [" ".join(p) for p in cfg.key_phrases],
            "stop_phrases": [" ".join(p) for p in cfg.stop_phrases],
            "file": out.name,
        }
        print(f"  {cfg.name:20s} pos={int((df.label==1).sum()):4d} "
              f"neg={int((df.label==0).sum()):4d} -> {out.name}")
    with open(DATA_DIR / "manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"manifest -> {DATA_DIR / 'manifest.json'}")


if __name__ == "__main__":
    main()
