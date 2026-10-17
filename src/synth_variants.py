"""In-memory variants of the synthetic corpus, for sensitivity checks only.

This module never writes to data/synth/. It re-implements the string assembly
of make_synth._assemble with two additions, and nothing else:

  * every token carries its provenance (which block it came from: grade,
    brand, core, unit or filler), including the parts of a token produced by
    concatenating two adjacent tokens;
  * the noise process is a parameter: its rate (make_synth.NOISE_RATE is 0.30)
    and its kind, either "char" (the released character-level corruption) or
    "word" (word-level noise only, defined below).

With noise_rate=0.30 and noise_kind="char" the random draws are consumed in
exactly the order make_synth consumes them, so generate_variant reproduces the
released corpus string for string. check_reproduces_release() asserts this
against data/synth/*.parquet; every script that uses this module calls it
first, so a variant can never silently drift from the released generator.

Word-level noise ("word"). The released generator has one word-level
operation, concatenation of two adjacent tokens, and three character-level
ones (truncation, vowel deletion, transposition). The "word" kind keeps the
concatenation unchanged and replaces the three character-level operations by
two word-level ones applied with the same probabilities: in a noised string,
every token longer than three characters is, with probability 0.5, either
deleted or duplicated, chosen uniformly. A string whose every token would be
deleted is left with its tokens unchanged. No token is altered internally, so
every surviving token is a correctly spelt vocabulary word.

Grade-block removal (drop_grade=True). The grade block is removed wherever it
ended up after noise: as a token of its own, in corrupted form, or as the
leading part of a concatenated token. Tokens of the key phrase are never
touched, so product words that are also grade words ("white" in "white
sugar", "whole" in "whole milk", "pale" in "pale ale") are kept. All random
draws are identical with and without removal, so the two corpora differ only
in the grade block.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import make_synth as ms

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "synth"


def _abbrev_char(token: str, rng: np.random.RandomState) -> list[str]:
    """make_synth._maybe_abbrev, unchanged; returns a one-element list."""
    return [ms._maybe_abbrev(token, rng)]


def _word_noise(token: str, rng: np.random.RandomState) -> list[str]:
    """Word-level replacement for _maybe_abbrev: delete or duplicate."""
    if len(token) <= 3 or rng.rand() > 0.5:
        return [token]
    return [] if rng.randint(2) == 0 else [token, token]


def assemble_tracked(core, cfg, rng, positive, noise_rate=ms.NOISE_RATE,
                     noise_kind="char"):
    """Mirror of make_synth._assemble that also returns provenance.

    Returns a list of tokens, each a list of (text, block) segments; a token
    has more than one segment only if it was produced by concatenation.
    """
    parts: list[tuple[str, list[str]]] = []
    if positive and rng.rand() < 0.7:
        parts.append(("grade", [rng.choice(cfg.grades)]))
    if rng.rand() < 0.6:
        parts.append(("brand", [rng.choice(cfg.brands)]))
    parts.append(("core", list(core)))
    if rng.rand() < 0.7:
        parts.append(("unit", [rng.choice(cfg.units)]))
    if rng.rand() < 0.4:
        parts.append(("filler", [rng.choice(cfg.fillers)]))
    # make_synth's re-insertion branch never fires (the core is always in
    # parts) and consumes no random draws, so it is omitted here.
    tokens = [[(str(t), blk)] for blk, block in parts for t in block]

    if rng.rand() < noise_rate:
        op = _abbrev_char if noise_kind == "char" else _word_noise
        new = []
        for tok in tokens:
            (text, blk), = tok
            new.extend([[(s, blk)] for s in op(text, rng)])
        if not new:          # word noise deleted every token: keep the string
            new = tokens
        tokens = new
        if len(tokens) >= 2 and rng.rand() < 0.3:
            j = rng.randint(len(tokens) - 1)
            tokens = tokens[:j] + [tokens[j] + tokens[j + 1]] + tokens[j + 2:]
    elif noise_kind not in ("char", "word"):
        raise ValueError(noise_kind)
    return tokens


def render(tokens, drop_grade=False) -> str:
    out = []
    for tok in tokens:
        s = "".join(t for t, blk in tok if not (drop_grade and blk == "grade"))
        if s:
            out.append(s)
    return " ".join(out)


def generate_variant(cfg, seed=ms.CORPUS_SEED, noise_rate=ms.NOISE_RATE,
                     noise_kind="char", drop_grade=False,
                     return_tokens=False):
    """make_synth.generate_category with the variant parameters above."""
    rng = np.random.RandomState(seed)
    rows, toks = [], []
    for _ in range(cfg.n_pos):
        # same unused draw as make_synth.generate_category (see the note there)
        _ = list(rng.permutation(cfg.key_phrases[rng.randint(len(cfg.key_phrases))]))
        core = cfg.key_phrases[rng.randint(len(cfg.key_phrases))]
        t = assemble_tracked(core, cfg, rng, True, noise_rate, noise_kind)
        rows.append((render(t, drop_grade), 1))
        toks.append(t)
    for _ in range(cfg.n_neg):
        core = cfg.negative_cores[rng.randint(len(cfg.negative_cores))]
        t = assemble_tracked(core, cfg, rng, False, noise_rate, noise_kind)
        rows.append((render(t, drop_grade), 0))
        toks.append(t)
    df = pd.DataFrame(rows, columns=["text", "label"])
    order = df.sample(frac=1.0, random_state=seed).index
    df = df.loc[order].reset_index(drop=True)
    if return_tokens:
        return df, [toks[i] for i in order]
    return df


def check_reproduces_release() -> None:
    """Assert that the default variant equals the released parquet files."""
    for cfg in ms._categories():
        rel = pd.read_parquet(DATA_DIR / f"{cfg.name}.parquet")
        gen = generate_variant(cfg)
        assert rel.text.tolist() == gen.text.tolist(), cfg.name
        assert rel.label.tolist() == gen.label.tolist(), cfg.name
        # and through make_synth itself, in memory
        ref = ms.generate_category(cfg, ms.CORPUS_SEED)
        assert ref.text.tolist() == rel.text.tolist(), cfg.name
    print("synth_variants: default variant reproduces data/synth exactly")


if __name__ == "__main__":
    check_reproduces_release()
