"""The frozen token vocabulary, shared by the training pipeline and the harness.

`js/vocab-data.js` is the single source of truth for token <-> id: its array
index IS the model's token id. This module parses that file (rather than
re-deriving the dictionary in Python) so the tokeniser, the packed training
data, the trained model, and the JS harness can never disagree on ids.

It also owns the translation from the *bracketed* surface syntax the tokeniser
emits (`[Nf3]`, `[x]`, `[??]`) to the *bare* forms the dictionary stores
(`Nf3`, `x`, `<blunder>`), and the game-framing used to pack games into one
stream:

    <bos> <elo-W> <elo-B>  <move tokens...>  <eos>

The <bos>/<eos>/<elo-*> tokens are "vocab v2" — appended after the nerf tokens
in js/vocab-data.js without disturbing any pre-existing id.
"""
from __future__ import annotations

import os
import re
from functools import lru_cache

# ---------------------------------------------------------------- locating the dictionary

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_VOCAB_JS = os.path.normpath(os.path.join(_HERE, "..", "js", "vocab-data.js"))

# ---------------------------------------------------------------- surface translation

# The three accuracy glyphs the tokeniser emits map to the dictionary's nerf
# words. Everything else in the dictionary is stored bare, so translating a
# tokeniser token is just "strip the brackets, then remap glyphs".
GLYPH_TO_NERF = {"??": "<blunder>", "?": "<mistake>", "?!": "<inaccuracy>"}

BOS = "<bos>"
EOS = "<eos>"


def translate(token: str) -> str:
    """Bracketed tokeniser token -> bare dictionary surface form.

    '[Nf3]' -> 'Nf3' ; '[x]' -> 'x' ; '[=Q]' -> '=Q' ; '[??]' -> '<blunder>'.
    Tolerant of already-bare input (returns it unchanged apart from glyphs).
    """
    inner = token[1:-1] if len(token) >= 2 and token[0] == "[" and token[-1] == "]" else token
    return GLYPH_TO_NERF.get(inner, inner)


def elo_bucket_token(elo: int | float | None) -> str:
    """Numeric Elo -> bucket token. 200-point steps, open-ended tails.

    Mirrors eloBucketToken() in js/vocab.js. Non-numeric / <800 -> '<elo-u800>'.
    """
    try:
        e = int(elo)
    except (TypeError, ValueError):
        return "<elo-u800>"
    if e < 800:
        return "<elo-u800>"
    if e >= 3000:
        return "<elo-3000p>"
    return f"<elo-{(e // 200) * 200}>"


# ---------------------------------------------------------------- the dictionary

_TOKENS_RE = re.compile(r"TOKENS\s*=\s*Object\.freeze\(\[(.*?)\]\)", re.S)
_QUOTED_RE = re.compile(r"'((?:[^'\\]|\\.)*)'")
_CORE_COUNT_RE = re.compile(r"CORE_COUNT\s*=\s*(\d+)")


class Vocab:
    """Ordered token list + id lookups, loaded from js/vocab-data.js."""

    def __init__(self, tokens: list[str], core_count: int):
        self.tokens = tokens
        self.core_count = core_count
        self.stoi = {t: i for i, t in enumerate(tokens)}
        self.itos = {i: t for i, t in enumerate(tokens)}
        self.size = len(tokens)
        # structural ids used during framing (fail loudly if vocab v2 is missing)
        missing = [t for t in (BOS, EOS, "<elo-u800>", "<elo-3000p>") if t not in self.stoi]
        if missing:
            raise ValueError(
                f"vocab-data.js is missing structural tokens {missing}; "
                "regenerate it with: node tools/build-vocab.mjs"
            )
        self.bos_id = self.stoi[BOS]
        self.eos_id = self.stoi[EOS]

    # -- framing -------------------------------------------------------------

    def encode_game(self, tokens_field: str, white_elo, black_elo):
        """Frame one game as ids: <bos> <elo-W> <elo-B> <moves...> <eos>.

        `tokens_field` is the space-joined bracketed stream from build_dataset.
        Returns (ids, unknown_tokens). If any move token is outside the frozen
        dictionary, ids is None and unknown_tokens names the offenders (the
        caller should drop the game). On real Lichess data this never fires —
        the dictionary is a superset of legal SAN — but synthetic corpora with
        geometrically-loose SANs can trip it, which is the point of the check.
        """
        ids = [self.bos_id, self.stoi[elo_bucket_token(white_elo)], self.stoi[elo_bucket_token(black_elo)]]
        unknown: set[str] = set()
        for tok in tokens_field.split():
            sid = self.stoi.get(translate(tok))
            if sid is None:
                unknown.add(tok)
            else:
                ids.append(sid)
        if unknown:
            return None, unknown
        ids.append(self.eos_id)
        return ids, unknown

    def decode(self, ids) -> list[str]:
        return [self.itos[i] for i in ids]


def load_vocab(path: str | None = None) -> Vocab:
    with open(path or DEFAULT_VOCAB_JS, encoding="utf-8") as f:
        src = f.read()
    m = _TOKENS_RE.search(src)
    if not m:
        raise ValueError(f"could not find TOKENS array in {path or DEFAULT_VOCAB_JS}")
    tokens = [t.replace("\\'", "'") for t in _QUOTED_RE.findall(m.group(1))]
    if not tokens:
        raise ValueError("parsed zero tokens from vocab-data.js")
    cc = _CORE_COUNT_RE.search(src)
    return Vocab(tokens, int(cc.group(1)) if cc else len(tokens))


@lru_cache(maxsize=None)
def get_vocab(path: str | None = None) -> Vocab:
    """Cached loader for the default (or a specific) dictionary file."""
    return load_vocab(path)


if __name__ == "__main__":
    v = get_vocab()
    print(f"{v.size} tokens ({v.core_count} cores) from {DEFAULT_VOCAB_JS}")
    print(f"bos={v.bos_id} eos={v.eos_id}")
    ids, unk = v.encode_game("[Nf3] [x] [Bb5] [??] [e5] [+] [Qf5]", 1834, 1790)
    print("sample ids:", ids)
    print("decoded:", v.decode(ids))
