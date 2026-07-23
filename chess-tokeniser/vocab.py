"""The frozen token vocabulary, shared by the training pipeline and the harness.

`js/vocab-data.js` is the single source of truth for token <-> id: its array
index IS the model's token id. This module parses that file (rather than
re-deriving the dictionary in Python) so the tokeniser, the packed training
data, the trained model, and the JS harness can never disagree on ids.

It also owns the translation from the *bracketed* surface syntax the tokeniser
emits (`[Nf3]`, `[x]`, `[??]`) to the *bare* forms the dictionary stores
(`Nf3`, `x`, `<blunder>`), and the game-framing used to pack games into one
stream:

    <bos> <elo-W> <elo-B>  <move tokens...>  [<resign>|<draw>]  <eos>

The <bos>/<eos>/<elo-*> tokens are "vocab v2", the <resign>/<draw> game-end
tokens are "vocab v3" — both pure appends to js/vocab-data.js that disturb no
pre-existing id.
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
ELO_ANY = "<elo-any>"        # "strength unspecified" sentinel (see encode_game)
# vocab v3 game-end tokens, emitted just before <eos> (see Vocab._end_id).
# Resigns/flags are colour-differentiated: resignations happen on either turn
# (often right after the loser's own move), so colour is not inferable from
# parity; flags get colour too for uniformity. <draw> is symmetric — no colour.
WHITE_RESIGN = "<white-resign>"
BLACK_RESIGN = "<black-resign>"
WHITE_FLAG = "<white-flag>"
BLACK_FLAG = "<black-flag>"
DRAW = "<draw>"


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
        # structural ids used during framing (fail loudly if vocab v2/v3 is missing)
        missing = [t for t in (BOS, EOS, ELO_ANY, "<elo-u800>", "<elo-3000p>",
                               WHITE_RESIGN, BLACK_RESIGN, WHITE_FLAG, BLACK_FLAG, DRAW)
                   if t not in self.stoi]
        if missing:
            raise ValueError(
                f"vocab-data.js is missing structural tokens {missing}; "
                "regenerate it with: node tools/build-vocab.mjs"
            )
        self.bos_id = self.stoi[BOS]
        self.eos_id = self.stoi[EOS]
        self.elo_any_id = self.stoi[ELO_ANY]
        self.white_resign_id = self.stoi[WHITE_RESIGN]
        self.black_resign_id = self.stoi[BLACK_RESIGN]
        self.white_flag_id = self.stoi[WHITE_FLAG]
        self.black_flag_id = self.stoi[BLACK_FLAG]
        self.draw_id = self.stoi[DRAW]

    # -- framing -------------------------------------------------------------

    def _elo_id(self, elo, elo_dropout, rng):
        """Bucket id for `elo`, or the <elo-any> sentinel with prob elo_dropout."""
        if elo_dropout and rng is not None and rng.random() < elo_dropout:
            return self.elo_any_id
        return self.stoi[elo_bucket_token(elo)]

    def _end_id(self, result, termination, saw_mate: bool):
        """Game-end token id for HOW the game ended, or None for a bare <eos>.

        Header-only classification (the pipeline never replays the board), with
        the actor's COLOUR taken from Result — never from parity, because
        players often resign right after their own move, i.e. on the winner's
        turn (flags, by the clock rules, always belong to the side to move, but
        the colour token keeps them uniform and parity-proof too):
          * <draw>  — Result 1/2-1/2 with Termination "Normal". Includes
            rule-draws (stalemate/repetition/...): those positions are terminal
            for the harness anyway, and "offers a draw in a dead position" is
            the behaviour we want. Timeout-draws (flag vs. insufficient
            material, Termination "Time forfeit") stay bare — a flag token
            implies a loss and <draw> would mislabel the clock death.
          * <white-resign>/<black-resign> — decisive Result with Termination
            "Normal" and no mate in the movetext (mate is its own signal); the
            token names the LOSER. Both parities are kept: at play the engine
            only ever samples its OWN colour's resign token on its own turn,
            so opposite-parity occurrences are gauge/context signal, never a
            "resign while winning" hazard.
          * <white-flag>/<black-flag> — decisive Result with Termination
            "Time forfeit"; the token names the LOSER. Never sampled in play
            (the harness has no clocks) — they exist so time-forfeit games
            don't end with an uninformative bare <eos> that teaches "games
            just stop after ordinary moves".
        Everything else (Abandoned, Rules infraction, '*', missing headers)
        keeps the bare <eos>.
        """
        if result == "1/2-1/2":
            return self.draw_id if termination == "Normal" else None
        if result not in ("1-0", "0-1"):
            return None
        white_lost = result == "0-1"
        if termination == "Normal" and not saw_mate:
            return self.white_resign_id if white_lost else self.black_resign_id
        if termination == "Time forfeit":
            return self.white_flag_id if white_lost else self.black_flag_id
        return None

    def encode_game(self, tokens_field: str, white_elo, black_elo, elo_dropout: float = 0.0, rng=None,
                    result=None, termination=None):
        """Frame one game as ids: <bos> <elo-W> <elo-B> <moves...> [<end>] <eos>.

        `tokens_field` is the space-joined bracketed stream from build_dataset.
        With `elo_dropout` > 0 and a `random.Random` `rng`, each Elo slot is
        INDEPENDENTLY replaced by <elo-any> with that probability — so the model
        also sees unconditioned / half-conditioned games and inference can omit
        or neutralise Elo. rng makes the choice reproducible; pass a seeded one.

        `result` ("1-0"/"0-1"/"1/2-1/2") and `termination` (the Lichess
        Termination header) drive the colour-differentiated game-end token
        (see _end_id); left at None — e.g. when packing pre-v3 shards — the
        game ends with a bare <eos> exactly as before.

        Returns (ids, unknown_tokens). If any move token is outside the frozen
        dictionary, ids is None and unknown_tokens names the offenders (the
        caller should drop the game). On real Lichess data this never fires —
        the dictionary is a superset of legal SAN — but synthetic corpora with
        geometrically-loose SANs can trip it, which is the point of the check.
        """
        ids = [self.bos_id, self._elo_id(white_elo, elo_dropout, rng), self._elo_id(black_elo, elo_dropout, rng)]
        unknown: set[str] = set()
        saw_mate = False
        for tok in tokens_field.split():
            surface = translate(tok)
            sid = self.stoi.get(surface)
            if sid is None:
                unknown.add(tok)
                continue
            ids.append(sid)
            if surface == "#":
                saw_mate = True
        if unknown:
            return None, unknown
        end_id = self._end_id(result, termination, saw_mate)
        if end_id is not None:
            ids.append(end_id)
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
