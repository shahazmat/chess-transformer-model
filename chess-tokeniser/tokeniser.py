"""Core library: parse Lichess movetext, extract SAN + [%eval] comments,
derive accuracy annotations, and tokenise into Shahar's scheme.

Token order per move:  [accuracy][mate/check][capture][promotion][base-move]
    exf8=Q+??  ->  [??][+][x][=Q][ef8]
    Nc6        ->  [Nc6]

Movetext format (Lichess open database / HF parquet):
    1. d4 { [%eval 0.0] [%clk 0:03:00] } 1... d5?! { [%eval 0.27] } ...
Analyzed games carry glyphs (?! ? ??) inline on the SAN and an [%eval]
comment after every ply (the final mating move may lack one).
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

# ---------------------------------------------------------------- constants

MATE_CP = 30_000          # base magnitude for mate scores (mate in n -> ±(MATE_CP - n))
CP_CLAMP = 10_000         # clamp before win% formula (avoids exp overflow)
START_CP = 15             # nominal eval of the initial position

ACCURACY_GLYPHS = ("??", "?!", "?")   # order matters: match ?? and ?! before ?
GOOD_GLYPHS = ("!!", "!?", "!")       # rare in dumps; stripped & counted, not tokenised

# master scanner over movetext
_SCAN_RE = re.compile(
    r"\{(?P<comment>[^}]*)\}"          # { comment }
    r"|(?P<movenum>\d+\.(?:\.\.)?)"    # 12.  or  12...
    r"|(?P<nag>\$\d+)"                 # numeric NAG
    r"|(?P<tok>[^\s{}]+)"              # SAN token / result
)

_EVAL_RE = re.compile(r"\[%eval\s+(#?-?\d+(?:\.\d+)?)\]")

_SAN_RE = re.compile(
    r"^(?P<base>O-O-O|O-O"
    r"|[KQRBN][a-h]?[1-8]?x?[a-h][1-8]"
    r"|[a-h]x[a-h][1-8]"
    r"|[a-h][1-8])"
    r"(?P<promo>=[QRBN])?"
    r"(?P<chk>[+#])?$"
)

_RESULTS = {"1-0", "0-1", "1/2-1/2", "*"}


# ---------------------------------------------------------------- eval maths

def eval_to_cp(s: str) -> int:
    """'0.17' -> 17 ;  '#4' -> 29996 ;  '#-3' -> -29997  (White POV)."""
    if s.startswith("#"):
        n = int(s[1:])
        return (MATE_CP - n) if n >= 0 else (-MATE_CP - n)
    return round(float(s) * 100)


def win_pct(cp: int) -> float:
    cp = max(-CP_CLAMP, min(CP_CLAMP, cp))
    return 50 + 50 * (2 / (1 + math.exp(-0.00368208 * cp)) - 1)


def classify(cp_before: int, cp_after: int, white_moved: bool) -> str | None:
    """Lichess thresholds on win% drop for the mover: ?! >=10, ? >=20, ?? >=30."""
    drop = win_pct(cp_before) - win_pct(cp_after)
    if not white_moved:
        drop = -drop
    if drop >= 30:
        return "??"
    if drop >= 20:
        return "?"
    if drop >= 10:
        return "?!"
    return None


# ---------------------------------------------------------------- parsing

@dataclass
class Ply:
    san: str                    # SAN with glyph stripped, e.g. 'exf8=Q+'
    glyph: str | None = None    # inline glyph from the dump: '?!' '?' '??' (or good-move glyph)
    cp: int | None = None       # eval after this ply, centipawns White POV


@dataclass
class ParsedGame:
    plies: list[Ply] = field(default_factory=list)
    result: str | None = None
    has_evals: bool = False


class MovetextError(ValueError):
    pass


def _split_glyph(tok: str) -> tuple[str, str | None]:
    for g in ACCURACY_GLYPHS + GOOD_GLYPHS:
        if tok.endswith(g):
            return tok[: -len(g)], g
    return tok, None


def parse_movetext(movetext: str) -> ParsedGame:
    game = ParsedGame()
    for m in _SCAN_RE.finditer(movetext):
        if m.group("comment") is not None:
            ev = _EVAL_RE.search(m.group("comment"))
            if ev and game.plies:
                game.plies[-1].cp = eval_to_cp(ev.group(1))
                game.has_evals = True
        elif m.group("tok") is not None:
            tok = m.group("tok")
            if tok in _RESULTS:
                game.result = tok
                continue
            san, glyph = _split_glyph(tok)
            game.plies.append(Ply(san=san, glyph=glyph))
    return game


# ---------------------------------------------------------------- tokenising

def tokenise_san(san: str, accuracy: str | None) -> list[str]:
    m = _SAN_RE.match(san)
    if not m:
        raise MovetextError(f"unparseable SAN: {san!r}")
    out: list[str] = []
    if accuracy:
        out.append(f"[{accuracy}]")
    if m.group("chk"):
        out.append(f"[{m.group('chk')}]")
    base = m.group("base")
    if "x" in base:
        out.append("[x]")
        base = base.replace("x", "")
    if m.group("promo"):
        out.append(f"[{m.group('promo')}]")
    out.append(f"[{base}]")
    return out


def tokenise_game(
    game: ParsedGame,
    accuracy_source: str = "glyph",   # 'glyph' | 'computed' | 'none'
) -> tuple[list[str], list[int], int]:
    """Returns (tokens, cp_list, n_disagreements).

    cp_list uses MATE_CP+1 sentinel for missing evals (final mating move).
    Disagreements = plies where inline glyph and computed label differ
    (only counted when evals are present).
    """
    tokens: list[str] = []
    cps: list[int] = []
    disagreements = 0
    cp_before = START_CP
    for i, ply in enumerate(game.plies):
        white_moved = i % 2 == 0
        computed = None
        if game.has_evals and ply.cp is not None:
            computed = classify(cp_before, ply.cp, white_moved)
        inline = ply.glyph if ply.glyph in ACCURACY_GLYPHS else None
        if game.has_evals and ply.cp is not None and inline != computed:
            disagreements += 1
        if accuracy_source == "glyph":
            acc = inline
        elif accuracy_source == "computed":
            acc = computed
        else:
            acc = None
        tokens.extend(tokenise_san(ply.san, acc))
        cps.append(ply.cp if ply.cp is not None else MATE_CP + 1)
        if ply.cp is not None:
            cp_before = ply.cp
    return tokens, cps, disagreements
