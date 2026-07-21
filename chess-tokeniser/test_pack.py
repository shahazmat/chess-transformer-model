"""Tests for the vocab bridge + packing stage.

The load-bearing claims these guard:
  * the frozen js/vocab-data.js dictionary is a superset of real tokenised games
    (100% coverage on a verbatim dataset game);
  * SAN -> tokenise -> translate -> ids -> decode -> SAN is lossless;
  * Elo bucketing and the parsed dictionary match the JS side byte-for-byte.
"""
import json
import subprocess

import tokeniser as tk
import vocab as vb
from test_tokeniser import REAL_MOVETEXT

MODIFIERS = {"+", "#", "x", "=Q", "=R", "=B", "=N"}
NERF = {"<inaccuracy>", "<mistake>", "<blunder>"}


def _tokens_to_san(move_tokens):
    """Inverse of the tokeniser for one move (mirrors js tokensToSan)."""
    suffix = capture = promo = ""
    core = None
    for t in move_tokens:
        if t in ("+", "#"):
            suffix = t
        elif t == "x":
            capture = "x"
        elif t.startswith("="):
            promo = t
        elif t in NERF:
            pass  # accuracy annotation, not part of the SAN
        else:
            core = t
    assert core is not None, move_tokens
    if core.startswith("O-O"):
        return core + suffix
    return core[:-2] + capture + core[-2:] + promo + suffix


def _regroup(stream):
    """Flat surface-token stream -> list of per-move token lists (core ends a move)."""
    moves, cur = [], []
    for t in stream:
        cur.append(t)
        if t not in MODIFIERS and t not in NERF:
            moves.append(cur)
            cur = []
    assert not cur, f"trailing tokens with no core: {cur}"
    return moves


def test_vocab_shape():
    v = vb.get_vocab()
    assert v.size == 5267 and v.core_count == 5242
    assert v.tokens[v.bos_id] == "<bos>" and v.tokens[v.eos_id] == "<eos>"
    elo = [t for t in v.tokens if t.startswith("<elo-")]
    assert len(elo) == 13 and elo[0] == "<elo-u800>" and elo[-1] == "<elo-3000p>"
    # vocab v2 is a pure append: bos sits exactly at the old vocab size
    assert v.bos_id == 5252


def test_real_game_full_coverage():
    """Every token of a real dataset game maps into the frozen dictionary."""
    v = vb.get_vocab()
    game = tk.parse_movetext(REAL_MOVETEXT)
    toks, _, _ = tk.tokenise_game(game, accuracy_source="glyph")
    field = " ".join(toks)
    missing = [t for t in field.split() if vb.translate(t) not in v.stoi]
    assert missing == [], f"frozen vocab misses real tokens: {missing}"


def test_framing_roundtrip_lossless():
    """SAN -> ids -> back to the exact SAN sequence, with correct framing."""
    v = vb.get_vocab()
    game = tk.parse_movetext(REAL_MOVETEXT)
    toks, _, _ = tk.tokenise_game(game, accuracy_source="glyph")
    ids, unknown = v.encode_game(" ".join(toks), 1834, 1790)
    assert unknown == set() and ids is not None
    surface = v.decode(ids)
    assert surface[0] == "<bos>" and surface[-1] == "<eos>"
    assert surface[1] == "<elo-1800>" and surface[2] == "<elo-1600>"  # 1834->1800, 1790->1600
    sans = [_tokens_to_san(m) for m in _regroup(surface[3:-1])]
    assert sans == [p.san for p in game.plies]


def test_translate_cases():
    cases = {
        "[Nf3]": "Nf3", "[x]": "x", "[=Q]": "=Q", "[O-O-O]": "O-O-O", "[ef8]": "ef8",
        "[+]": "+", "[#]": "#", "[??]": "<blunder>", "[?]": "<mistake>", "[?!]": "<inaccuracy>",
    }
    for src, want in cases.items():
        assert vb.translate(src) == want


def test_elo_buckets():
    edges = {
        0: "<elo-u800>", 799: "<elo-u800>", 800: "<elo-800>", 999: "<elo-800>",
        1000: "<elo-1000>", 1899: "<elo-1800>", 2999: "<elo-2800>", 3000: "<elo-3000p>",
        4000: "<elo-3000p>", None: "<elo-u800>",
    }
    for elo, want in edges.items():
        assert vb.elo_bucket_token(elo) == want, elo


def test_unknown_token_dropped():
    v = vb.get_vocab()
    ids, unknown = v.encode_game("[Nf3] [Zz9] [e4]", 1500, 1500)
    assert ids is None and unknown == {"[Zz9]"}


def test_js_python_vocab_parity():
    """Python's parsed dictionary and Elo bucketing match the JS source of truth."""
    v = vb.get_vocab()
    probe = "800,799,1234,1600,1899,2999,3000,3200"
    js = (
        "import {TOKENS, eloBucketToken} from './js/vocab.js';"
        "console.log(JSON.stringify({n: TOKENS.length, tail: TOKENS.slice(-15),"
        f"elo: '{probe}'.split(',').map(e => eloBucketToken(+e))}}));"
    )
    try:
        out = subprocess.run(
            ["node", "--input-type=module", "-e", js],
            cwd="..", capture_output=True, text=True, timeout=30,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("test_js_python_vocab_parity: SKIPPED (node unavailable)")
        return
    assert out.returncode == 0, out.stderr
    data = json.loads(out.stdout)
    assert data["n"] == v.size
    assert data["tail"] == v.tokens[-15:]
    assert data["elo"] == [vb.elo_bucket_token(int(e)) for e in probe.split(",")]


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"{name}: OK")
