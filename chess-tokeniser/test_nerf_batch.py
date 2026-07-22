"""Tests for the bare-history batch transform (nerf_batch.py).

The load-bearing claims these guard:
  * a window is cut into segments exactly at annotated moves, and each
    possible "active segment" choice reproduces the hand-worked training
    rows: bare history, the nerf only as final-history/target token, loss
    only on that segment (the worked example from the design discussion);
  * a move whose nerf was stripped is NEVER graded as if it were clean
    (deletion-seam masking — the "clean branch poisoning" guard);
  * framing targets (<bos>, <elo-*>) are never graded; <eos> is;
  * multi-token moves ([nerf][x][core]...) stay whole: the segment runs to
    the core, and mid-move continuation tokens are graded with the nerf in
    context — and never without it;
  * the per-span segment choice is uniform, and windows without nerfs
    degrade to (near-)vanilla next-token batches;
  * the copy of nerf_batch.py embedded in train_chess_hf.py (which runs as
    a single file on HF Jobs) is byte-identical to the module tested here.
"""
import re

import numpy as np

import nerf_batch as nb
import vocab as vb

V = vb.get_vocab()
SPEC = nb.Spec(V.stoi, V.core_count)


def I(*names):  # noqa: E743 — token name(s) -> id(s)
    ids = [V.stoi[n] for n in names]
    return ids[0] if len(ids) == 1 else ids


BOS, EOS = I("<bos>"), I("<eos>")
EW, EB = I("<elo-1800>"), I("<elo-1600>")
M, BL = I("<mistake>"), I("<blunder>")
E4, C5, NF3, NG3, X = I("e4", "c5", "Nf3", "Ng3", "x")

# The worked example: 1. e4 c5? 2. Nf3 Ng3?? — fully annotated as packed.
GAME = [BOS, EW, EB, E4, M, C5, NF3, BL, NG3, EOS]
W = np.asarray(GAME + GAME, dtype=np.int64)  # two copies: window crosses a game boundary
B = 9  # block_size: one full game per row


def test_plan_segments():
    spans = nb.plan(W, SPEC)
    assert len(spans) == 2
    # cut after each annotated move's core; one tail segment through <eos>
    assert spans[0] == [(0, 5, 4), (6, 8, 7), (9, 9, None)]
    assert spans[1] == [(10, 15, 14), (16, 18, 17), (19, 19, None)]


def test_worked_example_rows():
    """Each active-segment choice yields exactly the hand-derived row."""
    spans = nb.plan(W, SPEC)
    tail2 = spans[1][2]  # keep game 2 on its (inert) tail choice throughout

    # -- active = first segment: grades  ->e4  ->[?]  ->c5  (instances 1-3)
    x, y = nb.materialize(W, SPEC, [spans[0][0], tail2], B)
    assert x.tolist() == [BOS, EW, EB, E4, M, C5, NF3, NG3, EOS]  # [??] stripped
    assert y.tolist() == [-1, -1, E4, M, C5, -1, -1, -1, -1]

    # -- active = second segment: grades  ->Nf3  ->[??]  ->Ng3  (instances 4-6)
    x, y = nb.materialize(W, SPEC, [spans[0][1], tail2], B)
    assert x.tolist() == [BOS, EW, EB, E4, C5, NF3, BL, NG3, EOS]  # [?] stripped
    assert y.tolist() == [-1, -1, -1, -1, NF3, BL, NG3, -1, -1]
    # the seam guard: c5 (whose [?] was stripped) is NOT graded as a clean move

    # -- active = tail: grades  -><eos>  (instance 7); both nerfs stripped
    x, y = nb.materialize(W, SPEC, [spans[0][2], spans[1][0]], B)
    assert x.tolist() == [BOS, EW, EB, E4, C5, NF3, NG3, EOS, BOS]
    assert y.tolist() == [-1, -1, -1, -1, -1, -1, EOS, -1, -1]
    # (game 2's first segment is active but its graded targets — its own
    #  e4/[?]/c5 — sit beyond this row's block, so only <eos> is graded;
    #  the trailing <bos> target is framing and stays masked)


def test_multitoken_move_stays_whole():
    """[nerf][x][core]: the segment runs to the core; continuation tokens are
    graded with the nerf in context, and never as bare-history clean moves."""
    game = [BOS, EW, EB, E4, BL, X, NF3, EOS]
    w = np.asarray(game + game, dtype=np.int64)
    b = 7
    spans = nb.plan(w, SPEC)
    assert spans[0] == [(0, 6, 4), (7, 7, None)]

    # annotated segment active: nerf, modifier, and core all graded
    x, y = nb.materialize(w, SPEC, [spans[0][0], spans[1][1]], b)
    assert x.tolist() == [BOS, EW, EB, E4, BL, X, NF3]
    assert y.tolist() == [-1, -1, E4, BL, X, NF3, -1]

    # tail active: blunder stripped -> its x/Nf3 must NOT be graded bare
    x, y = nb.materialize(w, SPEC, [spans[0][1], spans[1][1]], b)
    assert x.tolist() == [BOS, EW, EB, E4, X, NF3, EOS]
    assert y.tolist() == [-1, -1, -1, -1, -1, EOS, -1]


def test_window_cut_annotated_move():
    """A nerf whose move is cut off by the window edge: stripped, and nothing
    at or past it is graded (it would otherwise look clean)."""
    w = np.asarray([BOS, EW, EB, E4, BL], dtype=np.int64)  # window ends mid-group
    spans = nb.plan(w, SPEC)
    assert spans == [[(0, 3, None)]]
    x, y = nb.materialize(w, SPEC, [spans[0][0]], 3)
    assert x.tolist() == [BOS, EW, EB]
    assert y.tolist() == [-1, -1, E4]


def test_window_starts_midgame():
    """No leading <bos>: the partial span still segments and grades."""
    w = W[3:]  # starts at e4
    spans = nb.plan(w, SPEC)
    assert spans[0] == [(0, 2, 1), (3, 5, 4), (6, 6, None)]


def test_no_nerf_window_is_near_vanilla():
    game = [BOS, EW, EB, E4, C5, NF3, NG3, EOS]
    w = np.asarray(game + game, dtype=np.int64)
    spans = nb.plan(w, SPEC)
    assert [len(s) for s in spans] == [1, 1]
    x, y = nb.materialize(w, SPEC, [spans[0][0], spans[1][0]], 7)
    assert x.tolist() == game[:7]
    # everything graded except the framing targets
    assert y.tolist() == [-1, -1, E4, C5, NF3, NG3, EOS]


def test_uniform_segment_choice():
    """Each of a game's A+1 segments is active equally often, so every
    position is graded at the same 1/(A+1) rate across draws."""
    rng = np.random.default_rng(0)
    counts = {"seg0": 0, "seg1": 0, "tail": 0}
    n = 3000
    for _ in range(n):
        _, y = nb.build_row(W, SPEC, rng, B)
        if y[2] == E4:
            counts["seg0"] += 1
        elif y[4] == NF3:
            counts["seg1"] += 1
        elif y[6] == EOS:
            counts["tail"] += 1
        else:
            raise AssertionError(f"row matches no known signature: {y.tolist()}")
    for k, c in counts.items():
        assert 0.28 < c / n < 0.39, (k, c / n, counts)


def test_row_shapes_and_dtypes():
    rng = np.random.default_rng(1)
    x, y = nb.build_row(W, SPEC, rng, B)
    assert x.shape == (B,) and y.shape == (B,)
    assert x.dtype == np.int64 and y.dtype == np.int64
    graded = y[y != -1]
    assert len(graded) > 0 and all(0 <= t < V.size for t in graded)


def test_embedded_copy_in_sync():
    """train_chess_hf.py runs alone on HF Jobs, so it embeds nerf_batch.py.
    The embedded copy must be byte-identical to the module under test."""
    with open("train_chess_hf.py", encoding="utf-8") as f:
        src = f.read()
    m = re.search(r"NERF_BATCH_SRC = r'''\n(.*?)'''", src, re.S)
    assert m, "train_chess_hf.py has no embedded NERF_BATCH_SRC block"
    with open("nerf_batch.py", encoding="utf-8") as f:
        module = f.read()
    assert m.group(1) == module, (
        "embedded nerf_batch.py in train_chess_hf.py is out of sync — "
        "copy the module file into the NERF_BATCH_SRC block verbatim"
    )


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"{name}: OK")
