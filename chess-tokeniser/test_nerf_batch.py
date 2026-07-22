"""Tests for the bare-history batch transform (nerf_batch.py).

The load-bearing claims these guard:
  * a game is cut into segments exactly at annotated moves, and each
    possible "active segment" choice reproduces the hand-worked training
    rows: bare history, the nerf only as final-history/target token, loss
    only on that segment plus the end group;
  * a move whose nerf was stripped is NEVER graded as if it were clean
    (deletion-seam masking — the "clean branch poisoning" guard);
  * framing targets (<bos>, <elo-*>) are never graded; the END GROUP (the
    colour-differentiated game-end token + <eos>) is graded in EVERY row;
  * multi-token moves ([nerf][x][core]...) stay whole: the segment runs to
    the core, and mid-move continuation tokens are graded with the nerf in
    context — and never without it;
  * batching is STRICT one-game-per-row: every row is one complete game,
    <bos> at column 0, right-padded (x pad = <eos>, y pad = -1) to a
    length-bucket cap — games can never attend to another game, and a
    corrupt/stale game index RAISES instead of silently mixing games;
  * the game index round-trips: pack-written .idx.npy is loaded verbatim,
    a missing/stale index is rebuilt by the <bos> scan to identical
    offsets, and the rebuilt index is re-cached to disk;
  * bucket caps, per-cap row counts, and token-mass bucket sampling follow
    the documented formulas, so torch.compile sees a fixed shape family;
  * the copy of nerf_batch.py embedded in train_chess_hf.py (which runs as
    a single file on HF Jobs) is byte-identical to the module tested here.
"""
import os
import re
import tempfile

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
WRSN, DRW = I("<white-resign>"), I("<draw>")

# The worked example: 1. e4 c5? 2. Nf3 Ng3?? — fully annotated as packed.
GAME = [BOS, EW, EB, E4, M, C5, NF3, BL, NG3, EOS]
G = np.asarray(GAME, dtype=np.int64)


def _game(n_moves, nerf_before=()):
    """Synthetic whole game of n_moves cores (+ a nerf before chosen moves)."""
    pool = [E4, C5, NF3, NG3]
    body = []
    for k in range(n_moves):
        if k in nerf_before:
            body.append(M)
        body.append(pool[k % 4])
    return [BOS, EW, EB, *body, EOS]


def _corpus(*games):
    """Concatenate whole games -> (flat int64 array, uint64 offsets+sentinel)."""
    flat = np.asarray([t for g in games for t in g], dtype=np.int64)
    off = np.cumsum([0] + [len(g) for g in games]).astype(np.uint64)
    return flat, off


def test_plan_segments():
    spans = nb.plan(G, SPEC)
    assert len(spans) == 1  # one whole game -> exactly one span
    # cut after each annotated move's core; one tail segment through <eos>
    assert spans[0] == [(0, 5, 4), (6, 8, 7), (9, 9, None)]


def test_worked_example_rows():
    """Each active-segment choice yields exactly the hand-derived row.
    Rows are variable-length now (len(kept)-1); the end group (<eos> here)
    is graded whichever segment is active."""
    (segs,) = nb.plan(G, SPEC)

    # -- active = first segment: grades  ->e4  ->[?]  ->c5  + end group
    x, y = nb.materialize(G, SPEC, [segs[0]])
    assert x.tolist() == [BOS, EW, EB, E4, M, C5, NF3, NG3]  # [??] stripped
    assert y.tolist() == [-1, -1, E4, M, C5, -1, -1, EOS]

    # -- active = second segment: grades  ->Nf3  ->[??]  ->Ng3  + end group
    x, y = nb.materialize(G, SPEC, [segs[1]])
    assert x.tolist() == [BOS, EW, EB, E4, C5, NF3, BL, NG3]  # [?] stripped
    assert y.tolist() == [-1, -1, -1, -1, NF3, BL, NG3, EOS]
    # the seam guard: c5 (whose [?] was stripped) is NOT graded as a clean move

    # -- active = tail: both nerfs stripped; only the end group is graded
    x, y = nb.materialize(G, SPEC, [segs[2]])
    assert x.tolist() == [BOS, EW, EB, E4, C5, NF3, NG3]
    assert y.tolist() == [-1, -1, -1, -1, -1, -1, EOS]


def test_multitoken_move_stays_whole():
    """[nerf][x][core]: the segment runs to the core; continuation tokens are
    graded with the nerf in context, and never as bare-history clean moves."""
    g = np.asarray([BOS, EW, EB, E4, BL, X, NF3, EOS], dtype=np.int64)
    (segs,) = nb.plan(g, SPEC)
    assert segs == [(0, 6, 4), (7, 7, None)]

    # annotated segment active: nerf, modifier, and core all graded
    x, y = nb.materialize(g, SPEC, [segs[0]])
    assert x.tolist() == [BOS, EW, EB, E4, BL, X, NF3]
    assert y.tolist() == [-1, -1, E4, BL, X, NF3, EOS]

    # tail active: blunder stripped -> its x/Nf3 must NOT be graded bare
    x, y = nb.materialize(g, SPEC, [segs[1]])
    assert x.tolist() == [BOS, EW, EB, E4, X, NF3]
    assert y.tolist() == [-1, -1, -1, -1, -1, EOS]


def test_malformed_tail():
    """A nerf whose move never completes (corrupt/truncated data): stripped,
    and nothing at or past it is graded (it would otherwise look clean)."""
    w = np.asarray([BOS, EW, EB, E4, BL], dtype=np.int64)
    spans = nb.plan(w, SPEC)
    assert spans == [[(0, 3, None)]]
    x, y = nb.materialize(w, SPEC, [spans[0][0]])
    assert x.tolist() == [BOS, EW, EB]
    assert y.tolist() == [-1, -1, E4]  # no <eos> in this fragment -> no end group


def test_end_group_always_graded():
    """The game-end token and <eos> are graded in EVERY row — including rows
    whose active segment is elsewhere and rows whose adjacent nerf was
    stripped — but never across a deletion seam into the move itself."""
    g = np.asarray([BOS, EW, EB, E4, M, C5, WRSN, EOS], dtype=np.int64)
    (segs,) = nb.plan(g, SPEC)
    assert segs == [(0, 5, 4), (6, 7, None)]

    # annotated segment active: its targets AND the end group are graded
    x, y = nb.materialize(g, SPEC, [segs[0]])
    assert x.tolist() == [BOS, EW, EB, E4, M, C5, WRSN]
    assert y.tolist() == [-1, -1, E4, M, C5, WRSN, EOS]

    # tail active: [?] stripped -> c5 seam-masked, end group still graded
    x, y = nb.materialize(g, SPEC, [segs[1]])
    assert x.tolist() == [BOS, EW, EB, E4, C5, WRSN]
    assert y.tolist() == [-1, -1, -1, -1, WRSN, EOS]

    # same for <draw>
    g = np.asarray([BOS, EW, EB, E4, DRW, EOS], dtype=np.int64)
    (segs,) = nb.plan(g, SPEC)
    x, y = nb.materialize(g, SPEC, [segs[0]])
    assert y.tolist() == [-1, -1, E4, DRW, EOS]


def test_no_nerf_game_is_near_vanilla():
    g = np.asarray(_game(4), dtype=np.int64)
    spans = nb.plan(g, SPEC)
    assert [len(s) for s in spans] == [1]
    x, y = nb.materialize(g, SPEC, [spans[0][0]])
    assert x.tolist() == _game(4)[:-1]
    # everything graded except the framing targets
    assert y.tolist() == [-1, -1, E4, C5, NF3, NG3, EOS]


def test_uniform_segment_choice():
    """Each of a game's A+1 segments is active equally often, so every
    position is graded at the same 1/(A+1) rate across draws."""
    rng = np.random.default_rng(0)
    counts = {"seg0": 0, "seg1": 0, "tail": 0}
    n = 3000
    for _ in range(n):
        x, y = nb.build_row(G, SPEC, rng)
        if len(x) == 7:
            counts["tail"] += 1  # both nerfs stripped
        elif y[2] == E4:
            counts["seg0"] += 1
        elif y[4] == NF3:
            counts["seg1"] += 1
        else:
            raise AssertionError(f"row matches no known signature: {y.tolist()}")
    for k, c in counts.items():
        assert 0.28 < c / n < 0.39, (k, c / n, counts)


def test_build_row_rejects_non_games():
    """The isolation guard: a row that is not exactly one whole game raises
    (this is what stops a bad index from letting games attend across)."""
    rng = np.random.default_rng(0)
    bad = [
        np.asarray(GAME[1:], dtype=np.int64),        # no leading <bos>
        np.asarray(GAME[:-1], dtype=np.int64),       # no trailing <eos>
        np.asarray(GAME + GAME, dtype=np.int64),     # two games in one slice
        np.asarray([BOS], dtype=np.int64),           # too short
    ]
    for g in bad:
        try:
            nb.build_row(g, SPEC, rng)
        except ValueError:
            continue
        raise AssertionError(f"guard accepted a non-game slice: {g.tolist()}")


def test_rows_for_cap_table():
    """The full profile's fixed shape family (batch 64 x block 1024)."""
    want = {64: 512, 128: 512, 256: 256, 384: 171, 512: 128, 768: 85, 1024: 64}
    for cap, rows in want.items():
        assert nb.rows_for_cap(cap, 1024, 64) == rows, cap
    # smoke profile (batch 32 x block 512): clamp bites exactly at cap 64
    assert nb.rows_for_cap(64, 512, 32) == 256
    assert nb.rows_for_cap(512, 512, 32) == 32


def test_bucket_assignment():
    """A game of L tokens (L-1 input slots) lands in the smallest cap >= L-1;
    caps are filtered to block_size; too-long games are skipped+counted."""
    games = [_game(61), _game(62), _game(250), _game(600)]  # L = 65, 66, 254, 604
    _, off = _corpus(*games)
    bk = nb.Buckets(off, 512, 4)
    assert bk.caps == [64, 128, 256, 384, 512]  # filtered for block 512
    assert list(bk.game_ids[0]) == [0]          # L=65 -> need 64 -> cap 64
    assert list(bk.game_ids[1]) == [1]          # L=66 -> need 65 -> cap 128
    assert list(bk.game_ids[2]) == [2]          # L=254 -> cap 256
    assert bk.n_skipped == 1                    # L=604 > 513: skipped
    # sampling weight = token mass per bucket
    lengths = np.asarray([65, 66, 254], dtype=np.float64)
    np.testing.assert_allclose(bk.probs[:3], lengths / lengths.sum())
    assert bk.probs[3] == bk.probs[4] == 0.0


def test_pad_batch():
    X, Y = nb.pad_batch([(np.asarray([BOS, E4]), np.asarray([-1, E4]))], 4, EOS)
    assert X.tolist() == [[BOS, E4, EOS, EOS]]
    assert Y.tolist() == [[-1, E4, -1, -1]]
    try:
        nb.pad_batch([(np.zeros(5, dtype=np.int64), np.zeros(5, dtype=np.int64))], 4, EOS)
    except ValueError:
        pass
    else:
        raise AssertionError("pad_batch accepted a row longer than the cap")


def test_offsets_index_roundtrip():
    """pack-written index loads verbatim; scan fallback rebuilds identical
    offsets and re-caches them; a stale index is ignored and rescanned."""
    games = [_game(3), _game(7, nerf_before=(2,)), _game(5)]
    flat, off = _corpus(*games)
    # in-memory arrays always scan — and the scan must equal pack's offsets
    np.testing.assert_array_equal(nb.game_offsets(flat, SPEC), off)

    with tempfile.TemporaryDirectory() as td:
        bin_path = os.path.join(td, "train.bin")
        idx_path = os.path.join(td, "train.idx.npy")
        np.asarray(flat, dtype=np.uint16).tofile(bin_path)
        mm = np.memmap(bin_path, dtype=np.uint16, mode="r")

        # 1) pack-written index is loaded verbatim
        np.save(idx_path, off)
        nb._OFFSETS_CACHE.clear()
        np.testing.assert_array_equal(nb.game_offsets(mm, SPEC), off)

        # 2) missing index: scan rebuilds identical offsets and re-caches
        os.remove(idx_path)
        nb._OFFSETS_CACHE.clear()
        np.testing.assert_array_equal(nb.game_offsets(mm, SPEC), off)
        np.testing.assert_array_equal(np.load(idx_path), off)

        # 3) stale index (wrong sentinel): ignored, rescanned
        np.save(idx_path, np.asarray([0, len(flat) + 5], dtype=np.uint64))
        nb._OFFSETS_CACHE.clear()
        np.testing.assert_array_equal(nb.game_offsets(mm, SPEC), off)
        del mm  # Windows: release the file handle before tempdir cleanup
    nb._OFFSETS_CACHE.clear()


def test_poisoned_index_raises():
    """An index that validates but cuts mid-game must raise via the row
    guard — never silently train on a slice spanning two games."""
    flat, _ = _corpus(_game(3), _game(3))
    with tempfile.TemporaryDirectory() as td:
        bin_path = os.path.join(td, "train.bin")
        np.asarray(flat, dtype=np.uint16).tofile(bin_path)
        poisoned = np.asarray([0, 4, len(flat)], dtype=np.uint64)  # true cut is at 7
        np.save(os.path.join(td, "train.idx.npy"), poisoned)
        nb._OFFSETS_CACHE.clear()
        nb._BUCKETS_CACHE.clear()
        mm = np.memmap(bin_path, dtype=np.uint16, mode="r")
        rng = np.random.default_rng(0)
        try:
            nb.sample_batch_np(mm, 512, 4, SPEC, rng)
        except ValueError:
            pass
        else:
            raise AssertionError("sample_batch_np accepted a mid-game index")
        del mm  # Windows: release the file handle before tempdir cleanup
    nb._OFFSETS_CACHE.clear()
    nb._BUCKETS_CACHE.clear()


def test_sample_batch_np_end_to_end():
    """Strict one-game-per-row over a mixed-length corpus: fixed (rows, cap)
    shapes, <bos> only at column 0, end group graded, padding fully masked,
    and reproducible under a seed."""
    games = [_game(10), _game(20, nerf_before=(3,)), _game(40), _game(90),
             _game(58), _game(130, nerf_before=(5, 60))]
    flat, _ = _corpus(*games)
    block_size, batch_size = 512, 4

    for seed in (0, 1, 2):
        rng = np.random.default_rng(seed)
        X, Y = nb.sample_batch_np(flat, block_size, batch_size, SPEC, rng)
        cap = X.shape[1]
        assert cap in (64, 128, 256, 384, 512)
        assert X.shape[0] == nb.rows_for_cap(cap, block_size, batch_size)
        for x, y in zip(X, Y):
            # exactly one game per row, <bos> at column 0 only
            assert x[0] == BOS and np.count_nonzero(x == BOS) == 1
            # x holds no <eos> inside the game (it is only ever a target), so
            # the first <eos> in x marks the padding; everything from there on
            # is loss-masked, and the last real target is the graded <eos>
            pads = np.flatnonzero(x == EOS)
            pad0 = int(pads[0]) if len(pads) else len(x)
            assert np.all(pads == np.arange(pad0, len(x)))  # padding is one tail block
            assert y[pad0 - 1] == EOS                       # end group graded in every row
            assert np.all(y[pad0:] == -1)                   # padding contributes no loss
            graded = y[y != -1]
            assert len(graded) > 0 and all(0 <= t < V.size for t in graded)

    a = nb.sample_batch_np(flat, block_size, batch_size, SPEC, np.random.default_rng(7))
    b = nb.sample_batch_np(flat, block_size, batch_size, SPEC, np.random.default_rng(7))
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])


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
