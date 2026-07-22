"""Bare-history batch sampling for nanoGPT (the nerf-token training rule).

pack.py stores games FULLY annotated: every nerf token (<inaccuracy> /
<mistake> / <blunder>) sits immediately before the move it labels, and the
token order within a move is [nerf][+|#][x][=Q..][core] — the core token
ends the move:

    <bos> <eloW> <eloB> e4 <mistake> c5 Nf3 <blunder> Ng3 <eos>

At inference the model's context is BARE — the human's moves are never
annotated (there is no engine to judge them), and the harness keeps its own
emitted nerfs out of the history it feeds back (js/app.js). Training must
match:

    a nerf token may appear in the context only while the move it labels is
    being predicted (as the final history token, or mid-move); every later
    prediction must see the game with that nerf removed.

One packed row cannot serve every position at once — a token is either in
the row or not — so each sampled window becomes ONE consistent view:

  1. nerf tokens cut each game into SEGMENTS. A segment runs from just
     after the previous annotated move through the end of its own annotated
     move; one tail segment covers everything after the last annotated move
     (through <eos>).
  2. per game-span in the window, ONE segment is chosen uniformly, and its
     targets are the graded ones;
  3. every nerf except the chosen segments' own is deleted and the stream
     closes up (the window is over-read so the row stays block_size long);
  4. targets outside chosen segments, targets across a deletion seam (a
     move whose nerf was stripped must NOT be graded as if it were clean),
     and framing targets the harness always supplies (<bos>, <elo-*>) are
     set to -1, which nanoGPT's cross_entropy(ignore_index=-1) skips.
     <eos> IS graded — end-of-game prediction is real signal.

Known approximations (documented in TRAINING.md): (a) choosing uniformly
per game weights each position by 1/(segments in its game), so heavily
annotated games contribute slightly less per token than clean ones; (b) an
annotated move cut off by the window edge is stripped and excluded from
grading, and a move whose nerf fell just before the window start is
indistinguishable from a clean move (both bounded to at most one move per
row edge). Grading density is roughly 10-25% of a vanilla batch — budget
more iterations, and never compare losses against a run without this
transform.

Pure numpy except sample_batch's torch conversion (lazy import), so the
unit tests (test_nerf_batch.py) run without torch installed.
"""
from __future__ import annotations

import os
import pickle

import numpy as np

# a move is at most [nerf][+|#][x][=Q..][core]: the core must appear within
# this many tokens after its nerf, or the group is treated as cut/malformed
MAX_GROUP_SPAN = 4

N_MODIFIERS = 7  # ids core_count .. core_count+6, per the frozen vocab tail


class Spec:
    """Vocabulary facts the transform needs, from meta.pkl mappings."""

    def __init__(self, stoi: dict, core_count: int):
        try:
            nerf_ids = [stoi[t] for t in ("<inaccuracy>", "<mistake>", "<blunder>")]
            self.bos_id = stoi["<bos>"]
            unpredicted = [self.bos_id] + [i for t, i in stoi.items() if t.startswith("<elo-")]
        except KeyError as e:
            raise KeyError(f"vocab is missing structural token {e} — wrong meta.pkl?") from None
        self.core_count = int(core_count)
        if sorted(nerf_ids) != [self.core_count + N_MODIFIERS + k for k in range(3)]:
            raise ValueError(
                "vocab layout drift: nerf tokens are not directly after the "
                f"{N_MODIFIERS} modifiers (core_count={core_count}, nerf ids={sorted(nerf_ids)})"
            )
        self._nerf_arr = np.asarray(sorted(nerf_ids), dtype=np.int64)
        self._unpred_arr = np.asarray(sorted(unpredicted), dtype=np.int64)


_SPEC_CACHE: dict[str, Spec] = {}


def load_spec(meta_path: str) -> Spec:
    spec = _SPEC_CACHE.get(meta_path)
    if spec is None:
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        if "core_count" not in meta:
            raise KeyError("meta.pkl has no core_count — re-run pack.py (old format)")
        spec = Spec(meta["stoi"], meta["core_count"])
        _SPEC_CACHE[meta_path] = spec
    return spec


def _group_end(w: np.ndarray, nerf_pos: int, stop: int, spec: Spec):
    """Index of the core token ending the move annotated by the nerf at
    `nerf_pos`, or None when the move is cut off by `stop` or malformed
    (anything but modifier tokens before the core)."""
    for q in range(nerf_pos + 1, min(stop, nerf_pos + 1 + MAX_GROUP_SPAN)):
        t = int(w[q])
        if t < spec.core_count:
            return q
        if not (spec.core_count <= t < spec.core_count + N_MODIFIERS):
            return None
    return None


def plan(w: np.ndarray, spec: Spec) -> list[list[tuple]]:
    """Cut a window into per-game-span segment lists.

    Returns a list of spans; each span is a non-empty list of segments
    (t_lo, t_hi, nerf_pos): an inclusive range of TARGET token indices into
    `w`, plus the segment's own nerf position (None for the tail segment).
    """
    n = len(w)
    is_nerf = np.isin(w, spec._nerf_arr)
    starts = [0] + [int(b) for b in np.flatnonzero(w == spec.bos_id) if b != 0]
    spans = []
    for si, s in enumerate(starts):
        e = starts[si + 1] if si + 1 < len(starts) else n
        segs = []
        prev_end = s      # first segment starts at the span's first token
        tail_end = e - 1  # pulled in if an annotated move is cut/malformed
        for p in (np.flatnonzero(is_nerf[s:e]) + s):
            p = int(p)
            if p < prev_end:
                continue  # nerf inside a previous (malformed) group — deleted, never active
            g = _group_end(w, p, e, spec)
            if g is None:
                # cut off by the window / malformed: nothing at or past this
                # nerf may be graded (its move would look clean once stripped)
                tail_end = p - 1
                break
            segs.append((prev_end, g, p))
            prev_end = g + 1
        if prev_end <= tail_end:
            segs.append((prev_end, tail_end, None))
        if segs:
            spans.append(segs)
    return spans


def choose(spans: list[list[tuple]], rng) -> list[tuple]:
    """One segment per span, uniformly — every target position then has the
    same 1/len(span) chance of being graded across draws."""
    return [span[int(rng.integers(len(span)))] for span in spans]


def materialize(w: np.ndarray, spec: Spec, chosen: list[tuple], block_size: int):
    """Build one (x, y) training row from a window and its chosen segments.

    Keeps only the chosen segments' nerfs, deletes every other nerf (the
    stream closes up), and grades exactly the chosen segments' surviving
    targets: y[p] = -1 unless the target sits in a chosen segment, was
    originally adjacent to its left neighbour (no deletion seam), and is
    not a framing token (<bos>/<elo-*>) the harness always supplies.
    """
    keep = ~np.isin(w, spec._nerf_arr)
    active_t = np.zeros(len(w), dtype=bool)
    for (t0, t1, nerf_pos) in chosen:
        active_t[t0:t1 + 1] = True
        if nerf_pos is not None:
            keep[nerf_pos] = True
    old_of_new = np.flatnonzero(keep)
    if len(old_of_new) < block_size + 1:
        raise ValueError(
            f"window too short after nerf deletion ({len(old_of_new)} < {block_size + 1}) "
            "— over-read a larger slice"
        )
    new = w[old_of_new]
    tgt_old = old_of_new[1:block_size + 1]
    adjacent = tgt_old == old_of_new[:block_size] + 1
    targets = new[1:block_size + 1]
    graded = active_t[tgt_old] & adjacent & ~np.isin(targets, spec._unpred_arr)
    x = new[:block_size].astype(np.int64)
    y = np.where(graded, targets, -1).astype(np.int64)
    return x, y


def build_row(w: np.ndarray, spec: Spec, rng, block_size: int):
    return materialize(w, spec, choose(plan(w, spec), rng), block_size)


_RNG = None


def _rng():
    global _RNG
    if _RNG is None:  # RANK offset keeps DDP ranks from drawing identical batches
        _RNG = np.random.default_rng(1234 + int(os.environ.get("RANK", "0")))
    return _RNG


def sample_batch(data, block_size: int, batch_size: int, meta_path: str):
    """Drop-in replacement for nanoGPT get_batch's window assembly.

    `data` is the uint16 memmap of train.bin/val.bin. Returns CPU torch
    tensors (x, y) of shape (batch_size, block_size); y is -1 wherever no
    loss applies (nanoGPT's cross_entropy uses ignore_index=-1).
    """
    import torch  # lazy: the unit tests run torch-free

    spec = load_spec(meta_path)
    rng = _rng()
    # Over-read 2x: nerfs are at most every other token (each move has a
    # core), so deletion can never eat the row below block_size+1 tokens.
    slack = 2 * (block_size + 1)
    if len(data) <= slack:
        raise ValueError(f"dataset too small ({len(data)} tokens) for block_size {block_size}")
    xs, ys = [], []
    for i in rng.integers(0, len(data) - slack, size=batch_size):
        w = np.asarray(data[int(i):int(i) + slack], dtype=np.int64)
        x, y = build_row(w, spec, rng, block_size)
        xs.append(x)
        ys.append(y)
    return torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ys))
