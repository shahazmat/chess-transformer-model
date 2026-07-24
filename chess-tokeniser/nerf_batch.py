"""Bare-history batch sampling for nanoGPT (the nerf-token training rule).

pack.py stores games FULLY annotated: every nerf token (<inaccuracy> /
<mistake> / <blunder>) sits immediately before the move it labels, and the
token order within a move is [nerf][+|#][x][=Q..][core] — the core token
ends the move:

    <bos> <eloW> <eloB> e4 <mistake> c5 Nf3 <blunder> Ng3 <white-resign> <eos>

At inference the model's context is BARE — the human's moves are never
annotated (there is no engine to judge them), and the harness keeps its own
emitted nerfs out of the history it feeds back (js/app.js). Training must
match:

    a nerf token may appear in the context only while the move it labels is
    being predicted (as the final history token, or mid-move); every later
    prediction must see the game with that nerf removed.

Batching is STRICT one-game-per-row: every row is ONE complete game with
<bos> at position 0 (exactly the layout inference sees), right-padded to a
length-bucket cap. Games can never attend to other games — there is no other
game in the row, and the padding sits after the game where causal attention
cannot look; pad targets are -1, so padding contributes no loss and no
gradient. Each batch draws one bucket (probability = the bucket's share of
corpus tokens) and rows_for_cap(cap) games uniformly within it; row counts
are fixed per cap so torch.compile sees a small, bounded family of
(rows, cap) shapes, with tokens/step roughly level across buckets.

Game boundaries come from the <split>.idx.npy index pack.py writes next to
each bin (uint64 game-start offsets + a final total-length sentinel); when it
is missing or stale, a one-time chunked scan for <bos> rebuilds and re-caches
it. Every sampled slice is hard-checked to be exactly one whole game (<bos>
only at index 0, <eos> last) — a corrupt index raises instead of silently
mixing games.

Within its game the nerf rule works as before:

  1. nerf tokens cut the game into SEGMENTS. A segment runs from just
     after the previous annotated move through the end of its own annotated
     move; one tail segment covers everything after the last annotated move
     (through <eos>).
  2. ONE segment is chosen uniformly, and its targets are the graded ones;
  3. every nerf except the chosen segment's own is deleted and the stream
     closes up;
  4. targets outside the chosen segment, targets across a deletion seam (a
     move whose nerf was stripped must NOT be graded as if it were clean),
     and framing targets the harness always supplies (<bos>, <elo-*>) are
     set to -1, which nanoGPT's cross_entropy(ignore_index=-1) skips;
  5. the END GROUP — the final game-end token (<white/black-resign>,
     <white/black-flag>, <draw>, when present) and <eos> — is graded in
     EVERY row, whichever segment was chosen: how games end is exactly what
     the end tokens exist to learn, the prediction is never
     quality-conditioned, and its bare context matches inference;
  6. WINNER-ONLY grading (default ON; env WINNER_ONLY=0 disables): move
     targets played by the game's LOSER are additionally masked to -1.
     Lichess nerf annotations are win-probability based, so in an already
     lost position almost no move gets flagged — grading both sides would
     teach the model that hopeless flailing is clean play. The loser is
     read from the game's own tokens (loser_colour): the colour-
     differentiated end token names the loser; a bare-<eos> game containing
     '#' ended in mate, so the side that did NOT play the last move lost;
     <draw> and any other bare <eos> (timeout draws, abandoned, pre-v3
     shards) grade both sides. Loser moves always stay in the CONTEXT —
     the model must condition on opponent play, it just never imitates it —
     and the end group stays graded in every row regardless.

Known approximation (documented in TRAINING.md): choosing one segment
uniformly per game weights each position by 1/(segments in its game), so
heavily annotated games contribute slightly less per token than clean ones.
Grading density is a fraction of a vanilla batch — budget more iterations,
and never compare losses across batching schemes.

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

# Row-length bucket caps (multiples of 64). At table build the list is
# filtered to caps <= block_size (appending block_size itself if missing), so
# every profile keeps a small, fixed shape family for torch.compile.
BUCKET_CAPS = (64, 128, 256, 384, 512, 768, 1024)
MAX_ROWS_FACTOR = 8   # rows per batch never exceed MAX_ROWS_FACTOR * batch_size
SCAN_CHUNK = 1 << 24  # tokens per chunk of the fallback <bos> scan (~32 MB uint16)

# Winner-only grading (docstring item 6). Read once at import; WINNER_ONLY=0
# restores both-sides grading for comparison runs.
WINNER_ONLY = os.environ.get("WINNER_ONLY", "1") == "1"


class Spec:
    """Vocabulary facts the transform needs, from meta.pkl mappings."""

    def __init__(self, stoi: dict, core_count: int):
        try:
            nerf_ids = [stoi[t] for t in ("<inaccuracy>", "<mistake>", "<blunder>")]
            self.bos_id = stoi["<bos>"]
            self.eos_id = stoi["<eos>"]
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
        # the always-graded end group: <eos> plus, when the vocab has them
        # (vocab v3+; membership checks keep older meta.pkl loadable), the
        # colour-differentiated game-end tokens
        end_names = ("<white-resign>", "<black-resign>", "<white-flag>", "<black-flag>", "<draw>")
        end_ids = [self.eos_id] + [stoi[t] for t in end_names if t in stoi]
        self._end_arr = np.asarray(sorted(end_ids), dtype=np.int64)
        # winner-only grading: the colour-differentiated end tokens name the
        # LOSER; '#' marks mate. .get keeps pre-v3 meta.pkl loadable (those
        # games have no end token and simply grade both sides).
        self._white_loses = {stoi[t] for t in ("<white-resign>", "<white-flag>") if t in stoi}
        self._black_loses = {stoi[t] for t in ("<black-resign>", "<black-flag>") if t in stoi}
        self.mate_id = stoi.get("#")


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
    """Cut a token array into per-game-span segment lists.

    On a whole-game row this yields exactly one span (the <bos> at index 0
    opens it); the multi-span path remains only as a corruption guard.
    Each span is a non-empty list of segments (t_lo, t_hi, nerf_pos): an
    inclusive range of TARGET token indices into `w`, plus the segment's own
    nerf position (None for the tail segment).
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
                # cut off / malformed: nothing at or past this nerf may be
                # graded (its move would look clean once stripped)
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


def loser_colour(g: np.ndarray, spec: Spec):
    """The losing colour of one whole game: 0 white, 1 black, None unknown.

    Read purely from the game's own tokens: the colour-differentiated end
    token (second-to-last, before <eos>) names the loser; a bare-<eos> game
    containing the mate glyph '#' ended in checkmate, so the loser is the
    side that did NOT play the last move (move colour IS parity of the core
    tokens — every game starts at move 1). <draw>, timeout draws, abandoned
    games, and pre-v3 framings all return None (grade both sides).
    """
    end = int(g[-2]) if len(g) >= 2 else -1
    if end in spec._white_loses:
        return 0
    if end in spec._black_loses:
        return 1
    if spec.mate_id is not None and bool(np.any(g == spec.mate_id)):
        n_moves = int(np.count_nonzero(g < spec.core_count))
        return n_moves % 2  # last mover (parity n_moves-1) mated; other side lost
    return None


def winner_mask(g: np.ndarray, spec: Spec) -> np.ndarray:
    """Per-position grading permission under winner-only training: False on
    every token of a move played by the loser, True everywhere else. Frame
    and end-group positions get parity-arbitrary values here — their grading
    is decided by _unpred_arr/_end_arr in materialize, never by this mask."""
    loser = loser_colour(g, spec)
    if loser is None:
        return np.ones(len(g), dtype=bool)
    is_core = g < spec.core_count
    move_idx = np.cumsum(is_core) - is_core  # cores strictly before p = move number
    return (move_idx % 2) != loser           # a move's nerf/modifiers/core share its parity


def materialize(g: np.ndarray, spec: Spec, chosen: list[tuple], winner_t: np.ndarray | None = None):
    """Build one variable-length (x, y) training row from one whole game.

    Keeps only the chosen segment's nerf, deletes every other nerf (the
    stream closes up), and grades the chosen segment's surviving targets
    plus the end group (<resign>/<draw>/<eos> at the game's end, graded in
    every row): y[p] = -1 unless the target is in the chosen segment or the
    end group, was originally adjacent to its left neighbour (no deletion
    seam), and is not a framing token (<bos>/<elo-*>) the harness always
    supplies. `winner_t` (winner_mask) further restricts segment targets to
    the winning side's moves; the end group ignores it. x/y have length
    len(kept)-1.
    """
    keep = ~np.isin(g, spec._nerf_arr)
    active_t = np.zeros(len(g), dtype=bool)
    for (t0, t1, nerf_pos) in chosen:
        active_t[t0:t1 + 1] = True
        if nerf_pos is not None:
            keep[nerf_pos] = True
    if winner_t is not None:
        active_t &= winner_t
    old_of_new = np.flatnonzero(keep)
    new = g[old_of_new]
    tgt_old = old_of_new[1:]
    adjacent = tgt_old == old_of_new[:-1] + 1
    targets = new[1:]
    graded = ((active_t[tgt_old] | np.isin(targets, spec._end_arr))
              & adjacent & ~np.isin(targets, spec._unpred_arr))
    x = new[:-1].astype(np.int64)
    y = np.where(graded, targets, -1).astype(np.int64)
    return x, y


def build_row(g: np.ndarray, spec: Spec, rng, winner_only: bool | None = None):
    """One whole game -> (x, y). Hard isolation guard first: a row must be
    exactly one game, or games could silently attend across a bad index."""
    if (len(g) < 2 or g[0] != spec.bos_id or g[-1] != spec.eos_id
            or int(np.count_nonzero(g == spec.bos_id)) != 1):
        raise ValueError(
            "game slice is not exactly one whole game (<bos> ... <eos>) — "
            "stale or corrupt .idx.npy game index?"
        )
    if winner_only is None:
        winner_only = WINNER_ONLY
    winner_t = winner_mask(g, spec) if winner_only else None
    return materialize(g, spec, choose(plan(g, spec), rng), winner_t)


# ------------------------------------------------------------- game index

_OFFSETS_CACHE: dict[str, np.ndarray] = {}


def _bin_path(data):
    """Backing file of a memmap (as str), or None for in-memory arrays."""
    name = getattr(data, "filename", None)
    return os.fspath(name) if name is not None else None


def _idx_path(bin_path: str) -> str:
    root, _ = os.path.splitext(bin_path)  # '.../train.bin' -> '.../train.idx.npy'
    return root + ".idx.npy"


def _scan_game_starts(data, bos_id: int) -> np.ndarray:
    """Chunked scan for <bos> positions — the fallback when no index exists.
    Sound because <bos> is emitted only as the frame head of each game."""
    parts = [np.flatnonzero(np.asarray(data[i:i + SCAN_CHUNK]) == bos_id) + i
             for i in range(0, len(data), SCAN_CHUNK)]
    starts = np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64)
    return starts.astype(np.uint64)


def _valid_offsets(off: np.ndarray, n_tokens: int) -> bool:
    return (len(off) >= 2 and int(off[0]) == 0 and int(off[-1]) == n_tokens
            and bool(np.all(np.diff(off.astype(np.int64)) > 0)))


def game_offsets(data, spec: Spec) -> np.ndarray:
    """uint64 game-start offsets for `data`, with a final len(data) sentinel.

    File-backed data: load <split>.idx.npy when present and valid (a stale
    index against a re-packed bin falls through to a rescan), else scan once
    and best-effort re-cache the .npy next to the bin. Cached in-process by
    path — nanoGPT's get_batch re-creates the memmap every call. In-memory
    arrays (tests) are scanned fresh each call.
    """
    path = _bin_path(data)
    if path is not None:
        cached = _OFFSETS_CACHE.get(path)
        if cached is not None and int(cached[-1]) == len(data):
            return cached
    off = None
    if path is not None and os.path.exists(_idx_path(path)):
        loaded = np.load(_idx_path(path)).astype(np.uint64)
        if _valid_offsets(loaded, len(data)):
            off = loaded
    if off is None:
        off = np.concatenate([_scan_game_starts(data, spec.bos_id),
                              np.asarray([len(data)], dtype=np.uint64)])
        if not _valid_offsets(off, len(data)):
            raise ValueError("could not derive game offsets — empty bin, or data does not start with <bos>?")
        if path is not None:
            try:
                np.save(_idx_path(path), off)
            except OSError:
                pass  # read-only mount: keep the offsets in memory only
    if path is not None:
        _OFFSETS_CACHE[path] = off
    return off


# ---------------------------------------------------------- length buckets

def rows_for_cap(cap: int, block_size: int, batch_size: int) -> int:
    """Rows per batch at this cap — keeps tokens/step ~= batch_size *
    block_size, deterministic per cap so the (rows, cap) shape set is fixed."""
    return max(1, min(MAX_ROWS_FACTOR * batch_size, round(batch_size * block_size / cap)))


class Buckets:
    """Length-bucket table for one (bin, block_size, batch_size) combo."""

    def __init__(self, offsets: np.ndarray, block_size: int, batch_size: int):
        caps = [c for c in BUCKET_CAPS if c <= block_size]
        if not caps or caps[-1] != block_size:
            caps.append(block_size)
        self.caps = caps
        self.rows = [rows_for_cap(c, block_size, batch_size) for c in caps]
        lengths = np.diff(offsets.astype(np.int64))
        need = lengths - 1  # a game of L tokens fills L-1 input slots
        bucket_of = np.searchsorted(caps, need, side="left")
        self.n_skipped = int(np.count_nonzero(bucket_of >= len(caps)))
        if self.n_skipped:
            print(f"nerf_batch: skipping {self.n_skipped} games longer than "
                  f"{caps[-1] + 1} tokens (block_size {block_size})")
        self.game_ids = [np.flatnonzero(bucket_of == b) for b in range(len(caps))]
        mass = np.asarray([int(lengths[ids].sum()) for ids in self.game_ids], dtype=np.float64)
        if mass.sum() <= 0:
            raise ValueError("no game fits any bucket cap — wrong bin / block_size?")
        self.probs = mass / mass.sum()


_BUCKETS_CACHE: dict[tuple, Buckets] = {}


def get_buckets(data, offsets: np.ndarray, block_size: int, batch_size: int) -> Buckets:
    path = _bin_path(data)
    if path is None:
        return Buckets(offsets, block_size, batch_size)
    key = (path, len(data), block_size, batch_size)
    bk = _BUCKETS_CACHE.get(key)
    if bk is None:
        bk = _BUCKETS_CACHE[key] = Buckets(offsets, block_size, batch_size)
    return bk


def pad_batch(rows: list, cap: int, eos_id: int):
    """Stack variable-length rows into (n, cap) int64 arrays. x pads with
    <eos> (strictly after the game, causally invisible to it), y with -1."""
    X = np.full((len(rows), cap), eos_id, dtype=np.int64)
    Y = np.full((len(rows), cap), -1, dtype=np.int64)
    for r, (x, y) in enumerate(rows):
        if len(x) > cap:
            raise ValueError(f"row of {len(x)} tokens exceeds bucket cap {cap} — stale bucket table?")
        X[r, :len(x)] = x
        Y[r, :len(y)] = y
    return X, Y


# ---------------------------------------------------------------- sampling

_RNG = None


def _rng():
    global _RNG
    if _RNG is None:  # RANK offset keeps DDP ranks from drawing identical batches
        _RNG = np.random.default_rng(1234 + int(os.environ.get("RANK", "0")))
    return _RNG


def sample_batch_np(data, block_size: int, batch_size: int, spec: Spec, rng):
    """Torch-free core: one complete game per row, length-bucketed.

    Draws a bucket with probability proportional to its games' token mass,
    then rows_for_cap(cap) games uniformly (with replacement) within it.
    Returns numpy (X, Y) of shape (rows, cap).
    """
    offsets = game_offsets(data, spec)
    bk = get_buckets(data, offsets, block_size, batch_size)
    b = int(rng.choice(len(bk.caps), p=bk.probs))
    ids = bk.game_ids[b]
    picks = ids[rng.integers(0, len(ids), size=bk.rows[b])]
    rows = []
    for gi in picks:
        lo, hi = int(offsets[int(gi)]), int(offsets[int(gi) + 1])
        rows.append(build_row(np.asarray(data[lo:hi], dtype=np.int64), spec, rng))
    return pad_batch(rows, bk.caps[b], spec.eos_id)


def sample_batch(data, block_size: int, batch_size: int, meta_path: str):
    """Drop-in replacement for nanoGPT get_batch's window assembly.

    `data` is the uint16 memmap of train.bin/val.bin (its game index lives
    next to it, found via data.filename). Returns CPU torch tensors (x, y)
    of shape (rows, cap) — one whole game per row, <bos> at column 0; y is
    -1 wherever no loss applies (nanoGPT's cross_entropy uses
    ignore_index=-1), including the whole padding tail.
    """
    import torch  # lazy: the unit tests run torch-free

    X, Y = sample_batch_np(data, block_size, batch_size, load_spec(meta_path), _rng())
    return torch.from_numpy(X), torch.from_numpy(Y)
