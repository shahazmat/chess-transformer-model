# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "numpy", "huggingface_hub"]
# ///
"""Training entrypoint for Hugging Face Jobs (a UV script — deps are declared
above and installed by `hf jobs uv run`). It is deliberately thin: it does the
HF-specific glue and delegates the actual transformer to nanoGPT, matching
TRAINING.md.

What it does, in order:
  1. fetch nanoGPT (zip, no git needed in the container; PINNED to a commit),
  2. install the bare-history nerf patch: drop the embedded nerf_batch.py next
     to train.py and reroute get_batch through it, so nerf tokens are only ever
     in context beside the move they label (see chess-tokeniser/nerf_batch.py —
     the canonical copy, byte-identical to the blob below; test_nerf_batch.py
     enforces the sync),
  3. download the packed data (train.bin/val.bin/meta.pkl + *.idx.npy game
     indexes) from your dataset repo into nanoGPT/data/chess/,
  4. (optional) resume from the last checkpoint pushed to your model repo,
  5. write a nanoGPT config and run training,
  6. push the checkpoint (+ meta.pkl) back to your model repo — Jobs storage is
     EPHEMERAL, so anything not pushed is lost when the job ends.

Configure via environment variables (set with `-e` / `-s` on `hf jobs uv run`):
  DATA_REPO   (required)  e.g. you/lichess-chess-tokens   — dataset repo with the bins
  MODEL_REPO  (required)  e.g. you/chess-gpt              — model repo for checkpoints
  HF_TOKEN    (secret)    write token; huggingface_hub reads it automatically
  PROFILE     smoke|full  smoke = tiny 4x128 validation net (default); full = 8x512
  MAX_ITERS   absolute iteration this job stops at (raise it per chunked run)
  TOTAL_ITERS pin the lr-decay horizon to the final goal, so every chunk of a
              chunked run decays on the same schedule (default: MAX_ITERS)
  RESUME      1 to continue from the checkpoint in MODEL_REPO (default 0)
  PUSH_EVERY_MIN  minutes between periodic safety pushes of ckpt.pt while
              training (default 20; 0 disables) — a killed/timed-out job then
              costs at most that much progress instead of everything

NOTE: the bare-history transform grades only a fraction of each batch's
positions (the rest are context or padding), so budget more iterations than a
vanilla run for the same supervision — e.g. -e MAX_ITERS=6000 for a smoke run
— and never compare losses across batching schemes. Batches are strict
one-game-per-row, (rows, cap)-shaped per length bucket: torch.compile sees a
handful of fixed shapes (expect a slow first eval while they compile), and the
tokens_per_iter/MFU banners — which assume the nominal batch_size x block_size
— misreport slightly. NEVER RESUME=1 a pre-vocab-v3 checkpoint onto v3 data:
its 5268-wide embedding cannot index the new game-end token ids.
"""
import io
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile

from huggingface_hub import snapshot_download, upload_folder

DATA_REPO = os.environ["DATA_REPO"]
MODEL_REPO = os.environ["MODEL_REPO"]
PROFILE = os.environ.get("PROFILE", "smoke")
RESUME = os.environ.get("RESUME", "0") == "1"

# nanoGPT is deprecated upstream (frozen); pin it so the get_batch patch below
# can never drift under us.
NANO_COMMIT = "3adf61e154c3fe3fca428ad6bc3818b27a3b8291"
NANO = f"nanoGPT-{NANO_COMMIT}"

# --------------------------------------------------------------------------
# chess-tokeniser/nerf_batch.py, embedded verbatim (this script runs as a
# single file on HF Jobs — there are no sibling files in the container).
# EDIT THE MODULE FILE, then paste it back here; test_nerf_batch.py fails
# loudly if the two copies differ by a single byte.
# --------------------------------------------------------------------------
NERF_BATCH_SRC = r'''
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
     quality-conditioned, and its bare context matches inference.

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


def materialize(g: np.ndarray, spec: Spec, chosen: list[tuple]):
    """Build one variable-length (x, y) training row from one whole game.

    Keeps only the chosen segment's nerf, deletes every other nerf (the
    stream closes up), and grades the chosen segment's surviving targets
    plus the end group (<resign>/<draw>/<eos> at the game's end, graded in
    every row): y[p] = -1 unless the target is in the chosen segment or the
    end group, was originally adjacent to its left neighbour (no deletion
    seam), and is not a framing token (<bos>/<elo-*>) the harness always
    supplies. x/y have length len(kept)-1.
    """
    keep = ~np.isin(g, spec._nerf_arr)
    active_t = np.zeros(len(g), dtype=bool)
    for (t0, t1, nerf_pos) in chosen:
        active_t[t0:t1 + 1] = True
        if nerf_pos is not None:
            keep[nerf_pos] = True
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


def build_row(g: np.ndarray, spec: Spec, rng):
    """One whole game -> (x, y). Hard isolation guard first: a row must be
    exactly one game, or games could silently attend across a bad index."""
    if (len(g) < 2 or g[0] != spec.bos_id or g[-1] != spec.eos_id
            or int(np.count_nonzero(g == spec.bos_id)) != 1):
        raise ValueError(
            "game slice is not exactly one whole game (<bos> ... <eos>) — "
            "stale or corrupt .idx.npy game index?"
        )
    return materialize(g, spec, choose(plan(g, spec), rng))


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
'''

# ------------------------------------------------------------------ 1. nanoGPT
print(f"== fetching nanoGPT @ {NANO_COMMIT[:8]}", flush=True)
if not os.path.isdir(NANO):
    data = urllib.request.urlopen(f"https://github.com/karpathy/nanoGPT/archive/{NANO_COMMIT}.zip").read()
    zipfile.ZipFile(io.BytesIO(data)).extractall(".")

# ------------------------------------------------- 2. bare-history nerf patch
with open(os.path.join(NANO, "nerf_batch.py"), "w") as f:
    f.write(NERF_BATCH_SRC[1:])  # [1:] strips the literal's leading newline

_OLD_GET_BATCH = """    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])"""
_NEW_GET_BATCH = """    # bare-history nerf sampling (nerf_batch.py), STRICT one-game-per-row:
    # each row is one complete game (<bos> at position 0, never two games in
    # one attention window), length-bucketed and right-padded, so batches are
    # (rows, cap)-shaped per bucket. y is -1-masked (ignored by the loss)
    # outside the graded segment + end group, and on all padding.
    import nerf_batch
    x, y = nerf_batch.sample_batch(data, block_size, batch_size, os.path.join(data_dir, 'meta.pkl'))"""

train_py = os.path.join(NANO, "train.py")
with open(train_py) as f:
    train_src = f.read()
if "import nerf_batch" in train_src:
    print("== get_batch already patched", flush=True)
elif _OLD_GET_BATCH in train_src:
    with open(train_py, "w") as f:
        f.write(train_src.replace(_OLD_GET_BATCH, _NEW_GET_BATCH))
    print("== patched get_batch: bare-history nerf sampling", flush=True)
else:
    sys.exit("nanoGPT train.py does not contain the expected get_batch block — "
             "did the pinned commit change? Update _OLD_GET_BATCH in train_chess_hf.py.")

# ------------------------------------------------------------------ 3. data
print(f"== downloading data from {DATA_REPO}", flush=True)
data_dir = os.path.join(NANO, "data", "chess")
os.makedirs(data_dir, exist_ok=True)
snapshot_download(repo_id=DATA_REPO, repo_type="dataset", local_dir=data_dir,
                  allow_patterns=["*.bin", "*.idx.npy", "meta.pkl"])
assert os.path.exists(os.path.join(data_dir, "train.bin")), "train.bin not found in DATA_REPO"

# ------------------------------------------------------------------ 4. resume
init_from = "scratch"
if RESUME:
    print(f"== resuming from {MODEL_REPO}", flush=True)
    out_dir = os.path.join(NANO, "out")
    os.makedirs(out_dir, exist_ok=True)
    snapshot_download(repo_id=MODEL_REPO, repo_type="model", local_dir=out_dir, allow_patterns=["ckpt.pt"])
    if os.path.exists(os.path.join(out_dir, "ckpt.pt")):
        init_from = "resume"

# ------------------------------------------------------------------ 5. config + train
# Two profiles from TRAINING.md: a cheap pipeline-validation net, then the real one.
# vocab_size is read automatically by nanoGPT from data/chess/meta.pkl.
cfg = dict(
    smoke=dict(n_layer=4, n_head=4, n_embd=128, block_size=512, batch_size=32,
               max_iters=2000, lr_decay_iters=2000, eval_interval=250, warmup_iters=100),
    full=dict(n_layer=12, n_head=12, n_embd=768, block_size=512, batch_size=128,
              max_iters=300_000, lr_decay_iters=100_000, eval_interval=1000, warmup_iters=1000),
)[PROFILE]
if "MAX_ITERS" in os.environ:
    cfg["max_iters"] = cfg["lr_decay_iters"] = int(os.environ["MAX_ITERS"])
# Chunked runs: MAX_ITERS is where THIS job stops; TOTAL_ITERS pins the lr
# schedule to the final goal so every chunk decays on the same curve.
if "TOTAL_ITERS" in os.environ:
    cfg["lr_decay_iters"] = int(os.environ["TOTAL_ITERS"])

config = f"""
dataset = 'chess'
out_dir = 'out'
init_from = '{init_from}'
eval_iters = 100
log_interval = 50
always_save_checkpoint = True   # push whatever we have; storage is ephemeral
gradient_accumulation_steps = 1
learning_rate = 1e-3
min_lr = 1e-4
dropout = 0.0
device = 'cuda'
dtype = 'bfloat16'
compile = True
n_layer = {cfg['n_layer']}
n_head = {cfg['n_head']}
n_embd = {cfg['n_embd']}
block_size = {cfg['block_size']}
batch_size = {cfg['batch_size']}
max_iters = {cfg['max_iters']}
lr_decay_iters = {cfg['lr_decay_iters']}
eval_interval = {cfg['eval_interval']}
warmup_iters = {cfg['warmup_iters']}
"""
with open(os.path.join(NANO, "config", "train_chess.py"), "w") as f:
    f.write(config)
print("== config\n" + config, flush=True)

# ----------------------------------------------------- 5b. periodic safety push
# Jobs storage is ephemeral and the final upload only happens if train.py exits
# cleanly — a timeout or crash mid-run would lose everything. So while training,
# push the freshest checkpoint every PUSH_EVERY_MIN minutes. Only push files
# that have sat still for >60s so we never upload a checkpoint mid-torch.save.
push_every = int(os.environ.get("PUSH_EVERY_MIN", "20"))
if push_every > 0:
    import threading
    import time

    def _pusher():
        out = os.path.join(NANO, "out")
        ck = os.path.join(out, "ckpt.pt")
        last = 0.0
        while True:
            time.sleep(60 * push_every)
            try:
                if os.path.exists(ck):
                    mt = os.path.getmtime(ck)
                    if mt > last and time.time() - mt > 60:
                        upload_folder(repo_id=MODEL_REPO, repo_type="model", folder_path=out,
                                      allow_patterns=["ckpt.pt"], commit_message="periodic checkpoint push")
                        last = mt
                        print("== periodic checkpoint push done", flush=True)
            except Exception as e:  # a push hiccup must never kill training
                print(f"== periodic push failed (will retry): {e}", flush=True)

    threading.Thread(target=_pusher, daemon=True).start()

print("== training", flush=True)
subprocess.run([sys.executable, "train.py", "config/train_chess.py"], cwd=NANO, check=True)

# ------------------------------------------------------------------ 6. push checkpoint
print(f"== uploading checkpoint to {MODEL_REPO}", flush=True)
out_dir = os.path.join(NANO, "out")
shutil.copyfile(os.path.join(data_dir, "meta.pkl"), os.path.join(out_dir, "meta.pkl"))  # bundle for sampling
upload_folder(repo_id=MODEL_REPO, repo_type="model", folder_path=out_dir,
              allow_patterns=["ckpt.pt", "meta.pkl"], commit_message=f"checkpoint ({PROFILE}, bare-history)")
print("== done", flush=True)
