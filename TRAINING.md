# Training-data plan: evaluated Lichess games → nanoGPT

Goal: a token stream in this repo's vocabulary (see [README](README.md)), with
nerf tokens derived from real engine evaluations, packed into nanoGPT's
`train.bin` / `val.bin` format.

> **Status (2026-07-21): Stages A–D are implemented in
> [`chess-tokeniser/`](chess-tokeniser/README.md).** `build_dataset.py` covers
> filtering + labelling + tokenising (Stages A–C) and `pack.py` covers packing
> (Stage D). It reads the Hugging Face `Lichess/standard-chess-games` parquet via
> DuckDB rather than streaming raw `.pgn.zst` — same games, less plumbing — so
> the Stage A skeleton below is historical. **Vocab v2 is done**: `<bos>`,
> `<eos>`, 13 `<elo-*>` buckets, and an `<elo-any>` "unspecified" sentinel are
> appended in [`js/vocab-data.js`](js/vocab-data.js) (ids 5252–5267, total
> 5,268). `pack.py --elo-dropout` swaps buckets for `<elo-any>` on a fraction of
> games so the model also learns unconditioned play (inference can omit Elo). What
> remains is *running* the training (Stages/milestones 4–6), not building the
> data pipeline.
>
> **Update (2026-07-22): bare-history nerf training is implemented.**
> `train_chess_hf.py` (nanoGPT now pinned) reroutes `get_batch` through
> [`chess-tokeniser/nerf_batch.py`](chess-tokeniser/nerf_batch.py) so a nerf
> token is only ever in context beside the move it labels (see the
> "Bare-history rule" in §5), and the harness keeps `historyTokens` bare to
> match ([`js/app.js`](js/app.js)). Any checkpoint trained before this scheme
> is obsolete — re-run the smoke before trusting any metric.
>
> **Update (2026-07-22, later): vocab v3 game-end tokens + strict
> one-game-per-row batching.** Five colour-differentiated game-end tokens are
> appended: `<white-resign>` (5268), `<black-resign>` (5269), `<white-flag>`
> (5270), `<black-flag>` (5271), `<draw>` (5272) — total **5,273**.
> `build_dataset.py` now carries the Lichess `Termination` header through, and
> `pack.py` emits `… <end-token> <eos>` per `vocab.Vocab._end_id`: the token
> names the LOSER from the Result header — never from parity, since players
> often resign right after their own move (both parities are kept; at play the
> engine only samples its OWN colour's resign token, so opposite-parity
> occurrences are context/gauge signal, not a "resign while winning" hazard).
> Decisive time forfeits get the loser's flag token (training/gauge only —
> never sampled; the harness has no clocks); mates, timeout-draws, and junk
> terminations stay bare. Batching is now **strict one-game-per-row**:
> `nerf_batch.py` samples whole games via the `train/val.idx.npy` indexes
> `pack.py` writes — length-bucketed, right-padded, `(rows, cap)`-shaped
> batches — so **games can never attend to earlier games**, and the end group
> (game-end token + `<eos>`) is graded in every row. The harness offers you a
> draw when the model samples `<draw>` (declining masks it for the rest of the
> game) and resigns on its own resign token. Pre-v3 datasets and checkpoints
> are incompatible: rebuild Stage 1+2 and train from scratch — **never
> `RESUME=1` a pre-v3 checkpoint onto v3 data** (its 5,268-wide embedding
> cannot index the new ids).

## 1. Source: the Lichess open database

- **What:** https://database.lichess.org — monthly `.pgn.zst` dumps of every
  rated standard game (billions of games, ~100M/month currently). License is
  **CC0**, explicitly free for any use.
- **The part we want:** games where someone requested server analysis carry
  inline Stockfish evals in the movetext:
  `1. e4 { [%eval 0.24] } e5 { [%eval 0.17] } ...` — roughly **~6% of games**
  (stated on the database page), so a recent month yields on the order of
  **5–6M evaluated games**. Three to four months ≈ the 15–20M games target for
  the 8-layer model; one month is plenty for the pipeline-validation model.
- **Format facts that make parsing easy:** one game = a block of `[Tag "..."]`
  header lines + ONE line of movetext (Lichess never wraps it). Evals are
  White-POV pawns (`[%eval -1.53]`) or mate distances (`[%eval #-3]`).
- Download: `curl -O https://database.lichess.org/standard/lichess_db_standard_rated_YYYY-MM.pgn.zst`
  (~30 GB/month; torrents available). Never decompress to disk — stream with
  the `zstandard` pip package (pure Python, works on Windows).

## 2. Stage A — filter (PGN.zst → evaluated-games JSONL)

Stream each dump once; keep a game iff:

- movetext contains `%eval` (the entire point);
- speed is blitz / rapid / classical — read it from the `Event` header
  ("Rated Blitz game", ...). Skip bullet/ultrabullet: time-scramble noise, not
  the skill signal we condition on (revisit later if wanted);
- `Termination` is not `Abandoned` / `Rules infraction`;
- both `WhiteElo` / `BlackElo` are numeric;
- ≥ 10 plies;
- every ply has an eval, allowing only the final ply to lack one (the mating
  move gets no eval; anything else missing means partial analysis — drop).

Emit one JSON line per game — this intermediate means later stages never touch
PGN again:

```json
{"w": 1834, "b": 1790, "speed": "blitz",
 "moves": ["e4", "c5", "Nf3", ...],
 "evals": [0.3, 0.35, 0.28, ...]}     // "#3" / "#-2" kept as strings
```

Skeleton (`tools/filter_lichess.py`):

```python
import zstandard, json, io, sys
dctx = zstandard.ZstdDecompressor()
stream = io.TextIOWrapper(dctx.stream_reader(open(sys.argv[1], "rb")), encoding="utf-8")
headers = {}
for line in stream:
    if line.startswith("["):
        k, v = line[1:line.index(' ')], line[line.index('"')+1:line.rindex('"')]
        headers[k] = v
    elif line.startswith("1.") or line.startswith("1..."):
        if "%eval" in line and keep(headers):
            emit(headers, parse_movetext(line))   # regex out SANs + evals
        headers = {}
```

`parse_movetext` is one regex pass: match `(SAN)( \{ \[%eval (V)\] ... \})?`
pairs, strip move numbers and the result token. Validate the regex against a
few hundred games by replaying the SANs with `python-chess` before trusting it
at scale. Expect the filter to be I/O-bound: well under an hour per month on
one core, parallelize by month. Store output as `YYYY-MM.jsonl.zst`
(~2–4 GB/month).

## 3. Stage B — label nerf tokens (Lichess's own thresholds)

Evals are position evals *after* each ply, White-POV. Convert to win% and
label each move by how much it dropped the **mover's** win%:

```python
import math
def win_pct(ev):                     # ev: float pawns, or "#n"/"#-n"
    if isinstance(ev, str):
        return 100.0 if not ev.startswith("#-") else 0.0
    cp = max(-1000, min(1000, ev * 100))
    return 50 + 50 * (2 / (1 + math.exp(-0.00368208 * cp)) - 1)  # lila's curve

# for ply k (0-based; white moves on even k):
#   before = win_pct(evals[k-1])  (k=0: use eval 0.2 for the start position)
#   after  = win_pct(evals[k])
#   drop   = (before - after)          if white moved
#          = (100-before) - (100-after) if black moved
# label:  drop >= 30 -> <blunder> | >= 20 -> <mistake> | >= 10 -> <inaccuracy>
```

That is exactly Lichess's ?!/?/?? algorithm, so the glyphs the harness shows
will mean what players expect. Notes:

- Label **both** sides' moves; the model plays both colors in training.
- The final (mating) ply has no eval after it — a mating move is never a nerf;
  label it clean.
- Recall beats precision here: an unlabelled blunder poisons the model's
  "clean" branch, while an over-flagged decent move just wastes a nerf sample.
  If in doubt, round drops *up* (e.g. treat 29.5 as a blunder).
- **Sanity plot before proceeding:** blunder-rate per move vs player rating.
  It must decrease monotonically from ~800 to ~2500. If it doesn't, the
  labeller is wrong.

## 4. Stage C — tokenize into THIS repo's vocabulary

Sequence layout per game (nerf token *precedes* its move, matching what the
harness feeds back as `historyTokens`):

```
<bos> <elo-1800> <elo-1700>   ['e4'] ['x','Nf3'] ['<blunder>','x','Qf6'] ... <eos>
       (white)    (black)
```

Either Elo slot can instead be `<elo-any>` (the "strength unspecified" sentinel):
`pack.py --elo-dropout` swaps buckets for it on a fraction of games, per side and
independently, so the model learns conditioned, half-conditioned, and fully
unconditioned play — at inference you feed `<elo-any>` to omit/neutralise Elo.

- Move tokens: port `sanToTokens()` from [js/vocab.js](js/vocab.js) to Python
  (it is ~20 lines: two regexes). **Token ids must come from
  [js/vocab-data.js](js/vocab-data.js)** — parse the quoted strings out of
  that file rather than re-deriving them, so Python, the harness, and the
  trained model can never disagree on ids.
- New structural tokens (not yet in the dictionary): `<bos>`, `<eos>`, and
  shared Elo-bucket tokens in 200-point steps (`<elo-u800>`, `<elo-800>` …
  `<elo-2800p>`, ~13 of them). Position disambiguates white's vs black's
  bucket. → extend `tools/build-vocab.mjs` to append them after the nerf
  tokens and adjust the tail-assert in `vocab.js`; vocab becomes ~5,267,
  still `uint16`-friendly. (Freeze this **before** training anything real.)
- Per-game validation while tokenizing: replay SANs with `python-chess`, and
  round-trip `tokens -> san` — any failure drops the game with a logged count.

Budget: ~70 plies × ~1.35 tokens + nerf + framing ≈ **~105 tokens/game**, so
~0.6B tokens per month of dumps (~1.2 GB of uint16).

## 5. Stage D — pack for nanoGPT

Mirror nanoGPT's `data/shakespeare_char/prepare.py` shape: a
`data/chess/prepare.py` that concatenates all game token ids (games separated
by `<eos>`), memmap-appends into `train.bin` / `val.bin` as `np.uint16`, and
writes `meta.pkl` with `vocab_size`, `stoi`, `itos` (so `sample.py` prints
readable token streams).

- **Split by time, not randomly:** e.g. train = 2025-01…05, val = 2025-06.
  Zero leakage debates, and val tracks the real distribution.
- Config to start (from the sizing discussion): validate the pipeline with
  `n_layer=4, n_head=4, n_embd=128, block_size=512, vocab padded to 5,312`
  on ~1M games; then the real run at `n_layer=8, n_head=8, n_embd=512` on
  15–20M games (~2B tokens — single-GPU, hours-to-a-day in bf16).
- Metrics that matter beyond val loss, all cheap to script against a
  checkpoint: (1) legal-move rate of raw samples, (2) **pre-mask legal mass**
  (the purest board-representation probe), (3) blunder-token rate vs
  conditioned Elo bucket, (4) per-move NLL on val (comparable across
  tokenizer changes; per-token loss is not).

### The bare-history rule (nerf tokens in training)

The packed stream stores games **fully annotated**, but the model must never
*rely* on annotated history: at inference the human's moves are never
annotated (there is no engine), and the harness keeps even the computer's own
emitted nerfs out of `historyTokens`. So training enforces, at batch time:

> a nerf token may appear in the context only while the move it labels is
> being predicted (final history token, or mid-move); every later prediction
> sees the game with that nerf removed.

Batching is **strict one-game-per-row**: a training row is ONE complete game,
`<bos>` at position 0 (exactly the layout inference sees), right-padded to a
length-bucket cap (pad targets `-1`, pad inputs `<eos>` — causally invisible
to the game). Games can never attend to another game — there is no other game
in the row. Rows come from the `train/val.idx.npy` game index `pack.py`
writes (with a one-time `<bos>`-scan fallback for older data repos), buckets
are drawn ∝ token mass with fixed rows-per-cap so `torch.compile` sees a
small fixed shape family, and a hard guard raises if an index would ever
slice mid-game.

Within its game, [`nerf_batch.py`](chess-tokeniser/nerf_batch.py) — installed
into nanoGPT's `get_batch` by `train_chess_hf.py` — builds one consistent
view: nerf tokens cut the game into segments (one per annotated move, cut
*after* the move, plus a tail); one segment is chosen uniformly; every other
nerf is deleted; and loss applies to the chosen segment **plus the end group
— the game's final `<resign>`/`<draw>` (when present) and `<eos>`, graded in
every row** (how games end is exactly what the end tokens exist to learn, and
their bare context matches inference). Targets elsewhere, targets across a
deletion seam (a stripped move must not be graded as clean — that would
poison the clean branch), and `<bos>`/`<elo-*>` framing targets are masked to
`-1`, which nanoGPT's `cross_entropy(ignore_index=-1)` skips.
`test_nerf_batch.py` pins the exact hand-worked rows, the isolation guard,
and the index round-trip.

**Winner-only grading** (default on; `-e WINNER_ONLY=0` disables): move
targets played by the game's **loser** are additionally masked to `-1`.
Rationale: the nerf labels are Lichess win-probability deltas, so in an
already-lost position almost nothing gets flagged — grading both sides
teaches the model that hopeless flailing is clean play, which showed up as
severe mid/endgame deterioration in the first full run. The loser is read
from the game's own tokens (`loser_colour`): the colour-differentiated end
token names the loser; a bare-`<eos>` game containing `#` ended in mate, so
the side that didn't play the last move lost; `<draw>` and every other bare
`<eos>` (timeout draws, abandoned, pre-v3 shards) grade both sides. Loser
moves stay in the **context** (the model must condition on opponent play,
never imitate it), and the end group stays graded in every row.

Consequences to plan around: only a fraction of each batch is graded —
winner-only masking roughly halves the graded move targets of decisive games
on top of the segment rule — so budget more iterations than a vanilla run
(smoke: `-e MAX_ITERS=6000`);
**losses are not comparable** across batching schemes (different task,
different averaging set); and the first eval is slow while `torch.compile`
builds the handful of `(rows, cap)` shape variants.

## 6. Milestones

| # | deliverable | check | status |
|---|---|---|---|
| 0 | evaluated games sourced (HF parquet, DuckDB `contains('[%eval')`) | ~6% of games | ✅ |
| 1 | `build_dataset.py` → tokenised parquet | 0 SAN parse errors on real games | ✅ |
| 2 | labeller (`tokeniser.classify`, Lichess win%-drop thresholds) | blunder-rate vs Elo plot is monotone | 🟡 code done; sanity plot not yet scripted |
| 3 | vocab v2 (`<bos>`/`<eos>`/`<elo-*>`) + Python tokenizer (`vocab.py`) | round-trips vs `vocab-data.js` ids | ✅ `test_pack.py` (round-trip + JS↔Python parity) |
| 4 | `pack.py` → `train.bin`/`val.bin`/`meta.pkl` | tiny 4×128 smoke run overfits a shard | 🟡 bins produced + verified; smoke run pending (re-run: pre-bare-history ckpt is obsolete) |
| 4.5 | bare-history nerf batching (`nerf_batch.py`, patched `get_batch`, bare `historyTokens`) | `test_nerf_batch.py` reproduces the hand-worked rows; probe model sees bare history in the harness | ✅ |
| 4.6 | vocab v3 (colour-differentiated `<white/black-resign>`, `<white/black-flag>`, `<draw>`) + strict one-game-per-row bucketed batching + harness draw-offer/resign flow | `test_pack.py` end-token framing (both parities, flags); `test_nerf_batch.py` isolation guard + end-group rows; mock model exercises both flows | ✅ code done; needs Stage 1+2 rebuild & fresh run |
| 5 | 8×512 run on 15–20M games | pre-mask legal mass ≫ 99%; nerf rate tracks Elo | ⬜ |
| 6 | export checkpoint behind `predict(ctx)` (ONNX / transformers.js or a local server) and plug into `window.chessGpt.setModel` | play it in the harness | ⬜ |

Known bias to accept for now: analysis-requested games skew toward engaged
players and decisive games. The scale-up path is labelling unevaluated games
with your own shallow Stockfish pass — reuse Stage B's exact thresholds so
labels stay comparable.
